"""Fast-path GenBank flatfile scanner.

High-throughput, low-allocation scanner for GenBank flatfiles.
Extracts record metadata (id, name, description), nucleotide sequence,
and target genomic features (CDS, gene, tRNA, rRNA) with minimal object
instantiation and zero-copy string slicing.
"""

from __future__ import annotations

import logging
from os import PathLike
from typing import TextIO

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition, Location, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

log = logging.getLogger(__name__)

_DEFAULT_TARGET_FEATURES = frozenset({"CDS", "gene", "tRNA", "rRNA"})
_STRIP_SEQ_TRANS = str.maketrans("", "", " 0123456789\r\n\t")


def fast_read_genbank(
    source: str | PathLike | TextIO,
    target_types: frozenset[str] | set[str] | None = _DEFAULT_TARGET_FEATURES,
) -> SeqRecord:
    """Fast scanner reading a single GenBank record.

    Extracts record ID, locus name, definition, sequence, and requested
    features. If parsing fails or non-standard formatting is encountered,
    transparently falls back to BioPython's `SeqIO.read()`.

    Args:
        source: File path or open text file handle.
        target_types: Feature types to extract. If None, extracts all features.

    Returns:
        A populated Bio.SeqRecord.SeqRecord.
    """
    try:
        return _scan_genbank_block(source, target_types)
    except Exception as exc:
        log.debug(
            "FastGenBankScanner fallback on %s: %s",
            getattr(source, "name", source),
            exc,
        )
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        return SeqIO.read(source, "genbank")


def _read_content(source: str | PathLike | TextIO) -> str:
    if isinstance(source, (str, PathLike)):
        with open(source, "r", encoding="latin-1", errors="replace") as f:
            return f.read()
    content = source.read()
    return content


def _parse_location(loc_str: str, seq_len: int) -> Location:
    """Fast-path location parsing for simple, complement, and compound features."""
    if ".." in loc_str and "(" not in loc_str:
        start_str, end_str = loc_str.split("..", 1)
        if start_str.isdigit() and end_str.isdigit():
            return SimpleLocation(
                ExactPosition(int(start_str) - 1),
                ExactPosition(int(end_str)),
                strand=1,
            )
    elif (
        loc_str.startswith("complement(")
        and loc_str.endswith(")")
        and "join" not in loc_str
    ):
        inner = loc_str[11:-1]
        if ".." in inner:
            start_str, end_str = inner.split("..", 1)
            if start_str.isdigit() and end_str.isdigit():
                return SimpleLocation(
                    ExactPosition(int(start_str) - 1),
                    ExactPosition(int(end_str)),
                    strand=-1,
                )
    return Location.fromstring(loc_str, seq_len)


def _scan_genbank_block(
    source: str | PathLike | TextIO,
    target_types: frozenset[str] | set[str] | None,
) -> SeqRecord:
    text = _read_content(source)
    if not text.startswith("LOCUS"):
        # Not a standard GenBank file; trigger fallback
        raise ValueError("File does not start with LOCUS")

    feat_idx = text.find("\nFEATURES ")
    orig_idx = text.find("\nORIGIN", feat_idx if feat_idx != -1 else 0)
    end_idx = text.find("\n//", orig_idx if orig_idx != -1 else 0)

    if feat_idx == -1 or orig_idx == -1:
        raise ValueError("Missing FEATURES or ORIGIN section")
    if end_idx == -1:
        end_idx = len(text)

    # 1. Header parsing
    rec_name = ""
    rec_id = ""
    description = ""
    def_parts: list[str] = []
    in_def = False

    header_block = text[:feat_idx]
    for line in header_block.splitlines():
        if line.startswith("LOCUS"):
            parts = line.split()
            if len(parts) >= 2:
                rec_name = parts[1]
                rec_id = rec_name
            in_def = False
        elif line.startswith("DEFINITION"):
            def_parts.append(line[10:].strip())
            in_def = True
        elif line.startswith("ACCESSION"):
            parts = line.split()
            if len(parts) >= 2 and (not rec_id or rec_id == rec_name):
                rec_id = parts[1]
            in_def = False
        elif line.startswith("VERSION"):
            parts = line.split()
            if len(parts) >= 2:
                rec_id = parts[1]
            in_def = False
        elif in_def:
            if line.startswith("   ") or line.startswith("\t"):
                def_parts.append(line.strip())
            else:
                in_def = False

    if def_parts:
        description = " ".join(def_parts).strip()
        if description.endswith("."):
            description = description[:-1]

    # 2. Sequence translation
    seq_start = text.find("\n", orig_idx + 1)
    if seq_start == -1:
        seq_start = orig_idx + 7
    seq_str = text[seq_start:end_idx].translate(_STRIP_SEQ_TRANS).upper()
    seq_len = len(seq_str)

    # 3. Features extraction
    features: list[SeqFeature] = []
    feat_text = text[feat_idx:orig_idx]
    feat_lines = feat_text.splitlines()

    curr_type: str | None = None
    curr_loc_str: str = ""
    curr_quals: dict[str, list[str]] = {}
    curr_qual_key: str | None = None
    curr_qual_val_parts: list[str] = []
    in_quoted_qualifier = False
    keep_curr_feature = False

    def flush_qual():
        nonlocal curr_qual_key, curr_qual_val_parts, in_quoted_qualifier
        if curr_qual_key is not None:
            val = " ".join(curr_qual_val_parts).strip()
            if val.startswith('"') and val.endswith('"') and len(val) >= 2:
                val = val[1:-1]
            elif val.startswith('"'):
                val = val[1:]
            elif val.endswith('"'):
                val = val[:-1]
            curr_quals.setdefault(curr_qual_key, []).append(val)
            curr_qual_key = None
            curr_qual_val_parts = []
            in_quoted_qualifier = False

    def flush_feat():
        nonlocal curr_type, curr_loc_str, curr_quals, keep_curr_feature
        flush_qual()
        if curr_type is not None and keep_curr_feature:
            loc = _parse_location(curr_loc_str, seq_len)
            features.append(
                SeqFeature(location=loc, type=curr_type, qualifiers=curr_quals)
            )
        curr_type = None
        curr_loc_str = ""
        curr_quals = {}
        keep_curr_feature = False

    for line in feat_lines[1:]:
        if line.startswith("     ") and len(line) > 5 and line[5] != " ":
            flush_feat()
            feature_type = line[5:21].strip()
            keep_curr_feature = target_types is None or feature_type in target_types
            curr_type = feature_type
            curr_loc_str = line[21:].strip()
            continue

        if not keep_curr_feature:
            continue

        stripped = line[21:].strip() if len(line) >= 21 else line.strip()
        if not stripped:
            continue

        if in_quoted_qualifier:
            if stripped.endswith('"') and not stripped.endswith('\\"'):
                curr_qual_val_parts.append(stripped[:-1])
                flush_qual()
            else:
                curr_qual_val_parts.append(stripped)
            continue

        if stripped.startswith("/"):
            flush_qual()
            if "=" in stripped:
                k, v = stripped[1:].split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith('"'):
                    if v.endswith('"') and len(v) > 1:
                        curr_quals.setdefault(k, []).append(v[1:-1])
                    else:
                        curr_qual_key = k
                        curr_qual_val_parts = [v[1:]]
                        in_quoted_qualifier = True
                else:
                    curr_quals.setdefault(k, []).append(v)
            else:
                curr_quals.setdefault(stripped[1:].strip(), []).append("")
        elif curr_qual_key is None:
            curr_loc_str += stripped

    flush_feat()

    return SeqRecord(
        Seq(seq_str),
        id=rec_id,
        name=rec_name,
        description=description,
        features=features,
    )
