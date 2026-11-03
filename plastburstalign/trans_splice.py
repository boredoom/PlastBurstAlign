"""Trans-spliced plastid CDS reconstruction (rps12).

Assembles split / join-annotated ribosomal protein S12 exons into one
canonical ``rps12`` coding sequence before storage. IGS exon splitting
(:class:`~plastburstalign.extraction_ops.ExonSpliceHandler`) is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .qc import (
    ERR_RPS12_AMBIGUOUS_PARTS,
    ERR_RPS12_INCOMPLETE_EXONS,
    ERR_RPS12_JOIN_PARSE,
    location_raw_coords,
)
from .seqfeature_ops import GeneFeature, PlastidFeature

# Cleaned / raw gene names: optional 5'/3' prefix, optional a/b/c suffix.
_RPS12_NAME_RE = re.compile(
    r"^(?:5[_']?|3[_']?)?rps12[abc]?$",
    re.IGNORECASE,
)
_PRODUCT_S12_RE = re.compile(r"ribosomal\s+protein\s+s12", re.IGNORECASE)

# Length envelopes (±~20% of synth truth 114 / 232 / 26).
_EXON1_RANGE = (90, 140)
_EXON2_RANGE = (185, 280)  # ~232 ±20%
_EXON3_RANGE = (20, 35)  # ~26 ±20%
_EXON23_RANGE = (230, 280)  # merged 2+3 (~258)
_FULL_RANGE = (300, 450)  # assembled ~372


@dataclass
class Rps12Part:
    """One extracted rps12-related CDS fragment."""

    feature: SeqFeature
    seq: Seq
    start: int
    end: int
    strand: int | None
    coords: str


@dataclass
class Rps12AssemblyResult:
    """Outcome of :meth:`Rps12Reconstructor.reconstruct`."""

    success: bool
    gene: GeneFeature | None = None
    reason_code: str | None = None
    raw_coords: str = ""
    observed_len: int = 0
    consumed_features: list[SeqFeature] = field(default_factory=list)


class Rps12Reconstructor:
    """Detect and assemble plastid trans-spliced ``rps12`` exons."""

    @classmethod
    def is_candidate(cls, feature: SeqFeature, cleaned_name: str | None = None) -> bool:
        """Return True when a CDS looks like an rps12 (fragment or full)."""
        name = cleaned_name
        if not name:
            name = PlastidFeature.name_cleaner.get_safe_gene(feature) or ""
        raw = PlastidFeature.name_cleaner.get_gene(feature) or ""
        if _RPS12_NAME_RE.match(str(name)) or _RPS12_NAME_RE.match(str(raw)):
            return True
        quals = getattr(feature, "qualifiers", None) or {}
        if "trans_splicing" in quals:
            # Only claim trans_splicing when the gene/product also points at s12,
            # or the cleaned name already matched above.
            product = " ".join(str(p) for p in (quals.get("product") or []))
            if _PRODUCT_S12_RE.search(product) or _RPS12_NAME_RE.match(str(name)):
                return True
        product = " ".join(str(p) for p in (quals.get("product") or []))
        if _PRODUCT_S12_RE.search(product):
            return True
        return False

    @classmethod
    def reconstruct(
        cls,
        record: SeqRecord,
        features: Sequence[SeqFeature],
    ) -> Rps12AssemblyResult:
        """Assemble candidate CDS parts into one ``rps12`` GeneFeature.

        Returns a failed result (with reason code) when parts are present but
        cannot be assembled. Returns ``success=False`` and no reason when no
        candidates exist (caller treats as no-op).
        """
        candidates = [f for f in features if f.type == "CDS" and cls.is_candidate(f)]
        if not candidates:
            return Rps12AssemblyResult(success=False)

        parts: list[Rps12Part] = []
        for feature in candidates:
            try:
                extracted = feature.extract(record).seq
            except Exception:
                return Rps12AssemblyResult(
                    success=False,
                    reason_code=ERR_RPS12_JOIN_PARSE,
                    raw_coords=location_raw_coords(getattr(feature, "location", None)),
                    observed_len=0,
                    consumed_features=list(candidates),
                )
            if extracted is None:
                return Rps12AssemblyResult(
                    success=False,
                    reason_code=ERR_RPS12_JOIN_PARSE,
                    raw_coords=location_raw_coords(getattr(feature, "location", None)),
                    observed_len=0,
                    consumed_features=list(candidates),
                )
            start, end, strand = _span(feature)
            parts.append(
                Rps12Part(
                    feature=feature,
                    seq=Seq(str(extracted)),
                    start=start,
                    end=end,
                    strand=strand,
                    coords=location_raw_coords(feature.location),
                )
            )

        # Prefer an already-complete join (~372 bp); keep lowest-start IR copy.
        full = [p for p in parts if _in_range(len(p.seq), _FULL_RANGE)]
        if full:
            full.sort(key=lambda p: p.start)
            chosen = full[0]
            return cls._finalize(
                record,
                chosen.seq,
                consumed=candidates,
                coords=";".join(p.coords for p in full),
            )

        exon1 = _unique_by_start(
            [p for p in parts if _in_range(len(p.seq), _EXON1_RANGE)]
        )
        exon3 = _unique_by_start(
            [p for p in parts if _in_range(len(p.seq), _EXON3_RANGE)]
        )
        # Mid-length 3′ pieces: ~232 (exon2) vs ~258 (merged exon2+3). Ranges
        # overlap, so prefer merged-block classification only when exon3 is absent.
        mid = [
            p
            for p in parts
            if p not in exon1 and p not in exon3 and _in_range(len(p.seq), _EXON2_RANGE)
        ]
        if exon3:
            exon2 = _unique_by_start(mid)
            exon23: list[Rps12Part] = []
        else:
            exon23 = _unique_by_start(
                [p for p in mid if _in_range(len(p.seq), _EXON23_RANGE)]
            )
            exon2 = _unique_by_start(
                [p for p in mid if not _in_range(len(p.seq), _EXON23_RANGE)]
            )

        if len(exon1) != 1:
            if not exon1:
                return Rps12AssemblyResult(
                    success=False,
                    reason_code=ERR_RPS12_INCOMPLETE_EXONS,
                    raw_coords=";".join(p.coords for p in parts),
                    observed_len=sum(len(p.seq) for p in parts),
                    consumed_features=list(candidates),
                )
            return Rps12AssemblyResult(
                success=False,
                reason_code=ERR_RPS12_AMBIGUOUS_PARTS,
                raw_coords=";".join(p.coords for p in parts),
                observed_len=sum(len(p.seq) for p in parts),
                consumed_features=list(candidates),
            )

        e1 = exon1[0]
        if len(exon23) == 1 and not exon2 and not exon3:
            # Split merged 3′ block into exon2 (~232) + exon3 (~26) when possible.
            block = str(exon23[0].seq)
            if len(block) >= 250:
                e2_seq, e3_seq = block[:-26], block[-26:]
            else:
                # Ambiguous split point — treat whole block as exon2+3 concat
                assembled = Seq(str(e1.seq) + block)
                return cls._finalize(
                    record,
                    assembled,
                    consumed=candidates,
                    coords=f"{e1.coords};{exon23[0].coords}",
                )
            assembled = Seq(str(e1.seq) + e2_seq + e3_seq)
            return cls._finalize(
                record,
                assembled,
                consumed=candidates,
                coords=f"{e1.coords};{exon23[0].coords}",
            )

        if len(exon2) == 1 and len(exon3) == 1:
            assembled = Seq(str(e1.seq) + str(exon2[0].seq) + str(exon3[0].seq))
            return cls._finalize(
                record,
                assembled,
                consumed=candidates,
                coords=f"{e1.coords};{exon2[0].coords};{exon3[0].coords}",
            )

        if not exon2 and not exon3 and not exon23:
            return Rps12AssemblyResult(
                success=False,
                reason_code=ERR_RPS12_INCOMPLETE_EXONS,
                raw_coords=";".join(p.coords for p in parts),
                observed_len=sum(len(p.seq) for p in parts),
                consumed_features=list(candidates),
            )

        return Rps12AssemblyResult(
            success=False,
            reason_code=ERR_RPS12_AMBIGUOUS_PARTS,
            raw_coords=";".join(p.coords for p in parts),
            observed_len=sum(len(p.seq) for p in parts),
            consumed_features=list(candidates),
        )

    @classmethod
    def _finalize(
        cls,
        record: SeqRecord,
        assembled: Seq,
        *,
        consumed: Iterable[SeqFeature],
        coords: str,
    ) -> Rps12AssemblyResult:
        seq = Seq(str(assembled))
        feature = SeqFeature(
            FeatureLocation(0, len(seq), strand=1),
            type="CDS",
            qualifiers={
                "gene": ["rps12"],
                "transl_table": ["11"],
                "product": ["ribosomal protein S12"],
            },
        )
        shadow = SeqRecord(seq, id=getattr(record, "id", "rps12"), name=record.name)
        gene = GeneFeature(shadow, feature)
        if gene.seq_obj is None:
            reason = gene._qc_reason or ERR_RPS12_JOIN_PARSE
            return Rps12AssemblyResult(
                success=False,
                reason_code=str(reason),
                raw_coords=coords,
                observed_len=len(seq),
                consumed_features=list(consumed),
            )
        return Rps12AssemblyResult(
            success=True,
            gene=gene,
            raw_coords=coords,
            observed_len=len(gene.seq_obj),
            consumed_features=list(consumed),
        )


def classify_ir_copies(
    copies: Sequence[GeneFeature],
    ir_mode: str = "single",
    *,
    length_tol: float = 0.05,
) -> tuple[list[GeneFeature], list[dict[str, Any]]]:
    """Apply IR copy policy to same-locus GeneFeatures for one accession.

    Returns ``(features_to_store, quarantine_event_dicts)``. Event dicts are
    kwargs suitable for :class:`~plastburstalign.qc.QuarantineRecord`.
    """
    from .qc import (
        ACTION_DEDUPLICATED_IR_SECONDARY,
        INFO_IR_SECONDARY_DEDUPLICATED,
    )

    if len(copies) < 2:
        return (list(copies), [])

    ordered = sorted(copies, key=_gene_start)
    # Length / identity gate: keep pairing when any secondary is similar to
    # the canonical (or always when ≥2 names collide — annotation duplicates).
    canonical = ordered[0]
    events: list[dict[str, Any]] = []
    mode = (ir_mode or "single").strip().lower()

    if mode == "both":
        kept: list[GeneFeature] = []
        for idx, gene in enumerate(ordered, start=1):
            base = gene.feat_name
            gene.feat_name = f"{base}_IR{idx}"
            gene.seq_name = f"{gene.feat_name}_{gene.rec_name}"
            kept.append(gene)
        return (kept, [])

    # single: keep lowest-start; quarantine the rest
    kept = [canonical]
    can_coords = _gene_coords(canonical)
    for secondary in ordered[1:]:
        sec_coords = _gene_coords(secondary)
        similar = _length_similar(canonical, secondary, length_tol) or _seq_similar(
            canonical, secondary
        )
        detail = f"canonical={can_coords};secondary={sec_coords}" + (
            "" if similar else ";length_divergent"
        )
        events.append(
            {
                "taxon_id": str(secondary.rec_name or ""),
                "locus_id": str(canonical.feat_name or ""),
                "reason_code": INFO_IR_SECONDARY_DEDUPLICATED,
                "raw_coords": detail,
                "observed_len": int(len(secondary.seq_obj) if secondary.seq_obj else 0),
                "locus_median_len": None,
                "upper_fence": None,
                "action_taken": ACTION_DEDUPLICATED_IR_SECONDARY,
            }
        )
    return (kept, events)


# Alias used by the plan / docs.
IrCopyClassifier = classify_ir_copies


def _in_range(n: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= n <= bounds[1]


def _span(feature: SeqFeature) -> tuple[int, int, int | None]:
    loc = feature.location
    try:
        parts = list(getattr(loc, "parts", None) or [loc])
        starts = [int(p.start) for p in parts]
        ends = [int(p.end) for p in parts]
        strand = getattr(loc, "strand", None)
        return (min(starts), max(ends), strand)
    except Exception:
        return (0, 0, None)


def _unique_by_start(parts: list[Rps12Part]) -> list[Rps12Part]:
    if len(parts) <= 1:
        return parts
    parts = sorted(parts, key=lambda p: p.start)
    # Collapse near-identical IR copies (same length) to the first.
    kept: list[Rps12Part] = []
    for part in parts:
        if any(abs(len(part.seq) - len(k.seq)) <= 2 for k in kept):
            continue
        kept.append(part)
    return kept


def _gene_start(gene: GeneFeature) -> int:
    feat = getattr(gene, "feature", None)
    if feat is None:
        return 0
    start, _, _ = _span(feat)
    return start


def _gene_coords(gene: GeneFeature) -> str:
    feat = getattr(gene, "feature", None)
    if feat is None:
        return ""
    return location_raw_coords(getattr(feat, "location", None))


def _length_similar(a: GeneFeature, b: GeneFeature, tol: float) -> bool:
    la = len(a.seq_obj) if a.seq_obj is not None else 0
    lb = len(b.seq_obj) if b.seq_obj is not None else 0
    if la <= 0 or lb <= 0:
        return False
    return abs(la - lb) / max(la, lb) <= tol


def _seq_similar(a: GeneFeature, b: GeneFeature) -> bool:
    if a.seq_obj is None or b.seq_obj is None:
        return False
    sa, sb = str(a.seq_obj).upper(), str(b.seq_obj).upper()
    if sa == sb:
        return True
    try:
        return sa == str(Seq(sb).reverse_complement())
    except Exception:
        return False
