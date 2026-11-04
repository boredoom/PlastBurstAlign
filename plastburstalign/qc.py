"""Fault-tolerant poison-pill isolation and statistical QC.

Four tiers:
  1. Coordinate / topology validation at extract time
  2. Per-locus Tukey length outlier quarantine (post DataCleaning)
  3. Auditable TSV + ``qc_audit`` manifest block
  4. Aligner subprocess deadlines (records returned via task payload)

``QuarantineLedger`` is **main-process only**. Alignment / extraction workers
must return :class:`QuarantineRecord` instances in their task payload; the
coordinator merges them on the parent thread. Do not share a ledger across
process boundaries or protect it with cross-process locks.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Bio.SeqFeature import CompoundLocation, FeatureLocation


# ---------------------------------------------------------------------------
# Reason codes (stable string constants)
# ---------------------------------------------------------------------------

ERR_COORDINATE_OVERRUN = "ERR_COORDINATE_OVERRUN"
ERR_TOPOLOGY_INVERSION = "ERR_TOPOLOGY_INVERSION"
ERR_POPULATION_LENGTH_OUTLIER = "ERR_POPULATION_LENGTH_OUTLIER"
ERR_ALIGNMENT_TIMEOUT = "ERR_ALIGNMENT_TIMEOUT"

# Biological edge-case codes (IR / rps12 / CDS frame / strand).
INFO_IR_SECONDARY_DEDUPLICATED = "INFO_IR_SECONDARY_DEDUPLICATED"
INFO_PSEUDOGENE_EXCLUDED = "INFO_PSEUDOGENE_EXCLUDED"
ERR_RPS12_INCOMPLETE_EXONS = "ERR_RPS12_INCOMPLETE_EXONS"
ERR_RPS12_JOIN_PARSE = "ERR_RPS12_JOIN_PARSE"
ERR_RPS12_AMBIGUOUS_PARTS = "ERR_RPS12_AMBIGUOUS_PARTS"
ERR_FRAMESHIFT_NON_TRIPLET = "ERR_FRAMESHIFT_NON_TRIPLET"
ERR_INTERNAL_STOP_CODON = "ERR_INTERNAL_STOP_CODON"
ERR_ANTISENSE_STRAND_INVERSION = "ERR_ANTISENSE_STRAND_INVERSION"

REASON_CODES = (
    ERR_COORDINATE_OVERRUN,
    ERR_TOPOLOGY_INVERSION,
    ERR_POPULATION_LENGTH_OUTLIER,
    ERR_ALIGNMENT_TIMEOUT,
    INFO_IR_SECONDARY_DEDUPLICATED,
    INFO_PSEUDOGENE_EXCLUDED,
    ERR_RPS12_INCOMPLETE_EXONS,
    ERR_RPS12_JOIN_PARSE,
    ERR_RPS12_AMBIGUOUS_PARTS,
    ERR_FRAMESHIFT_NON_TRIPLET,
    ERR_INTERNAL_STOP_CODON,
    ERR_ANTISENSE_STRAND_INVERSION,
)

# Mission-approved biological envelopes for small-N Tukey fallback (bp).
BIO_ENVELOPE_INTRON_BP = 10_000
BIO_ENVELOPE_CDS_BP = 15_000

#: TSV column order for ``qc/quarantine_report.tsv``.
QUARANTINE_TSV_COLUMNS = (
    "taxon_id",
    "locus_id",
    "reason_code",
    "raw_coords",
    "observed_len",
    "locus_median_len",
    "upper_fence",
    "action_taken",
)

ACTION_REJECTED_AT_EXTRACT = "rejected_at_extract"
ACTION_EXCLUDED_FROM_FASTA = "excluded_from_fasta"
ACTION_MOVED_FAILED_ALIGNMENT = "moved_failed_alignment"
ACTION_DEDUPLICATED_IR_SECONDARY = "deduplicated_ir_secondary"

MAX_SUBPROCESS_TIMEOUT_S = 14400.0  # 4 hours hard ceiling

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuarantineRecord:
    """One auditable quarantine / rejection event."""

    taxon_id: str
    locus_id: str
    reason_code: str
    raw_coords: str
    observed_len: int
    locus_median_len: float | None
    upper_fence: float | None
    action_taken: str

    def to_tsv_row(self) -> dict[str, str]:
        """Serialize for the quarantine TSV (empty string for missing floats)."""

        def _fmt(value: float | None) -> str:
            if value is None:
                return ""
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return ""
            return f"{value:.6g}"

        return {
            "taxon_id": self.taxon_id,
            "locus_id": self.locus_id,
            "reason_code": self.reason_code,
            "raw_coords": self.raw_coords,
            "observed_len": str(int(self.observed_len)),
            "locus_median_len": _fmt(self.locus_median_len),
            "upper_fence": _fmt(self.upper_fence),
            "action_taken": self.action_taken,
        }


@dataclass
class QCAuditSummary:
    """Aggregate counters for the run manifest ``qc_audit`` block."""

    total_features_scanned: int = 0
    total_quarantined: int = 0
    quarantined_by_code: dict[str, int] = field(default_factory=dict)
    report_path: str = "qc/quarantine_report.tsv"

    def note_scan(self, n: int = 1) -> None:
        self.total_features_scanned += max(0, int(n))

    def note_quarantine(self, reason_code: str) -> None:
        self.total_quarantined += 1
        self.quarantined_by_code[reason_code] = (
            self.quarantined_by_code.get(reason_code, 0) + 1
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "total_features_scanned": int(self.total_features_scanned),
            "total_quarantined": int(self.total_quarantined),
            "quarantined_by_code": dict(self.quarantined_by_code),
            "report_path": str(self.report_path),
        }


class QuarantineLedger:
    """Mutable run-scoped quarantine log (main process only).

    Workers must not mutate a shared ledger. Return :class:`QuarantineRecord`
    instances in task payloads and call :meth:`add` / :meth:`extend` on the
    coordinator thread.
    """

    def __init__(self) -> None:
        self.records: list[QuarantineRecord] = []
        self.summary = QCAuditSummary()

    def add(self, record: QuarantineRecord) -> None:
        self.records.append(record)
        self.summary.note_quarantine(record.reason_code)

    def extend(self, records: Sequence[QuarantineRecord]) -> None:
        for record in records:
            self.add(record)

    def note_scan(self, n: int = 1) -> None:
        self.summary.note_scan(n)

    def write_tsv(self, out_dir: str | Path) -> str:
        """Write ``qc/quarantine_report.tsv`` under ``out_dir``; return its path."""
        path = write_quarantine_tsv(out_dir, self.records)
        self.summary.report_path = os.path.relpath(path, start=str(out_dir)).replace(
            "\\", "/"
        )
        return path


# ---------------------------------------------------------------------------
# Tukey fence
# ---------------------------------------------------------------------------


def tukey_upper_fence(
    lengths: Sequence[int],
    *,
    min_n: int = 10,
    k: float = 3.0,
    small_n_envelope: int,
) -> tuple[float, float, float]:
    """Return ``(median, iqr, upper_fence)``.

    * ``N < min_n``: ``fence = small_n_envelope`` (biological envelope fallback).
    * ``IQR == 0``: ``fence = k * median`` (if median is 0, use envelope).
    * else: ``fence = Q3 + k * IQR`` (Tukey extreme outlier rule, ``k=3``).
    """
    values = [int(x) for x in lengths]
    n = len(values)
    if n == 0:
        return (0.0, 0.0, float(small_n_envelope))
    med = float(statistics.median(values))
    if n < min_n:
        return (med, 0.0, float(small_n_envelope))
    # statistics.quantiles(..., n=4) → [Q1, Q2, Q3]
    q1, _q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = float(q3) - float(q1)
    if iqr == 0.0:
        fence = (k * med) if med > 0.0 else float(small_n_envelope)
        return (med, 0.0, float(fence))
    return (med, float(iqr), float(q3) + k * iqr)


def bio_envelope_for_mode(mode: str | None) -> int:
    """Envelope keyed only by select-mode (preferred at Tukey call sites)."""
    key = (mode or "").rsplit(".", 1)[-1].lower()
    if key == "cds":
        return BIO_ENVELOPE_CDS_BP
    return BIO_ENVELOPE_INTRON_BP


def envelope_for_locus(locus_id: str, *, mode: str | None = None) -> int:
    """Pick the small-N biological envelope for a locus name / mode."""
    if mode is not None:
        return bio_envelope_for_mode(mode)
    name = (locus_id or "").lower()
    if "intron" in name or "_" in name or name.startswith(("trn", "rrn")):
        return BIO_ENVELOPE_INTRON_BP
    return BIO_ENVELOPE_CDS_BP


# ---------------------------------------------------------------------------
# Tier-1 validators
# ---------------------------------------------------------------------------


def location_raw_coords(location: Any) -> str:
    """Human-readable coordinate string for quarantine rows."""
    if location is None:
        return ""
    try:
        return str(location)
    except Exception:
        return ""


def feature_span_ok(
    location: Any,
    parent_len: int,
    *,
    topology: str | None,
) -> tuple[bool, str | None, int]:
    """Validate a feature location against the parent replicon.

    Returns ``(ok, reason_code|None, observed_len)``.
    """
    del topology  # topology gates origin wraps at construction; span uses length
    if location is None:
        return (False, ERR_TOPOLOGY_INVERSION, 0)
    parent = max(0, int(parent_len))
    try:
        observed = int(len(location))
    except Exception:
        return (False, ERR_TOPOLOGY_INVERSION, 0)
    if observed < 0:
        return (False, ERR_TOPOLOGY_INVERSION, observed)
    # Unjoined simple location with start > end (or negative coords).
    parts = getattr(location, "parts", None)
    if not parts:
        try:
            start = int(location.start)
            end = int(location.end)
        except Exception:
            return (False, ERR_TOPOLOGY_INVERSION, observed)
        if start < 0 or end < 0 or start > end:
            return (False, ERR_TOPOLOGY_INVERSION, observed)
    else:
        for part in parts:
            try:
                start = int(part.start)
                end = int(part.end)
            except Exception:
                return (False, ERR_TOPOLOGY_INVERSION, observed)
            if start < 0 or end < 0 or start > end:
                return (False, ERR_TOPOLOGY_INVERSION, observed)
    if parent > 0 and observed > parent:
        return (False, ERR_COORDINATE_OVERRUN, observed)
    return (True, None, observed)


def circular_intron_bounds(
    exon1: Any,
    exon2: Any,
    parent_len: int,
    topology: str | None,
) -> FeatureLocation | CompoundLocation | None:
    """Build the minimal-arc intron location between two exon parts.

    Enforces the shortest-arc topological invariant on S^1 circular manifolds,
    invariant to transcript strand polarity and Biopython part indexing order.
    """
    from Bio.SeqFeature import CompoundLocation, FeatureLocation

    try:
        s1, e1 = int(exon1.start), int(exon1.end)
        s2, e2 = int(exon2.start), int(exon2.end)
        strand = getattr(exon1, "strand", None)
        if strand is None:
            strand = getattr(exon2, "strand", None)
    except Exception:
        return None

    parent = int(parent_len)
    if parent <= 0:
        return None

    # Compute candidate linear arc vs wrap arc based on part order
    if s2 >= e1:
        linear_len = s2 - e1
        linear_coords = (e1, s2)
        wrap_len = (parent - e2) + s1
        wrap_coords = ((e2, parent), (0, s1))
    elif s1 >= e2:
        linear_len = s1 - e2
        linear_coords = (e2, s1)
        wrap_len = (parent - e1) + s2
        wrap_coords = ((e1, parent), (0, s2))
    else:
        # Overlapping exons cannot define an internal intron
        return None

    # Intron connects exons across the shortest arc on S^1
    if linear_len <= wrap_len:
        if linear_len <= 0:
            return None
        return FeatureLocation(linear_coords[0], linear_coords[1], strand=strand)

    # Origin wrap: legally valid ONLY on circular replicons
    topo = (topology or "").strip().lower()
    if topo != "circular":
        return None

    if wrap_len <= 0:
        return None

    part_a = FeatureLocation(wrap_coords[0][0], wrap_coords[0][1], strand=strand)
    part_b = FeatureLocation(wrap_coords[1][0], wrap_coords[1][1], strand=strand)
    return CompoundLocation([part_a, part_b])


def record_topology(record: Any) -> str | None:
    """Read ``circular`` / ``linear`` from record annotations if present."""
    ann = getattr(record, "annotations", None) or {}
    topo = ann.get("topology")
    if topo is None:
        return None
    text = str(topo).strip().lower()
    if text in {"circular", "linear"}:
        return text
    return None


# ---------------------------------------------------------------------------
# CDS coding-sequence validator (GeneFeature / CDS extract path only)
# ---------------------------------------------------------------------------


def _qualifier_first(qualifiers: Any, key: str) -> str | None:
    """Return the first string value for a GenBank qualifier key, if any."""
    if not qualifiers:
        return None
    values = qualifiers.get(key)
    if not values:
        return None
    try:
        return str(values[0])
    except (TypeError, IndexError):
        return str(values)


class CdsCodingValidator:
    """Validate CDS nucleotide sequences for frame and premature stops.

    **CDS gating:** intended only for protein-coding CDS sequences on the
    :class:`~plastburstalign.seqfeature_ops.GeneFeature` extract path. RNA /
    IGS / intron modes do not call this validator.

    Checks (in order):

    1. ``/pseudo`` or ``/pseudogene`` → ``INFO_PSEUDOGENE_EXCLUDED`` (audit skip).
    2. Honor ``/codon_start`` (1-based) when present.
    3. Modulo-3: ATG + trailing incomplete codon → legacy trim; else
       ``ERR_FRAMESHIFT_NON_TRIPLET``.
    4. Translate with ``/transl_table`` or default 11; internal ``*`` before
       the terminal codon → ``ERR_INTERNAL_STOP_CODON``. Missing terminal stop
       is allowed.
    """

    @staticmethod
    def validate(nuc_seq: Any, feature: Any) -> tuple[Any | None, str | None]:
        """Validate a CDS coding sequence.

        Returns:
            ``(seq, None)`` on success (``seq`` may be ATG-trimmed), or
            ``(None, reason_code)`` when the feature should be rejected /
            excluded from matrices.
        """
        if nuc_seq is None:
            return (None, ERR_FRAMESHIFT_NON_TRIPLET)

        quals = getattr(feature, "qualifiers", None) or {}
        if "pseudo" in quals or "pseudogene" in quals:
            return (None, INFO_PSEUDOGENE_EXCLUDED)

        seq = nuc_seq
        codon_start_raw = _qualifier_first(quals, "codon_start")
        if codon_start_raw is not None:
            try:
                codon_start = int(codon_start_raw)
            except (TypeError, ValueError):
                codon_start = 1
            if codon_start > 1:
                seq = seq[codon_start - 1 :]

        remainder = len(seq) % 3
        if remainder > 0:
            # Legacy: keep CDS that start with ATG by dropping a trailing
            # incomplete codon; otherwise fail closed.
            if seq[:3] == "ATG":
                seq = seq[:-remainder]
            else:
                return (None, ERR_FRAMESHIFT_NON_TRIPLET)
            if len(seq) % 3 != 0:
                return (None, ERR_FRAMESHIFT_NON_TRIPLET)

        table_raw = _qualifier_first(quals, "transl_table")
        table = 11
        if table_raw is not None:
            try:
                table = int(table_raw)
            except (TypeError, ValueError):
                table = 11

        try:
            from Bio.Seq import Seq

            aa = str(Seq(str(seq)).translate(table=table, cds=False))
        except Exception:
            return (None, ERR_INTERNAL_STOP_CODON)

        body = aa[:-1] if aa.endswith("*") else aa
        if "*" in body:
            return (None, ERR_INTERNAL_STOP_CODON)

        return (seq, None)

CANONICAL_PLASTID_STARTS = frozenset({
    "ATG", "GTG", "TTG", "CTG",  # Canonical Table 11
    "ATT", "ATC", "ATA",         # Isoleucine initiation
    "ACG",                       # Plastid C-to-U RNA editing
})
CANONICAL_STOPS = frozenset({"TAA", "TAG", "TGA"})

def validate_cds_invariants(seq_dna: str) -> tuple[bool, str | None]:
    """Validate CDS integrity against NCBI Genetic Code 11 invariants.

    Returns (is_valid, reason_code).
    """
    if len(seq_dna) % 3 != 0:
        return False, ERR_FRAMESHIFT_NON_TRIPLET

    if len(seq_dna) < 45:  # Minimum functional peptide (< 15 aa)
        return False, "ERR_CDS_SUB_MINIMAL_LENGTH"

    start_codon = seq_dna[:3].upper()
    if start_codon not in CANONICAL_PLASTID_STARTS:
        return False, "ERR_NON_CANONICAL_START_CODON"

    # Translate using NCBI Plastid Genetic Code 11
    from Bio.Seq import Seq
    prot = str(Seq(seq_dna).translate(table=11))

    # Check for internal premature termination
    internal_stops = prot[:-1].count("*")
    if internal_stops > 0:
        return False, ERR_INTERNAL_STOP_CODON

    return True, None


# ---------------------------------------------------------------------------
# Strand polarity auditor (post-DataCleaning, CDS mode only)
# ---------------------------------------------------------------------------


class StrandPolarityAuditor:
    """Detect antisense-inverted plastomes from stored CDS nucleotide sequences.

    Runs **after** Tukey length QC on the SQLite feature store. For each taxon,
    scores forward vs reverse-complement coding potential under NCBI table 11
    across anchor loci (or all active CDS loci when anchors are absent). When
    ≥70% of evaluated loci prefer the reverse complement, every stored CDS row
    for that taxon is surgically deleted and ledgered as
    ``ERR_ANTISENSE_STRAND_INVERSION``.

    No automatic reverse-complement correction is applied (phylogenetic safety).
    """

    ANCHOR_LOCI: tuple[str, ...] = ("rbcL", "psbA", "atpB", "matK")
    PREFERENCE_THRESHOLD: float = 0.70
    _START_CODONS: frozenset[str] = frozenset({"ATG", "GTG"})
    _IR_SUFFIX = ("_IR1", "_IR2", "_ir1", "_ir2")

    @classmethod
    def base_locus_name(cls, region_name: str) -> str:
        """Strip ``_IR1`` / ``_IR2`` suffixes for anchor matching."""
        name = str(region_name)
        for suffix in cls._IR_SUFFIX:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        # Case-insensitive IR suffix (e.g. ``_Ir1``)
        lower = name.lower()
        if lower.endswith("_ir1") or lower.endswith("_ir2"):
            return name[:-4]
        return name

    @classmethod
    def score_coding(cls, nuc_seq: str, *, table: int = 11) -> tuple[int, bool]:
        """Return ``(internal_stop_count, has_valid_start)`` under ``table``."""
        from Bio.Seq import Seq

        s = str(nuc_seq).upper().replace("U", "T")
        if len(s) < 3:
            return (0, False)
        has_start = s[:3] in cls._START_CODONS
        usable = s[: len(s) - (len(s) % 3)]
        if len(usable) < 3:
            return (0, has_start)
        try:
            aa = str(Seq(usable).translate(table=table, cds=False))
        except Exception:
            return (max(1, len(usable) // 3), has_start)
        body = aa[:-1] if aa.endswith("*") else aa
        return (body.count("*"), has_start)

    @classmethod
    def prefers_reverse_complement(cls, nuc_seq: str, *, table: int = 11) -> bool:
        """True when RC is a valid coding frame and forward is not.

        Valid RC: zero internal stops and ATG/GTG start. Forward fails when it
        has multiple internal stops or otherwise fails translation (no start /
        any internal stop).
        """
        from Bio.Seq import Seq

        s_fwd, start_fwd = cls.score_coding(nuc_seq, table=table)
        rc = str(Seq(str(nuc_seq)).reverse_complement())
        s_rev, start_rev = cls.score_coding(rc, table=table)
        rev_valid = s_rev == 0 and start_rev
        fwd_valid = s_fwd == 0 and start_fwd
        fwd_bad = (s_fwd >= 2) or (not fwd_valid)
        return bool(rev_valid and fwd_bad)

    @classmethod
    def select_eval_regions(cls, active_regions: Sequence[str]) -> list[str]:
        """Prefer anchor loci when present; else all active CDS region names."""
        anchors: list[str] = []
        for region in active_regions:
            base = cls.base_locus_name(region)
            if base in cls.ANCHOR_LOCI or any(
                base.startswith(a) for a in cls.ANCHOR_LOCI
            ):
                anchors.append(region)
        return anchors if anchors else list(active_regions)

    @classmethod
    def audit(
        cls,
        store: Any,
        ledger: QuarantineLedger | None,
        *,
        remove_region: Any = None,
        min_num_taxa: int | float | None = 2,
        num_taxa: int | None = None,
    ) -> int:
        """Quarantine antisense taxa from ``store``; return deletion count.

        ``remove_region`` is an optional callable ``(region_name) -> None`` used
        to deactivate loci left with fewer than ``min_num_taxa`` (or N < 2)
        sequences after surgical deletes — same pattern as Tukey QC.
        """
        active = list(store.get_active_regions())
        if not active:
            return 0
        eval_regions = cls.select_eval_regions(active)
        if not eval_regions:
            return 0

        # taxon -> list[(region, sequence)] for evaluation loci
        by_taxon: dict[str, list[tuple[str, str]]] = {}
        for region in eval_regions:
            for taxon, sequence in store.iter_region_sequences(region, "nucl"):
                by_taxon.setdefault(taxon, []).append((region, sequence))

        antisense_taxa: list[str] = []
        for taxon, pairs in by_taxon.items():
            if not pairs:
                continue
            prefer_rc = sum(
                1 for _region, seq in pairs if cls.prefers_reverse_complement(seq)
            )
            if (prefer_rc / len(pairs)) >= cls.PREFERENCE_THRESHOLD:
                antisense_taxa.append(taxon)

        if not antisense_taxa:
            return 0

        antisense_set = set(antisense_taxa)
        deleted = 0
        affected_regions: set[str] = set()
        # Quarantine every stored CDS row for flagged taxa (all active loci).
        for region in list(store.get_active_regions()):
            for taxon, sequence in list(store.iter_region_sequences(region, "nucl")):
                if taxon not in antisense_set:
                    continue
                store.delete_sequence(region, taxon, "nucl")
                store.delete_sequence(region, taxon, "prot")
                deleted += 1
                affected_regions.add(region)
                if ledger is not None:
                    ledger.add(
                        QuarantineRecord(
                            taxon_id=taxon,
                            locus_id=region,
                            reason_code=ERR_ANTISENSE_STRAND_INVERSION,
                            raw_coords="",
                            observed_len=len(sequence),
                            locus_median_len=None,
                            upper_fence=None,
                            action_taken=ACTION_EXCLUDED_FROM_FASTA,
                        )
                    )

        for region in affected_regions:
            remaining = list(store.iter_region_lengths(region, "nucl"))
            if cls._should_deactivate(
                len(remaining), min_num_taxa=min_num_taxa, num_taxa=num_taxa
            ):
                if remove_region is not None:
                    remove_region(region)
                else:
                    store.remove_region(region)

        return deleted

    @staticmethod
    def _should_deactivate(
        remaining: int,
        *,
        min_num_taxa: int | float | None,
        num_taxa: int | None,
    ) -> bool:
        if remaining < 2:
            return True
        if min_num_taxa is None:
            return False
        if isinstance(min_num_taxa, float):
            if num_taxa is None or num_taxa <= 0:
                return False
            return (remaining / num_taxa) < min_num_taxa
        return remaining < int(min_num_taxa)


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


def alignment_timeout_seconds(
    *,
    work: float,
    tier: int,
    alphabet: str,
    leased_threads: int,
    time_model: Any = None,
    min_seconds: float = 60.0,
    max_seconds: float = MAX_SUBPROCESS_TIMEOUT_S,
    slack: float = 3.0,
) -> float:
    """Subprocess deadline clamped to ``[min_seconds, max_seconds]``.

    Calculates ``slack * t_pred`` where ``t_pred`` is the Amdahl wall estimate from
    :class:`~plastburstalign.estimate.TimeModel` plus fixed overhead.
    """
    from .estimate import DEFAULT_TIME_MODEL

    tm = time_model if time_model is not None else DEFAULT_TIME_MODEL
    t1 = float(tm.align_seconds(work, tier, alphabet=alphabet))
    threads = max(1, int(leased_threads))
    capped = max(1, min(threads, int(tm.max_useful_threads)))
    tier_i = min(max(int(tier), 0), len(tm.serial_fraction) - 1)
    serial = float(tm.serial_fraction[tier_i])

    # Amdahl: T(p) = T1 * (s + (1 - s) / p)
    parallel = serial + (1.0 - serial) / float(capped)
    t_pred = t1 * parallel + float(tm.fixed_overhead_s)

    # Clamp strictly between min_seconds and max_seconds to prevent 32-bit epoll overflow
    return float(min(max_seconds, max(min_seconds, slack * t_pred)))


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def write_quarantine_tsv(
    out_dir: str | Path,
    records: Sequence[QuarantineRecord],
    *,
    relative_name: str = "qc/quarantine_report.tsv",
) -> str:
    """Write quarantine rows to ``out_dir/qc/quarantine_report.tsv``."""
    root = Path(out_dir)
    path = root / relative_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(QUARANTINE_TSV_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_tsv_row())
    return str(path)


def quarantine_failed_alignment(
    input_fasta: str,
    out_dir: str | Path,
    *,
    locus_id: str,
    observed_len: int = 0,
    raw_coords: str = "",
) -> QuarantineRecord:
    """Move (copy+remove) a timed-out FASTA into ``quarantine/failed_alignments/``."""
    root = Path(out_dir)
    dest_dir = root / "quarantine" / "failed_alignments"
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(input_fasta)
    dest = dest_dir / src.name
    try:
        if src.exists():
            # Prefer rename within the same filesystem; fall back to copy.
            try:
                os.replace(str(src), str(dest))
            except OSError:
                import shutil

                shutil.copy2(str(src), str(dest))
                try:
                    src.unlink()
                except OSError:
                    pass
    except OSError:
        pass
    return QuarantineRecord(
        taxon_id="*",
        locus_id=locus_id,
        reason_code=ERR_ALIGNMENT_TIMEOUT,
        raw_coords=raw_coords or str(src),
        observed_len=int(observed_len),
        locus_median_len=None,
        upper_fence=None,
        action_taken=ACTION_MOVED_FAILED_ALIGNMENT,
    )


__all__ = [
    "ACTION_DEDUPLICATED_IR_SECONDARY",
    "ACTION_EXCLUDED_FROM_FASTA",
    "ACTION_MOVED_FAILED_ALIGNMENT",
    "ACTION_REJECTED_AT_EXTRACT",
    "BIO_ENVELOPE_CDS_BP",
    "BIO_ENVELOPE_INTRON_BP",
    "CdsCodingValidator",
    "StrandPolarityAuditor",
    "ERR_ALIGNMENT_TIMEOUT",
    "ERR_ANTISENSE_STRAND_INVERSION",
    "ERR_COORDINATE_OVERRUN",
    "ERR_FRAMESHIFT_NON_TRIPLET",
    "ERR_INTERNAL_STOP_CODON",
    "ERR_POPULATION_LENGTH_OUTLIER",
    "ERR_RPS12_AMBIGUOUS_PARTS",
    "ERR_RPS12_INCOMPLETE_EXONS",
    "ERR_RPS12_JOIN_PARSE",
    "ERR_TOPOLOGY_INVERSION",
    "INFO_IR_SECONDARY_DEDUPLICATED",
    "INFO_PSEUDOGENE_EXCLUDED",
    "QCAuditSummary",
    "QUARANTINE_TSV_COLUMNS",
    "QuarantineLedger",
    "QuarantineRecord",
    "REASON_CODES",
    "alignment_timeout_seconds",
    "bio_envelope_for_mode",
    "circular_intron_bounds",
    "envelope_for_locus",
    "feature_span_ok",
    "location_raw_coords",
    "quarantine_failed_alignment",
    "record_topology",
    "tukey_upper_fence",
    "write_quarantine_tsv",
]
