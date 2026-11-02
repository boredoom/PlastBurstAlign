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

    from .estimate import TimeModel

# ---------------------------------------------------------------------------
# Reason codes (stable string constants)
# ---------------------------------------------------------------------------

ERR_COORDINATE_OVERRUN = "ERR_COORDINATE_OVERRUN"
ERR_TOPOLOGY_INVERSION = "ERR_TOPOLOGY_INVERSION"
ERR_POPULATION_LENGTH_OUTLIER = "ERR_POPULATION_LENGTH_OUTLIER"
ERR_ALIGNMENT_TIMEOUT = "ERR_ALIGNMENT_TIMEOUT"

REASON_CODES = (
    ERR_COORDINATE_OVERRUN,
    ERR_TOPOLOGY_INVERSION,
    ERR_POPULATION_LENGTH_OUTLIER,
    ERR_ALIGNMENT_TIMEOUT,
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
    """Build the short-arc intron location between two exon parts.

    When exons are in genomic order, returns a simple
    ``FeatureLocation(exon1.end, exon2.start)``. When they wrap the origin and
    ``topology == "circular"``, returns a two-part :class:`CompoundLocation`
    spanning the short arc. Otherwise returns ``None``
    (``ERR_TOPOLOGY_INVERSION``).
    """
    from Bio.SeqFeature import CompoundLocation, FeatureLocation

    try:
        e1_end = int(exon1.end)
        e2_start = int(exon2.start)
        strand = getattr(exon1, "strand", None)
        if strand is None:
            strand = getattr(exon2, "strand", None)
    except Exception:
        return None

    parent = int(parent_len)
    if parent <= 0:
        return None

    if e2_start >= e1_end:
        if e2_start == e1_end:
            return None
        return FeatureLocation(e1_end, e2_start, strand=strand)

    # Origin wrap: only legal on circular replicons. The linear alternative
    # FeatureLocation(e2_start, e1_end) is the long-arc poison pill.
    topo = (topology or "").strip().lower()
    if topo != "circular":
        return None
    # Short arc: [e1_end .. parent) + [0 .. e2_start)
    if e1_end > parent or e2_start < 0:
        return None
    if e1_end == parent and e2_start == 0:
        return None
    left = FeatureLocation(e1_end, parent, strand=strand)
    right = FeatureLocation(0, e2_start, strand=strand)
    wrap_len = int(len(left)) + int(len(right))
    if wrap_len <= 0:
        return None
    return CompoundLocation([left, right])


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
# Timeout helper
# ---------------------------------------------------------------------------


def alignment_timeout_seconds(
    *,
    work: float,
    tier: int,
    alphabet: str,
    leased_threads: int,
    time_model: TimeModel | None = None,
    min_seconds: float = 60.0,
    slack: float = 3.0,
) -> float:
    """Subprocess deadline: ``max(min_seconds, slack * t_pred)``.

    ``t_pred`` is the Amdahl wall estimate from :class:`~plastburstalign.estimate.TimeModel`
    plus fixed overhead. Reuses existing serial_fraction / max_useful_threads.
    """
    from .estimate import DEFAULT_TIME_MODEL

    tm = time_model or DEFAULT_TIME_MODEL
    t1 = float(tm.align_seconds(work, tier, alphabet=alphabet))
    threads = max(1, int(leased_threads))
    capped = max(1, min(threads, int(tm.max_useful_threads)))
    tier_i = min(max(int(tier), 0), len(tm.serial_fraction) - 1)
    serial = float(tm.serial_fraction[tier_i])
    # Amdahl: T(p) = T1 * (s + (1 - s) / p)
    parallel = serial + (1.0 - serial) / float(capped)
    t_pred = t1 * parallel + float(tm.fixed_overhead_s)
    return float(max(min_seconds, slack * t_pred))


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
    "ACTION_EXCLUDED_FROM_FASTA",
    "ACTION_MOVED_FAILED_ALIGNMENT",
    "ACTION_REJECTED_AT_EXTRACT",
    "BIO_ENVELOPE_CDS_BP",
    "BIO_ENVELOPE_INTRON_BP",
    "ERR_ALIGNMENT_TIMEOUT",
    "ERR_COORDINATE_OVERRUN",
    "ERR_POPULATION_LENGTH_OUTLIER",
    "ERR_TOPOLOGY_INVERSION",
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
