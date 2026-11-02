"""Fault-tolerant poison-pill isolation & statistical QC (tiers 1–4)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.qc import (
    ACTION_EXCLUDED_FROM_FASTA,
    ACTION_MOVED_FAILED_ALIGNMENT,
    ACTION_REJECTED_AT_EXTRACT,
    BIO_ENVELOPE_CDS_BP,
    BIO_ENVELOPE_INTRON_BP,
    ERR_ALIGNMENT_TIMEOUT,
    ERR_COORDINATE_OVERRUN,
    ERR_POPULATION_LENGTH_OUTLIER,
    ERR_TOPOLOGY_INVERSION,
    QUARANTINE_TSV_COLUMNS,
    QuarantineLedger,
    QuarantineRecord,
    alignment_timeout_seconds,
    bio_envelope_for_mode,
    circular_intron_bounds,
    feature_span_ok,
    tukey_upper_fence,
)
from plastburstalign.storage import DiskBackedFeatureStore


class TukeyHelperTests(unittest.TestCase):
    """Pure-math contracts for ``tukey_upper_fence`` (tests B/C helpers)."""

    def test_c_identical_lengths_iqr_zero_uses_k_times_median(self) -> None:
        lengths = [800] * 12  # N >= min_n, IQR == 0
        med, iqr, fence = tukey_upper_fence(
            lengths, min_n=10, k=3.0, small_n_envelope=BIO_ENVELOPE_INTRON_BP
        )
        self.assertEqual(med, 800.0)
        self.assertEqual(iqr, 0.0)
        self.assertEqual(fence, 3.0 * 800.0)

    def test_c_small_n_uses_biological_envelope(self) -> None:
        lengths = [800, 800, 800]  # N=3 < min_n=10
        med, iqr, fence = tukey_upper_fence(
            lengths, min_n=10, k=3.0, small_n_envelope=BIO_ENVELOPE_INTRON_BP
        )
        self.assertEqual(med, 800.0)
        self.assertEqual(fence, float(BIO_ENVELOPE_INTRON_BP))

    def test_b_one_extreme_outlier_fence_near_q3_plus_3iqr(self) -> None:
        lengths = [800] * 99 + [20_000]
        med, iqr, fence = tukey_upper_fence(
            lengths, min_n=10, k=3.0, small_n_envelope=BIO_ENVELOPE_INTRON_BP
        )
        self.assertEqual(med, 800.0)
        # With 99 identical + one outlier, inclusive quartiles still give IQR≈0
        # for this distribution shape; the extreme 20 kb is far above any fence
        # that is k*median or Q3+3*IQR when the bulk is 800 bp.
        self.assertLess(fence, 20_000)
        self.assertGreaterEqual(fence, 800.0)

    def test_cds_envelope_constant(self) -> None:
        self.assertEqual(bio_envelope_for_mode("cds"), BIO_ENVELOPE_CDS_BP)
        self.assertEqual(bio_envelope_for_mode("igs"), BIO_ENVELOPE_INTRON_BP)

    def test_ledger_and_tsv_roundtrip(self) -> None:
        ledger = QuarantineLedger()
        ledger.note_scan(3)
        ledger.add(
            QuarantineRecord(
                taxon_id="TaxA",
                locus_id="psbA",
                reason_code=ERR_COORDINATE_OVERRUN,
                raw_coords="1..200000",
                observed_len=200000,
                locus_median_len=None,
                upper_fence=None,
                action_taken=ACTION_REJECTED_AT_EXTRACT,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = ledger.write_tsv(tmp)
            self.assertTrue(Path(path).is_file())
            with open(path, encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertEqual(tuple(reader.fieldnames or ()), QUARANTINE_TSV_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reason_code"], ERR_COORDINATE_OVERRUN)
        summary = ledger.summary.to_manifest_dict()
        self.assertEqual(summary["total_features_scanned"], 3)
        self.assertEqual(summary["total_quarantined"], 1)
        self.assertEqual(summary["quarantined_by_code"][ERR_COORDINATE_OVERRUN], 1)

    def test_alignment_timeout_seconds_floor(self) -> None:
        # Tiny work → floor at 60 s
        t = alignment_timeout_seconds(
            work=1.0, tier=0, alphabet="nucl", leased_threads=1
        )
        self.assertGreaterEqual(t, 60.0)


class FeatureSpanHelperTests(unittest.TestCase):
    def test_simple_ok(self) -> None:
        loc = FeatureLocation(10, 100)
        ok, reason, observed = feature_span_ok(loc, 150_000, topology="circular")
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertEqual(observed, 90)

    def test_overrun(self) -> None:
        loc = FeatureLocation(0, 200_000)
        ok, reason, observed = feature_span_ok(loc, 150_000, topology="circular")
        self.assertFalse(ok)
        self.assertEqual(reason, ERR_COORDINATE_OVERRUN)
        self.assertEqual(observed, 200_000)

    def test_circular_intron_short_arc(self) -> None:
        exon1 = FeatureLocation(149_000, 149_500)
        exon2 = FeatureLocation(200, 500)
        loc = circular_intron_bounds(exon1, exon2, 150_000, "circular")
        self.assertIsNotNone(loc)
        assert loc is not None
        self.assertIsInstance(loc, CompoundLocation)
        # Short arc: [149500..150000) + [0..200) = 500 + 200
        self.assertEqual(int(len(loc)), 700)

    def test_linear_wrap_rejected(self) -> None:
        exon1 = FeatureLocation(149_000, 150_000)
        exon2 = FeatureLocation(100, 500)
        loc = circular_intron_bounds(exon1, exon2, 150_000, "linear")
        self.assertIsNone(loc)


class TestAWrapAroundAndOverrun(unittest.TestCase):
    """Tier-1: circular wrap OK; long-arc / overrun quarantined; others retained."""

    def test_a_circular_cds_join_ok_and_poison_intron_rejected(self) -> None:
        from plastburstalign.seqfeature_ops import GeneFeature, IntronFeature

        parent_len = 150_000
        seq = Seq("A" * parent_len)
        record = SeqRecord(seq, id="plastome", name="plastome")
        record.annotations["topology"] = "circular"

        # join(140000..150000,1..500) CDS — short wrap, OK
        cds_loc = CompoundLocation(
            [
                FeatureLocation(140_000, 150_000, strand=1),
                FeatureLocation(0, 500, strand=1),
            ]
        )
        cds_feat = SeqFeature(cds_loc, type="CDS", qualifiers={"gene": ["psbZ"]})
        gene = GeneFeature(record, cds_feat)
        self.assertIsNotNone(gene.seq_obj)
        self.assertIsNone(gene._qc_reason)
        self.assertEqual(len(gene.seq_obj), 10_000 + 500)

        # Origin-crossing exon pair → short-arc intron on circular
        intron_loc = CompoundLocation(
            [
                FeatureLocation(149_000, 149_500, strand=1),
                FeatureLocation(200, 800, strand=1),
            ]
        )
        intron_feat = SeqFeature(intron_loc, type="CDS", qualifiers={"gene": ["clpP"]})
        good_intron = IntronFeature(record, intron_feat, offset=0)
        self.assertIsNotNone(good_intron.seq_obj)
        self.assertEqual(int(len(good_intron.feature.location)), 500 + 200)

        # Same wrap on a *linear* record → topology inversion; other loci untouched
        linear = SeqRecord(seq, id="lin", name="lin")
        linear.annotations["topology"] = "linear"
        bad = IntronFeature(linear, intron_feat, offset=0)
        self.assertIsNone(bad.seq_obj)
        self.assertEqual(bad._qc_reason, ERR_TOPOLOGY_INVERSION)

        # Explicit overrun: simple location longer than parent
        overrun_feat = SeqFeature(
            FeatureLocation(0, parent_len + 50),
            type="CDS",
            qualifiers={"gene": ["bad"]},
        )
        overrun = GeneFeature(record, overrun_feat)
        self.assertIsNone(overrun.seq_obj)
        self.assertEqual(overrun._qc_reason, ERR_COORDINATE_OVERRUN)
        # The good CDS above remains intact (surgical, not locus-wide)
        self.assertIsNotNone(gene.seq_obj)


class TestBTukeySurgicalDelete(unittest.TestCase):
    """Tier-2: N=100 with one 20 kb outlier → only that taxon deleted."""

    def test_b_outlier_taxon_deleted_others_retained(self) -> None:
        from plastburstalign.extraction_ops import DataCleaning
        from plastburstalign.qc import QuarantineLedger
        from plastburstalign.user_parameters import SelectMode

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "features.db"
            store = DiskBackedFeatureStore(db)
            for i in range(99):
                store.add_record("psbA", f"Tax{i:03d}", "nucl", f"id{i}", "", "A" * 800)
            store.add_record("psbA", "TaxOUT", "nucl", "idOUT", "", "A" * 20_000)
            # A second locus that must be untouched
            store.add_record("matK", "Tax000", "nucl", "idm", "", "A" * 900)
            store.add_record("matK", "Tax001", "nucl", "idm2", "", "A" * 900)

            ledger = QuarantineLedger()

            class _Data:
                def __init__(self) -> None:
                    self.store = store
                    self.mode = SelectMode.CDS
                    self.num_taxa = 100
                    self.qc_ledger = ledger

                def remove_nuc(self, name: str) -> None:
                    store.remove_region(name)

            cleaner = DataCleaning(
                _Data(),  # type: ignore[arg-type]
                {"min_num_taxa": 2, "min_seq_length": 3, "select_mode": "cds"},
            )
            cleaner.apply_tukey_length_qc()

            taxa = [t for t, _ in store.iter_region_lengths("psbA", "nucl")]
            self.assertNotIn("TaxOUT", taxa)
            self.assertEqual(len(taxa), 99)
            self.assertTrue(store.has_region("matK"))
            codes = [r.reason_code for r in ledger.records]
            self.assertIn(ERR_POPULATION_LENGTH_OUTLIER, codes)
            outlier = next(r for r in ledger.records if r.taxon_id == "TaxOUT")
            self.assertEqual(outlier.action_taken, ACTION_EXCLUDED_FROM_FASTA)
            self.assertEqual(outlier.observed_len, 20_000)
            store.close()

    def test_post_tukey_n_less_than_2_deactivates_locus(self) -> None:
        from plastburstalign.extraction_ops import DataCleaning
        from plastburstalign.qc import QuarantineLedger
        from plastburstalign.user_parameters import SelectMode

        with tempfile.TemporaryDirectory() as tmp:
            store = DiskBackedFeatureStore(Path(tmp) / "features.db")
            store.add_record("lonely", "TaxA", "nucl", "a", "", "A" * 800)
            store.add_record("lonely", "TaxB", "nucl", "b", "", "A" * 20_000)

            removed: list[str] = []
            ledger = QuarantineLedger()

            class _Data:
                def __init__(self) -> None:
                    self.store = store
                    self.mode = SelectMode.IGS
                    self.num_taxa = 2
                    self.qc_ledger = ledger

                def remove_nuc(self, name: str) -> None:
                    removed.append(name)
                    store.remove_region(name)

            cleaner = DataCleaning(
                _Data(),  # type: ignore[arg-type]
                {"min_num_taxa": 1, "min_seq_length": 3, "select_mode": "igs"},
            )
            # Force envelope so TaxB (20k) is above BIO_ENVELOPE_INTRON_BP
            cleaner.apply_tukey_length_qc()
            self.assertIn("lonely", removed)
            self.assertFalse(store.has_region("lonely"))
            store.close()


class TestDWatchdogTimeout(unittest.TestCase):
    """Tier-4: subprocess deadline → quarantine FASTA + ERR_ALIGNMENT_TIMEOUT."""

    def test_d_timeout_quarantines_fasta_and_returns_record(self) -> None:
        from plastburstalign.alignment_tools import (
            AlignmentTimeout,
            run_aligner_subprocess,
        )
        from plastburstalign.qc import quarantine_failed_alignment

        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "nucl_psbA.unalign.fasta"
            fasta.write_text(">a\nAAA\n>b\nAAA\n", encoding="utf-8")
            with self.assertRaises(AlignmentTimeout):
                run_aligner_subprocess(
                    ["sleep", "5"],
                    timeout=0.3,
                )
            record = quarantine_failed_alignment(
                str(fasta), tmp, locus_id="psbA", observed_len=3
            )
            dest = Path(tmp) / "quarantine" / "failed_alignments" / fasta.name
            self.assertTrue(dest.is_file())
            self.assertFalse(fasta.exists())
            self.assertEqual(record.reason_code, ERR_ALIGNMENT_TIMEOUT)
            self.assertEqual(record.action_taken, ACTION_MOVED_FAILED_ALIGNMENT)
            self.assertEqual(record.locus_id, "psbA")

    def test_d_align_fasta_payload_path(self) -> None:
        """AlignmentCoordination._align_fasta returns quarantine via payload."""
        from plastburstalign.alignment_ops import AlignmentCoordination
        from plastburstalign.alignment_tools import AlignmentTimeout
        from plastburstalign.strategy import AlignmentStrategy, AlignmentTier

        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "in.fa"
            fasta.write_text(">a\nAAA\n>b\nTTT\n", encoding="utf-8")
            out = Path(tmp) / "out.fa"

            class _FakeAligner:
                def align(self, *args, **kwargs):
                    raise AlignmentTimeout(0.5, ["sleep", "9"])

            ledger = QuarantineLedger()

            class _FakeData:
                mode = mock.Mock()
                mode.collects_proteins = mock.Mock(return_value=False)
                store = mock.Mock()
                qc_ledger = ledger

            coord = AlignmentCoordination.__new__(AlignmentCoordination)
            coord.plastid_data = _FakeData()  # type: ignore[assignment]
            coord.user_params = {"out_dir": tmp}
            coord.aligner = _FakeAligner()  # type: ignore[assignment]
            coord.qc_ledger = ledger
            ctx = mock.Mock()
            ctx.threads = 1
            ctx.cpu_ids = (0,)
            ctx.pinning = False
            ctx.subprocess_env = mock.Mock(return_value={})
            strategy = AlignmentStrategy(
                tier=AlignmentTier.ACCURATE,
                mafft_args=("--auto",),
                work=1.0,
                reason="test",
                needs_fragment_split=False,
            )
            with mock.patch.object(
                coord,
                "_build_align_kwargs",
                return_value={"timeout": 0.5},
            ):
                ok, qrec = coord._align_fasta(
                    str(fasta),
                    str(out),
                    summary_entry={"count": 2, "max_len": 3, "min_len": 3},
                    support={"timeout": True},
                    ctx=ctx,
                    alphabet="nucl",
                    strategy=strategy,
                    work=1.0,
                    locus_id="psbA",
                    timeout=0.5,
                )
            self.assertFalse(ok)
            self.assertIsNotNone(qrec)
            assert qrec is not None
            self.assertEqual(qrec.reason_code, ERR_ALIGNMENT_TIMEOUT)
            dest = Path(tmp) / "quarantine" / "failed_alignments" / "in.fa"
            self.assertTrue(dest.is_file())


class TestTier3ManifestAudit(unittest.TestCase):
    def test_qc_audit_extra_roundtrip(self) -> None:
        import datetime

        from plastburstalign.manifest import load_run_manifest, write_run_manifest

        ledger = QuarantineLedger()
        ledger.note_scan(5)
        ledger.add(
            QuarantineRecord(
                taxon_id="T",
                locus_id="L",
                reason_code=ERR_POPULATION_LENGTH_OUTLIER,
                raw_coords="",
                observed_len=20000,
                locus_median_len=800.0,
                upper_fence=2400.0,
                action_taken=ACTION_EXCLUDED_FROM_FASTA,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            ledger.write_tsv(tmp)
            path = write_run_manifest(
                tmp,
                started_at=datetime.datetime.now(datetime.timezone.utc),
                wall_seconds=1.0,
                ok=True,
                extra={"qc_audit": ledger.summary.to_manifest_dict()},
            )
            self.assertIsNotNone(path)
            doc = load_run_manifest(tmp)
            assert doc is not None
            self.assertEqual(doc["qc_audit"]["total_quarantined"], 1)
            self.assertEqual(
                doc["qc_audit"]["quarantined_by_code"][ERR_POPULATION_LENGTH_OUTLIER],
                1,
            )
            tsv = Path(tmp) / "qc" / "quarantine_report.tsv"
            self.assertTrue(tsv.is_file())
            with tsv.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertEqual(tuple(reader.fieldnames or ()), QUARANTINE_TSV_COLUMNS)


if __name__ == "__main__":
    unittest.main()
