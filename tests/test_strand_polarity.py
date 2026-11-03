"""Post-clean StrandPolarityAuditor (antisense plastome quarantine)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Bio.Seq import Seq

from plastburstalign.extraction_ops import DataCleaning
from plastburstalign.qc import (
    ACTION_EXCLUDED_FROM_FASTA,
    ERR_ANTISENSE_STRAND_INVERSION,
    QuarantineLedger,
    StrandPolarityAuditor,
)
from plastburstalign.storage import DiskBackedFeatureStore
from plastburstalign.user_parameters import SelectMode


def _valid_cds(n_codons: int = 40, *, seed: str = "ATG") -> str:
    """In-frame CDS without internal stops (table 11)."""
    body = ("TCT", "GCT", "GGT", "TTA")
    codons = [seed if seed == "ATG" else "ATG"]
    i = 0
    while len(codons) < n_codons - 1:
        codons.append(body[i % len(body)])
        i += 1
    codons.append("TAA")
    return "".join(codons)


def _rc(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


class _MockPlastidData:
    def __init__(
        self,
        store: DiskBackedFeatureStore,
        ledger: QuarantineLedger,
        *,
        mode: SelectMode = SelectMode.CDS,
        num_taxa: int = 3,
    ) -> None:
        self.store = store
        self.mode = mode
        self.num_taxa = num_taxa
        self.qc_ledger = ledger
        self.removed: list[str] = []

    def remove_nuc(self, name: str) -> None:
        self.removed.append(name)
        self.store.remove_region(name)


class StrandPolarityUnitTests(unittest.TestCase):
    def test_base_locus_strips_ir_suffix(self) -> None:
        self.assertEqual(StrandPolarityAuditor.base_locus_name("ndhB_IR1"), "ndhB")
        self.assertEqual(StrandPolarityAuditor.base_locus_name("ndhB_IR2"), "ndhB")
        self.assertEqual(StrandPolarityAuditor.base_locus_name("rbcL"), "rbcL")

    def test_prefers_rc_for_inverted_cds(self) -> None:
        fwd = _valid_cds(30)
        self.assertFalse(StrandPolarityAuditor.prefers_reverse_complement(fwd))
        self.assertTrue(StrandPolarityAuditor.prefers_reverse_complement(_rc(fwd)))

    def test_select_eval_regions_prefers_anchors(self) -> None:
        regions = ["ycf1", "rbcL", "psbA_IR1", "ndhF"]
        chosen = StrandPolarityAuditor.select_eval_regions(regions)
        self.assertEqual(chosen, ["rbcL", "psbA_IR1"])

    def test_select_eval_regions_falls_back_to_all(self) -> None:
        regions = ["ycf1", "ndhF"]
        self.assertEqual(StrandPolarityAuditor.select_eval_regions(regions), regions)


class StrandPolarityIntegrationTests(unittest.TestCase):
    def test_clean_in_frame_taxa_no_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskBackedFeatureStore(Path(tmp) / "features.db")
            cds_rbc = _valid_cds(40)
            cds_psb = _valid_cds(35, seed="GTG")
            for taxon in ("TaxA", "TaxB", "TaxC"):
                store.add_record("rbcL", taxon, "nucl", taxon, "", cds_rbc)
                store.add_record("psbA", taxon, "nucl", taxon, "", cds_psb)
                store.add_record("matK", taxon, "nucl", taxon, "", _valid_cds(28))

            ledger = QuarantineLedger()
            data = _MockPlastidData(store, ledger, num_taxa=3)
            cleaner = DataCleaning(
                data,  # type: ignore[arg-type]
                {"min_num_taxa": 2, "min_seq_length": 3, "select_mode": "cds"},
            )
            cleaner.apply_strand_polarity_audit()

            for region in ("rbcL", "psbA", "matK"):
                taxa = [t for t, _ in store.iter_region_lengths(region, "nucl")]
                self.assertEqual(sorted(taxa), ["TaxA", "TaxB", "TaxC"])
            codes = [r.reason_code for r in ledger.records]
            self.assertNotIn(ERR_ANTISENSE_STRAND_INVERSION, codes)
            store.close()

    def test_antisense_inverted_taxon_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskBackedFeatureStore(Path(tmp) / "features.db")
            anchors = {
                "rbcL": _valid_cds(40),
                "psbA": _valid_cds(36),
                "atpB": _valid_cds(32),
                "matK": _valid_cds(30),
            }
            for taxon in ("TaxA", "TaxB", "TaxC"):
                for locus, seq in anchors.items():
                    store.add_record(locus, taxon, "nucl", taxon, "", seq)
                # Extra non-anchor CDS also present
                store.add_record("ycf1", taxon, "nucl", taxon, "", _valid_cds(50))

            # Invert all CDS for TaxB (antisense plastome)
            for locus in list(anchors) + ["ycf1"]:
                seq = next(
                    s
                    for t, s in store.iter_region_sequences(locus, "nucl")
                    if t == "TaxB"
                )
                store.delete_sequence(locus, "TaxB", "nucl")
                store.add_record(locus, "TaxB", "nucl", "TaxB", "", _rc(seq))

            ledger = QuarantineLedger()
            data = _MockPlastidData(store, ledger, num_taxa=3)
            cleaner = DataCleaning(
                data,  # type: ignore[arg-type]
                {"min_num_taxa": 2, "min_seq_length": 3, "select_mode": "cds"},
            )
            deleted = cleaner.apply_strand_polarity_audit()
            self.assertGreater(deleted, 0)

            for locus in list(anchors) + ["ycf1"]:
                taxa = [t for t, _ in store.iter_region_lengths(locus, "nucl")]
                self.assertNotIn("TaxB", taxa)
                self.assertEqual(sorted(taxa), ["TaxA", "TaxC"])

            anti = [
                r
                for r in ledger.records
                if r.reason_code == ERR_ANTISENSE_STRAND_INVERSION
            ]
            self.assertTrue(anti)
            self.assertTrue(all(r.taxon_id == "TaxB" for r in anti))
            self.assertTrue(
                all(r.action_taken == ACTION_EXCLUDED_FROM_FASTA for r in anti)
            )
            store.close()

    def test_mode_invariance_igs_skips_auditor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskBackedFeatureStore(Path(tmp) / "features.db")
            good = _valid_cds(40)
            bad = _rc(good)
            for taxon, seq in (("TaxA", good), ("TaxB", bad), ("TaxC", good)):
                store.add_record("rbcL", taxon, "nucl", taxon, "", seq)
                store.add_record("psbA", taxon, "nucl", taxon, "", seq)

            ledger = QuarantineLedger()
            data = _MockPlastidData(store, ledger, mode=SelectMode.IGS, num_taxa=3)
            cleaner = DataCleaning(
                data,  # type: ignore[arg-type]
                {"min_num_taxa": 2, "min_seq_length": 3, "select_mode": "igs"},
            )
            with mock.patch.object(
                StrandPolarityAuditor, "audit", wraps=StrandPolarityAuditor.audit
            ) as spy:
                n = cleaner.apply_strand_polarity_audit()
                spy.assert_not_called()
            self.assertEqual(n, 0)
            taxa = [t for t, _ in store.iter_region_lengths("rbcL", "nucl")]
            self.assertIn("TaxB", taxa)
            self.assertNotIn(
                ERR_ANTISENSE_STRAND_INVERSION,
                [r.reason_code for r in ledger.records],
            )
            store.close()

    def test_mode_invariance_int_skips_even_via_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskBackedFeatureStore(Path(tmp) / "features.db")
            good = _valid_cds(40)
            for taxon, seq in (
                ("TaxA", good),
                ("TaxB", _rc(good)),
                ("TaxC", good),
            ):
                store.add_record("spacer", taxon, "nucl", taxon, "", seq)

            ledger = QuarantineLedger()
            data = _MockPlastidData(store, ledger, mode=SelectMode.INT, num_taxa=3)
            cleaner = DataCleaning(
                data,  # type: ignore[arg-type]
                {"min_num_taxa": 2, "min_seq_length": 3, "select_mode": "int"},
            )
            cleaner.clean()
            self.assertNotIn(
                ERR_ANTISENSE_STRAND_INVERSION,
                [r.reason_code for r in ledger.records],
            )
            taxa = [t for t, _ in store.iter_region_lengths("spacer", "nucl")]
            self.assertEqual(sorted(taxa), ["TaxA", "TaxB", "TaxC"])
            store.close()

    def test_locus_deactivation_when_n_equals_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskBackedFeatureStore(Path(tmp) / "features.db")
            # Two taxa only: removing the inverted one leaves N=1 → deactivate
            seqs = {
                "rbcL": _valid_cds(40),
                "psbA": _valid_cds(36),
                "atpB": _valid_cds(32),
            }
            for locus, seq in seqs.items():
                store.add_record(locus, "TaxGood", "nucl", "TaxGood", "", seq)
                store.add_record(locus, "TaxBad", "nucl", "TaxBad", "", _rc(seq))

            ledger = QuarantineLedger()
            data = _MockPlastidData(store, ledger, num_taxa=2)
            cleaner = DataCleaning(
                data,  # type: ignore[arg-type]
                {"min_num_taxa": 2, "min_seq_length": 3, "select_mode": "cds"},
            )
            cleaner.apply_strand_polarity_audit()

            self.assertTrue(
                any(
                    r.reason_code == ERR_ANTISENSE_STRAND_INVERSION
                    for r in ledger.records
                )
            )
            for locus in seqs:
                self.assertIn(locus, data.removed)
                self.assertFalse(store.has_region(locus))
            store.close()


if __name__ == "__main__":
    unittest.main()
