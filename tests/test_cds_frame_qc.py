"""CDS frame / internal-stop Tier-1 quarantine (CdsCodingValidator + GeneFeature)."""

from __future__ import annotations

import unittest

from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.qc import (
    ERR_FRAMESHIFT_NON_TRIPLET,
    ERR_INTERNAL_STOP_CODON,
    INFO_PSEUDOGENE_EXCLUDED,
    CdsCodingValidator,
)
from plastburstalign.seqfeature_ops import GeneFeature


def _cds_feature(
    start: int,
    end: int,
    *,
    qualifiers: dict | None = None,
) -> SeqFeature:
    return SeqFeature(
        location=FeatureLocation(start, end, strand=1),
        type="CDS",
        qualifiers=qualifiers or {"gene": ["test"]},
    )


def _record_with_cds(seq: str, feature: SeqFeature) -> SeqRecord:
    return SeqRecord(Seq(seq), id="plastome", name="TaxA")


class CdsCodingValidatorUnitTests(unittest.TestCase):
    def test_non_triplet_without_atg_frameshift(self) -> None:
        # 10 bp, does not start with ATG → frameshift
        seq = Seq("GCTAGCTAGC")
        feat = _cds_feature(0, 10)
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(out)
        self.assertEqual(reason, ERR_FRAMESHIFT_NON_TRIPLET)

    def test_atg_trims_trailing_incomplete_codon(self) -> None:
        # ATG + 7 coding nt → trim 1 trailing base → 9 bp kept
        seq = Seq("ATGAAACCCT")
        feat = _cds_feature(0, 10)
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(reason)
        self.assertEqual(str(out), "ATGAAACCC")
        self.assertEqual(len(out) % 3, 0)

    def test_internal_stop_codon(self) -> None:
        # Met-Ser-Stop-Ala — internal * before a following codon
        seq = Seq("ATGTCCTAAGCT")  # M S * A
        feat = _cds_feature(0, 12)
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(out)
        self.assertEqual(reason, ERR_INTERNAL_STOP_CODON)

    def test_terminal_stop_ok(self) -> None:
        seq = Seq("ATGTCCTAA")  # M S *
        feat = _cds_feature(0, 9)
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(reason)
        self.assertEqual(str(out), "ATGTCCTAA")

    def test_missing_terminal_stop_ok(self) -> None:
        seq = Seq("ATGTCC")  # M S — no stop
        feat = _cds_feature(0, 6)
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(reason)
        self.assertEqual(str(out), "ATGTCC")

    def test_pseudogene_info_excluded(self) -> None:
        seq = Seq("ATGTCCTAA")
        feat = _cds_feature(0, 9, qualifiers={"gene": ["ycf1"], "pseudo": [""]})
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(out)
        self.assertEqual(reason, INFO_PSEUDOGENE_EXCLUDED)

    def test_pseudogene_qualifier_variant(self) -> None:
        seq = Seq("ATGTCCTAA")
        feat = _cds_feature(
            0, 9, qualifiers={"gene": ["ycf1"], "pseudogene": ["unknown"]}
        )
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(out)
        self.assertEqual(reason, INFO_PSEUDOGENE_EXCLUDED)

    def test_codon_start_honored(self) -> None:
        # Leading base then ATG frame of length 9
        seq = Seq("CATGTCCTAA")
        feat = _cds_feature(0, 10, qualifiers={"gene": ["test"], "codon_start": ["2"]})
        out, reason = CdsCodingValidator.validate(seq, feat)
        self.assertIsNone(reason)
        self.assertEqual(str(out), "ATGTCCTAA")


class GeneFeatureFrameQcTests(unittest.TestCase):
    def test_non_triplet_rejects_with_frameshift_code(self) -> None:
        parent = "GCTAGCTAGC" + "A" * 100
        feat = _cds_feature(0, 10, qualifiers={"gene": ["psbA"]})
        gene = GeneFeature(_record_with_cds(parent, feat), feat)
        self.assertIsNone(gene.seq_obj)
        self.assertEqual(gene._qc_reason, ERR_FRAMESHIFT_NON_TRIPLET)

    def test_atg_trim_keeps_sequence(self) -> None:
        cds = "ATGAAACCCT"  # 10 bp → trim to 9
        parent = cds + "A" * 100
        feat = _cds_feature(0, 10, qualifiers={"gene": ["psbA"]})
        gene = GeneFeature(_record_with_cds(parent, feat), feat)
        self.assertIsNotNone(gene.seq_obj)
        self.assertIsNone(gene._qc_reason)
        self.assertEqual(str(gene.seq_obj), "ATGAAACCC")

    def test_internal_stop_rejects(self) -> None:
        cds = "ATGTCCTAAGCT"  # M S * A
        parent = cds + "A" * 100
        feat = _cds_feature(0, 12, qualifiers={"gene": ["psbA"]})
        gene = GeneFeature(_record_with_cds(parent, feat), feat)
        self.assertIsNone(gene.seq_obj)
        self.assertEqual(gene._qc_reason, ERR_INTERNAL_STOP_CODON)

    def test_pseudo_info_excluded(self) -> None:
        cds = "ATGTCCTAA"
        parent = cds + "A" * 100
        feat = _cds_feature(0, 9, qualifiers={"gene": ["ycf1"], "pseudo": [""]})
        gene = GeneFeature(_record_with_cds(parent, feat), feat)
        self.assertIsNone(gene.seq_obj)
        self.assertEqual(gene._qc_reason, INFO_PSEUDOGENE_EXCLUDED)


if __name__ == "__main__":
    unittest.main()
