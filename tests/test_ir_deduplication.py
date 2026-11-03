"""IR copy deduplication (``--ir-mode`` single / both)."""

from __future__ import annotations

import unittest

from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.extraction_ops import ExtractAndCollect
from plastburstalign.qc import (
    ACTION_DEDUPLICATED_IR_SECONDARY,
    INFO_IR_SECONDARY_DEDUPLICATED,
    QuarantineRecord,
)
from plastburstalign.seqfeature_ops import PlastidDict
from plastburstalign.trans_splice import classify_ir_copies
from plastburstalign.user_parameters import UserParameters


def _cds(seq: str, start: int, *, gene: str, strand: int = 1) -> tuple[str, SeqFeature]:
    """Return (padded_parent_fragment contribution, feature) for building a record."""
    end = start + len(seq)
    feat = SeqFeature(
        location=FeatureLocation(start, end, strand=strand),
        type="CDS",
        qualifiers={"gene": [gene], "transl_table": ["11"]},
    )
    return seq, feat


def _record_from_parts(
    name: str,
    parts: list[tuple[str, SeqFeature]],
    *,
    genome_len: int | None = None,
) -> SeqRecord:
    """Build a linear SeqRecord with CDS features at absolute coordinates."""
    length = genome_len or max(int(f.location.end) for _, f in parts)
    chars = ["A"] * length
    features: list[SeqFeature] = []
    for seq, feat in parts:
        start = int(feat.location.start)
        end = int(feat.location.end)
        strand = int(feat.location.strand or 1)
        body = seq if strand == 1 else str(Seq(seq).reverse_complement())
        chars[start:end] = list(body)
        features.append(feat)
    return SeqRecord(Seq("".join(chars)), id=name, name=name, features=features)


def _valid_cds(n_codons: int = 40, *, seed: str = "ATG") -> str:
    """In-frame CDS without internal stops (table 11), length ``3 * n_codons``."""
    # Ser / Ala / Gly / Leu — no stop when read in frame.
    body = ("TCT", "GCT", "GGT", "TTA")
    codons = [seed if seed == "ATG" else "ATG"]
    i = 0
    while len(codons) < n_codons - 1:
        codons.append(body[i % len(body)])
        i += 1
    codons.append("TAA")
    return "".join(codons)


def _extractor(
    *,
    ir_mode: str = "single",
    exclude_fullcds: list[str] | None = None,
) -> ExtractAndCollect:
    ext = ExtractAndCollect.__new__(ExtractAndCollect)
    ext.exclude_cds = ["rps12"] if exclude_fullcds is None else list(exclude_fullcds)
    ext.exclude_region = []
    ext.include_trna = True
    ext.include_rrna = True
    ext.min_seq_length = 3
    ext.ir_mode = ir_mode
    ext._quarantine = []
    ext._features_scanned = 0
    ext._report = None
    return ext


class IrDeduplicationUnitTests(unittest.TestCase):
    def test_classify_single_emits_info(self) -> None:
        from plastburstalign.seqfeature_ops import GeneFeature

        cds = _valid_cds(30)
        parent = cds + str(Seq(cds).reverse_complement()) + "A" * 50
        f1 = SeqFeature(
            FeatureLocation(0, len(cds), strand=1),
            type="CDS",
            qualifiers={"gene": ["ndhB"], "transl_table": ["11"]},
        )
        f2 = SeqFeature(
            FeatureLocation(len(cds), 2 * len(cds), strand=-1),
            type="CDS",
            qualifiers={"gene": ["ndhB"], "transl_table": ["11"]},
        )
        rec = SeqRecord(Seq(parent), id="t", name="TaxIR")
        g1, g2 = GeneFeature(rec, f1), GeneFeature(rec, f2)
        self.assertIsNotNone(g1.seq_obj)
        self.assertIsNotNone(g2.seq_obj)
        kept, events = classify_ir_copies([g1, g2], "single")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].feat_name, "ndhB")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reason_code"], INFO_IR_SECONDARY_DEDUPLICATED)
        self.assertEqual(events[0]["action_taken"], ACTION_DEDUPLICATED_IR_SECONDARY)

    def test_classify_both_suffixes(self) -> None:
        from plastburstalign.seqfeature_ops import GeneFeature

        cds = _valid_cds(30)
        parent = cds * 2 + "A" * 20
        f1 = SeqFeature(
            FeatureLocation(0, len(cds), strand=1),
            type="CDS",
            qualifiers={"gene": ["ndhB"], "transl_table": ["11"]},
        )
        f2 = SeqFeature(
            FeatureLocation(len(cds), 2 * len(cds), strand=1),
            type="CDS",
            qualifiers={"gene": ["ndhB"], "transl_table": ["11"]},
        )
        rec = SeqRecord(Seq(parent), id="t", name="TaxIR")
        kept, events = classify_ir_copies(
            [GeneFeature(rec, f1), GeneFeature(rec, f2)], "both"
        )
        self.assertEqual(events, [])
        self.assertEqual({g.feat_name for g in kept}, {"ndhB_IR1", "ndhB_IR2"})


class IrDeduplicationExtractTests(unittest.TestCase):
    def test_dual_ndhb_single_keeps_one_and_quarantines(self) -> None:
        cds = _valid_cds(36)
        # Opposite-strand IR-like pair; first has lower start.
        parts = [
            _cds(cds, 10, gene="ndhB", strand=1),
            _cds(cds, 10 + len(cds) + 50, gene="ndhB", strand=-1),
        ]
        # Rebuild with absolute coords for second feature.
        s1, f1 = parts[0]
        start2 = 10 + len(cds) + 50
        f2 = SeqFeature(
            FeatureLocation(start2, start2 + len(cds), strand=-1),
            type="CDS",
            qualifiers={"gene": ["ndhB"], "transl_table": ["11"]},
        )
        rec = _record_from_parts("TaxDual", [(s1, f1), (cds, f2)])
        ext = _extractor(ir_mode="single")
        nuc, prot = PlastidDict(), PlastidDict()
        ext._extract_cds(rec, nuc, prot)
        self.assertIn("ndhB", nuc)
        self.assertEqual(len(nuc["ndhB"]), 1)
        self.assertNotIn("ndhB_IR1", nuc)
        qc: list[QuarantineRecord] = ext._quarantine
        ir_rows = [r for r in qc if r.reason_code == INFO_IR_SECONDARY_DEDUPLICATED]
        self.assertEqual(len(ir_rows), 1)
        self.assertEqual(ir_rows[0].action_taken, ACTION_DEDUPLICATED_IR_SECONDARY)
        self.assertIn("canonical=", ir_rows[0].raw_coords)
        self.assertIn("secondary=", ir_rows[0].raw_coords)

    def test_dual_ndhb_both_keeps_ir1_ir2(self) -> None:
        cds = _valid_cds(36)
        start2 = 10 + len(cds) + 50
        f1 = SeqFeature(
            FeatureLocation(10, 10 + len(cds), strand=1),
            type="CDS",
            qualifiers={"gene": ["ndhB"], "transl_table": ["11"]},
        )
        f2 = SeqFeature(
            FeatureLocation(start2, start2 + len(cds), strand=-1),
            type="CDS",
            qualifiers={"gene": ["ndhB"], "transl_table": ["11"]},
        )
        rec = _record_from_parts("TaxBoth", [(cds, f1), (cds, f2)])
        ext = _extractor(ir_mode="both")
        nuc = PlastidDict()
        ext._extract_cds(rec, nuc, PlastidDict())
        self.assertNotIn("ndhB", nuc)
        self.assertIn("ndhB_IR1", nuc)
        self.assertIn("ndhB_IR2", nuc)
        self.assertEqual(
            [
                r
                for r in ext._quarantine
                if r.reason_code == INFO_IR_SECONDARY_DEDUPLICATED
            ],
            [],
        )

    def test_single_copy_genome_no_ir_events(self) -> None:
        cds = _valid_cds(20)
        f1 = SeqFeature(
            FeatureLocation(0, len(cds), strand=1),
            type="CDS",
            qualifiers={"gene": ["psbA"], "transl_table": ["11"]},
        )
        rec = _record_from_parts("TaxSingle", [(cds, f1)])
        ext = _extractor(ir_mode="single")
        nuc = PlastidDict()
        ext._extract_cds(rec, nuc, PlastidDict())
        self.assertIn("psbA", nuc)
        self.assertEqual(
            [
                r
                for r in ext._quarantine
                if r.reason_code == INFO_IR_SECONDARY_DEDUPLICATED
            ],
            [],
        )

    def test_user_parameters_ir_mode(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in"
            inp.mkdir()
            (inp / "dummy.gb").write_text("LOCUS\n//\n", encoding="ascii")
            params = UserParameters(
                in_dir=str(inp),
                out_dir=str(Path(tmp) / "out"),
                ir_mode="both",
            )
            self.assertEqual(params["ir_mode"], "both")
            with self.assertRaises(ValueError):
                UserParameters(
                    in_dir=str(inp),
                    out_dir=str(Path(tmp) / "out2"),
                    ir_mode="all",
                )


if __name__ == "__main__":
    unittest.main()
