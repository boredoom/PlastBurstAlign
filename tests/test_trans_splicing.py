"""Trans-spliced rps12 reconstruction and ``--exclcds none`` sentinel."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.extraction_ops import ExtractAndCollect
from plastburstalign.qc import ERR_RPS12_INCOMPLETE_EXONS
from plastburstalign.seqfeature_ops import PlastidDict
from plastburstalign.trans_splice import Rps12Reconstructor
from plastburstalign.user_parameters import UserParameters


def _coding_exons() -> tuple[str, str, str]:
    """Return exon1 (114), exon2 (232), exon3 (26) that concat to a valid CDS."""
    # 124 codons = 372 nt. Build then split at synth boundaries 114 / 232 / 26
    # but biological order is exon1 + exon2 + exon3 (114+232+26).
    body = ("TCT", "GCT", "GGT", "TTA")
    codons = ["ATG"]
    while len(codons) < 123:
        codons.append(body[(len(codons) - 1) % len(body)])
    codons.append("TAA")
    full = "".join(codons)
    assert len(full) == 372
    return full[:114], full[114:346], full[346:]


def _extractor(
    *,
    exclude_fullcds: list[str] | None = None,
    ir_mode: str = "single",
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


class Rps12ReconstructorUnitTests(unittest.TestCase):
    def test_full_114_232_26_assembles_372(self) -> None:
        e1, e2, e3 = _coding_exons()
        # Place exons at disjoint genomic coords (not biological order).
        s3, s1, s2 = 50, 100, 250
        features = [
            SeqFeature(
                FeatureLocation(s1, s1 + len(e1), strand=1),
                type="CDS",
                qualifiers={"gene": ["5_rps12"], "transl_table": ["11"]},
            ),
            SeqFeature(
                FeatureLocation(s2, s2 + len(e2), strand=1),
                type="CDS",
                qualifiers={"gene": ["rps12"], "transl_table": ["11"]},
            ),
            SeqFeature(
                FeatureLocation(s3, s3 + len(e3), strand=1),
                type="CDS",
                qualifiers={"gene": ["3_rps12"], "transl_table": ["11"]},
            ),
        ]
        chars = list("A" * (s2 + len(e2) + 40))
        chars[s1 : s1 + len(e1)] = list(e1)
        chars[s2 : s2 + len(e2)] = list(e2)
        chars[s3 : s3 + len(e3)] = list(e3)
        rec = SeqRecord(Seq("".join(chars)), id="t", name="TaxRps", features=features)
        result = Rps12Reconstructor.reconstruct(rec, features)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.gene)
        self.assertEqual(result.gene.feat_name, "rps12")
        self.assertEqual(len(result.gene.seq_obj), 372)
        self.assertEqual(str(result.gene.seq_obj), e1 + e2 + e3)

    def test_missing_exon1_incomplete(self) -> None:
        e1, e2, e3 = _coding_exons()
        block = e2 + e3
        features = [
            SeqFeature(
                FeatureLocation(0, len(block), strand=1),
                type="CDS",
                qualifiers={
                    "gene": ["3_rps12"],
                    "transl_table": ["11"],
                    "trans_splicing": [""],
                },
            )
        ]
        rec = SeqRecord(
            Seq(block + "A" * 10), id="t", name="TaxMiss", features=features
        )
        result = Rps12Reconstructor.reconstruct(rec, features)
        self.assertFalse(result.success)
        self.assertEqual(result.reason_code, ERR_RPS12_INCOMPLETE_EXONS)

    def test_split_labels_collapse_to_rps12(self) -> None:
        e1, e2, e3 = _coding_exons()
        block = e2 + e3  # 258 bp 3′ block
        features = [
            SeqFeature(
                FeatureLocation(0, 114, strand=-1),
                type="CDS",
                qualifiers={"gene": ["5_rps12"], "transl_table": ["11"]},
            ),
            SeqFeature(
                FeatureLocation(200, 200 + len(block), strand=1),
                type="CDS",
                qualifiers={"gene": ["3_rps12"], "transl_table": ["11"]},
            ),
        ]
        # Minus-strand exon1: store RC in genome so extract yields e1.
        genome = ["A"] * (200 + len(block) + 10)
        genome[0:114] = list(str(Seq(e1).reverse_complement()))
        genome[200 : 200 + len(block)] = list(block)
        rec = SeqRecord(
            Seq("".join(genome)), id="t", name="TaxSplit", features=features
        )
        result = Rps12Reconstructor.reconstruct(rec, features)
        self.assertTrue(result.success)
        self.assertEqual(result.gene.feat_name, "rps12")
        self.assertEqual(len(result.gene.seq_obj), 372)


class Rps12ExtractIntegrationTests(unittest.TestCase):
    def test_extract_assembles_when_not_excluded(self) -> None:
        e1, e2, e3 = _coding_exons()
        features = [
            SeqFeature(
                FeatureLocation(0, 114, strand=1),
                type="CDS",
                qualifiers={"gene": ["5_rps12"], "transl_table": ["11"]},
            ),
            SeqFeature(
                FeatureLocation(200, 432, strand=1),
                type="CDS",
                qualifiers={"gene": ["3_rps12"], "transl_table": ["11"]},
            ),
            SeqFeature(
                FeatureLocation(500, 500 + 26, strand=1),
                type="CDS",
                # decoy — should be ignored as non-rps12
                qualifiers={"gene": ["psbA"], "transl_table": ["11"]},
            ),
        ]
        # 3_rps12 as separate exon2+exon3 would need two feats; use merged 258 block.
        features[1] = SeqFeature(
            FeatureLocation(200, 200 + 258, strand=1),
            type="CDS",
            qualifiers={"gene": ["3_rps12"], "transl_table": ["11"]},
        )
        genome = ["A"] * 800
        genome[0:114] = list(e1)
        genome[200:458] = list(e2 + e3)
        psb = "ATG" + "GCT" * 7 + "TAA"  # 27 bp
        genome[500 : 500 + len(psb)] = list(psb)
        features[2] = SeqFeature(
            FeatureLocation(500, 500 + len(psb), strand=1),
            type="CDS",
            qualifiers={"gene": ["psbA"], "transl_table": ["11"]},
        )
        rec = SeqRecord(Seq("".join(genome)), id="t", name="TaxIn", features=features)
        ext = _extractor(exclude_fullcds=[])
        nuc = PlastidDict()
        ext._extract_cds(rec, nuc, PlastidDict())
        self.assertIn("rps12", nuc)
        self.assertEqual(len(str(nuc["rps12"][0].seq)), 372)
        self.assertNotIn("5_rps12", nuc)
        self.assertNotIn("3_rps12", nuc)
        self.assertIn("psbA", nuc)

    def test_missing_exon1_quarantines(self) -> None:
        _e1, e2, e3 = _coding_exons()
        block = e2 + e3
        features = [
            SeqFeature(
                FeatureLocation(0, len(block), strand=1),
                type="CDS",
                qualifiers={"gene": ["3_rps12"], "transl_table": ["11"]},
            )
        ]
        rec = SeqRecord(Seq(block + "A" * 5), id="t", name="TaxQ", features=features)
        ext = _extractor(exclude_fullcds=[])
        nuc = PlastidDict()
        ext._extract_cds(rec, nuc, PlastidDict())
        self.assertNotIn("rps12", nuc)
        self.assertTrue(
            any(r.reason_code == ERR_RPS12_INCOMPLETE_EXONS for r in ext._quarantine)
        )

    def test_default_exclude_omits_rps12(self) -> None:
        e1, e2, e3 = _coding_exons()
        full = e1 + e2 + e3
        feat = SeqFeature(
            FeatureLocation(0, len(full), strand=1),
            type="CDS",
            qualifiers={"gene": ["rps12"], "transl_table": ["11"]},
        )
        rec = SeqRecord(Seq(full + "A" * 10), id="t", name="TaxEx", features=[feat])
        ext = _extractor()  # default exclude rps12
        nuc = PlastidDict()
        ext._extract_cds(rec, nuc, PlastidDict())
        self.assertNotIn("rps12", nuc)

    def test_exclcds_none_sentinel_clears_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in"
            out = Path(tmp) / "out"
            inp.mkdir()
            (inp / "x.gb").write_text("LOCUS\n//\n", encoding="ascii")
            params = UserParameters(
                in_dir=str(inp),
                out_dir=str(out),
                select_mode="cds",
                exclude_fullcds=["none"],
            )
            self.assertEqual(params["exclude_fullcds"], [])
            params_dash = UserParameters(
                in_dir=str(inp),
                out_dir=str(Path(tmp) / "out2"),
                select_mode="cds",
                exclude_fullcds=["-"],
            )
            self.assertEqual(params_dash["exclude_fullcds"], [])
            params_igs = UserParameters(
                in_dir=str(inp),
                out_dir=str(Path(tmp) / "out3"),
                select_mode="igs",
                exclude_fullcds=["none"],
            )
            self.assertEqual(params_igs["exclude_fullcds"], ["matK"])


class JoinParseHelpers(unittest.TestCase):
    def test_compound_join_full_length(self) -> None:
        e1, e2, e3 = _coding_exons()
        # Single join feature already ~372 bp after extract.
        loc = CompoundLocation(
            [
                FeatureLocation(0, 114, strand=1),
                FeatureLocation(200, 200 + 232, strand=1),
                FeatureLocation(500, 500 + 26, strand=1),
            ]
        )
        feat = SeqFeature(
            location=loc,
            type="CDS",
            qualifiers={
                "gene": ["rps12"],
                "transl_table": ["11"],
                "trans_splicing": [""],
            },
        )
        genome = ["A"] * 600
        genome[0:114] = list(e1)
        genome[200:432] = list(e2)
        genome[500:526] = list(e3)
        rec = SeqRecord(Seq("".join(genome)), id="t", name="TaxJoin", features=[feat])
        result = Rps12Reconstructor.reconstruct(rec, [feat])
        self.assertTrue(result.success)
        self.assertEqual(len(result.gene.seq_obj), 372)


if __name__ == "__main__":
    unittest.main()
