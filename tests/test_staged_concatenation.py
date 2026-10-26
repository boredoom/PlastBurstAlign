import io
import unittest
from Bio.Nexus.Nexus import Nexus as NexusAlignment, combine as nexus_combine
from plastburstalign.alignment_ops import IncrementalNexusAccumulator


class StagedConcatenationTests(unittest.TestCase):
    def test_accumulator_byte_identity_with_nexus_combine(self):
        nexus_a = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=6;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon1 ATGCAT\n"
            "concat_taxon2 ATGC--\n"
            ";\n"
            "end;\n"
        )
        nexus_b = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=4;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon1 GGGG\n"
            "concat_taxon2 CCCC\n"
            ";\n"
            "end;\n"
        )

        m_a1 = NexusAlignment(nexus_a)
        m_b1 = NexusAlignment(nexus_b)
        combined_batch = nexus_combine([("geneA", m_a1), ("geneB", m_b1)])

        m_a2 = NexusAlignment(nexus_a)
        m_b2 = NexusAlignment(nexus_b)
        acc = IncrementalNexusAccumulator()
        acc.add("geneA", m_a2)
        acc.add("geneB", m_b2)
        combined_inc = acc.finalize()

        buf_batch = io.StringIO()
        combined_batch.write_nexus_data(buf_batch)

        buf_inc = io.StringIO()
        combined_inc.write_nexus_data(buf_inc)

        self.assertEqual(buf_batch.getvalue(), buf_inc.getvalue())

    def test_accumulator_disjoint_taxa_padding(self):
        nexus_a = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=3;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon1 ATG\n"
            "concat_taxon2 ATG\n"
            ";\n"
            "end;\n"
        )
        nexus_b = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=3;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon2 CCC\n"
            "concat_taxon3 CCC\n"
            ";\n"
            "end;\n"
        )

        acc = IncrementalNexusAccumulator()
        acc.add("gene1", NexusAlignment(nexus_a))
        acc.add("gene2", NexusAlignment(nexus_b))
        res = acc.finalize()

        self.assertEqual(res.ntax, 3)
        self.assertEqual(res.nchar, 6)
        # taxon1 missing gene2 -> ???
        self.assertEqual(str(res.matrix["concat_taxon1"]), "ATG???")
        # taxon3 missing gene1 -> ???
        self.assertEqual(str(res.matrix["concat_taxon3"]), "???CCC")
        # taxon2 present in both
        self.assertEqual(str(res.matrix["concat_taxon2"]), "ATGCCC")

    def test_accumulator_empty_and_single(self):
        acc = IncrementalNexusAccumulator()
        self.assertIsNone(acc.finalize())

        nexus_single = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=1 nchar=4;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_t1 AAAA\n"
            ";\n"
            "end;\n"
        )
        acc.add("g1", NexusAlignment(nexus_single))
        res = acc.finalize()
        self.assertEqual(res.nchar, 4)
        self.assertEqual(str(res.matrix["concat_t1"]), "AAAA")


if __name__ == "__main__":
    unittest.main()
