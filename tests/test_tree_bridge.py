"""Verification for tree bridge script emission."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from plastburstalign.tree_bridge import (
    coerce_tree_cores,
    data_type_from_mode,
    generate_tree_commands,
    resolve_matrix_path,
    resolve_partition_path,
    write_tree_runner_scripts,
)


class GenerateTreeCommandsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_unpartitioned_dna_defaults(self) -> None:
        msa = self.root / "supermatrix.fasta"
        msa.touch()
        cmds = generate_tree_commands(
            matrix_path=msa,
            partition_path=None,
            out_prefix=self.root / "tree",
            cores=8,
        )
        self.assertIn("raxml-ng", cmds["raxml_ng"])
        self.assertIn("--model GTR+G", cmds["raxml_ng"])
        self.assertIn("--threads 8", cmds["raxml_ng"])
        self.assertIn("iqtree2", cmds["iqtree2"])
        self.assertIn("-m GTR+G", cmds["iqtree2"])
        self.assertIn("-T 8", cmds["iqtree2"])

    def test_partitioned_uses_partition_file(self) -> None:
        msa = self.root / "supermatrix.fasta"
        part = self.root / "partitions.txt"
        msa.touch()
        part.touch()
        cmds = generate_tree_commands(
            matrix_path=msa,
            partition_path=part,
            out_prefix=self.root / "tree",
            cores=16,
        )
        self.assertIn(f"--model {part}", cmds["raxml_ng"])
        self.assertIn(f"-p {part}", cmds["iqtree2"])
        self.assertIn("--threads 16", cmds["raxml_ng"])
        self.assertNotIn("--model GTR+G", cmds["raxml_ng"])

    def test_aa_unpartitioned_uses_lg(self) -> None:
        msa = self.root / "supermatrix.fasta"
        msa.touch()
        cmds = generate_tree_commands(
            matrix_path=msa,
            partition_path=None,
            out_prefix=self.root / "tree",
            cores=4,
            data_type="aa",
        )
        self.assertIn("--model LG+G", cmds["raxml_ng"])
        self.assertIn("-m LG+G", cmds["iqtree2"])


class WriteTreeRunnerScriptsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_write_tree_runner_scripts(self) -> None:
        msa = self.root / "supermatrix.fasta"
        msa.touch()
        written = write_tree_runner_scripts(
            out_dir=self.root,
            matrix_path=msa,
            partition_path=None,
            cores=4,
        )
        self.assertIn("raxml_ng", written)
        self.assertIn("iqtree2", written)
        self.assertTrue(written["raxml_ng"].is_file())
        self.assertTrue(written["iqtree2"].is_file())
        self.assertTrue(os.access(written["raxml_ng"], os.X_OK))
        self.assertTrue(os.access(written["iqtree2"], os.X_OK))
        content = written["raxml_ng"].read_text(encoding="utf-8")
        self.assertIn("#!/usr/bin/env bash", content)
        self.assertIn("set -euo pipefail", content)
        self.assertIn("raxml-ng --msa", content)
        iq = written["iqtree2"].read_text(encoding="utf-8")
        self.assertIn("#!/usr/bin/env bash", iq)
        self.assertIn("iqtree2 -s", iq)


class PathResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_resolve_prefers_write_all_names(self) -> None:
        matrix = self.root / "nucl_3concat.aligned.fasta"
        part = self.root / "nucl_3concat.partitions.txt"
        matrix.write_text(">a\nATGC\n", encoding="utf-8")
        part.write_text("DNA, locus = 1-4\n", encoding="utf-8")
        written = {
            "fasta": str(matrix),
            "raxml": str(part),
        }
        self.assertEqual(resolve_matrix_path(self.root, written), matrix)
        self.assertEqual(resolve_partition_path(self.root, written), part)

    def test_helpers(self) -> None:
        self.assertEqual(coerce_tree_cores("auto", default=8), 8)
        self.assertEqual(coerce_tree_cores(16), 16)
        self.assertEqual(coerce_tree_cores("4"), 4)
        self.assertEqual(data_type_from_mode("prot"), "aa")
        self.assertEqual(data_type_from_mode("cds"), "dna")


if __name__ == "__main__":
    unittest.main()
