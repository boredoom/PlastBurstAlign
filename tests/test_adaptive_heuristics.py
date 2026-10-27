from __future__ import annotations

import os
import tempfile
import unittest

from plastburstalign.alignment_tools import MAFFT


class TestAdaptiveHeuristics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.temp_dir.name, "test.fasta")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_fasta(self, n_seq: int):
        with open(self.in_file, "w") as f:
            for i in range(n_seq):
                f.write(f">seq_{i}\nATGCATGC\n")

    def test_small_dataset_uses_auto(self):
        # N < 1000 should use standard --auto
        self._create_fasta(50)
        mafft = MAFFT()
        cmd = mafft._build_command(self.in_file)
        self.assertIn("--auto", cmd)
        self.assertNotIn("--parttree", cmd)

    def test_medium_dataset_uses_parttree_retree2(self):
        # 1000 <= N < 5000 should use --parttree --retree 2
        self._create_fasta(1200)
        mafft = MAFFT()
        cmd = mafft._build_command(self.in_file)
        self.assertIn("--parttree", cmd)
        self.assertIn("--retree", cmd)
        retree_idx = cmd.index("--retree")
        self.assertEqual(cmd[retree_idx + 1], "2")
        self.assertNotIn("--auto", cmd)

    def test_massive_dataset_uses_parttree_retree1(self):
        # N >= 5000 should use --parttree --retree 1
        self._create_fasta(5500)
        mafft = MAFFT()
        cmd = mafft._build_command(self.in_file)
        self.assertIn("--parttree", cmd)
        self.assertIn("--retree", cmd)
        retree_idx = cmd.index("--retree")
        self.assertEqual(cmd[retree_idx + 1], "1")
        self.assertNotIn("--auto", cmd)

    def test_explicit_algorithm_overrides_heuristics(self):
        # If user explicitly specifies algorithm, heuristics must not override
        self._create_fasta(5500)
        user_params = {"alignment_tool_config": {"algorithm": "localpair"}}
        mafft = MAFFT(user_params)
        cmd = mafft._build_command(self.in_file)
        self.assertIn("--localpair", cmd)
        self.assertNotIn("--parttree", cmd)

    def test_disabled_heuristics_falls_back_to_auto(self):
        self._create_fasta(5500)
        user_params = {"adaptive_heuristics": False}
        mafft = MAFFT(user_params)
        cmd = mafft._build_command(self.in_file)
        self.assertIn("--auto", cmd)
        self.assertNotIn("--parttree", cmd)

    def test_sequence_count_passed_avoids_scan(self):
        # Passing sequence_count directly avoids reading input_file
        mafft = MAFFT()
        cmd = mafft._build_command("non_existent_file.fasta", sequence_count=13724)
        self.assertIn("--parttree", cmd)
        self.assertIn("--retree", cmd)
        retree_idx = cmd.index("--retree")
        self.assertEqual(cmd[retree_idx + 1], "1")


if __name__ == "__main__":
    unittest.main()
