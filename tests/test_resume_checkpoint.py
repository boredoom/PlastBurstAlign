"""Tests for locus checkpoint validation and --resume lifecycle."""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from plastburstalign.align_runtime import clear_stale_outputs
from plastburstalign.alignment_ops import (
    AlignmentCoordination,
    is_locus_checkpoint_valid,
    locus_checkpoint_paths,
)
from plastburstalign.plast_burst_align import PlastBurstAlign
from plastburstalign.seqfeature_ops import PlastidData
from plastburstalign.user_parameters import UserParameters


def _write_fasta(
    path: Path, records: list[tuple[str, str]], *, trailing_nl: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f">{name}\n{seq}\n" for name, seq in records)
    if not trailing_nl and body.endswith("\n"):
        body = body[:-1]
    path.write_text(body, encoding="utf-8")


class RecordingAligner:
    exec_path = "recording-aligner"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def align(
        self,
        input_file: str,
        output_file: str,
        **kwargs: Any,
    ) -> bool:
        with self._lock:
            self.calls.append({"input": input_file, "output": output_file, **kwargs})
        shutil.copyfile(input_file, output_file)
        return True


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_fasta_passes(self) -> None:
        path = self.root / "nucl_psbA.aligned.fasta"
        _write_fasta(path, [("t1", "ATGC"), ("t2", "ATGG")])
        self.assertTrue(is_locus_checkpoint_valid(path, expected_ntax=2))

    def test_rejects_wrong_header_count_empty_or_missing_newline(self) -> None:
        path = self.root / "nucl_psbA.aligned.fasta"
        _write_fasta(path, [("t1", "ATGC")])
        self.assertFalse(is_locus_checkpoint_valid(path, expected_ntax=2))

        path.write_text("", encoding="utf-8")
        self.assertFalse(is_locus_checkpoint_valid(path, expected_ntax=1))

        _write_fasta(path, [("t1", "ATGC")], trailing_nl=False)
        self.assertFalse(is_locus_checkpoint_valid(path, expected_ntax=1))

        missing = self.root / "missing.fasta"
        self.assertFalse(is_locus_checkpoint_valid(missing, expected_ntax=1))

    def test_sha256_mismatch(self) -> None:
        path = self.root / "nucl_x.aligned.fasta"
        _write_fasta(path, [("t1", "AAA")])
        self.assertFalse(
            is_locus_checkpoint_valid(path, expected_ntax=1, expected_sha256="deadbeef")
        )

    def test_cds_requires_dual_artifacts(self) -> None:
        from plastburstalign.user_parameters import SelectMode

        paths = locus_checkpoint_paths(self.root, "psbA", "cds")
        self.assertEqual(
            [p.name for p in paths],
            ["prot_psbA.aligned.fasta", "nucl_psbA.aligned.fasta"],
        )
        # Enum members must resolve the same way as plain "cds".
        self.assertEqual(
            locus_checkpoint_paths(self.root, "psbA", SelectMode.CDS),
            paths,
        )
        self.assertTrue(all(p.parent.name == "fasta" for p in paths))
        igs = locus_checkpoint_paths(self.root, "psbA", "igs")
        self.assertEqual([p.name for p in igs], ["nucl_psbA.aligned.fasta"])
        prot = locus_checkpoint_paths(self.root, "psbA", "prot")
        self.assertEqual([p.name for p in prot], ["prot_psbA.aligned.fasta"])


class ResumeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.in_dir = self.root / "in"
        self.out_dir = self.root / "out"
        self.in_dir.mkdir()
        self.out_dir.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _params(self, *, resume: bool = False, mode: str = "igs") -> UserParameters:
        return UserParameters(
            in_dir=str(self.in_dir),
            out_dir=str(self.out_dir),
            select_mode=mode,
            num_threads=1,
            min_seq_length=1,
            concat=False,
            individual_nexus=False,
            keep_unaligned=True,
            resume=resume,
            subdivide="off",
        )

    def _seed_store(self, params: UserParameters, regions: list[str]) -> PlastidData:
        data = PlastidData(params)
        for region in regions:
            for i, taxon in enumerate(("t1", "t2")):
                data.store.add_record(
                    region,
                    taxon,
                    "nucl",
                    f"{region}_{taxon}",
                    "",
                    "ATGCATGC",
                    file_index=i,
                )
        data.store._conn.commit()
        return data

    def test_resume_skips_clear_stale_outputs(self) -> None:
        store = self.out_dir / "feature_store.db"
        store.write_bytes(b"keep-me")
        unalign = self.out_dir / "nucl_psbA.unalign.fasta"
        unalign.write_text(">t1\nAAA\n", encoding="utf-8")

        with patch("plastburstalign.plast_burst_align.PlastidData") as mock_data:
            mock_data.return_value = object()
            PlastBurstAlign(self._params(resume=True))
        self.assertTrue(store.exists())
        self.assertTrue(unalign.exists())

        clear_stale_outputs(str(self.out_dir))
        self.assertFalse(store.exists())
        self.assertFalse(unalign.exists())

    def test_resume_skips_valid_checkpoint_and_realigns_invalid(self) -> None:
        params = self._params(resume=True, mode="igs")
        data = self._seed_store(params, ["locA", "locB"])
        self.addCleanup(data.store.close)

        fasta_dir = self.out_dir / "fasta"
        valid = fasta_dir / "nucl_locA.aligned.fasta"
        invalid = fasta_dir / "nucl_locB.aligned.fasta"
        _write_fasta(valid, [("locA_t1", "ATGCATGC"), ("locA_t2", "ATGCATGC")])
        # Wrong taxon count → must re-align.
        _write_fasta(invalid, [("locB_t1", "ATGCATGC")])

        aligner = RecordingAligner()
        with patch(
            "plastburstalign.alignment_ops.get_alignment_tool",
            return_value=aligner,
        ):
            coord = AlignmentCoordination(data, params)
            # Pretend unaligned export already happened so resume can skip it.
            coord._unaligned_saved = True
            coord.perform_MSA()

        hit_regions = {
            r["region"]
            for r in coord._locus_records
            if r.get("checkpoint") == "CHECKPOINT_HIT"
        }
        self.assertEqual(hit_regions, {"locA"})
        aligned_names = {Path(c["output"]).name for c in aligner.calls}
        self.assertIn("nucl_locB.aligned.fasta", aligned_names)
        self.assertNotIn("nucl_locA.aligned.fasta", aligned_names)

    def test_resume_skips_extract_when_store_exists(self) -> None:
        params = self._params(resume=True, mode="igs")
        data = self._seed_store(params, ["locA"])
        data.store.close()

        fasta_dir = self.out_dir / "fasta"
        _write_fasta(
            fasta_dir / "nucl_locA.aligned.fasta",
            [("locA_t1", "ATGCATGC"), ("locA_t2", "ATGCATGC")],
        )

        aligner = RecordingAligner()
        extract_calls: list[str] = []

        class FakeExtract:
            def __init__(self, *_a: Any, **_k: Any) -> None:
                pass

            def extract(self) -> None:
                extract_calls.append("extract")

        class FakeClean:
            def __init__(self, *_a: Any, **_k: Any) -> None:
                pass

            def clean(self) -> None:
                extract_calls.append("clean")

        with (
            patch(
                "plastburstalign.alignment_ops.get_alignment_tool",
                return_value=aligner,
            ),
            patch(
                "plastburstalign.plast_burst_align.ExtractAndCollect",
                FakeExtract,
            ),
            patch(
                "plastburstalign.plast_burst_align.DataCleaning",
                FakeClean,
            ),
        ):
            PlastBurstAlign(params).execute()

        self.assertEqual(extract_calls, [])
        self.assertEqual(aligner.calls, [])

    def test_resume_runs_extract_when_store_missing(self) -> None:
        params = self._params(resume=True, mode="igs")
        # No feature_store.db — resume must extract/clean.
        extract_calls: list[str] = []

        class FakeExtract:
            def __init__(self, data: PlastidData, *_a: Any, **_k: Any) -> None:
                self.data = data

            def extract(self) -> None:
                extract_calls.append("extract")
                self.data.store.add_record(
                    "locA", "t1", "nucl", "locA_t1", "", "ATGC", file_index=0
                )
                self.data.store.add_record(
                    "locA", "t2", "nucl", "locA_t2", "", "ATGG", file_index=1
                )

        class FakeClean:
            def __init__(self, *_a: Any, **_k: Any) -> None:
                pass

            def clean(self) -> None:
                extract_calls.append("clean")

        aligner = RecordingAligner()
        with (
            patch(
                "plastburstalign.alignment_ops.get_alignment_tool",
                return_value=aligner,
            ),
            patch(
                "plastburstalign.plast_burst_align.ExtractAndCollect",
                FakeExtract,
            ),
            patch(
                "plastburstalign.plast_burst_align.DataCleaning",
                FakeClean,
            ),
        ):
            PlastBurstAlign(params).execute()

        self.assertEqual(extract_calls, ["extract", "clean"])
        self.assertEqual(len(aligner.calls), 1)


class CdsDualArtifactCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.in_dir = self.root / "in"
        self.out_dir = self.root / "out"
        self.in_dir.mkdir()
        self.out_dir.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cds_hit_requires_both_prot_and_nucl(self) -> None:
        params = UserParameters(
            in_dir=str(self.in_dir),
            out_dir=str(self.out_dir),
            select_mode="cds",
            num_threads=1,
            min_seq_length=1,
            concat=False,
            individual_nexus=False,
            keep_unaligned=True,
            resume=True,
            subdivide="off",
        )
        data = PlastidData(params)
        self.addCleanup(data.store.close)
        data.store.add_record(
            "psbA", "t1", "nucl", "psbA_t1", "", "ATGAAA", file_index=0
        )
        data.store.add_record(
            "psbA", "t2", "nucl", "psbA_t2", "", "ATGGAA", file_index=1
        )
        data.store.add_record("psbA", "t1", "prot", "psbA_t1", "", "MK", file_index=0)
        data.store.add_record("psbA", "t2", "prot", "psbA_t2", "", "MG", file_index=1)

        fasta = self.out_dir / "fasta"
        _write_fasta(fasta / "nucl_psbA.aligned.fasta", [("a", "ATG"), ("b", "ATG")])
        _write_fasta(
            self.out_dir / "prot_psbA.unalign.fasta",
            [("psbA_t1", "MK"), ("psbA_t2", "MG")],
        )
        _write_fasta(
            self.out_dir / "nucl_psbA.unalign.fasta",
            [("psbA_t1", "ATGAAA"), ("psbA_t2", "ATGGAA")],
        )

        class BT:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args

            def backtranslate(self) -> None:
                Path(self.args[3]).write_text(
                    ">a\nATGAAA\n>b\nATGGAA\n", encoding="utf-8"
                )

        # Missing prot aligned artifact → must not checkpoint-hit; aligner runs.
        aligner = RecordingAligner()
        with (
            patch(
                "plastburstalign.alignment_ops.get_alignment_tool",
                return_value=aligner,
            ),
            patch("plastburstalign.alignment_ops.BackTranslation", BT),
        ):
            coord = AlignmentCoordination(data, params)
            coord._unaligned_saved = True
            coord.perform_MSA()

        self.assertFalse(
            any(r.get("checkpoint") == "CHECKPOINT_HIT" for r in coord._locus_records)
        )
        self.assertGreaterEqual(len(aligner.calls), 1)

        # Both artifacts present → CHECKPOINT_HIT and no aligner dispatch.
        _write_fasta(fasta / "prot_psbA.aligned.fasta", [("a", "MK"), ("b", "MG")])
        aligner2 = RecordingAligner()
        with patch(
            "plastburstalign.alignment_ops.get_alignment_tool",
            return_value=aligner2,
        ):
            coord2 = AlignmentCoordination(data, params)
            coord2._unaligned_saved = True
            coord2.perform_MSA()
        self.assertTrue(
            any(r.get("checkpoint") == "CHECKPOINT_HIT" for r in coord2._locus_records)
        )
        self.assertEqual(aligner2.calls, [])


if __name__ == "__main__":
    unittest.main()
