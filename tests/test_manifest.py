"""Tests for atomic run-manifest helpers and PARTIAL/COMPLETE status."""

from __future__ import annotations

import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plastburstalign.manifest import (
    MANIFEST_NAME,
    load_run_manifest,
    write_atomic_json,
    write_run_manifest,
)


class WriteAtomicJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_atomic_replace_and_no_temp_leak(self) -> None:
        target = self.root / "doc.json"
        write_atomic_json(target, {"a": 1, "b": [2, 3]})
        self.assertTrue(target.is_file())
        with target.open(encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data, {"a": 1, "b": [2, 3]})
        leftovers = list(self.root.glob("*.tmp.*"))
        self.assertEqual(leftovers, [])

    def test_failed_write_unlinks_temp(self) -> None:
        target = self.root / "broken.json"

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        with mock.patch("plastburstalign.manifest.json.dump", side_effect=boom):
            with self.assertRaises(OSError):
                write_atomic_json(target, {"x": 1})
        self.assertFalse(target.exists())
        leftovers = list(self.root.glob("*.tmp.*"))
        self.assertEqual(leftovers, [])


class RunManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = self.tmp.name
        self.started = datetime.datetime(
            2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_complete_status_schema(self) -> None:
        path = write_run_manifest(
            self.out_dir,
            started_at=self.started,
            wall_seconds=1.5,
            params={"select_mode": "igs", "out_dir": self.out_dir},
            locus_records=[{"region": "psbA", "mode": "igs", "ok": True}],
            ok=True,
        )
        self.assertIsNotNone(path)
        assert path is not None
        doc = load_run_manifest(self.out_dir)
        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc["run"]["ok"], True)
        self.assertEqual(doc["run"]["status"], "COMPLETE")
        self.assertEqual(doc["loci"][0]["region"], "psbA")
        self.assertTrue(path.endswith(MANIFEST_NAME))

    def test_partial_and_failed_status(self) -> None:
        write_run_manifest(
            self.out_dir,
            started_at=self.started,
            wall_seconds=0.25,
            locus_records=[{"region": "matK", "ok": True}],
            ok=True,
            status="PARTIAL",
            supermatrix=None,
        )
        doc = load_run_manifest(self.out_dir)
        assert doc is not None
        self.assertEqual(doc["run"]["status"], "PARTIAL")
        self.assertIsNone(doc["supermatrix"])

        write_run_manifest(
            self.out_dir,
            started_at=self.started,
            wall_seconds=0.5,
            ok=False,
        )
        doc = load_run_manifest(self.out_dir)
        assert doc is not None
        self.assertEqual(doc["run"]["ok"], False)
        self.assertEqual(doc["run"]["status"], "FAILED")

    def test_load_missing_returns_none(self) -> None:
        self.assertIsNone(load_run_manifest(self.out_dir))

    def test_trailing_newline_on_manifest(self) -> None:
        write_run_manifest(
            self.out_dir,
            started_at=self.started,
            wall_seconds=0.1,
            ok=True,
        )
        raw = Path(self.out_dir, MANIFEST_NAME).read_bytes()
        self.assertTrue(raw.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
