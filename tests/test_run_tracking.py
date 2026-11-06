"""Tests for immutable run configuration and resumable state recording."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mics.experiments import RunRecorder


class RunRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.config = {"method": "uniform_random", "rounds": 2}
        self.metadata = {"source_revision": "abc123"}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_records_jsonl_status_and_checkpoint(self) -> None:
        recorder = RunRecorder(
            self.root,
            "run-1",
            resolved_config=self.config,
            runtime_metadata=self.metadata,
        )
        recorder.mark_running(next_round=0)
        recorder.record_selection({"round": 0, "selected_clients": ["C1"]})
        recorder.record_round({"round": 0, "loss": 1.25})
        recorder.save_checkpoint(
            {
                "config_fingerprint": recorder.config_fingerprint,
                "next_round": 1,
                "tensor": torch.tensor([1.0]),
            }
        )
        recorder.mark_round_complete(next_round=1)

        checkpoint = recorder.load_checkpoint()
        status = recorder.read_status()
        metric = json.loads(recorder.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["next_round"], 1)
        self.assertTrue(
            torch.equal(checkpoint["tensor"], torch.tensor([1.0]))
        )
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["next_round"], 1)
        self.assertEqual(metric, {"loss": 1.25, "round": 0})

    def test_refuses_overwrite_and_mismatched_resume(self) -> None:
        recorder = RunRecorder(
            self.root,
            "run-1",
            resolved_config=self.config,
            runtime_metadata=self.metadata,
        )
        recorder.save_checkpoint(
            {
                "config_fingerprint": recorder.config_fingerprint,
                "next_round": 0,
            }
        )
        with self.assertRaises(FileExistsError):
            RunRecorder(
                self.root,
                "run-1",
                resolved_config=self.config,
                runtime_metadata=self.metadata,
            )
        with self.assertRaisesRegex(ValueError, "configuration"):
            RunRecorder(
                self.root,
                "run-1",
                resolved_config={"method": "different"},
                runtime_metadata=self.metadata,
                resume=True,
            )

    def test_rejects_nonfinite_json_metric(self) -> None:
        recorder = RunRecorder(
            self.root,
            "run-1",
            resolved_config=self.config,
            runtime_metadata=self.metadata,
        )
        with self.assertRaises(ValueError):
            recorder.record_round({"round": 0, "loss": float("nan")})


if __name__ == "__main__":
    unittest.main()
