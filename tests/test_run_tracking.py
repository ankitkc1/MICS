"""Tests for immutable run configuration and resumable state recording."""

from __future__ import annotations

import json
import shutil
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
        archive = recorder.save_checkpoint(
            {
                "config_fingerprint": recorder.config_fingerprint,
                "next_round": 1,
                "tensor": torch.tensor([1.0]),
            },
            archive_completed_rounds=1,
        )
        recorder.mark_round_complete(next_round=1)

        checkpoint = recorder.load_checkpoint()
        status = recorder.read_status()
        metric = json.loads(recorder.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["next_round"], 1)
        self.assertEqual(
            archive,
            recorder.checkpoint_dir / "after_round_0001.pt",
        )
        self.assertTrue(archive.is_file())
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

    def test_archive_collision_does_not_replace_last_checkpoint(self) -> None:
        recorder = RunRecorder(
            self.root,
            "run-1",
            resolved_config=self.config,
            runtime_metadata=self.metadata,
        )
        recorder.save_checkpoint(
            {
                "config_fingerprint": recorder.config_fingerprint,
                "next_round": 1,
            },
            archive_completed_rounds=1,
        )
        recorder.save_checkpoint(
            {
                "config_fingerprint": recorder.config_fingerprint,
                "next_round": 2,
            }
        )

        with self.assertRaises(FileExistsError):
            recorder.save_checkpoint(
                {
                    "config_fingerprint": recorder.config_fingerprint,
                    "next_round": 1,
                },
                archive_completed_rounds=1,
            )

        self.assertEqual(recorder.load_checkpoint()["next_round"], 2)

    def test_resume_recovers_latest_archive_and_trims_ahead_logs(self) -> None:
        recorder = RunRecorder(
            self.root,
            "run-1",
            resolved_config=self.config,
            runtime_metadata=self.metadata,
        )
        for round_index in range(4):
            recorder.record_selection(
                {"round": round_index, "selected_clients": ["C1"]}
            )
            recorder.record_round({"round": round_index, "loss": 1.0})
            recorder.save_checkpoint(
                {
                    "config_fingerprint": recorder.config_fingerprint,
                    "next_round": round_index + 1,
                },
                archive_completed_rounds=(
                    round_index + 1
                    if round_index + 1 in (2, 4)
                    else None
                ),
            )
        shutil.copyfile(
            recorder.checkpoint_dir / "after_round_0002.pt",
            recorder.checkpoint_path,
        )
        recorder.record_selection({"round": 1, "selected_clients": ["C2"]})
        recorder.record_round({"round": 1, "loss": 2.0})
        recorder.mark_failed(next_round=1, error="simulated stale resume")

        recovered = RunRecorder(
            self.root,
            "run-1",
            resolved_config=self.config,
            runtime_metadata=self.metadata,
            resume=True,
        )

        self.assertEqual(recovered.load_checkpoint()["next_round"], 4)
        for path in (recovered.selection_path, recovered.metrics_path):
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["round"] for record in records],
                [0, 1, 2, 3],
            )
        recovery_records = [
            json.loads(line)
            for line in recovered.recovery_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(recovery_records[-1]["next_round"], 4)
        self.assertTrue((recovered.run_dir / "recovery_backups").is_dir())
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
