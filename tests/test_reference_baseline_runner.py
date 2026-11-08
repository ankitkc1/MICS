"""End-to-end CPU checks for probe-based baseline orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_scenario(root: Path) -> tuple[Path, Path]:
    dataset_root = root / "dataset"
    (dataset_root / "vol").mkdir(parents=True)
    (dataset_root / "seg").mkdir()
    for index in range(1, 9):
        case_id = f"Case{index:03d}"
        generator = np.random.default_rng(index)
        np.save(
            dataset_root / "vol" / f"{case_id}.npy",
            generator.normal(size=(4, 4, 4, 4)).astype(np.float32),
        )
        np.save(
            dataset_root / "seg" / f"{case_id}.npy",
            np.zeros((4, 4, 4), dtype=np.uint8),
        )

    bundle = root / "bundle"
    scenario_dir = bundle / "balanced"
    scenario_dir.mkdir(parents=True)
    assignments = {
        "C1": {
            "train": "Case001",
            "validation": "Case003",
            "test": "Case005",
        },
        "C2": {
            "train": "Case002",
            "validation": "Case004",
            "test": "Case006",
        },
    }
    clients = {}
    for client_id, partitions in assignments.items():
        files = {}
        for partition, case_id in partitions.items():
            path = scenario_dir / f"{client_id}_{partition}.csv"
            path.write_text(f"{case_id}\n", encoding="utf-8")
            files[partition] = str(path)
        clients[client_id] = {"files": files}
    global_test = bundle / "global_test.csv"
    global_test.write_text("Case007\nCase008\n", encoding="utf-8")
    scenario_config = scenario_dir / "materialized_config.json"
    scenario_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_id": "balanced",
                "data_seed": 1,
                "modality_order": ["FLAIR", "T1ce", "T1", "T2"],
                "masks": {
                    "C1": [1, 1, 0, 0],
                    "C2": [0, 0, 1, 1],
                },
                "expected_modality_pool_sizes": {
                    "FLAIR": 1,
                    "T1ce": 1,
                    "T1": 1,
                    "T2": 1,
                },
                "global_test_file": str(global_test),
                "clients": clients,
            }
        ),
        encoding="utf-8",
    )
    return dataset_root, scenario_config


class ReferenceBaselineRunnerTests(unittest.TestCase):
    def test_power_of_choice_and_divfl_write_costed_runs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            dataset_root, scenario_config = _write_scenario(root)
            output_root = root / "results"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "src")

            for method, probe_kind in (
                ("power_of_choice", "local_loss"),
                ("divfl", "gradient_vector"),
            ):
                with self.subTest(method=method):
                    run_id = f"{method}-cpu-smoke"
                    command = [
                        sys.executable,
                        str(
                            project_root
                            / "scripts"
                            / "run_reference_baseline.py"
                        ),
                        "--method",
                        method,
                        "--dataset-root",
                        str(dataset_root),
                        "--scenario-config",
                        str(scenario_config),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        run_id,
                        "--rounds",
                        "1",
                        "--clients-per-round",
                        "1",
                        "--candidate-size",
                        "2",
                        "--probe-max-batches",
                        "1",
                        "--base-channels",
                        "2",
                        "--num-levels",
                        "2",
                        "--fusion",
                        "mean",
                        "--modality-dropout",
                        "0",
                        "--patch-size",
                        "4",
                        "4",
                        "4",
                        "--max-batches-per-epoch",
                        "1",
                        "--device",
                        "cpu",
                        "--no-amp",
                    ]
                    result = subprocess.run(
                        command,
                        cwd=project_root,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        self.fail(
                            f"{method} baseline subprocess failed:\n"
                            f"stdout:\n{result.stdout}\n"
                            f"stderr:\n{result.stderr}"
                        )

                    run_dir = output_root / run_id
                    status = json.loads(
                        (run_dir / "status.json").read_text(encoding="utf-8")
                    )
                    record = json.loads(
                        (run_dir / "round_metrics.jsonl")
                        .read_text(encoding="utf-8")
                        .strip()
                    )
                    self.assertEqual(status["state"], "completed")
                    self.assertEqual(record["selection_probe"]["kind"], probe_kind)
                    self.assertEqual(
                        record["selection_probe"]["partition"],
                        "train",
                    )
                    self.assertFalse(record["selection_probe"]["model_updated"])
                    self.assertGreater(
                        record["communication"][
                            "selection_probe_download_bytes"
                        ],
                        0,
                    )
                    self.assertGreater(
                        record["communication"][
                            "selection_probe_upload_bytes"
                        ],
                        0,
                    )
                    self.assertTrue(
                        (run_dir / "checkpoints" / "last.pt").is_file()
                    )


if __name__ == "__main__":
    unittest.main()
