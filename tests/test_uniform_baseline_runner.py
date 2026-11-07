"""Small end-to-end test of the uniform baseline orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class UniformBaselineRunnerTests(unittest.TestCase):
    def test_one_cpu_round_writes_completed_checkpointed_run(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            (dataset_root / "vol").mkdir(parents=True)
            (dataset_root / "seg").mkdir()
            for index in range(1, 9):
                case_id = f"Case{index:03d}"
                np.save(
                    dataset_root / "vol" / f"{case_id}.npy",
                    np.zeros((4, 4, 4, 4), dtype=np.float32),
                )
                np.save(
                    dataset_root / "seg" / f"{case_id}.npy",
                    np.zeros((4, 4, 4), dtype=np.uint8),
                )

            bundle = root / "bundle"
            scenario_dir = bundle / "splitA"
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
                        "split_id": "A",
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

            output_root = root / "results"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "src")
            result = subprocess.run(
                (
                    sys.executable,
                    str(project_root / "scripts" / "run_uniform_baseline.py"),
                    "--dataset-root",
                    str(dataset_root),
                    "--scenario-config",
                    str(scenario_config),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "cpu-smoke",
                    "--rounds",
                    "1",
                    "--clients-per-round",
                    "1",
                    "--base-channels",
                    "2",
                    "--num-levels",
                    "2",
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
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.fail(
                    "Uniform baseline subprocess failed:\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

            run_dir = output_root / "cpu-smoke"
            status = json.loads(
                (run_dir / "status.json").read_text(encoding="utf-8")
            )
            metrics = (run_dir / "round_metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertIn(
                "MICS UNIFORM-RANDOM BASELINE TRAINING PASSED",
                result.stdout,
            )
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["next_round"], 1)
            self.assertEqual(len(metrics), 1)
            self.assertTrue((run_dir / "checkpoints" / "last.pt").is_file())
            self.assertTrue(
                (run_dir / "checkpoints" / "after_round_0001.pt").is_file()
            )

            evaluation = subprocess.run(
                (
                    sys.executable,
                    str(
                        project_root
                        / "scripts"
                        / "evaluate_brats_checkpoint.py"
                    ),
                    "--run-dir",
                    str(run_dir),
                    "--dataset-root",
                    str(dataset_root),
                    "--scenario-config",
                    str(scenario_config),
                    "--partition",
                    "validation",
                    "--max-cases-per-client",
                    "1",
                    "--roi-size",
                    "4",
                    "4",
                    "4",
                    "--device",
                    "cpu",
                    "--no-amp",
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if evaluation.returncode != 0:
                self.fail(
                    "Checkpoint evaluation subprocess failed:\n"
                    f"stdout:\n{evaluation.stdout}\n"
                    f"stderr:\n{evaluation.stderr}"
                )
            evaluation_path = (
                run_dir
                / "evaluations"
                / "validation_subset_after_round_0001.json"
            )
            evaluation_payload = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            self.assertIn(
                "MICS CHECKPOINT WHOLE-VOLUME EVALUATION PASSED",
                evaluation.stdout,
            )
            self.assertEqual(
                evaluation_payload["purpose"],
                "integration_smoke_not_model_quality",
            )
            self.assertEqual(evaluation_payload["evaluation"]["case_count"], 2)

            importance = subprocess.run(
                (
                    sys.executable,
                    str(
                        project_root
                        / "scripts"
                        / "estimate_modality_importance.py"
                    ),
                    "--run-dir",
                    str(run_dir),
                    "--dataset-root",
                    str(dataset_root),
                    "--scenario-config",
                    str(scenario_config),
                    "--max-cases-per-client",
                    "1",
                    "--roi-size",
                    "4",
                    "4",
                    "4",
                    "--device",
                    "cpu",
                    "--no-amp",
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if importance.returncode != 0:
                self.fail(
                    "Modality-importance subprocess failed:\n"
                    f"stdout:\n{importance.stdout}\n"
                    f"stderr:\n{importance.stderr}"
                )
            importance_path = (
                run_dir
                / "evaluations"
                / "modality_importance_subset_after_round_0001.json"
            )
            importance_payload = json.loads(
                importance_path.read_text(encoding="utf-8")
            )
            self.assertIn(
                "MICS VALIDATION-ONLY MODALITY IMPORTANCE ESTIMATION PASSED",
                importance.stdout,
            )
            self.assertEqual(
                importance_payload["purpose"],
                "integration_smoke_not_modality_importance",
            )
            self.assertFalse(
                importance_payload["importance"]["test_data_accessed"]
            )
            self.assertEqual(
                importance_payload["importance"]["probe_cost"]
                ["total_case_evaluations"],
                6,
            )

            selector_output = root / "mics_selector_smoke.json"
            selector_smoke = subprocess.run(
                (
                    sys.executable,
                    str(project_root / "scripts" / "smoke_mics_selector.py"),
                    "--dataset-root",
                    str(dataset_root),
                    "--scenario-config",
                    str(scenario_config),
                    "--importance-output",
                    str(importance_path),
                    "--output",
                    str(selector_output),
                    "--rounds",
                    "3",
                    "--clients-per-round",
                    "2",
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if selector_smoke.returncode != 0:
                self.fail(
                    "MICS selector smoke subprocess failed:\n"
                    f"stdout:\n{selector_smoke.stdout}\n"
                    f"stderr:\n{selector_smoke.stderr}"
                )
            selector_payload = json.loads(
                selector_output.read_text(encoding="utf-8")
            )
            self.assertIn(
                "MICS BALANCED-SCENARIO SELECTOR SMOKE TEST PASSED",
                selector_smoke.stdout,
            )
            self.assertEqual(
                selector_payload["purpose"],
                "selector_integration_smoke_not_model_quality",
            )
            self.assertFalse(
                selector_payload["test_partition_used_for_selection"]
            )
            self.assertEqual(len(selector_payload["rounds"]), 3)

            mics_training = subprocess.run(
                (
                    sys.executable,
                    str(project_root / "scripts" / "run_mics_training.py"),
                    "--dataset-root",
                    str(dataset_root),
                    "--scenario-config",
                    str(scenario_config),
                    "--importance-output",
                    str(importance_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "mics-cpu-smoke",
                    "--rounds",
                    "1",
                    "--clients-per-round",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--base-channels",
                    "2",
                    "--num-levels",
                    "2",
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
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if mics_training.returncode != 0:
                self.fail(
                    "MICS training subprocess failed:\n"
                    f"stdout:\n{mics_training.stdout}\n"
                    f"stderr:\n{mics_training.stderr}"
                )
            mics_run = output_root / "mics-cpu-smoke"
            mics_status = json.loads(
                (mics_run / "status.json").read_text(encoding="utf-8")
            )
            mics_config = json.loads(
                (mics_run / "resolved_config.json").read_text(encoding="utf-8")
            )
            mics_metrics = json.loads(
                (mics_run / "round_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn(
                "MICS TRACKED FEDERATED TRAINING PASSED",
                mics_training.stdout,
            )
            self.assertEqual(mics_status["state"], "completed")
            self.assertEqual(mics_config["method"], "mics_impact")
            self.assertEqual(mics_metrics["round"], 0)
            self.assertIn(
                "component_queues_after_round",
                mics_metrics["selector_outcome"],
            )
            self.assertTrue((mics_run / "checkpoints" / "last.pt").is_file())

            mics_evaluation = subprocess.run(
                (
                    sys.executable,
                    str(
                        project_root
                        / "scripts"
                        / "evaluate_brats_checkpoint.py"
                    ),
                    "--run-dir",
                    str(mics_run),
                    "--dataset-root",
                    str(dataset_root),
                    "--scenario-config",
                    str(scenario_config),
                    "--partition",
                    "validation",
                    "--max-cases-per-client",
                    "1",
                    "--roi-size",
                    "4",
                    "4",
                    "4",
                    "--device",
                    "cpu",
                    "--no-amp",
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if mics_evaluation.returncode != 0:
                self.fail(
                    "MICS evaluation subprocess failed:\n"
                    f"stdout:\n{mics_evaluation.stdout}\n"
                    f"stderr:\n{mics_evaluation.stderr}"
                )
            mics_evaluation_path = (
                mics_run
                / "evaluations"
                / "validation_subset_after_round_0001.json"
            )
            comparison_path = root / "paired_comparison.json"
            comparison = subprocess.run(
                (
                    sys.executable,
                    str(project_root / "scripts" / "compare_tracked_runs.py"),
                    "--baseline-run-dir",
                    str(run_dir),
                    "--candidate-run-dir",
                    str(mics_run),
                    "--baseline-evaluation",
                    str(evaluation_path),
                    "--candidate-evaluation",
                    str(mics_evaluation_path),
                    "--output",
                    str(comparison_path),
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if comparison.returncode != 0:
                self.fail(
                    "Paired comparison subprocess failed:\n"
                    f"stdout:\n{comparison.stdout}\n"
                    f"stderr:\n{comparison.stderr}"
                )
            comparison_payload = json.loads(
                comparison_path.read_text(encoding="utf-8")
            )
            self.assertIn(
                "MICS PAIRED TRACKED-RUN COMPARISON PASSED",
                comparison.stdout,
            )
            self.assertEqual(
                comparison_payload["purpose"],
                "paired_integration_audit_not_model_quality",
            )
            self.assertFalse(comparison_payload["interpretation_allowed"])
            self.assertEqual(
                len(
                    comparison_payload["paired_invariants"]
                    ["evaluation_cases_and_masks"]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
