"""End-to-end CPU checks for the registered MICS ablation variants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mics.experiments import generate_system_trace


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
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

    scenario_dir = root / "bundle" / "balanced"
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
    global_test = root / "bundle" / "global_test.csv"
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
    importance = root / "importance.json"
    importance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "integration_smoke_not_modality_importance",
                "checkpoint": {"sha256": "fixture"},
                "importance": {
                    "test_data_accessed": False,
                    "complete_validation_probe": False,
                    "normalised_selection_weights": {
                        "flair": 0.4,
                        "t1ce": 0.3,
                        "t1": 0.2,
                        "t2": 0.1,
                    },
                    "probe_cost": {"total_case_evaluations": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_root, scenario_config, importance


class MicsAblationRunnerTests(unittest.TestCase):
    def test_v2_selector_preflight_records_calibration_and_inputs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            dataset_root, scenario_config, importance = _write_fixture(root)
            trace = generate_system_trace(
                ("C1", "C2"),
                rounds=8,
                seed=19,
                availability_probabilities={"C1": 1.0, "C2": 1.0},
                completion_probabilities={"C1": 1.0, "C2": 1.0},
                minimum_available_clients=2,
                trace_id="v2-preflight-fixture",
            )
            trace_path = root / "trace.json"
            trace_path.write_text(
                json.dumps(trace.as_dict()),
                encoding="utf-8",
            )
            output = root / "preflight.json"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "src")
            result = subprocess.run(
                (
                    sys.executable,
                    str(
                        project_root
                        / "scripts"
                        / "simulate_mics_v2_selector.py"
                    ),
                    "--dataset-root",
                    str(dataset_root),
                    "--scenario-config",
                    str(scenario_config),
                    "--system-trace",
                    str(trace_path),
                    "--importance-output",
                    str(importance),
                    "--budget",
                    "1",
                    "--source-revision",
                    "fixture-revision",
                    "--output",
                    str(output),
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.fail(
                    "MICS-v2 selector preflight failed:\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_revision"], "fixture-revision")
            self.assertEqual(
                set(payload["methods"]),
                {
                    "mics_v2_impact",
                    "mics_v2_core",
                    "mics_v2_impact_no_client_queue",
                },
            )
            self.assertEqual(set(payload["client_targets"]), {"C1", "C2"})
            self.assertIn("clients", payload["calibration"])
            self.assertIsNotNone(
                payload["inputs"]["system_trace"]["sha256"]
            )

    def test_registered_ablation_variants_write_distinct_provenance(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            dataset_root, scenario_config, importance = _write_fixture(root)
            output_root = root / "results"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "src")
            variants = (
                (
                    "core",
                    ("--impact-weight", "0"),
                    "mics_core",
                    1.0,
                    "submodular_service",
                    "none",
                    "v1",
                    False,
                ),
                (
                    "no-queue",
                    (
                        "--importance-output",
                        str(importance),
                        "--ablation",
                        "no_component_queue",
                    ),
                    "mics_impact_no_queue",
                    0.0,
                    "submodular_service",
                    "validation_derived",
                    "v1",
                    False,
                ),
                (
                    "modular",
                    (
                        "--importance-output",
                        str(importance),
                        "--ablation",
                        "modular_additive",
                    ),
                    "mics_impact_modular",
                    1.0,
                    "modular_additive",
                    "validation_derived",
                    "v1",
                    False,
                ),
                (
                    "uniform",
                    ("--ablation", "uniform_importance"),
                    "mics_impact_uniform",
                    1.0,
                    "submodular_service",
                    "uniform",
                    "v1",
                    False,
                ),
                (
                    "v2-impact",
                    (
                        "--framework-version",
                        "v2",
                        "--importance-output",
                        str(importance),
                    ),
                    "mics_v2_impact",
                    1.0,
                    "submodular_service",
                    "validation_derived",
                    "v2",
                    True,
                ),
                (
                    "v2-core",
                    ("--framework-version", "v2", "--impact-weight", "0"),
                    "mics_v2_core",
                    1.0,
                    "submodular_service",
                    "none",
                    "v2",
                    True,
                ),
                (
                    "v2-no-client-queue",
                    (
                        "--framework-version",
                        "v2",
                        "--importance-output",
                        str(importance),
                        "--ablation",
                        "no_client_queue",
                    ),
                    "mics_v2_impact_no_client_queue",
                    1.0,
                    "submodular_service",
                    "validation_derived",
                    "v2",
                    False,
                ),
            )
            for (
                run_id,
                variant_arguments,
                expected_method,
                expected_queue_weight,
                expected_subset_utility,
                expected_importance_mode,
                expected_framework_version,
                expected_client_queue,
            ) in variants:
                with self.subTest(run_id=run_id):
                    command = (
                        sys.executable,
                        str(project_root / "scripts" / "run_mics_training.py"),
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
                        "--checkpoint-every",
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
                        *variant_arguments,
                    )
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
                            f"Ablation {run_id} failed:\n"
                            f"stdout:\n{result.stdout}\n"
                            f"stderr:\n{result.stderr}"
                        )
                    run_dir = output_root / run_id
                    config = json.loads(
                        (run_dir / "resolved_config.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    selection = config["selection"]
                    self.assertEqual(config["method"], expected_method)
                    self.assertEqual(
                        selection["component_queue_weight"],
                        expected_queue_weight,
                    )
                    self.assertEqual(
                        selection["subset_utility"],
                        expected_subset_utility,
                    )
                    self.assertEqual(
                        selection["importance_mode"],
                        expected_importance_mode,
                    )
                    self.assertEqual(
                        selection["framework_version"],
                        expected_framework_version,
                    )
                    self.assertEqual(
                        selection["client_queue_enabled"],
                        expected_client_queue,
                    )
                    if expected_client_queue:
                        self.assertEqual(
                            set(selection["client_targets"]),
                            {"C1", "C2"},
                        )
                        self.assertIsNotNone(
                            selection["client_target_calibration"]
                        )
                    else:
                        self.assertEqual(selection["client_targets"], {})
                        self.assertIsNone(
                            selection["client_target_calibration"]
                        )
                    self.assertTrue(
                        (run_dir / "checkpoints" / "last.pt").is_file()
                    )


if __name__ == "__main__":
    unittest.main()
