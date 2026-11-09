"""Tests for client-masked whole-volume federated evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from mics import MultiEncoderUNet3D
from mics.data import load_brats_scenario
from mics.evaluation import (
    estimate_leave_one_modality_out_importance,
    evaluate_brats_client_partition,
    normalise_nonnegative_importance,
)


class FederatedEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        (self.dataset_root / "vol").mkdir(parents=True)
        (self.dataset_root / "seg").mkdir()
        for index in range(1, 11):
            case_id = f"Case{index:03d}"
            volume = np.zeros((4, 4, 4, 4), dtype=np.float32)
            volume[..., index % 4] = 1.0
            target = np.zeros((4, 4, 4), dtype=np.uint8)
            target[1:3, 1:3, 1:3] = 3
            np.save(self.dataset_root / "vol" / f"{case_id}.npy", volume)
            np.save(self.dataset_root / "seg" / f"{case_id}.npy", target)

        scenario_dir = self.root / "balanced"
        scenario_dir.mkdir()
        assignments = {
            "C1": {
                "train": "Case001",
                "calibration": "Case003",
                "validation": "Case005",
                "test": "Case007",
            },
            "C2": {
                "train": "Case002",
                "calibration": "Case004",
                "validation": "Case006",
                "test": "Case008",
            },
        }
        clients = {}
        for client_id, partitions in assignments.items():
            files = {}
            for partition, case_id in partitions.items():
                path = scenario_dir / f"{client_id}_{partition}.csv"
                path.write_text(f"{case_id}\n", encoding="utf-8")
                files[partition] = path.name
            clients[client_id] = {"files": files}
        global_test = scenario_dir / "global_test.csv"
        global_test.write_text("Case009\nCase010\n", encoding="utf-8")
        config = scenario_dir / "materialized_config.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 2,
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
                    "global_test_file": global_test.name,
                    "clients": clients,
                }
            ),
            encoding="utf-8",
        )
        self.scenario = load_brats_scenario(
            config,
            dataset_root=self.dataset_root,
        )
        self.model = MultiEncoderUNet3D(
            modalities=OrderedDict(
                (("flair", 1), ("t1ce", 1), ("t1", 1), ("t2", 1))
            ),
            num_classes=4,
            base_channels=2,
            num_levels=2,
            fusion="mean_var",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_evaluates_all_clients_and_both_reductions(self) -> None:
        self.model.train()
        result = evaluate_brats_client_partition(
            self.model,
            self.scenario,
            self.dataset_root,
            partition="validation",
            roi_size=(4, 4, 4),
            overlap=0.5,
            device="cpu",
            amp=False,
        )

        self.assertTrue(result["complete_partition"])
        self.assertEqual(result["client_count"], 2)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["per_client"]["C1"]["modalities"], ["flair", "t1ce"])
        self.assertEqual(result["per_client"]["C2"]["modalities"], ["t1", "t2"])
        expected_macro = sum(
            client["dice"]["mean"]
            for client in result["per_client"].values()
        ) / 2
        self.assertAlmostEqual(result["macro_client"]["dice"]["mean"], expected_macro)
        self.assertEqual(result["sample_weighted"]["case_count"], 2)
        self.assertEqual(result["peak_cuda_allocated_bytes"], 0)
        self.assertTrue(self.model.training)
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")

    def test_modality_overrides_are_recorded_as_incomplete_evaluation(self) -> None:
        result = evaluate_brats_client_partition(
            self.model,
            self.scenario,
            self.dataset_root,
            partition="validation",
            modality_overrides={"C1": ("flair",), "C2": ("t2",)},
            roi_size=(4, 4, 4),
            device="cpu",
            amp=False,
        )

        self.assertFalse(result["complete_partition"])
        self.assertEqual(result["per_client"]["C1"]["modalities"], ["flair"])
        self.assertEqual(result["per_client"]["C2"]["modalities"], ["t2"])

    def test_validation_only_leave_one_out_importance_and_probe_cost(self) -> None:
        result = estimate_leave_one_modality_out_importance(
            self.model,
            self.scenario,
            self.dataset_root,
            roi_size=(4, 4, 4),
            overlap=0.5,
            device="cpu",
            amp=False,
        )

        self.assertEqual(result["partition"], "validation")
        self.assertFalse(result["test_data_accessed"])
        self.assertTrue(result["complete_validation_probe"])
        self.assertEqual(result["primary_reduction"], "macro_client")
        self.assertEqual(
            result["region_weights"],
            {"WT": 1.0 / 3.0, "TC": 1.0 / 3.0, "ET": 1.0 / 3.0},
        )
        self.assertEqual(set(result["modalities"]), {"flair", "t1ce", "t1", "t2"})
        self.assertAlmostEqual(
            sum(result["normalised_selection_weights"].values()),
            1.0,
        )
        for modality in result["modalities"].values():
            self.assertEqual(modality["client_count"], 1)
            self.assertEqual(modality["case_count"], 1)
            self.assertEqual(len(modality["case_deltas"]), 1)
            self.assertIsNone(
                modality["uncertainty_across_client_means"]["standard_error"]
            )
        self.assertEqual(
            result["probe_cost"]["native_baseline_case_evaluations"], 2
        )
        self.assertEqual(
            result["probe_cost"]["additional_ablation_case_evaluations"], 4
        )
        self.assertEqual(result["probe_cost"]["total_case_evaluations"], 6)
        self.assertEqual(result["probe_cost"]["total_sliding_windows"], 6)

    def test_nonpositive_importance_uses_explicit_uniform_fallback(self) -> None:
        weights, fallback = normalise_nonnegative_importance(
            {"flair": -0.1, "t1ce": 0.0, "t1": -0.2, "t2": 0.0}
        )

        self.assertTrue(fallback)
        self.assertEqual(set(weights.values()), {0.25})

    def test_calibration_importance_is_complete_and_disjoint(self) -> None:
        result = estimate_leave_one_modality_out_importance(
            self.model,
            self.scenario,
            self.dataset_root,
            partition="calibration",
            roi_size=(4, 4, 4),
            device="cpu",
            amp=False,
        )

        self.assertEqual(result["partition"], "calibration")
        self.assertTrue(result["complete_partition_probe"])
        self.assertTrue(result["complete_calibration_probe"])
        self.assertFalse(result["complete_validation_probe"])
        calibration_cases = {
            record["case_id"] for record in result["baseline_evaluation"]["cases"]
        }
        validation_cases = {
            case_id
            for client in self.scenario.clients
            for case_id in client.validation_case_ids
        }
        self.assertFalse(calibration_cases.intersection(validation_cases))


if __name__ == "__main__":
    unittest.main()
