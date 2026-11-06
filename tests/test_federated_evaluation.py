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
from mics.evaluation import evaluate_brats_client_partition


class FederatedEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        (self.dataset_root / "vol").mkdir(parents=True)
        (self.dataset_root / "seg").mkdir()
        for index in range(1, 9):
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
            "C1": {"train": "Case001", "validation": "Case003", "test": "Case005"},
            "C2": {"train": "Case002", "validation": "Case004", "test": "Case006"},
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
        global_test.write_text("Case007\nCase008\n", encoding="utf-8")
        config = scenario_dir / "materialized_config.json"
        config.write_text(
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


if __name__ == "__main__":
    unittest.main()
