"""Tests for authoritative BraTS data-and-modality scenarios."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mics.data import (
    build_brats_scenario_dataset,
    load_brats_scenario,
    validate_matched_brats_scenarios,
)


class BraTSScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        (self.dataset_root / "vol").mkdir(parents=True)
        (self.dataset_root / "seg").mkdir()
        for index in range(1, 9):
            case_id = f"Case{index:03d}"
            np.save(
                self.dataset_root / "vol" / f"{case_id}.npy",
                np.zeros((4, 4, 4, 4), dtype=np.float32),
            )
            np.save(
                self.dataset_root / "seg" / f"{case_id}.npy",
                np.zeros((4, 4, 4), dtype=np.uint8),
            )
        self.bundle = self.root / "bundle"
        self.scenario_a = self._write_scenario(
            "splitA",
            "A",
            {"C1": [1, 0, 1, 1], "C2": [1, 1, 0, 0]},
        )
        self.scenario_b = self._write_scenario(
            "splitB",
            "B",
            {"C1": [0, 1, 1, 1], "C2": [1, 1, 0, 0]},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_scenario(
        self,
        directory_name: str,
        scenario_id: str,
        masks: dict[str, list[int]],
    ) -> Path:
        directory = self.bundle / directory_name
        directory.mkdir(parents=True)
        assignments = {
            "C1": {"train": "Case001", "validation": "Case003", "test": "Case005"},
            "C2": {"train": "Case002", "validation": "Case004", "test": "Case006"},
        }
        clients = {}
        for client_id, partitions in assignments.items():
            files = {}
            for partition, case_id in partitions.items():
                filename = f"{client_id}_{partition}.csv"
                (directory / filename).write_text(f"{case_id}\n", encoding="utf-8")
                files[partition] = f"/stale/host/path/{filename}"
            clients[client_id] = {"files": files}

        global_test = self.bundle / "global_test.csv"
        global_test.write_text("Case007\nCase008\n", encoding="utf-8")
        pool_sizes = {
            modality: sum(bool(mask[index]) for mask in masks.values())
            for index, modality in enumerate(("FLAIR", "T1ce", "T1", "T2"))
        }
        config = {
            "schema_version": 1,
            "split_id": scenario_id,
            "data_seed": 17,
            "modality_order": ["FLAIR", "T1ce", "T1", "T2"],
            "modality_masks": masks,
            "expected_modality_pool_sizes": pool_sizes,
            "global_test_file": "/stale/host/path/global_test.csv",
            "clients": clients,
        }
        config_path = directory / "materialized_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_loads_portable_bundle_and_validates_complete_dataset(self) -> None:
        scenario = load_brats_scenario(
            self.scenario_a,
            dataset_root=self.dataset_root,
        )

        self.assertEqual(
            scenario.partition_counts,
            {"train": 2, "validation": 2, "test": 2},
        )
        self.assertEqual(scenario.covered_case_count, 8)
        self.assertEqual(
            scenario.modality_pool_sizes,
            {"flair": 2, "t1ce": 1, "t1": 1, "t2": 1},
        )
        self.assertEqual(scenario.client("C1").modalities, ("flair", "t1", "t2"))
        self.assertEqual(scenario.client(2).train_case_ids, ("Case002",))

    def test_builds_dataset_from_recorded_partition_and_mask(self) -> None:
        scenario = load_brats_scenario(self.scenario_a, dataset_root=self.dataset_root)
        dataset = build_brats_scenario_dataset(
            self.dataset_root,
            scenario,
            "C1",
            partition="train",
        )

        self.assertEqual(dataset.case_ids, ("Case001",))
        self.assertEqual(dataset.available_modalities, ("flair", "t1", "t2"))

    def test_matched_scenarios_require_same_cases_and_changed_masks(self) -> None:
        scenario_a = load_brats_scenario(
            self.scenario_a,
            dataset_root=self.dataset_root,
        )
        scenario_b = load_brats_scenario(
            self.scenario_b,
            dataset_root=self.dataset_root,
        )

        self.assertEqual(
            validate_matched_brats_scenarios(scenario_a, scenario_b),
            ("C1",),
        )

    def test_rejects_cross_partition_leakage(self) -> None:
        leaked = self.scenario_a.parent / "C1_test.csv"
        leaked.write_text("Case001\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            load_brats_scenario(self.scenario_a, dataset_root=self.dataset_root)

    def test_rejects_incorrect_modality_pool_metadata(self) -> None:
        payload = json.loads(self.scenario_a.read_text(encoding="utf-8"))
        payload["expected_modality_pool_sizes"]["FLAIR"] = 1
        self.scenario_a.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pool sizes"):
            load_brats_scenario(self.scenario_a)


if __name__ == "__main__":
    unittest.main()
