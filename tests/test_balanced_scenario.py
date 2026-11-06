"""Tests for the single modality-symmetric BraTS experiment scenario."""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from mics.data import (
    BRATS_BALANCED_PROFILE_SEED,
    BRATS_MODALITIES,
    balanced_brats_modalities,
    materialize_balanced_brats_scenario,
    validate_balanced_brats_modalities,
)


class BalancedScenarioTests(unittest.TestCase):
    def test_profile_is_deterministic_and_modality_symmetric(self) -> None:
        first = balanced_brats_modalities(seed=BRATS_BALANCED_PROFILE_SEED)
        second = balanced_brats_modalities(seed=BRATS_BALANCED_PROFILE_SEED)
        validate_balanced_brats_modalities(first)

        self.assertEqual(first, second)
        self.assertEqual(
            Counter(
                modality
                for modalities in first.values()
                for modality in modalities
            ),
            Counter({modality: 5 for modality in BRATS_MODALITIES}),
        )
        pair_counts = Counter()
        for modalities in first.values():
            for pair in itertools.combinations(sorted(modalities), 2):
                pair_counts[frozenset(pair)] += 1
        self.assertEqual(set(pair_counts.values()), {3})

    def test_materializer_preserves_cases_and_replaces_only_masks(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            (dataset_root / "vol").mkdir(parents=True)
            (dataset_root / "seg").mkdir()
            source_dir = root / "splitA"
            source_dir.mkdir()

            clients = {}
            case_index = 1
            for client_index in range(1, 9):
                client_id = f"C{client_index}"
                files = {}
                for partition in ("train", "validation", "test"):
                    case_id = f"Case{case_index:03d}"
                    case_index += 1
                    np.save(
                        dataset_root / "vol" / f"{case_id}.npy",
                        np.zeros((1, 1, 1, 4), dtype=np.float32),
                    )
                    np.save(
                        dataset_root / "seg" / f"{case_id}.npy",
                        np.zeros((1, 1, 1), dtype=np.uint8),
                    )
                    split_file = source_dir / f"{client_id}_{partition}.csv"
                    split_file.write_text(f"{case_id}\n", encoding="utf-8")
                    files[partition] = str(split_file)
                clients[client_id] = {"files": files}

            global_ids = []
            for _ in range(2):
                case_id = f"Case{case_index:03d}"
                case_index += 1
                global_ids.append(case_id)
                np.save(
                    dataset_root / "vol" / f"{case_id}.npy",
                    np.zeros((1, 1, 1, 4), dtype=np.float32),
                )
                np.save(
                    dataset_root / "seg" / f"{case_id}.npy",
                    np.zeros((1, 1, 1), dtype=np.uint8),
                )
            global_test = root / "global_test.csv"
            global_test.write_text("\n".join(global_ids) + "\n", encoding="utf-8")

            source_config = source_dir / "materialized_config.json"
            source_config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split_id": "A",
                        "data_seed": 19,
                        "modality_order": ["FLAIR", "T1ce", "T1", "T2"],
                        "masks": {
                            client_id: [1, 1, 1, 1]
                            for client_id in clients
                        },
                        "expected_modality_pool_sizes": {
                            modality: 8
                            for modality in ("FLAIR", "T1ce", "T1", "T2")
                        },
                        "global_test_file": str(global_test),
                        "clients": clients,
                        "mapping_fingerprint": "source-fingerprint",
                    }
                ),
                encoding="utf-8",
            )

            output = root / "balanced"
            scenario = materialize_balanced_brats_scenario(
                source_config,
                output,
                dataset_root=dataset_root,
            )
            payload = json.loads(
                (output / "materialized_config.json").read_text(encoding="utf-8")
            )

            self.assertEqual(scenario.scenario_id, "balanced")
            self.assertEqual(
                scenario.modality_pool_sizes,
                {modality: 5 for modality in BRATS_MODALITIES},
            )
            self.assertEqual(scenario.covered_case_count, 26)
            self.assertEqual(
                payload["source_scenario"]["mapping_fingerprint"],
                "source-fingerprint",
            )
            self.assertFalse(payload["source_scenario"]["modality_masks_used"])
            self.assertTrue(
                payload["modality_profile"]["derived_independently_of_patient_data"]
            )
            self.assertTrue((output / "global_test.csv").is_file())


if __name__ == "__main__":
    unittest.main()
