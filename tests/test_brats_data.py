"""Tests for the verified BraTS NumPy data contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mics.data import (
    BraTSNpyDataset,
    build_brats_client_dataset,
    read_case_ids,
)


class BraTSNpyDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name) / "dataset"
        (self.root / "vol").mkdir(parents=True)
        (self.root / "seg").mkdir()
        self.split_root = Path(self.temporary.name) / "splits"
        self.split_root.mkdir()

        shape = (10, 12, 8)
        volume = np.zeros((*shape, 4), dtype=np.float32)
        for index in range(4):
            volume[..., index] = float(index + 1)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[4:7, 5:8, 3:6] = 3

        np.save(self.root / "vol" / "Case001.npy", volume)
        np.save(self.root / "seg" / "Case001.npy", segmentation)
        (self.split_root / "c1.csv").write_text("Case001\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_volume_returns_named_dh_w_tensors(self) -> None:
        dataset = BraTSNpyDataset(
            self.root,
            split_file=self.split_root / "c1.csv",
            available_modalities=("flair", "t1ce"),
        )

        sample = dataset[0]

        self.assertEqual(tuple(sample["inputs"]), ("flair", "t1ce"))
        self.assertEqual(sample["inputs"]["flair"].shape, (1, 8, 10, 12))
        self.assertEqual(sample["target"].shape, (8, 10, 12))
        self.assertEqual(sample["target"].dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                sample["availability"],
                torch.tensor([True, True, False, False]),
            )
        )
        self.assertTrue(torch.all(sample["inputs"]["flair"] == 1.0))
        self.assertTrue(torch.all(sample["inputs"]["t1ce"] == 2.0))

    def test_training_patch_is_deterministic_and_contains_foreground(self) -> None:
        dataset = build_brats_client_dataset(
            self.root,
            self.split_root,
            "C1",
            available_modalities=("t1",),
            patch_size=(6, 6, 6),
            training=True,
            foreground_probability=1.0,
            seed=11,
        )

        first = dataset[0]
        second = dataset[0]

        self.assertTrue(torch.equal(first["target"], second["target"]))
        self.assertEqual(first["target"].shape, (6, 6, 6))
        self.assertTrue(torch.any(first["target"] > 0))
        self.assertEqual(first["inputs"]["t1"].shape, (1, 6, 6, 6))

    def test_case_list_supports_header_and_rejects_duplicates(self) -> None:
        header_file = self.split_root / "header.csv"
        header_file.write_text("case_id\nCase001\n", encoding="utf-8")
        self.assertEqual(read_case_ids(header_file), ("Case001",))

        duplicate_file = self.split_root / "duplicates.csv"
        duplicate_file.write_text("Case001\nCase001\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_case_ids(duplicate_file)

    def test_unknown_modality_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown BraTS modalities"):
            BraTSNpyDataset(
                self.root,
                split_file=self.split_root / "c1.csv",
                available_modalities=("pet",),
            )


if __name__ == "__main__":
    unittest.main()
