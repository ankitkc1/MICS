"""Tests for segmentation losses and BraTS region metrics."""

from __future__ import annotations

import unittest

import torch

from mics.evaluation import brats_dice_from_logits, brats_region_dice
from mics.training import DiceCrossEntropyLoss, soft_dice_loss


class BraTSMetricTests(unittest.TestCase):
    def test_perfect_prediction_has_unit_region_dice(self) -> None:
        target = torch.tensor(
            [
                [
                    [[0, 1], [2, 3]],
                    [[0, 0], [1, 3]],
                ]
            ]
        )
        scores = brats_region_dice(target.clone(), target)
        for region in ("WT", "TC", "ET"):
            self.assertAlmostEqual(float(scores[region]), 1.0)

    def test_empty_regions_use_declared_empty_score(self) -> None:
        empty = torch.zeros(1, 2, 2, 2, dtype=torch.long)
        scores = brats_region_dice(empty, empty, empty_score=1.0)
        for score in scores.values():
            self.assertEqual(float(score), 1.0)

    def test_logits_are_converted_to_hard_predictions(self) -> None:
        target = torch.randint(0, 4, (2, 3, 4, 5))
        logits = torch.full((2, 4, 3, 4, 5), -10.0)
        logits.scatter_(1, target.unsqueeze(1), 10.0)

        scores = brats_dice_from_logits(logits, target)

        for score in scores.values():
            self.assertAlmostEqual(float(score), 1.0)

    def test_dice_cross_entropy_is_finite_and_differentiable(self) -> None:
        logits = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
        target = torch.randint(0, 4, (2, 4, 4, 4))
        criterion = DiceCrossEntropyLoss()

        loss = criterion(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))

    def test_soft_dice_rejects_invalid_labels(self) -> None:
        logits = torch.randn(1, 4, 2, 2, 2)
        target = torch.full((1, 2, 2, 2), 4)
        with self.assertRaisesRegex(ValueError, "outside"):
            soft_dice_loss(logits, target)


if __name__ == "__main__":
    unittest.main()
