"""Tests for whole-volume sliding-window reconstruction."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from mics.evaluation import (
    gaussian_importance_map,
    sliding_window_inference,
    sliding_window_starts,
)


class PointwiseModel(nn.Module):
    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        availability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del availability
        value = inputs["flair"]
        return torch.cat((-value, value), dim=1)


class SlidingWindowTests(unittest.TestCase):
    def test_starts_cover_far_edge(self) -> None:
        starts = sliding_window_starts(
            (155, 240, 240),
            (96, 96, 96),
            overlap=0.5,
        )
        self.assertEqual(starts[0][-1], 59)
        self.assertEqual(starts[1][-1], 144)
        self.assertEqual(starts[2][-1], 144)
        self.assertEqual(
            len(starts[0]) * len(starts[1]) * len(starts[2]),
            48,
        )

    def test_gaussian_map_is_positive_and_normalized(self) -> None:
        importance = gaussian_importance_map((5, 6, 7))
        self.assertEqual(tuple(importance.shape), (1, 1, 5, 6, 7))
        self.assertGreater(float(importance.min()), 0.0)
        self.assertAlmostEqual(float(importance.max()), 1.0)

    def test_reconstructs_pointwise_model_exactly(self) -> None:
        inputs = {
            "flair": torch.randn(1, 1, 5, 6, 7),
        }
        model = PointwiseModel()
        model.train()
        result = sliding_window_inference(
            model,
            inputs,
            availability=None,
            roi_size=(4, 4, 4),
            overlap=0.5,
            device="cpu",
            amp=False,
        )
        expected = model(inputs)

        self.assertTrue(torch.allclose(result.logits, expected, atol=1e-5))
        self.assertEqual(result.original_shape, (5, 6, 7))
        self.assertTrue(model.training)

    def test_pads_volume_smaller_than_roi_and_restores_shape(self) -> None:
        inputs = {"flair": torch.randn(1, 1, 2, 3, 4)}
        result = sliding_window_inference(
            PointwiseModel(),
            inputs,
            availability=None,
            roi_size=(4, 4, 4),
            overlap=0.5,
            device="cpu",
            amp=False,
        )

        self.assertEqual(tuple(result.logits.shape), (1, 2, 2, 3, 4))
        self.assertEqual(result.padded_shape, (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
