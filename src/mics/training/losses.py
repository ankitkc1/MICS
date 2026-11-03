"""Segmentation losses used by MICS experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    include_background: bool = False,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Multi-class soft Dice loss averaged over samples and classes."""

    if logits.ndim != 5:
        raise ValueError("logits must have shape [B, C, D, H, W].")
    if target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("target must have shape [B, D, H, W] matching logits.")
    if logits.shape[1] < 2:
        raise ValueError("At least two classes are required.")

    target = target.long()
    if target.numel() and (
        int(target.min()) < 0 or int(target.max()) >= logits.shape[1]
    ):
        raise ValueError("target contains a class outside the logits range.")

    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1)
    one_hot = one_hot.to(dtype=probabilities.dtype)

    if not include_background:
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]

    spatial_dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * one_hot).sum(spatial_dims)
    denominator = probabilities.sum(spatial_dims) + one_hot.sum(spatial_dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    """Weighted sum of soft Dice and voxel-wise cross-entropy."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        cross_entropy_weight: float = 1.0,
        class_weights: Sequence[float] | None = None,
        include_background_dice: bool = False,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or cross_entropy_weight < 0:
            raise ValueError("Loss weights cannot be negative.")
        if dice_weight == 0 and cross_entropy_weight == 0:
            raise ValueError("At least one loss weight must be positive.")
        self.dice_weight = float(dice_weight)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.include_background_dice = bool(include_background_dice)
        weights = (
            None
            if class_weights is None
            else torch.tensor(tuple(class_weights), dtype=torch.float32)
        )
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = soft_dice_loss(
            logits,
            target,
            include_background=self.include_background_dice,
        )
        cross_entropy = F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weights,
        )
        return (
            self.dice_weight * dice
            + self.cross_entropy_weight * cross_entropy
        )
