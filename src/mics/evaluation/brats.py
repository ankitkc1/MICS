"""BraTS Whole Tumour, Tumour Core, and Enhancing Tumour metrics."""

from __future__ import annotations

from collections.abc import Mapping

import torch

BRATS_REGIONS: Mapping[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}


def _region_mask(labels: torch.Tensor, region_labels: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for value in region_labels:
        mask |= labels == value
    return mask


def brats_region_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    empty_score: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute hard-label Dice for WT, TC, and ET.

    Inputs may have shape [D, H, W] or [B, D, H, W]. When both prediction and
    target are empty for a region, empty_score is returned for that sample.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes.")
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    if prediction.ndim != 4:
        raise ValueError("prediction and target must have shape [B, D, H, W].")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")

    prediction = prediction.long()
    target = target.long()
    scores: dict[str, torch.Tensor] = {}
    spatial_dims = (1, 2, 3)
    for name, labels in BRATS_REGIONS.items():
        predicted_region = _region_mask(prediction, labels)
        target_region = _region_mask(target, labels)
        intersection = (predicted_region & target_region).sum(spatial_dims).float()
        denominator = (
            predicted_region.sum(spatial_dims) + target_region.sum(spatial_dims)
        ).float()
        score = torch.where(
            denominator > 0,
            2.0 * intersection / denominator.clamp_min(1.0),
            torch.full_like(denominator, float(empty_score)),
        )
        if reduction == "mean":
            score = score.mean()
        elif reduction == "sum":
            score = score.sum()
        scores[name] = score
    return scores


def brats_dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    empty_score: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Convert class logits to hard predictions and compute region Dice."""

    if logits.ndim != 5:
        raise ValueError("logits must have shape [B, C, D, H, W].")
    return brats_region_dice(
        logits.argmax(dim=1),
        target,
        reduction=reduction,
        empty_score=empty_score,
    )
