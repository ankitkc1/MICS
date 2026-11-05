"""BraTS Whole Tumour, Tumour Core, and Enhancing Tumour metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from scipy import ndimage

BRATS_REGIONS: Mapping[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}
BRATS_HD95_PENALTY_MM = 373.13


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


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )
    return mask ^ eroded


def _binary_hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    one_empty_penalty: float,
) -> float:
    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())
    if not prediction_nonempty and not target_nonempty:
        return 0.0
    if prediction_nonempty != target_nonempty:
        return float(one_empty_penalty)

    predicted_surface = _surface(prediction)
    target_surface = _surface(target)
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface,
        sampling=spacing,
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        ~predicted_surface,
        sampling=spacing,
    )
    distances = np.concatenate(
        (
            distance_to_target[predicted_surface],
            distance_to_prediction[target_surface],
        )
    )
    return float(np.percentile(distances, 95))


def brats_region_hd95(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    reduction: str = "mean",
    one_empty_penalty: float = BRATS_HD95_PENALTY_MM,
) -> dict[str, torch.Tensor]:
    """Compute case-wise symmetric surface HD95 for WT, TC, and ET.

    BraTS 2020 volumes use one-millimetre isotropic spacing. Both-empty regions
    receive zero millimetres; exactly-one-empty regions receive the declared
    fixed penalty (373.13 mm by default).
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
    if len(spacing) != 3 or any(float(value) <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values.")
    if one_empty_penalty < 0:
        raise ValueError("one_empty_penalty cannot be negative.")

    prediction_cpu = prediction.detach().long().cpu()
    target_cpu = target.detach().long().cpu()
    scores: dict[str, torch.Tensor] = {}
    for name, labels in BRATS_REGIONS.items():
        values = []
        for predicted_case, target_case in zip(
            prediction_cpu,
            target_cpu,
            strict=True,
        ):
            predicted_region = np.isin(predicted_case.numpy(), labels)
            target_region = np.isin(target_case.numpy(), labels)
            values.append(
                _binary_hd95(
                    predicted_region,
                    target_region,
                    spacing=tuple(float(value) for value in spacing),
                    one_empty_penalty=one_empty_penalty,
                )
            )
        score = torch.tensor(values, dtype=torch.float32)
        if reduction == "mean":
            score = score.mean()
        elif reduction == "sum":
            score = score.sum()
        scores[name] = score
    return scores
