"""Availability-aware fusion for variable multimodal inputs."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import make_norm


def _validate_fusion_inputs(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if features.ndim != 6:
        raise ValueError("features must have shape [B, M, C, D, H, W].")
    if mask.ndim != 2 or mask.shape != features.shape[:2]:
        raise ValueError("mask must have shape [B, M] matching the feature tensor.")
    mask = mask.to(device=features.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every sample must contain at least one available modality.")
    return mask


class MaskedMeanFusion3D(nn.Module):
    """Parameter-free mean over the available modality representations."""

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _validate_fusion_inputs(features, mask)
        expanded = mask[:, :, None, None, None, None].to(features.dtype)
        denominator = expanded.sum(dim=1).clamp_min(1.0)
        fused = (features * expanded).sum(dim=1) / denominator
        weights = mask.to(features.dtype) / mask.sum(dim=1, keepdim=True).to(features.dtype)
        return fused, weights


class MaskedMeanVarianceFusion3D(nn.Module):
    """HeMIS-style mean/variance statistics followed by a learned projection."""

    def __init__(self, channels: int, *, norm: str = "group") -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False),
            make_norm(norm, channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _validate_fusion_inputs(features, mask)
        expanded = mask[:, :, None, None, None, None].to(features.dtype)
        denominator = expanded.sum(dim=1).clamp_min(1.0)
        mean = (features * expanded).sum(dim=1) / denominator
        centered = features - mean[:, None]
        variance = (centered.square() * expanded).sum(dim=1) / denominator
        fused = self.projection(torch.cat((mean, variance), dim=1))
        weights = mask.to(features.dtype) / mask.sum(dim=1, keepdim=True).to(features.dtype)
        return fused, weights


class MaskedAttentionFusion3D(nn.Module):
    """Spatial attention over available modalities with a shared scorer."""

    def __init__(self, channels: int, num_modalities: int) -> None:
        super().__init__()
        self.scorer = nn.Conv3d(channels, 1, kernel_size=1)
        self.modality_bias = nn.Parameter(torch.zeros(num_modalities))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _validate_fusion_inputs(features, mask)
        batch, modalities, channels, depth, height, width = features.shape
        flat = features.reshape(batch * modalities, channels, depth, height, width)
        scores = self.scorer(flat).reshape(batch, modalities, 1, depth, height, width)
        scores = scores + self.modality_bias[None, :, None, None, None, None]
        expanded_mask = mask[:, :, None, None, None, None]
        scores = scores.masked_fill(~expanded_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        fused = (weights * features).sum(dim=1)
        return fused, weights.squeeze(2)


def build_fusion(
    kind: str,
    channels: int,
    num_modalities: int,
    *,
    norm: str = "group",
) -> nn.Module:
    normalized = kind.lower().replace("-", "_")
    if normalized in {"mean", "masked_mean"}:
        return MaskedMeanFusion3D()
    if normalized in {"mean_var", "mean_variance", "masked_mean_variance"}:
        return MaskedMeanVarianceFusion3D(channels, norm=norm)
    if normalized in {"attention", "masked_attention"}:
        return MaskedAttentionFusion3D(channels, num_modalities)
    raise ValueError(f"Unsupported fusion method: {kind!r}")

