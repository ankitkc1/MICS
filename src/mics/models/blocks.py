"""Reusable building blocks for a memory-conscious 3D U-Net."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest useful GroupNorm divisor no greater than maximum."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(kind: str, channels: int) -> nn.Module:
    """Construct normalization suitable for small 3D medical-image batches."""

    normalized = kind.lower().replace("_", "")
    if normalized == "group":
        return nn.GroupNorm(_group_count(channels), channels)
    if normalized == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if normalized == "batch":
        return nn.BatchNorm3d(channels)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {kind!r}")


class ResidualConvBlock3D(nn.Module):
    """Two 3D convolutions with normalization, activation, and a residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: str = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm(norm, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(norm, out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class Encoder3D(nn.Module):
    """U-Net encoder returning features at every spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        feature_channels: Sequence[int],
        *,
        norm: str = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in feature_channels)
        if len(channels) < 2 or any(value <= 0 for value in channels):
            raise ValueError("feature_channels must contain at least two positive values.")

        stages: list[nn.Module] = []
        previous = in_channels
        for current in channels:
            stages.append(
                ResidualConvBlock3D(
                    previous,
                    current,
                    norm=norm,
                    dropout=dropout,
                )
            )
            previous = current
        self.stages = nn.ModuleList(stages)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if index < len(self.stages) - 1:
                x = self.pool(x)
        return tuple(features)


class UpBlock3D(nn.Module):
    """Upsample, concatenate a fused skip feature, and refine it."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        norm: str = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.refine = ResidualConvBlock3D(
            out_channels + skip_channels,
            out_channels,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class Decoder3D(nn.Module):
    """Shared U-Net decoder operating on fused multi-scale features."""

    def __init__(
        self,
        feature_channels: Sequence[int],
        *,
        norm: str = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in feature_channels)
        blocks: list[nn.Module] = []
        for level in range(len(channels) - 1, 0, -1):
            blocks.append(
                UpBlock3D(
                    in_channels=channels[level],
                    skip_channels=channels[level - 1],
                    out_channels=channels[level - 1],
                    norm=norm,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, fused_features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(fused_features) != len(self.blocks) + 1:
            raise ValueError("The number of fused feature levels does not match the decoder.")
        x = fused_features[-1]
        for block, skip in zip(self.blocks, reversed(fused_features[:-1]), strict=True):
            x = block(x, skip)
        return x

