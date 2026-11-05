"""Memory-bounded three-dimensional sliding-window inference."""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SlidingWindowResult:
    """Whole-volume logits and inference metadata."""

    logits: torch.Tensor
    window_count: int
    elapsed_seconds: float
    original_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]


def sliding_window_starts(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    *,
    overlap: float,
) -> tuple[tuple[int, ...], ...]:
    """Return starts that cover every voxel and always include the far edge."""

    image = tuple(int(value) for value in image_size)
    roi = tuple(int(value) for value in roi_size)
    if len(image) != 3 or len(roi) != 3:
        raise ValueError("image_size and roi_size must contain three values.")
    if any(value <= 0 for value in image + roi):
        raise ValueError("image_size and roi_size must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must lie in [0, 1).")

    starts: list[tuple[int, ...]] = []
    for current, window in zip(image, roi, strict=True):
        if current <= window:
            starts.append((0,))
            continue
        stride = max(int(round(window * (1.0 - overlap))), 1)
        last = current - window
        positions = list(range(0, last + 1, stride))
        if positions[-1] != last:
            positions.append(last)
        starts.append(tuple(positions))
    return tuple(starts)


def gaussian_importance_map(
    roi_size: Sequence[int],
    *,
    sigma_scale: float = 0.125,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create a positive separable Gaussian blend map with maximum one."""

    roi = tuple(int(value) for value in roi_size)
    if len(roi) != 3 or any(value <= 0 for value in roi):
        raise ValueError("roi_size must contain three positive values.")
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive.")

    axes = []
    for size in roi:
        coordinate = torch.arange(size, device=device, dtype=torch.float32)
        center = (size - 1) / 2.0
        sigma = max(size * sigma_scale, 1e-6)
        axes.append(torch.exp(-0.5 * ((coordinate - center) / sigma) ** 2))
    importance = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )
    importance /= importance.max()
    return importance.clamp_min(1e-6)[None, None]


def _pad_inputs(
    inputs: Mapping[str, torch.Tensor],
    roi_size: tuple[int, int, int],
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int]]:
    reference = next(iter(inputs.values()))
    original_shape = tuple(int(value) for value in reference.shape[2:])
    required = tuple(
        max(window - current, 0)
        for current, window in zip(original_shape, roi_size, strict=True)
    )
    padding = (
        0,
        required[2],
        0,
        required[1],
        0,
        required[0],
    )
    if not any(required):
        return dict(inputs), original_shape
    return {
        name: F.pad(tensor, padding, mode="constant", value=0.0)
        for name, tensor in inputs.items()
    }, original_shape


def sliding_window_inference(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    availability: torch.Tensor | None,
    *,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    device: str | torch.device = "cuda",
    amp: bool = True,
    gaussian_sigma_scale: float = 0.125,
) -> SlidingWindowResult:
    """Reconstruct whole-volume logits from overlapping 3D model windows.

    The current evaluator intentionally supports one case at a time so
    subject-level metrics and memory usage remain explicit.
    """

    if not inputs:
        raise ValueError("inputs cannot be empty.")
    roi = tuple(int(value) for value in roi_size)
    if len(roi) != 3 or any(value <= 0 for value in roi):
        raise ValueError("roi_size must contain three positive values.")

    reference = next(iter(inputs.values()))
    if reference.ndim != 5 or reference.shape[0] != 1:
        raise ValueError("inputs must contain one case shaped [1, C, D, H, W].")
    original_spatial = tuple(int(value) for value in reference.shape[2:])
    for name, tensor in inputs.items():
        if tensor.ndim != 5 or tensor.shape[0] != 1:
            raise ValueError(
                f"Input {name!r} must have shape [1, C, D, H, W]."
            )
        if tuple(tensor.shape[2:]) != original_spatial:
            raise ValueError("All modalities must share spatial dimensions.")

    target_device = torch.device(device)
    use_amp = bool(amp and target_device.type == "cuda")
    cpu_inputs = {
        name: tensor.detach().to("cpu")
        for name, tensor in inputs.items()
    }
    padded_inputs, original_shape = _pad_inputs(cpu_inputs, roi)
    padded_shape = tuple(
        int(value)
        for value in next(iter(padded_inputs.values())).shape[2:]
    )
    starts = sliding_window_starts(padded_shape, roi, overlap=overlap)
    positions = tuple(itertools.product(*starts))
    importance = gaussian_importance_map(
        roi,
        sigma_scale=gaussian_sigma_scale,
        device=target_device,
    )

    model.to(target_device)
    was_training = model.training
    model.eval()
    availability_device = (
        None
        if availability is None
        else availability.to(target_device, dtype=torch.bool)
    )
    logits_sum: torch.Tensor | None = None
    weight_sum = torch.zeros(
        (1, 1, *padded_shape),
        device=target_device,
        dtype=torch.float32,
    )
    started = time.perf_counter()

    try:
        with torch.inference_mode():
            for depth, height, width in positions:
                slices = (
                    slice(depth, depth + roi[0]),
                    slice(height, height + roi[1]),
                    slice(width, width + roi[2]),
                )
                window_inputs = {
                    name: tensor[(slice(None), slice(None), *slices)].to(
                        target_device,
                        non_blocking=True,
                    )
                    for name, tensor in padded_inputs.items()
                }
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    window_logits = model(window_inputs, availability_device)
                window_logits = window_logits.float()
                if logits_sum is None:
                    logits_sum = torch.zeros(
                        (1, window_logits.shape[1], *padded_shape),
                        device=target_device,
                        dtype=torch.float32,
                    )
                region = (slice(None), slice(None), *slices)
                logits_sum[region] += window_logits * importance
                weight_sum[(slice(None), slice(None), *slices)] += importance
    finally:
        model.train(was_training)

    if logits_sum is None:
        raise RuntimeError("Sliding-window inference produced no windows.")
    logits = logits_sum / weight_sum.clamp_min(1e-6)
    crop = (
        slice(None),
        slice(None),
        slice(0, original_shape[0]),
        slice(0, original_shape[1]),
        slice(0, original_shape[2]),
    )
    logits = logits[crop].detach().cpu()
    return SlidingWindowResult(
        logits=logits,
        window_count=len(positions),
        elapsed_seconds=time.perf_counter() - started,
        original_shape=original_shape,
        padded_shape=padded_shape,
    )
