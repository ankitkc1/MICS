"""Run one real BraTS optimization step on a CUDA device.

This is a systems/integration check, not a model-quality experiment. The model
is randomly initialized and the reported Dice values are measured on one
training patch only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch

from mics import MultiEncoderUNet3D
from mics.data import BRATS_MODALITIES, build_brats_client_dataset
from mics.evaluation import brats_dice_from_logits
from mics.training import DiceCrossEntropyLoss


def parse_patch_size(value: str) -> tuple[int, int, int]:
    result = tuple(int(part.strip()) for part in value.split(","))
    if len(result) != 3 or any(part <= 0 for part in result):
        raise argparse.ArgumentTypeError(
            "patch size must contain three positive comma-separated integers"
        )
    return result


def parse_modalities(value: str) -> tuple[str, ...]:
    requested = tuple(
        part.strip().lower()
        for part in value.split(",")
        if part.strip()
    )
    if not requested:
        raise argparse.ArgumentTypeError("at least one modality is required")
    if len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("modalities cannot contain duplicates")
    unknown = set(requested).difference(BRATS_MODALITIES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown modalities: {sorted(unknown)}"
        )
    return tuple(name for name in BRATS_MODALITIES if name in requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real BraTS forward/backward/optimizer step on CUDA."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MICS_DATA_ROOT"),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=os.environ.get("MICS_SPLIT_ROOT"),
    )
    parser.add_argument("--client", default="C1")
    parser.add_argument(
        "--modalities",
        type=parse_modalities,
        default=BRATS_MODALITIES,
        help="Comma-separated available modalities.",
    )
    parser.add_argument("--patch-size", type=parse_patch_size, default=(96, 96, 96))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument(
        "--fusion",
        choices=("mean", "mean_var", "attention"),
        default="mean_var",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--amp-initial-scale",
        type=float,
        default=4096.0,
        help="Initial CUDA AMP loss scale.",
    )
    parser.add_argument(
        "--amp-max-retries",
        type=int,
        default=8,
        help="Maximum retries after AMP gradient overflow.",
    )
    return parser


def gibibytes(value: int) -> float:
    return value / (1024**3)


def gradient_l2_norm(
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    """Return the global gradient norm without modifying gradients."""

    norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        raise RuntimeError("No gradients were produced for trainable parameters.")
    return torch.stack(norms).norm(2)


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.split_root is None:
        raise SystemExit("Set MICS_SPLIT_ROOT or pass --split-root.")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Submit this script through a Slurm GPU job."
        )
    if arguments.sample_index < 0:
        raise SystemExit("--sample-index cannot be negative.")
    if arguments.base_channels <= 0:
        raise SystemExit("--base-channels must be positive.")
    if arguments.num_levels < 2:
        raise SystemExit("--num-levels must be at least two.")
    if arguments.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive.")
    if arguments.amp_initial_scale <= 0:
        raise SystemExit("--amp-initial-scale must be positive.")
    if arguments.amp_max_retries < 0:
        raise SystemExit("--amp-max-retries cannot be negative.")

    divisor = 2 ** (arguments.num_levels - 1)
    if any(size % divisor for size in arguments.patch_size):
        raise SystemExit(
            f"Each patch dimension must be divisible by {divisor} "
            f"for {arguments.num_levels} levels."
        )

    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    device = torch.device("cuda")

    dataset = build_brats_client_dataset(
        arguments.dataset_root,
        arguments.split_root,
        arguments.client,
        available_modalities=arguments.modalities,
        patch_size=arguments.patch_size,
        training=True,
        foreground_probability=1.0,
        seed=arguments.seed,
    )
    if arguments.sample_index >= len(dataset):
        raise SystemExit(
            f"--sample-index {arguments.sample_index} exceeds "
            f"client dataset size {len(dataset)}."
        )

    sample = dataset[arguments.sample_index]
    inputs = {
        name: tensor.unsqueeze(0).to(device, non_blocking=True)
        for name, tensor in sample["inputs"].items()
    }
    target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
    availability = sample["availability"].unsqueeze(0).to(device)

    model = MultiEncoderUNet3D(
        modalities=OrderedDict((name, 1) for name in BRATS_MODALITIES),
        num_classes=4,
        base_channels=arguments.base_channels,
        num_levels=arguments.num_levels,
        fusion=arguments.fusion,
        modality_dropout=0.0,
    ).to(device)
    model.configure_client_modalities(arguments.modalities)
    model.train()

    criterion = DiceCrossEntropyLoss()
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=arguments.learning_rate,
    )
    use_amp = bool(arguments.amp)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=arguments.amp_initial_scale,
    )
    initial_amp_scale = float(scaler.get_scale())

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    maximum_attempts = 1 + (arguments.amp_max_retries if use_amp else 0)
    overflow_retries = 0
    successful_attempt = 0
    scale_before_success = float(scaler.get_scale())
    for attempt in range(1, maximum_attempts + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(inputs, availability)
            loss = criterion(logits, target)

        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"Non-finite forward loss encountered: {float(loss.detach())}"
            )

        scale_before_attempt = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = gradient_l2_norm(trainable_parameters)
        gradients_are_finite = bool(torch.isfinite(gradient_norm))

        if gradients_are_finite:
            scaler.step(optimizer)
            scaler.update()
            successful_attempt = attempt
            scale_before_success = scale_before_attempt
            break

        if not use_amp:
            raise RuntimeError(
                f"Non-finite gradient norm encountered: {float(gradient_norm)}"
            )

        # GradScaler skips the optimizer update because unscale_ recorded
        # non-finite gradients, then lowers the scale for the retry.
        scaler.step(optimizer)
        scaler.update()
        overflow_retries += 1
        print(
            "AMP gradient overflow: "
            f"attempt={attempt}, scale={scale_before_attempt:g}, "
            f"next_scale={float(scaler.get_scale()):g}"
        )
        if attempt == maximum_attempts:
            raise RuntimeError(
                "AMP gradients remained non-finite after "
                f"{maximum_attempts} attempts; "
                f"final scale={float(scaler.get_scale()):g}."
            )

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    dice = brats_dice_from_logits(logits.detach(), target)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    summary = {
        "status": "passed",
        "purpose": "integration_smoke_test_not_model_quality",
        "case_id": sample["case_id"],
        "client": str(arguments.client),
        "modalities": list(arguments.modalities),
        "patch_size": list(arguments.patch_size),
        "target_shape": list(target.shape),
        "logit_shape": list(logits.shape),
        "target_labels": torch.unique(target).detach().cpu().tolist(),
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "patch_dice_untrained": {
            name: float(value.detach())
            for name, value in dice.items()
        },
        "base_channels": arguments.base_channels,
        "num_levels": arguments.num_levels,
        "fusion": arguments.fusion,
        "amp": use_amp,
        "amp_initial_scale": initial_amp_scale,
        "amp_scale_before_success": scale_before_success,
        "amp_final_scale": float(scaler.get_scale()),
        "amp_overflow_retries": overflow_retries,
        "optimizer_attempts": successful_attempt,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameter_count,
        "trainable_components": list(model.trainable_component_names()),
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_allocated_gib": gibibytes(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_cuda_reserved_gib": gibibytes(
            torch.cuda.max_memory_reserved(device)
        ),
        "gpu": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }

    print("MICS REAL BRATS GPU SMOKE TEST")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        "NOTE: patch_dice_untrained is diagnostic only; "
        "it is not a validation/test-set result."
    )
    print("MICS REAL BRATS GPU SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
