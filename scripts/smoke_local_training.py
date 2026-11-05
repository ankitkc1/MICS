"""Exercise reusable local-client training on real BraTS patches."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mics import MultiEncoderUNet3D
from mics.data import BRATS_MODALITIES, build_brats_client_dataset
from mics.training import LocalTrainingConfig, train_local_client


def parse_modalities(value: str) -> tuple[str, ...]:
    requested = tuple(
        part.strip().lower()
        for part in value.split(",")
        if part.strip()
    )
    if not requested or len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError(
            "modalities must be a nonempty list without duplicates"
        )
    unknown = set(requested).difference(BRATS_MODALITIES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown modalities: {sorted(unknown)}"
        )
    return tuple(name for name in BRATS_MODALITIES if name in requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one logical client for a few real BraTS batches."
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
    )
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.split_root is None:
        raise SystemExit("Set MICS_SPLIT_ROOT or pass --split-root.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; submit through a Slurm GPU job.")
    if arguments.base_channels <= 0:
        raise SystemExit("--base-channels must be positive.")
    if arguments.max_batches <= 0:
        raise SystemExit("--max-batches must be positive.")

    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    dataset = build_brats_client_dataset(
        arguments.dataset_root,
        arguments.split_root,
        arguments.client,
        available_modalities=arguments.modalities,
        patch_size=(96, 96, 96),
        training=True,
        foreground_probability=1.0,
        seed=arguments.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = MultiEncoderUNet3D(
        modalities=OrderedDict((name, 1) for name in BRATS_MODALITIES),
        num_classes=4,
        base_channels=arguments.base_channels,
        num_levels=4,
        fusion="mean_var",
        modality_dropout=0.0,
    )
    config = LocalTrainingConfig(
        local_epochs=1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        gradient_clip_norm=5.0,
        amp=True,
        amp_initial_scale=4096.0,
        amp_max_retries=8,
        max_batches_per_epoch=arguments.max_batches,
    )

    torch.cuda.reset_peak_memory_stats()
    result = train_local_client(
        model,
        loader,
        client_id=arguments.client,
        available_modalities=arguments.modalities,
        device="cuda",
        config=config,
        aggregation_weight=len(dataset),
    )
    update_bytes = sum(
        tensor.numel() * tensor.element_size()
        for state in result.update.states.values()
        for tensor in state.values()
    )
    summary = {
        "status": "passed",
        "purpose": "local_training_integration_not_model_quality",
        "client": result.client_id,
        "modalities": list(arguments.modalities),
        "local_dataset_size": len(dataset),
        "examples_seen": result.examples_seen,
        "aggregation_weight": result.aggregation_weight,
        "batches_processed": result.batches_processed,
        "optimizer_steps": result.optimizer_steps,
        "amp_overflow_retries": result.amp_overflow_retries,
        "amp_final_scale": result.final_amp_scale,
        "mean_loss": result.mean_loss,
        "mean_gradient_norm": result.mean_gradient_norm,
        "max_gradient_norm": result.max_gradient_norm,
        "patch_dice_untrained": dict(result.region_dice),
        "elapsed_seconds": result.elapsed_seconds,
        "trainable_components": list(result.trainable_components),
        "update_components": list(result.update.states),
        "update_size_mib": update_bytes / (1024**2),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved() / (1024**3)
        ),
    }

    print("MICS REAL BRATS LOCAL-TRAINING SMOKE TEST")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        "NOTE: patch_dice_untrained is diagnostic only; "
        "it is not a validation/test-set result."
    )
    print("MICS REAL BRATS LOCAL-TRAINING SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
