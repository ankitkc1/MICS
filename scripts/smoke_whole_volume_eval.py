"""Validate full-case BraTS reconstruction and report diagnostic metrics."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch

from mics import MultiEncoderUNet3D
from mics.data import BRATS_MODALITIES, build_brats_client_dataset
from mics.evaluation import (
    BRATS_HD95_PENALTY_MM,
    brats_region_dice,
    brats_region_hd95,
    sliding_window_inference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--overlap", type=float, default=0.5)
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
    if arguments.sample_index < 0 or arguments.base_channels <= 0:
        raise SystemExit("Sample index and base channels are invalid.")
    if not 0.0 <= arguments.overlap < 1.0:
        raise SystemExit("--overlap must lie in [0, 1).")

    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    dataset = build_brats_client_dataset(
        arguments.dataset_root,
        arguments.split_root,
        arguments.client,
        available_modalities=BRATS_MODALITIES,
        patch_size=None,
        training=False,
        seed=arguments.seed,
    )
    if arguments.sample_index >= len(dataset):
        raise SystemExit("--sample-index exceeds the client dataset.")
    sample = dataset[arguments.sample_index]
    inputs = {
        name: tensor.unsqueeze(0)
        for name, tensor in sample["inputs"].items()
    }
    target = sample["target"].unsqueeze(0)
    availability = sample["availability"].unsqueeze(0)
    model = MultiEncoderUNet3D(
        modalities=OrderedDict((name, 1) for name in BRATS_MODALITIES),
        num_classes=4,
        base_channels=arguments.base_channels,
        num_levels=4,
        fusion="mean_var",
        modality_dropout=0.0,
    )

    torch.cuda.reset_peak_memory_stats()
    inference = sliding_window_inference(
        model,
        inputs,
        availability,
        roi_size=(96, 96, 96),
        overlap=arguments.overlap,
        device="cuda",
        amp=True,
    )
    prediction = inference.logits.argmax(dim=1)
    dice = brats_region_dice(prediction, target)
    hd95 = brats_region_hd95(
        prediction,
        target,
        spacing=(1.0, 1.0, 1.0),
        one_empty_penalty=BRATS_HD95_PENALTY_MM,
    )
    output = {
        "status": "passed",
        "purpose": "whole_volume_integration_not_model_quality",
        "case_id": sample["case_id"],
        "client": arguments.client,
        "original_shape": list(inference.original_shape),
        "padded_shape": list(inference.padded_shape),
        "logit_shape": list(inference.logits.shape),
        "target_shape": list(target.shape),
        "target_labels": torch.unique(target).tolist(),
        "roi_size": [96, 96, 96],
        "overlap": arguments.overlap,
        "window_count": inference.window_count,
        "inference_seconds": inference.elapsed_seconds,
        "whole_case_dice_untrained": {
            name: float(value)
            for name, value in dice.items()
        },
        "whole_case_hd95_mm_untrained": {
            name: float(value)
            for name, value in hd95.items()
        },
        "hd95_one_empty_penalty_mm": BRATS_HD95_PENALTY_MM,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved() / (1024**3)
        ),
    }
    print("MICS WHOLE-VOLUME BRATS EVALUATION SMOKE TEST")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(
        "NOTE: metrics are from a randomly initialized model and are "
        "integration diagnostics, not experimental results."
    )
    print("MICS WHOLE-VOLUME BRATS EVALUATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
