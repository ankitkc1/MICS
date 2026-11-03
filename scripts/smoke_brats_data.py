"""Inspect real BraTS client data without starting model training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from mics.data import BRATS_MODALITIES, build_brats_client_dataset


def parse_patch_size(value: str) -> tuple[int, int, int]:
    result = tuple(int(part.strip()) for part in value.split(","))
    if len(result) != 3 or any(part <= 0 for part in result):
        raise argparse.ArgumentTypeError(
            "patch size must contain three positive comma-separated integers"
        )
    return result


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
    parser.add_argument(
        "--modalities",
        default=",".join(BRATS_MODALITIES),
        help="Comma-separated available modalities.",
    )
    parser.add_argument("--patch-size", type=parse_patch_size, default=(96, 96, 96))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.split_root is None:
        raise SystemExit("Set MICS_SPLIT_ROOT or pass --split-root.")
    if arguments.samples <= 0:
        raise SystemExit("--samples must be positive.")

    modalities = tuple(
        value.strip().lower()
        for value in arguments.modalities.split(",")
        if value.strip()
    )
    dataset = build_brats_client_dataset(
        arguments.dataset_root,
        arguments.split_root,
        arguments.client,
        available_modalities=modalities,
        patch_size=arguments.patch_size,
        training=True,
        foreground_probability=1.0,
        seed=arguments.seed,
    )

    print("BRATS DATASET SMOKE TEST")
    print("Dataset root:", Path(arguments.dataset_root).resolve())
    print("Split root:", Path(arguments.split_root).resolve())
    print("Client:", arguments.client)
    print("Cases:", len(dataset))
    print("Modalities:", dataset.available_modalities)
    print("Patch size:", dataset.patch_size)

    for index in range(min(arguments.samples, len(dataset))):
        sample = dataset[index]
        print("Case:", sample["case_id"])
        print("Target shape:", tuple(sample["target"].shape))
        print("Labels:", torch.unique(sample["target"]).tolist())
        for name, tensor in sample["inputs"].items():
            print(
                f"{name}: shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, finite={bool(torch.isfinite(tensor).all())}"
            )

    print("BRATS DATASET SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
