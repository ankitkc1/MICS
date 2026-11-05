"""Run two deterministic federated rounds with real BraTS clients."""

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
from mics.federated import ClientRoundSpec, run_federated_round
from mics.training import LocalTrainingConfig


EXPECTED_UPDATE_COUNTS = {
    "encoder:flair": 1,
    "encoder:t1ce": 2,
    "encoder:t1": 1,
    "encoder:t2": 1,
    "fusion": 2,
    "decoder": 2,
    "head": 2,
}


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
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def component_distance(
    before: dict[str, dict[str, torch.Tensor]],
    after: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    distances: dict[str, float] = {}
    for component, state in before.items():
        total = 0.0
        for name, value in state.items():
            if value.is_floating_point():
                delta = after[component][name] - value
                total += float(delta.square().sum())
        distances[component] = total**0.5
    return distances


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.split_root is None:
        raise SystemExit("Set MICS_SPLIT_ROOT or pass --split-root.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; submit through a Slurm GPU job.")
    if arguments.base_channels <= 0 or arguments.max_batches <= 0:
        raise SystemExit("Channel and batch limits must be positive.")

    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    client_definitions = (
        ("C1", BRATS_MODALITIES),
        ("C6", ("t1ce",)),
    )
    clients: list[ClientRoundSpec] = []
    for client_id, modalities in client_definitions:
        dataset = build_brats_client_dataset(
            arguments.dataset_root,
            arguments.split_root,
            client_id,
            available_modalities=modalities,
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
        clients.append(
            ClientRoundSpec(
                client_id=client_id,
                modalities=tuple(modalities),
                batches=loader,
                aggregation_weight=float(len(dataset)),
            )
        )

    global_model = MultiEncoderUNet3D(
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
    summaries = []
    previous_state = global_model.component_state_dict(clone=True)
    for round_index in range(2):
        result = run_federated_round(
            global_model,
            clients,
            round_index=round_index,
            device="cuda",
            training_config=config,
        )
        current_state = global_model.component_state_dict(clone=True)
        distances = component_distance(previous_state, current_state)
        if result.component_update_counts != EXPECTED_UPDATE_COUNTS:
            raise RuntimeError(
                "Unexpected component update counts: "
                f"{result.component_update_counts}"
            )
        if any(value <= 0 for value in distances.values()):
            raise RuntimeError(
                f"A global component did not change in round {round_index}: "
                f"{distances}"
            )

        summaries.append(
            {
                "round": round_index,
                "selected_clients": list(result.selected_client_ids),
                "component_update_counts": dict(result.component_update_counts),
                "component_delta_l2": distances,
                "examples_seen": result.examples_seen,
                "elapsed_seconds": result.elapsed_seconds,
                "clients": {
                    local.client_id: {
                        "aggregation_weight": local.aggregation_weight,
                        "examples_seen": local.examples_seen,
                        "mean_loss": local.mean_loss,
                        "optimizer_steps": local.optimizer_steps,
                        "amp_overflow_retries": local.amp_overflow_retries,
                        "update_components": list(local.update.states),
                    }
                    for local in result.local_results
                },
            }
        )
        previous_state = current_state

    output = {
        "status": "passed",
        "purpose": "two_client_two_round_integration_not_model_quality",
        "rounds": summaries,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved() / (1024**3)
        ),
    }
    print("MICS TWO-CLIENT TWO-ROUND FEDERATED SMOKE TEST")
    print(json.dumps(output, indent=2, sort_keys=True))
    print("MICS TWO-CLIENT TWO-ROUND FEDERATED SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
