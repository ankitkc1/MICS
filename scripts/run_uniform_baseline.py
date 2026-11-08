#!/usr/bin/env python3
"""Run scenario-aware uniform-random federated training with full provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from mics import MultiEncoderUNet3D
from mics.data import (
    BRATS_MODALITIES,
    BraTSScenario,
    load_brats_scenario,
)
from mics.experiments import (
    FederatedTrainingLoopConfig,
    RunRecorder,
    SystemTrace,
    collect_runtime_metadata,
    load_system_trace,
    run_tracked_federated_training,
)
from mics.selection import UniformRandomSelector
from mics.training import LocalTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MICS_DATA_ROOT"),
    )
    parser.add_argument("--scenario-config", required=True, type=Path)
    parser.add_argument("--system-trace", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--clients-per-round", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--fusion", default="mean_var")
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
    )
    parser.add_argument("--foreground-probability", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--amp-initial-scale", type=float, default=4096.0)
    parser.add_argument("--amp-max-retries", type=int, default=8)
    parser.add_argument("--max-batches-per-epoch", type=int)
    parser.add_argument("--model-seed", type=int, default=101)
    parser.add_argument("--selection-seed", type=int, default=202)
    parser.add_argument("--data-seed", type=int, default=303)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_files(scenario: BraTSScenario) -> tuple[Path, ...]:
    files = [scenario.config_path, scenario.global_test_file]
    for client in scenario.clients:
        files.extend(
            (client.train_file, client.validation_file, client.test_file)
        )
    return tuple(files)


def _resolved_config(
    arguments: argparse.Namespace,
    scenario: BraTSScenario,
    system_trace: SystemTrace | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "uniform_random",
        "purpose": (
            "integration_smoke_not_model_quality"
            if arguments.max_batches_per_epoch is not None
            else "full_training_checkpoint_for_held_out_evaluation"
        ),
        "scenario": {
            "id": scenario.scenario_id,
            "data_seed": scenario.data_seed,
            "config": str(scenario.config_path),
            "config_sha256": _file_sha256(scenario.config_path),
            "file_sha256": {
                str(path): _file_sha256(path)
                for path in _scenario_files(scenario)
            },
            "partition_counts": scenario.partition_counts,
            "global_test_count": len(scenario.global_test_case_ids),
            "modality_pool_sizes": scenario.modality_pool_sizes,
            "clients": {
                client.client_id: {
                    "modalities": list(client.modalities),
                    "train_count": len(client.train_case_ids),
                    "validation_count": len(client.validation_case_ids),
                    "test_count": len(client.test_case_ids),
                }
                for client in scenario.clients
            },
        },
        "model": {
            "modalities": list(BRATS_MODALITIES),
            "num_classes": 4,
            "base_channels": arguments.base_channels,
            "num_levels": arguments.num_levels,
            "fusion": arguments.fusion,
            "modality_dropout": arguments.modality_dropout,
        },
        "federation": {
            "rounds": arguments.rounds,
            "clients_per_round": arguments.clients_per_round,
            "checkpoint_every": arguments.checkpoint_every,
            "aggregation": "component_weighted_mean",
        },
        "training": {
            "patch_size": list(arguments.patch_size),
            "foreground_probability": arguments.foreground_probability,
            "batch_size": arguments.batch_size,
            "num_workers": arguments.num_workers,
            "local_epochs": arguments.local_epochs,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "gradient_clip_norm": arguments.gradient_clip_norm,
            "amp": not arguments.no_amp,
            "amp_initial_scale": arguments.amp_initial_scale,
            "amp_max_retries": arguments.amp_max_retries,
            "max_batches_per_epoch": arguments.max_batches_per_epoch,
            "random_streams": {
                "data_loader": (
                    "sha256(data_seed:data_loader:round:client)"
                ),
                "local_training": (
                    "sha256(data_seed:local_training:round:client)"
                ),
            },
        },
        "seeds": {
            "model": arguments.model_seed,
            "selection": arguments.selection_seed,
            "data": arguments.data_seed,
        },
        "device": arguments.device,
        "system_trace": (
            None if system_trace is None else system_trace.source_record()
        ),
    }


def _validate_arguments(arguments: argparse.Namespace) -> torch.device:
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.rounds <= 0 or arguments.clients_per_round <= 0:
        raise SystemExit("Round count and client budget must be positive.")
    if arguments.checkpoint_every <= 0:
        raise SystemExit("Checkpoint interval must be positive.")
    if arguments.batch_size <= 0 or arguments.num_workers < 0:
        raise SystemExit("Batch size or worker count is invalid.")
    if min(arguments.model_seed, arguments.selection_seed, arguments.data_seed) < 0:
        raise SystemExit("Scientific seeds must be non-negative.")
    target_device = torch.device(arguments.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; submit this runner through a GPU job.")
    return target_device


def main() -> None:
    arguments = build_parser().parse_args()
    target_device = _validate_arguments(arguments)
    scenario = load_brats_scenario(
        arguments.scenario_config,
        dataset_root=arguments.dataset_root,
    )
    if arguments.clients_per_round > len(scenario.clients):
        raise SystemExit("Client budget exceeds the scenario client count.")
    system_trace = (
        None
        if arguments.system_trace is None
        else load_system_trace(arguments.system_trace)
    )

    torch.manual_seed(arguments.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.model_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    project_root = Path(__file__).resolve().parents[1]
    config = _resolved_config(arguments, scenario, system_trace)
    model = MultiEncoderUNet3D(
        modalities=OrderedDict((name, 1) for name in BRATS_MODALITIES),
        num_classes=4,
        base_channels=arguments.base_channels,
        num_levels=arguments.num_levels,
        fusion=arguments.fusion,
        modality_dropout=arguments.modality_dropout,
    )
    selector = UniformRandomSelector(arguments.selection_seed)
    settings = LocalTrainingConfig(
        local_epochs=arguments.local_epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip_norm=arguments.gradient_clip_norm,
        amp=not arguments.no_amp,
        amp_initial_scale=arguments.amp_initial_scale,
        amp_max_retries=arguments.amp_max_retries,
        max_batches_per_epoch=arguments.max_batches_per_epoch,
    )
    recorder = RunRecorder(
        arguments.output_root,
        arguments.run_id,
        resolved_config=config,
        runtime_metadata=collect_runtime_metadata(project_root),
        resume=arguments.resume,
    )
    loop = FederatedTrainingLoopConfig(
        dataset_root=arguments.dataset_root,
        rounds=arguments.rounds,
        clients_per_round=arguments.clients_per_round,
        checkpoint_every=arguments.checkpoint_every,
        patch_size=tuple(arguments.patch_size),
        foreground_probability=arguments.foreground_probability,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        data_seed=arguments.data_seed,
        device=target_device,
    )
    run_tracked_federated_training(
        scenario,
        recorder,
        model,
        selector,
        settings,
        loop,
        system_trace=system_trace,
        resume=arguments.resume,
    )

    print(f"Run directory: {recorder.run_dir}")
    if arguments.max_batches_per_epoch is not None:
        print(
            "NOTE: max_batches_per_epoch is set; this run validates integration "
            "and must not be reported as model quality."
        )
    print("MICS UNIFORM-RANDOM BASELINE TRAINING PASSED")


if __name__ == "__main__":
    main()
