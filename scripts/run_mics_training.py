#!/usr/bin/env python3
"""Run tracked federated training with MICS-Core or MICS-Impact selection."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mics import MICSSelector, MultiEncoderUNet3D
from mics.data import BRATS_MODALITIES, BraTSScenario, load_brats_scenario
from mics.experiments import (
    FederatedTrainingLoopConfig,
    LoadedModalityImportance,
    RunRecorder,
    SystemTrace,
    collect_runtime_metadata,
    load_modality_importance_record,
    load_system_trace,
    run_tracked_federated_training,
    sha256_file,
)
from mics.selection import (
    MICS_SERVICE_TYPES,
    PURE_COVERAGE,
    calibrate_static_service_targets,
)
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
    parser.add_argument("--importance-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--clients-per-round", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--service-type",
        choices=MICS_SERVICE_TYPES,
        default=PURE_COVERAGE,
    )
    parser.add_argument("--saturation-level", type=int, default=1)
    parser.add_argument("--safety-margin", type=float, default=0.1)
    parser.add_argument("--impact-weight", type=float, default=1.0)
    parser.add_argument("--reliability-aware", action="store_true")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--fusion", default="mean_var")
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
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


def _scenario_files(scenario: BraTSScenario) -> tuple[Path, ...]:
    files = [scenario.config_path, scenario.global_test_file]
    for client in scenario.clients:
        files.extend((client.train_file, client.validation_file, client.test_file))
    return tuple(files)


def _validate_arguments(arguments: argparse.Namespace) -> torch.device:
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.rounds <= 0 or arguments.clients_per_round <= 0:
        raise SystemExit("Round count and client budget must be positive.")
    if arguments.checkpoint_every <= 0:
        raise SystemExit("Checkpoint interval must be positive.")
    if arguments.saturation_level <= 0:
        raise SystemExit("Saturation level must be positive.")
    if (
        arguments.service_type == PURE_COVERAGE
        and arguments.saturation_level != 1
    ):
        raise SystemExit("Pure coverage requires --saturation-level 1.")
    if not 0.0 < arguments.safety_margin < 1.0:
        raise SystemExit("Safety margin must lie in (0, 1).")
    if not math.isfinite(arguments.impact_weight) or arguments.impact_weight < 0.0:
        raise SystemExit("Impact weight must be finite and non-negative.")
    if arguments.impact_weight > 0.0 and arguments.importance_output is None:
        raise SystemExit("MICS-Impact requires --importance-output.")
    if arguments.batch_size <= 0 or arguments.num_workers < 0:
        raise SystemExit("Batch size or worker count is invalid.")
    if min(arguments.model_seed, arguments.selection_seed, arguments.data_seed) < 0:
        raise SystemExit("Scientific seeds must be non-negative.")
    target_device = torch.device(arguments.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; submit this runner through a GPU job.")
    return target_device


def _load_importance(
    arguments: argparse.Namespace,
) -> LoadedModalityImportance | None:
    if arguments.impact_weight == 0.0:
        if arguments.importance_output is not None:
            raise SystemExit(
                "Do not pass --importance-output when --impact-weight is zero."
            )
        return None
    loaded = load_modality_importance_record(
        arguments.importance_output,
        modalities=BRATS_MODALITIES,
    )
    is_smoke = arguments.max_batches_per_epoch is not None
    if not is_smoke and not loaded.complete_validation_probe:
        raise SystemExit(
            "A full MICS run cannot use subset importance estimates. Run the "
            "complete validation probe after freezing the study protocol."
        )
    return loaded


def _resolved_config(
    arguments: argparse.Namespace,
    scenario: BraTSScenario,
    importance: LoadedModalityImportance | None,
    calibration: Mapping[str, Any],
    system_trace: SystemTrace | None,
) -> dict[str, Any]:
    is_smoke = arguments.max_batches_per_epoch is not None
    method = "mics_core" if arguments.impact_weight == 0.0 else "mics_impact"
    return {
        "schema_version": 1,
        "method": method,
        "purpose": (
            "integration_smoke_not_model_quality"
            if is_smoke
            else "full_training_checkpoint_for_held_out_evaluation"
        ),
        "scenario": {
            "id": scenario.scenario_id,
            "data_seed": scenario.data_seed,
            "config": str(scenario.config_path),
            "config_sha256": sha256_file(scenario.config_path),
            "file_sha256": {
                str(path): sha256_file(path) for path in _scenario_files(scenario)
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
        "selection": {
            "name": method,
            "service_type": arguments.service_type,
            "saturation_level": arguments.saturation_level,
            "impact_weight": arguments.impact_weight,
            "importance_source": (
                None if importance is None else importance.source_record()
            ),
            "importance_refresh": "fixed_for_this_run",
            "target_calibration": dict(calibration),
            "reliability_aware": arguments.reliability_aware,
            "completion_model": (
                "deterministic_success"
                if system_trace is None
                else "fixed_selection_independent_potential_outcome_trace"
            ),
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
    if system_trace is not None and any(
        probability < 1.0
        for probability in system_trace.availability_probabilities.values()
    ):
        raise SystemExit(
            "MICS availability traces require state-aware target calibration, "
            "which is not enabled in this runner yet."
        )
    importance = _load_importance(arguments)
    levels = {
        modality: arguments.saturation_level for modality in BRATS_MODALITIES
    }
    calibration = calibrate_static_service_targets(
        {
            client.client_id: client.modalities for client in scenario.clients
        },
        budget=arguments.clients_per_round,
        priorities={modality: 1.0 for modality in BRATS_MODALITIES},
        service_type=arguments.service_type,
        saturation_levels=levels,
        completion_probabilities=(
            None
            if system_trace is None
            else system_trace.completion_probabilities
        ),
        oracle_factor=1.0 - 1.0 / math.e,
        safety_margin=arguments.safety_margin,
    )

    torch.manual_seed(arguments.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.model_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    project_root = Path(__file__).resolve().parents[1]
    config = _resolved_config(
        arguments,
        scenario,
        importance,
        calibration.as_dict(),
        system_trace,
    )
    model = MultiEncoderUNet3D(
        modalities=OrderedDict((name, 1) for name in BRATS_MODALITIES),
        num_classes=4,
        base_channels=arguments.base_channels,
        num_levels=arguments.num_levels,
        fusion=arguments.fusion,
        modality_dropout=arguments.modality_dropout,
    )
    selector = MICSSelector(
        calibration.operational_targets,
        seed=arguments.selection_seed,
        service_type=arguments.service_type,
        saturation_levels=levels,
        impact_weight=arguments.impact_weight,
        reliability_aware=arguments.reliability_aware,
    )
    training = LocalTrainingConfig(
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
        training,
        loop,
        component_importance=(
            {} if importance is None else importance.weights
        ),
        system_trace=system_trace,
        resume=arguments.resume,
    )

    print(f"Run directory: {recorder.run_dir}")
    print(json.dumps({"calibration": calibration.as_dict()}, sort_keys=True))
    if arguments.max_batches_per_epoch is not None:
        print(
            "NOTE: max_batches_per_epoch is set; this run validates integration "
            "and must not be reported as model quality."
        )
    print("MICS TRACKED FEDERATED TRAINING PASSED")


if __name__ == "__main__":
    main()
