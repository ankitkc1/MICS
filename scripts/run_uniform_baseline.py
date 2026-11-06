#!/usr/bin/env python3
"""Run scenario-aware uniform-random federated training with full provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mics import ClientState, MultiEncoderUNet3D, SelectionContext
from mics.data import (
    BRATS_MODALITIES,
    BraTSScenario,
    build_brats_scenario_dataset,
    load_brats_scenario,
)
from mics.experiments import (
    RunRecorder,
    collect_runtime_metadata,
    component_state_nbytes,
    selection_round_diagnostics,
    validate_selection_plan,
)
from mics.federated import (
    ClientRoundSpec,
    FederatedRoundResult,
    run_federated_round,
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
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--clients-per-round", type=int, default=2)
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


def _stable_seed(base_seed: int, round_index: int, client_id: str) -> int:
    material = f"{base_seed}:{round_index}:{client_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _resolved_config(
    arguments: argparse.Namespace,
    scenario: BraTSScenario,
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
        },
        "seeds": {
            "model": arguments.model_seed,
            "selection": arguments.selection_seed,
            "data": arguments.data_seed,
        },
        "device": arguments.device,
    }


def _build_loader(
    dataset_root: Path,
    scenario: BraTSScenario,
    client_id: str,
    *,
    arguments: argparse.Namespace,
    round_index: int,
) -> DataLoader:
    seed = _stable_seed(arguments.data_seed, round_index, client_id)
    dataset = build_brats_scenario_dataset(
        dataset_root,
        scenario,
        client_id,
        partition="train",
        patch_size=tuple(arguments.patch_size),
        foreground_probability=arguments.foreground_probability,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=str(arguments.device).startswith("cuda"),
        generator=generator,
    )


def _checkpoint(
    model: MultiEncoderUNet3D,
    selector: UniformRandomSelector,
    *,
    recorder: RunRecorder,
    next_round: int,
    selection_counts: dict[str, int],
    last_selected_round: dict[str, int | None],
    modality_last_updated_round: dict[str, int | None],
    cumulative_download_bytes: int,
    cumulative_upload_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_fingerprint": recorder.config_fingerprint,
        "next_round": next_round,
        "global_components": model.component_state_dict(clone=True),
        "selector_state": dict(selector.state_dict()),
        "selection_counts": selection_counts,
        "last_selected_round": last_selected_round,
        "modality_last_updated_round": modality_last_updated_round,
        "cumulative_download_bytes": cumulative_download_bytes,
        "cumulative_upload_bytes": cumulative_upload_bytes,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _validate_arguments(arguments: argparse.Namespace) -> torch.device:
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.rounds <= 0 or arguments.clients_per_round <= 0:
        raise SystemExit("Round count and client budget must be positive.")
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

    torch.manual_seed(arguments.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.model_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    project_root = Path(__file__).resolve().parents[1]
    config = _resolved_config(arguments, scenario)
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
    selection_counts = {client.client_id: 0 for client in scenario.clients}
    last_selected_round: dict[str, int | None] = {
        client.client_id: None for client in scenario.clients
    }
    modality_last_updated_round: dict[str, int | None] = {
        modality: None for modality in BRATS_MODALITIES
    }
    cumulative_download_bytes = 0
    cumulative_upload_bytes = 0
    start_round = 0

    if arguments.resume:
        checkpoint = recorder.load_checkpoint()
        model.load_component_state_dict(checkpoint["global_components"])
        selector.load_state_dict(checkpoint["selector_state"])
        selection_counts = dict(checkpoint["selection_counts"])
        last_selected_round = dict(checkpoint["last_selected_round"])
        modality_last_updated_round = dict(
            checkpoint["modality_last_updated_round"]
        )
        cumulative_download_bytes = int(checkpoint["cumulative_download_bytes"])
        cumulative_upload_bytes = int(checkpoint["cumulative_upload_bytes"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state_all"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_round = int(checkpoint["next_round"])

    _run_rounds(
        arguments,
        scenario,
        recorder,
        model,
        selector,
        settings,
        target_device=target_device,
        start_round=start_round,
        selection_counts=selection_counts,
        last_selected_round=last_selected_round,
        modality_last_updated_round=modality_last_updated_round,
        cumulative_download_bytes=cumulative_download_bytes,
        cumulative_upload_bytes=cumulative_upload_bytes,
    )

    print(f"Run directory: {recorder.run_dir}")
    if arguments.max_batches_per_epoch is not None:
        print(
            "NOTE: max_batches_per_epoch is set; this run validates integration "
            "and must not be reported as model quality."
        )
    print("MICS UNIFORM-RANDOM BASELINE TRAINING PASSED")


def _run_rounds(
    arguments: argparse.Namespace,
    scenario: BraTSScenario,
    recorder: RunRecorder,
    model: MultiEncoderUNet3D,
    selector: UniformRandomSelector,
    settings: LocalTrainingConfig,
    *,
    target_device: torch.device,
    start_round: int,
    selection_counts: dict[str, int],
    last_selected_round: dict[str, int | None],
    modality_last_updated_round: dict[str, int | None],
    cumulative_download_bytes: int,
    cumulative_upload_bytes: int,
) -> None:
    current_round = start_round
    recorder.mark_running(next_round=start_round)
    try:
        for round_index in range(start_round, arguments.rounds):
            current_round = round_index
            context = SelectionContext(
                round_index=round_index,
                budget=arguments.clients_per_round,
                clients=tuple(
                    ClientState(
                        client_id=client.client_id,
                        modalities=frozenset(client.modalities),
                        num_samples=len(client.train_case_ids),
                        last_selected_round=last_selected_round[client.client_id],
                    )
                    for client in scenario.clients
                ),
                metadata={"scenario_id": scenario.scenario_id},
            )
            selection_started = time.perf_counter()
            plan = selector.plan_round(context)
            selection_seconds = time.perf_counter() - selection_started
            validate_selection_plan(context, plan)
            diagnostics = selection_round_diagnostics(
                context,
                plan,
                selection_counts=selection_counts,
                modality_last_updated_round=modality_last_updated_round,
            )
            recorder.record_selection(
                {
                    "round": round_index,
                    "selected_clients": list(plan.selected_client_ids),
                    "scores": dict(plan.scores),
                    "selector_metadata": dict(plan.metadata),
                    "selection_seconds": selection_seconds,
                    "eligible_clients": [
                        client.client_id for client in context.eligible_clients
                    ],
                    "diagnostics_before_training": diagnostics,
                }
            )

            component_states = model.component_state_dict(clone=False)
            component_bytes = {
                name: component_state_nbytes(state)
                for name, state in component_states.items()
            }
            specs = []
            client_download_bytes: dict[str, int] = {}
            for client_id in plan.selected_client_ids:
                client = scenario.client(client_id)
                loader = _build_loader(
                    arguments.dataset_root,
                    scenario,
                    client_id,
                    arguments=arguments,
                    round_index=round_index,
                )
                components = (
                    *(f"encoder:{modality}" for modality in client.modalities),
                    "fusion",
                    "decoder",
                    "head",
                )
                client_download_bytes[client_id] = sum(
                    component_bytes[component] for component in components
                )
                specs.append(
                    ClientRoundSpec(
                        client_id=client_id,
                        modalities=client.modalities,
                        batches=loader,
                        aggregation_weight=float(len(client.train_case_ids)),
                    )
                )

            result = run_federated_round(
                model,
                specs,
                round_index=round_index,
                device=target_device,
                training_config=settings,
            )
            upload_by_client = {
                local.client_id: sum(
                    component_state_nbytes(state)
                    for state in local.update.states.values()
                )
                for local in result.local_results
            }
            round_download_bytes = sum(client_download_bytes.values())
            round_upload_bytes = sum(upload_by_client.values())
            cumulative_download_bytes += round_download_bytes
            cumulative_upload_bytes += round_upload_bytes

            for client_id in plan.selected_client_ids:
                selection_counts[client_id] += 1
                last_selected_round[client_id] = round_index
            for modality, count in diagnostics[
                "selected_clients_per_modality"
            ].items():
                if count > 0:
                    modality_last_updated_round[modality] = round_index

            weighted_loss_numerator = sum(
                local.mean_loss * local.examples_seen
                for local in result.local_results
            )
            round_record = _round_record(
                scenario,
                result,
                diagnostics,
                round_index=round_index,
                selection_seconds=selection_seconds,
                weighted_loss_numerator=weighted_loss_numerator,
                round_download_bytes=round_download_bytes,
                round_upload_bytes=round_upload_bytes,
                cumulative_download_bytes=cumulative_download_bytes,
                cumulative_upload_bytes=cumulative_upload_bytes,
                client_download_bytes=client_download_bytes,
                upload_by_client=upload_by_client,
            )
            recorder.record_round(round_record)
            recorder.save_checkpoint(
                _checkpoint(
                    model,
                    selector,
                    recorder=recorder,
                    next_round=round_index + 1,
                    selection_counts=selection_counts,
                    last_selected_round=last_selected_round,
                    modality_last_updated_round=modality_last_updated_round,
                    cumulative_download_bytes=cumulative_download_bytes,
                    cumulative_upload_bytes=cumulative_upload_bytes,
                )
            )
            recorder.mark_round_complete(next_round=round_index + 1)
            print(json.dumps(round_record, sort_keys=True), flush=True)

        recorder.mark_completed(next_round=arguments.rounds)
    except Exception as error:
        recorder.mark_failed(next_round=current_round, error=repr(error))
        raise


def _round_record(
    scenario: BraTSScenario,
    result: FederatedRoundResult,
    diagnostics: dict[str, Any],
    *,
    round_index: int,
    selection_seconds: float,
    weighted_loss_numerator: float,
    round_download_bytes: int,
    round_upload_bytes: int,
    cumulative_download_bytes: int,
    cumulative_upload_bytes: int,
    client_download_bytes: dict[str, int],
    upload_by_client: dict[str, int],
) -> dict[str, Any]:
    return {
        "round": round_index,
        "selected_clients": list(result.selected_client_ids),
        "selection_seconds": selection_seconds,
        "training_seconds": result.elapsed_seconds,
        "examples_seen": result.examples_seen,
        "sample_weighted_training_loss": (
            weighted_loss_numerator / result.examples_seen
        ),
        "component_update_counts": dict(result.component_update_counts),
        "selection_diagnostics": diagnostics,
        "communication": {
            "accounting": (
                "logical_relevant_components_not_simulator_memory_copies"
            ),
            "round_download_bytes": round_download_bytes,
            "round_upload_bytes": round_upload_bytes,
            "cumulative_download_bytes": cumulative_download_bytes,
            "cumulative_upload_bytes": cumulative_upload_bytes,
            "download_bytes_by_client": client_download_bytes,
            "upload_bytes_by_client": upload_by_client,
        },
        "clients": {
            local.client_id: {
                "modalities": list(scenario.client(local.client_id).modalities),
                "dataset_size": len(
                    scenario.client(local.client_id).train_case_ids
                ),
                "aggregation_weight": local.aggregation_weight,
                "examples_seen": local.examples_seen,
                "batches_processed": local.batches_processed,
                "optimizer_steps": local.optimizer_steps,
                "mean_loss": local.mean_loss,
                "training_patch_dice_diagnostic": dict(local.region_dice),
                "mean_gradient_norm_before_clipping": local.mean_gradient_norm,
                "max_gradient_norm_before_clipping": local.max_gradient_norm,
                "amp_overflow_retries": local.amp_overflow_retries,
                "elapsed_seconds": local.elapsed_seconds,
                "updated_components": list(local.update.states),
            }
            for local in result.local_results
        },
    }


if __name__ == "__main__":
    main()
