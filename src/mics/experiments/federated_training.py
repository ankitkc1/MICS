"""Shared tracked training loop for controlled selector comparisons."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..data import (
    BRATS_MODALITIES,
    BraTSScenario,
    build_brats_scenario_dataset,
)
from ..federated import (
    ClientRoundSpec,
    FederatedRoundResult,
    run_federated_round,
)
from ..models import MultiEncoderUNet3D
from ..selection import ClientSelector, ClientState, SelectionContext
from ..training import LocalTrainingConfig
from .metrics import (
    component_state_nbytes,
    selection_round_diagnostics,
    validate_selection_plan,
)
from .tracking import RunRecorder


@dataclass(frozen=True)
class FederatedTrainingLoopConfig:
    """Policy-neutral settings consumed by the tracked FL loop."""

    dataset_root: Path
    rounds: int
    clients_per_round: int
    checkpoint_every: int
    patch_size: tuple[int, int, int]
    foreground_probability: float
    batch_size: int
    num_workers: int
    data_seed: int
    device: torch.device

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(
            self,
            "patch_size",
            tuple(int(value) for value in self.patch_size),
        )
        object.__setattr__(self, "device", torch.device(self.device))
        if self.rounds <= 0 or self.clients_per_round <= 0:
            raise ValueError("Round count and client budget must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("Checkpoint interval must be positive.")
        if len(self.patch_size) != 3 or any(value <= 0 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive values.")
        if not 0.0 <= self.foreground_probability <= 1.0:
            raise ValueError("foreground_probability must lie in [0, 1].")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("Batch size or worker count is invalid.")
        if self.data_seed < 0:
            raise ValueError("data_seed must be non-negative.")


def _stable_seed(base_seed: int, round_index: int, client_id: str) -> int:
    material = f"{base_seed}:{round_index}:{client_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _build_loader(
    scenario: BraTSScenario,
    client_id: str,
    *,
    loop: FederatedTrainingLoopConfig,
    round_index: int,
) -> DataLoader:
    seed = _stable_seed(loop.data_seed, round_index, client_id)
    dataset = build_brats_scenario_dataset(
        loop.dataset_root,
        scenario,
        client_id,
        partition="train",
        patch_size=loop.patch_size,
        foreground_probability=loop.foreground_probability,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=loop.batch_size,
        shuffle=True,
        num_workers=loop.num_workers,
        pin_memory=loop.device.type == "cuda",
        generator=generator,
    )


def _checkpoint(
    model: MultiEncoderUNet3D,
    selector: ClientSelector,
    *,
    recorder: RunRecorder,
    next_round: int,
    selection_counts: Mapping[str, int],
    last_selected_round: Mapping[str, int | None],
    modality_last_updated_round: Mapping[str, int | None],
    cumulative_download_bytes: int,
    cumulative_upload_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_fingerprint": recorder.config_fingerprint,
        "next_round": next_round,
        "global_components": model.component_state_dict(clone=True),
        "selector_state": dict(selector.state_dict()),
        "selection_counts": dict(selection_counts),
        "last_selected_round": dict(last_selected_round),
        "modality_last_updated_round": dict(modality_last_updated_round),
        "cumulative_download_bytes": cumulative_download_bytes,
        "cumulative_upload_bytes": cumulative_upload_bytes,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_or_initialize_state(
    scenario: BraTSScenario,
    recorder: RunRecorder,
    model: MultiEncoderUNet3D,
    selector: ClientSelector,
    *,
    resume: bool,
) -> tuple[
    int,
    dict[str, int],
    dict[str, int | None],
    dict[str, int | None],
    int,
    int,
]:
    selection_counts = {client.client_id: 0 for client in scenario.clients}
    last_selected_round = {client.client_id: None for client in scenario.clients}
    modality_last_updated_round = {
        modality: None for modality in BRATS_MODALITIES
    }
    if not resume:
        return (
            0,
            selection_counts,
            last_selected_round,
            modality_last_updated_round,
            0,
            0,
        )

    checkpoint = recorder.load_checkpoint()
    model.load_component_state_dict(checkpoint["global_components"])
    selector.load_state_dict(checkpoint["selector_state"])
    selection_counts = {
        str(key): int(value)
        for key, value in checkpoint["selection_counts"].items()
    }
    last_selected_round = dict(checkpoint["last_selected_round"])
    modality_last_updated_round = dict(
        checkpoint["modality_last_updated_round"]
    )
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return (
        int(checkpoint["next_round"]),
        selection_counts,
        last_selected_round,
        modality_last_updated_round,
        int(checkpoint["cumulative_download_bytes"]),
        int(checkpoint["cumulative_upload_bytes"]),
    )


def run_tracked_federated_training(
    scenario: BraTSScenario,
    recorder: RunRecorder,
    model: MultiEncoderUNet3D,
    selector: ClientSelector,
    training: LocalTrainingConfig,
    loop: FederatedTrainingLoopConfig,
    *,
    component_importance: Mapping[str, float] | None = None,
    resume: bool = False,
) -> None:
    """Run selector-controlled FL with identical policy-neutral mechanics."""

    if loop.clients_per_round > len(scenario.clients):
        raise ValueError("Client budget exceeds the scenario client count.")
    importance = dict(component_importance or {})
    (
        start_round,
        selection_counts,
        last_selected_round,
        modality_last_updated_round,
        cumulative_download_bytes,
        cumulative_upload_bytes,
    ) = _restore_or_initialize_state(
        scenario,
        recorder,
        model,
        selector,
        resume=resume,
    )
    current_round = start_round
    recorder.mark_running(next_round=start_round)
    try:
        for round_index in range(start_round, loop.rounds):
            current_round = round_index
            context = SelectionContext(
                round_index=round_index,
                budget=loop.clients_per_round,
                clients=tuple(
                    ClientState(
                        client_id=client.client_id,
                        modalities=frozenset(client.modalities),
                        num_samples=len(client.train_case_ids),
                        last_selected_round=last_selected_round[client.client_id],
                    )
                    for client in scenario.clients
                ),
                component_importance=importance,
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
                    scenario,
                    client_id,
                    loop=loop,
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
                device=loop.device,
                training_config=training,
            )
            successful_client_ids = tuple(
                local.client_id for local in result.local_results
            )
            selector_outcome = selector.observe_round(
                round_index,
                selected_client_ids=plan.selected_client_ids,
                successful_client_ids=successful_client_ids,
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
            for modality in BRATS_MODALITIES:
                component = f"encoder:{modality}"
                if result.component_update_counts.get(component, 0) > 0:
                    modality_last_updated_round[modality] = round_index

            weighted_loss_numerator = sum(
                local.mean_loss * local.examples_seen
                for local in result.local_results
            )
            round_record = _round_record(
                scenario,
                result,
                diagnostics,
                selector_outcome=dict(selector_outcome),
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
            completed_rounds = round_index + 1
            archive_round = (
                completed_rounds
                if completed_rounds % loop.checkpoint_every == 0
                or completed_rounds == loop.rounds
                else None
            )
            recorder.save_checkpoint(
                _checkpoint(
                    model,
                    selector,
                    recorder=recorder,
                    next_round=completed_rounds,
                    selection_counts=selection_counts,
                    last_selected_round=last_selected_round,
                    modality_last_updated_round=modality_last_updated_round,
                    cumulative_download_bytes=cumulative_download_bytes,
                    cumulative_upload_bytes=cumulative_upload_bytes,
                ),
                archive_completed_rounds=archive_round,
            )
            recorder.mark_round_complete(next_round=completed_rounds)
            print(json.dumps(round_record, sort_keys=True), flush=True)

        recorder.mark_completed(next_round=loop.rounds)
    except Exception as error:
        recorder.mark_failed(next_round=current_round, error=repr(error))
        raise


def _round_record(
    scenario: BraTSScenario,
    result: FederatedRoundResult,
    diagnostics: dict[str, Any],
    *,
    selector_outcome: dict[str, Any],
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
        "selector_outcome": selector_outcome,
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
