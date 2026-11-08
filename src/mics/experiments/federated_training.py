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
from .system_trace import SystemTrace
from .selection_probes import SelectionProbeOutcome, run_selection_probes


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


def _stable_seed(
    base_seed: int,
    round_index: int,
    client_id: str,
    stream: str,
) -> int:
    material = (
        f"{base_seed}:{stream}:{round_index}:{client_id}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _build_loader(
    scenario: BraTSScenario,
    client_id: str,
    *,
    loop: FederatedTrainingLoopConfig,
    round_index: int,
) -> DataLoader:
    seed = _stable_seed(
        loop.data_seed,
        round_index,
        client_id,
        "data_loader",
    )
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


def _context_with_probe_statistics(
    context: SelectionContext,
    outcome: SelectionProbeOutcome,
) -> SelectionContext:
    return SelectionContext(
        round_index=context.round_index,
        budget=context.budget,
        clients=tuple(
            ClientState(
                client_id=client.client_id,
                modalities=client.modalities,
                num_samples=client.num_samples,
                available=client.available,
                last_selected_round=client.last_selected_round,
                statistics={
                    **client.statistics,
                    **outcome.statistics.get(client.client_id, {}),
                },
            )
            for client in context.clients
        ),
        component_importance=context.component_importance,
        metadata=context.metadata,
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
    system_trace: SystemTrace | None = None,
    resume: bool = False,
) -> None:
    """Run selector-controlled FL with identical policy-neutral mechanics."""

    if loop.clients_per_round > len(scenario.clients):
        raise ValueError("Client budget exceeds the scenario client count.")
    scenario_client_ids = tuple(client.client_id for client in scenario.clients)
    if system_trace is not None:
        if system_trace.clients != scenario_client_ids:
            raise ValueError("System-trace clients do not match the scenario.")
        if len(system_trace.rounds) < loop.rounds:
            raise ValueError("System trace is shorter than the training run.")
        if system_trace.minimum_available_clients < loop.clients_per_round:
            raise ValueError("System trace cannot satisfy the selection budget.")
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
            trace_round = (
                None if system_trace is None else system_trace.round(round_index)
            )
            available_clients = (
                set(scenario_client_ids)
                if trace_round is None
                else set(trace_round.available_clients)
            )
            context = SelectionContext(
                round_index=round_index,
                budget=loop.clients_per_round,
                clients=tuple(
                    ClientState(
                        client_id=client.client_id,
                        modalities=frozenset(client.modalities),
                        num_samples=len(client.train_case_ids),
                        available=client.client_id in available_clients,
                        last_selected_round=last_selected_round[client.client_id],
                    )
                    for client in scenario.clients
                ),
                component_importance=importance,
                metadata={"scenario_id": scenario.scenario_id},
            )
            component_states = model.component_state_dict(clone=False)
            component_bytes = {
                name: component_state_nbytes(state)
                for name, state in component_states.items()
            }
            probe_request = selector.prepare_round(context)
            probe_diagnostics: Mapping[str, Any] | None = None
            probe_download_by_client: dict[str, int] = {}
            probe_upload_by_client: dict[str, int] = {}
            if probe_request is not None:
                if probe_request.round_index != round_index:
                    raise ValueError("Selection probe round does not match context.")
                eligible_ids = {
                    client.client_id for client in context.eligible_clients
                }
                unknown_probe_clients = set(
                    probe_request.client_ids
                ).difference(eligible_ids)
                if unknown_probe_clients:
                    raise ValueError(
                        "Selection probe contains unavailable clients: "
                        f"{sorted(unknown_probe_clients)}"
                    )
                probe_outcome = run_selection_probes(
                    model,
                    probe_request,
                    loaders={
                        client_id: _build_loader(
                            scenario,
                            client_id,
                            loop=loop,
                            round_index=round_index,
                        )
                        for client_id in probe_request.client_ids
                    },
                    client_modalities={
                        client.client_id: client.modalities
                        for client in scenario.clients
                    },
                    component_bytes=component_bytes,
                    device=loop.device,
                )
                context = _context_with_probe_statistics(context, probe_outcome)
                probe_diagnostics = probe_outcome.diagnostics
                probe_download_by_client = dict(
                    probe_outcome.download_bytes_by_client
                )
                probe_upload_by_client = dict(
                    probe_outcome.upload_bytes_by_client
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
                    "selection_probe": probe_diagnostics,
                    "eligible_clients": [
                        client.client_id for client in context.eligible_clients
                    ],
                    "diagnostics_before_training": diagnostics,
                }
            )

            specs = []
            training_download_by_client: dict[str, int] = {}
            successful_client_ids = (
                plan.selected_client_ids
                if system_trace is None
                else system_trace.successful_selected_clients(
                    round_index,
                    plan.selected_client_ids,
                )
            )
            successful_client_set = set(successful_client_ids)
            for client_id in plan.selected_client_ids:
                client = scenario.client(client_id)
                components = (
                    *(f"encoder:{modality}" for modality in client.modalities),
                    "fusion",
                    "decoder",
                    "head",
                )
                training_download_by_client[client_id] = (
                    0
                    if client_id in probe_download_by_client
                    else sum(
                        component_bytes[component]
                        for component in components
                    )
                )
                if client_id not in successful_client_set:
                    continue
                loader = _build_loader(
                    scenario,
                    client_id,
                    loop=loop,
                    round_index=round_index,
                )
                specs.append(
                    ClientRoundSpec(
                        client_id=client_id,
                        modalities=client.modalities,
                        batches=loader,
                        aggregation_weight=float(len(client.train_case_ids)),
                        random_seed=_stable_seed(
                            loop.data_seed,
                            round_index,
                            client_id,
                            "local_training",
                        ),
                    )
                )

            result = run_federated_round(
                model,
                specs,
                round_index=round_index,
                device=loop.device,
                training_config=training,
                attempted_client_ids=plan.selected_client_ids,
            )
            if result.successful_client_ids != successful_client_ids:
                raise RuntimeError("Federated result changed trace completion order.")
            selector_outcome = selector.observe_round(
                round_index,
                selected_client_ids=plan.selected_client_ids,
                successful_client_ids=successful_client_ids,
            )
            training_upload_by_client = {
                local.client_id: sum(
                    component_state_nbytes(state)
                    for state in local.update.states.values()
                )
                for local in result.local_results
            }
            total_download_by_client = dict(probe_download_by_client)
            for client_id, value in training_download_by_client.items():
                total_download_by_client[client_id] = (
                    total_download_by_client.get(client_id, 0) + value
                )
            total_upload_by_client = dict(probe_upload_by_client)
            for client_id, value in training_upload_by_client.items():
                total_upload_by_client[client_id] = (
                    total_upload_by_client.get(client_id, 0) + value
                )
            round_download_bytes = sum(total_download_by_client.values())
            round_upload_bytes = sum(total_upload_by_client.values())
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
                selection_probe=(
                    None
                    if probe_diagnostics is None
                    else dict(probe_diagnostics)
                ),
                available_client_ids=tuple(sorted(available_clients)),
                round_index=round_index,
                selection_seconds=selection_seconds,
                weighted_loss_numerator=weighted_loss_numerator,
                round_download_bytes=round_download_bytes,
                round_upload_bytes=round_upload_bytes,
                cumulative_download_bytes=cumulative_download_bytes,
                cumulative_upload_bytes=cumulative_upload_bytes,
                probe_download_by_client=probe_download_by_client,
                probe_upload_by_client=probe_upload_by_client,
                training_download_by_client=training_download_by_client,
                training_upload_by_client=training_upload_by_client,
                total_download_by_client=total_download_by_client,
                total_upload_by_client=total_upload_by_client,
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
    selection_probe: dict[str, Any] | None,
    available_client_ids: tuple[str, ...],
    round_index: int,
    selection_seconds: float,
    weighted_loss_numerator: float,
    round_download_bytes: int,
    round_upload_bytes: int,
    cumulative_download_bytes: int,
    cumulative_upload_bytes: int,
    probe_download_by_client: dict[str, int],
    probe_upload_by_client: dict[str, int],
    training_download_by_client: dict[str, int],
    training_upload_by_client: dict[str, int],
    total_download_by_client: dict[str, int],
    total_upload_by_client: dict[str, int],
) -> dict[str, Any]:
    return {
        "round": round_index,
        "selected_clients": list(result.selected_client_ids),
        "successful_clients": list(result.successful_client_ids),
        "failed_selected_clients": sorted(
            set(result.selected_client_ids).difference(
                result.successful_client_ids
            )
        ),
        "available_clients": list(available_client_ids),
        "selection_seconds": selection_seconds,
        "training_seconds": result.elapsed_seconds,
        "examples_seen": result.examples_seen,
        "sample_weighted_training_loss": (
            weighted_loss_numerator / result.examples_seen
            if result.examples_seen > 0
            else None
        ),
        "component_update_counts": dict(result.component_update_counts),
        "selection_diagnostics": diagnostics,
        "selector_outcome": selector_outcome,
        "selection_probe": selection_probe,
        "communication": {
            "accounting": (
                "logical_relevant_components_not_simulator_memory_copies"
            ),
            "round_download_bytes": round_download_bytes,
            "round_upload_bytes": round_upload_bytes,
            "cumulative_download_bytes": cumulative_download_bytes,
            "cumulative_upload_bytes": cumulative_upload_bytes,
            "selection_probe_download_bytes": sum(
                probe_download_by_client.values()
            ),
            "selection_probe_upload_bytes": sum(
                probe_upload_by_client.values()
            ),
            "training_download_bytes": sum(
                training_download_by_client.values()
            ),
            "training_upload_bytes": sum(
                training_upload_by_client.values()
            ),
            "selection_probe_download_bytes_by_client": (
                probe_download_by_client
            ),
            "selection_probe_upload_bytes_by_client": probe_upload_by_client,
            "training_download_bytes_by_client": training_download_by_client,
            "training_upload_bytes_by_client": training_upload_by_client,
            "download_bytes_by_client": total_download_by_client,
            "upload_bytes_by_client": total_upload_by_client,
        },
        "clients": {
            local.client_id: {
                "modalities": list(scenario.client(local.client_id).modalities),
                "local_training_random_seed": local.random_seed,
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
