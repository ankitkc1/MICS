"""Sequential synchronous rounds for reproducible single-node FL simulation."""

from __future__ import annotations

import copy
import gc
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..models import MultiEncoderUNet3D
from ..training import (
    LocalTrainingConfig,
    LocalTrainingResult,
    train_local_client,
)
from .aggregation import aggregate_components


@dataclass(frozen=True)
class ClientRoundSpec:
    """Data and metadata required to train one selected logical client."""

    client_id: str
    modalities: tuple[str, ...]
    batches: Iterable[Mapping[str, Any]]
    aggregation_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "modalities",
            tuple(str(name) for name in self.modalities),
        )
        if not self.client_id:
            raise ValueError("client_id cannot be empty.")
        if not self.modalities:
            raise ValueError("modalities cannot be empty.")
        if self.aggregation_weight <= 0:
            raise ValueError("aggregation_weight must be positive.")


@dataclass(frozen=True)
class FederatedRoundResult:
    """Auditable output of one synchronous communication round."""

    round_index: int
    selected_client_ids: tuple[str, ...]
    local_results: tuple[LocalTrainingResult, ...]
    component_update_counts: Mapping[str, int]
    examples_seen: int
    elapsed_seconds: float


def run_federated_round(
    global_model: MultiEncoderUNet3D,
    selected_clients: Sequence[ClientRoundSpec],
    *,
    round_index: int,
    device: str | torch.device,
    training_config: LocalTrainingConfig | None = None,
) -> FederatedRoundResult:
    """Train selected clients sequentially and aggregate their components.

    Each client receives an independent copy of the same pre-round global
    model. Local optimizers are reset per client and round, as in standard
    FedAvg. The global model remains on CPU between rounds.
    """

    if round_index < 0:
        raise ValueError("round_index cannot be negative.")
    clients = tuple(selected_clients)
    if not clients:
        raise ValueError("At least one selected client is required.")
    client_ids = tuple(client.client_id for client in clients)
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("Selected client identifiers must be unique.")

    settings = training_config or LocalTrainingConfig()
    target_device = torch.device(device)
    global_model.to("cpu")
    pre_round_state = global_model.component_state_dict(clone=True)
    local_results: list[LocalTrainingResult] = []
    started = time.perf_counter()

    for client in clients:
        local_model = copy.deepcopy(global_model)
        result = train_local_client(
            local_model,
            client.batches,
            client_id=client.client_id,
            available_modalities=client.modalities,
            device=target_device,
            config=settings,
            aggregation_weight=client.aggregation_weight,
            epoch_offset=round_index * settings.local_epochs,
        )
        local_results.append(result)

        del local_model
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    updates = [result.update for result in local_results]
    aggregated_state = aggregate_components(pre_round_state, updates)
    global_model.load_component_state_dict(aggregated_state)
    global_model.to("cpu")

    component_update_counts = {
        component: sum(
            component in result.update.states
            for result in local_results
        )
        for component in pre_round_state
    }
    return FederatedRoundResult(
        round_index=round_index,
        selected_client_ids=client_ids,
        local_results=tuple(local_results),
        component_update_counts=component_update_counts,
        examples_seen=sum(result.examples_seen for result in local_results),
        elapsed_seconds=time.perf_counter() - started,
    )
