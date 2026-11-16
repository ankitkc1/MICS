"""Oort training-selector adapter with observed, checkpointed client feedback."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

from .base import ClientSelector, SelectionContext, SelectionPlan
from .vendor.oort_training import _training_selector

OORT_REVISION = "05a3aa1677a10f8e621055b1626ef82e73d09759"
OORT_PARAMETERS = {
    "exploration_factor": 0.9,
    "exploration_decay": 0.95,
    "exploration_min": 0.2,
    "exploration_alpha": 0.3,
    "round_threshold": 10.0,
    "round_penalty": 2.0,
    "pacer_step": 20,
    "pacer_delta": 5.0,
    "sample_window": 5.0,
    "clip_bound": 0.98,
    "cut_off_util": 0.7,
    "blacklist_rounds": -1,
    "blacklist_max_len": 0.3,
}


class OortSelector(ClientSelector):
    """Adapt the pinned Oort bandit, pacer and exploration to logical clients.

    Utility is training-set size times RMS batch training loss. Duration is
    observed successful-client optimization time. Initial duration is one
    declared unit until an observation exists. No future trace is consulted.
    """

    def __init__(self, *, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        self.seed = int(seed)
        self._backend = _training_selector(
            SimpleNamespace(**OORT_PARAMETERS), sample_seed=self.seed
        )
        self._client_ids: tuple[str, ...] = ()
        self._last_round = -1
        self._pending_context: SelectionContext | None = None
        self._pending_selection: tuple[str, ...] | None = None
        self._feedback: dict[str, dict[str, Any]] | None = None

    def plan_round(self, context: SelectionContext) -> SelectionPlan:
        if self._pending_context is not None:
            raise RuntimeError("The previous Oort round has not been observed.")
        if context.round_index != self._last_round + 1:
            raise ValueError("Oort rounds must be consecutive.")
        client_ids = tuple(sorted(client.client_id for client in context.clients))
        if not self._client_ids:
            self._client_ids = client_ids
            clients = {client.client_id: client for client in context.clients}
            for index, client_id in enumerate(client_ids):
                self._backend.register_client(
                    index,
                    {
                        "reward": float(clients[client_id].num_samples),
                        "duration": 1.0,
                    },
                )
        if client_ids != self._client_ids:
            raise ValueError("Oort client identities changed within a run.")
        eligible = {
            self._client_ids.index(client.client_id)
            for client in context.eligible_clients
        }
        indices = self._backend.select_participant(
            context.budget, feasible_clients=eligible
        )
        selected = tuple(self._client_ids[int(index)] for index in indices)
        self._pending_context = context
        self._pending_selection = selected
        return SelectionPlan(
            round_index=context.round_index,
            selected_client_ids=selected,
            scores={
                self._client_ids[index]: float(self._backend.totalArms[index]["reward"])
                for index in sorted(eligible)
            },
            metadata={
                "selector": "oort_training",
                "upstream_revision": OORT_REVISION,
                "parameters": dict(OORT_PARAMETERS),
                "exploration": self._backend.exploration,
                "round_threshold": self._backend.round_threshold,
                "score_definition": "historical_raw_statistical_utility",
                "feedback_duration": "observed_local_optimization_seconds",
                "utility": "training_samples_times_rms_batch_loss",
            },
        )

    def record_training_feedback(
        self,
        round_index: int,
        feedback: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if (
            self._pending_context is None
            or round_index != self._pending_context.round_index
        ):
            raise ValueError("Oort training feedback has no matching plan.")
        if not set(feedback).issubset(self._pending_selection or ()):
            raise ValueError("Oort received feedback from an unselected client.")
        self._feedback = {
            client_id: dict(values) for client_id, values in feedback.items()
        }

    def observe_round(
        self,
        round_index: int,
        *,
        selected_client_ids: Iterable[str],
        successful_client_ids: Iterable[str],
    ) -> Mapping[str, Any]:
        if (
            self._pending_context is None
            or round_index != self._pending_context.round_index
        ):
            raise ValueError("Oort observation has no matching plan.")
        selected = tuple(selected_client_ids)
        successful = set(successful_client_ids)
        if selected != self._pending_selection or not successful.issubset(selected):
            raise ValueError("Oort observation differs from its selection plan.")
        if self._feedback is None or set(self._feedback) != successful:
            raise ValueError(
                "Oort needs training feedback for exactly the successful clients."
            )
        clients = {client.client_id: client for client in self._pending_context.clients}
        updates = {}
        for client_id in selected:
            index = self._client_ids.index(client_id)
            duration = float(self._backend.totalArms[index]["duration"])
            utility = 0.0
            if client_id in successful:
                feedback = self._feedback[client_id]
                loss_rms = float(feedback["loss_rms"])
                duration = float(feedback["elapsed_seconds"])
                if not math.isfinite(loss_rms) or loss_rms < 0.0:
                    raise ValueError("Oort RMS loss must be finite and non-negative.")
                if not math.isfinite(duration) or duration <= 0.0:
                    raise ValueError(
                        "Oort observed duration must be finite and positive."
                    )
                utility = clients[client_id].num_samples * loss_rms
            updates[client_id] = {
                "reward": utility,
                "duration": max(duration, 1e-4),
                "time_stamp": round_index + 1,
                "status": client_id in successful,
            }
        for client_id, feedback in updates.items():
            index = self._client_ids.index(client_id)
            self._backend.update_client_util(index, feedback)
            if client_id not in successful:
                self._backend.successfulClients.discard(index)
        self._last_round = round_index
        self._pending_context = None
        self._pending_selection = None
        self._feedback = None
        return {
            "round": round_index,
            "feedback_upload_bytes_by_client": {
                client_id: 16 for client_id in sorted(successful)
            },
            "client_feedback": {
                client_id: {
                    "statistical_utility": feedback["reward"],
                    "duration_seconds": feedback["duration"],
                    "successful": feedback["status"],
                }
                for client_id, feedback in updates.items()
            },
        }

    def state_dict(self) -> Mapping[str, Any]:
        if self._pending_context is not None:
            raise RuntimeError("Cannot checkpoint a pending Oort round.")
        return {
            "schema_version": 1,
            "seed": self.seed,
            "upstream_revision": OORT_REVISION,
            "parameters": dict(OORT_PARAMETERS),
            "last_round": self._last_round,
            "client_ids": self._client_ids,
            "backend": copy.deepcopy(vars(self._backend)),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or state.get("seed") != self.seed:
            raise ValueError("Oort checkpoint schema or seed differs.")
        if (
            state.get("upstream_revision") != OORT_REVISION
            or state.get("parameters") != OORT_PARAMETERS
        ):
            raise ValueError("Oort checkpoint implementation or parameters differ.")
        if self._pending_context is not None:
            raise RuntimeError("Cannot restore over a pending Oort round.")
        self._backend.__dict__ = copy.deepcopy(dict(state["backend"]))
        self._client_ids = tuple(state["client_ids"])
        self._last_round = int(state["last_round"])
