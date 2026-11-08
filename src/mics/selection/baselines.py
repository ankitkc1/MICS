"""Probe-based reference selectors used in controlled MICS comparisons."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .base import (
    ClientSelector,
    SelectionContext,
    SelectionPlan,
    SelectionProbeRequest,
)


def _weighted_sample_without_replacement(
    clients: tuple[tuple[str, float], ...],
    count: int,
    rng: random.Random,
) -> tuple[str, ...]:
    remaining = list(clients)
    selected = []
    for _ in range(count):
        total = sum(weight for _, weight in remaining)
        if total <= 0.0:
            raise ValueError("Candidate sampling weights must have positive mass.")
        threshold = rng.random() * total
        cumulative = 0.0
        chosen_index = len(remaining) - 1
        for index, (_, weight) in enumerate(remaining):
            cumulative += weight
            if threshold < cumulative:
                chosen_index = index
                break
        client_id, _ = remaining.pop(chosen_index)
        selected.append(client_id)
    return tuple(selected)


class PowerOfChoiceSelector(ClientSelector):
    """Power-of-Choice using current train-loss probes on ``d`` candidates."""

    def __init__(
        self,
        *,
        candidate_size: int,
        seed: int,
        max_probe_batches: int | None = None,
    ) -> None:
        if candidate_size <= 0:
            raise ValueError("candidate_size must be positive.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if max_probe_batches is not None and max_probe_batches <= 0:
            raise ValueError("max_probe_batches must be positive when provided.")
        self.candidate_size = int(candidate_size)
        self.seed = int(seed)
        self.max_probe_batches = max_probe_batches
        self._rng = random.Random(self.seed)
        self._pending_round: int | None = None
        self._pending_candidates: tuple[str, ...] | None = None
        self._pending_selection: tuple[str, ...] | None = None

    def prepare_round(self, context: SelectionContext) -> SelectionProbeRequest:
        if self._pending_round is not None:
            raise RuntimeError("Power-of-Choice already has a pending round.")
        if self.candidate_size < context.budget:
            raise ValueError("Power-of-Choice candidate size is below the budget.")
        eligible = tuple(
            sorted(context.eligible_clients, key=lambda item: item.client_id)
        )
        effective_size = min(self.candidate_size, len(eligible))
        candidates = _weighted_sample_without_replacement(
            tuple(
                (client.client_id, float(client.num_samples))
                for client in eligible
            ),
            effective_size,
            self._rng,
        )
        self._pending_round = context.round_index
        self._pending_candidates = candidates
        return SelectionProbeRequest(
            round_index=context.round_index,
            client_ids=candidates,
            kind="local_loss",
            max_batches=self.max_probe_batches,
        )

    def plan_round(self, context: SelectionContext) -> SelectionPlan:
        if self._pending_round != context.round_index:
            raise RuntimeError("Power-of-Choice has no matching prepared round.")
        if self._pending_candidates is None:
            raise RuntimeError("Power-of-Choice candidate set is missing.")
        clients = {client.client_id: client for client in context.clients}
        losses = {}
        for client_id in self._pending_candidates:
            raw_loss = clients[client_id].statistics.get("local_loss")
            if raw_loss is None:
                raise ValueError(f"Missing local-loss probe for {client_id}.")
            loss = float(raw_loss)
            if not math.isfinite(loss):
                raise ValueError(f"Non-finite local-loss probe for {client_id}.")
            losses[client_id] = loss

        groups: dict[float, list[str]] = {}
        for client_id, loss in losses.items():
            groups.setdefault(loss, []).append(client_id)
        ranked = []
        for loss in sorted(groups, reverse=True):
            tied = sorted(groups[loss])
            self._rng.shuffle(tied)
            ranked.extend(tied)
        selected = tuple(ranked[: context.budget])
        self._pending_selection = selected
        return SelectionPlan(
            round_index=context.round_index,
            selected_client_ids=selected,
            scores=losses,
            metadata={
                "selector": "power_of_choice",
                "variant": (
                    "pow_d_full_local_loss"
                    if self.max_probe_batches is None
                    else "cpow_d_minibatch_loss"
                ),
                "configured_candidate_size": self.candidate_size,
                "effective_candidate_size": len(self._pending_candidates),
                "candidate_clients": list(self._pending_candidates),
                "candidate_local_losses": dict(losses),
                "candidate_sampling": (
                    "successive_sample_weighted_without_replacement"
                ),
                "tie_breaking": "uniform_within_equal_loss",
                "seed": self.seed,
            },
        )

    def observe_round(
        self,
        round_index: int,
        *,
        selected_client_ids: Iterable[str],
        successful_client_ids: Iterable[str],
    ) -> Mapping[str, Any]:
        del successful_client_ids
        selected = tuple(selected_client_ids)
        if self._pending_round != int(round_index):
            raise ValueError("Observed round does not match Power-of-Choice state.")
        if selected != self._pending_selection:
            raise ValueError("Observed selection does not match Power-of-Choice plan.")
        candidates = self._pending_candidates
        self._pending_round = None
        self._pending_candidates = None
        self._pending_selection = None
        return {
            "round": int(round_index),
            "candidate_clients": list(candidates or ()),
        }

    def state_dict(self) -> Mapping[str, Any]:
        if self._pending_round is not None:
            raise RuntimeError("Cannot checkpoint a pending Power-of-Choice round.")
        return {
            "schema_version": 1,
            "seed": self.seed,
            "candidate_size": self.candidate_size,
            "max_probe_batches": self.max_probe_batches,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported Power-of-Choice state schema.")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("Power-of-Choice checkpoint seed differs.")
        if int(state.get("candidate_size", -1)) != self.candidate_size:
            raise ValueError("Power-of-Choice candidate size differs.")
        if state.get("max_probe_batches") != self.max_probe_batches:
            raise ValueError("Power-of-Choice probe batch limit differs.")
        self._rng.setstate(state["rng_state"])


class DivFLSelector(ClientSelector):
    """Full-greedy DivFL facility location over current gradient probes."""

    def __init__(
        self,
        *,
        seed: int,
        max_probe_batches: int | None = None,
        tie_tolerance: float = 1e-12,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if max_probe_batches is not None and max_probe_batches <= 0:
            raise ValueError("max_probe_batches must be positive when provided.")
        if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be finite and non-negative.")
        self.seed = int(seed)
        self.max_probe_batches = max_probe_batches
        self.tie_tolerance = float(tie_tolerance)
        self._rng = random.Random(self.seed)
        self._pending_round: int | None = None
        self._pending_clients: tuple[str, ...] | None = None
        self._pending_selection: tuple[str, ...] | None = None

    def prepare_round(self, context: SelectionContext) -> SelectionProbeRequest:
        if self._pending_round is not None:
            raise RuntimeError("DivFL already has a pending round.")
        clients = tuple(
            sorted(client.client_id for client in context.eligible_clients)
        )
        self._pending_round = context.round_index
        self._pending_clients = clients
        return SelectionProbeRequest(
            round_index=context.round_index,
            client_ids=clients,
            kind="gradient_vector",
            max_batches=self.max_probe_batches,
        )

    def plan_round(self, context: SelectionContext) -> SelectionPlan:
        if self._pending_round != context.round_index:
            raise RuntimeError("DivFL has no matching prepared round.")
        if self._pending_clients is None:
            raise RuntimeError("DivFL probe-client set is missing.")
        clients = {client.client_id: client for client in context.clients}
        vectors = []
        for client_id in self._pending_clients:
            value = clients[client_id].statistics.get("gradient_vector")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Missing gradient-vector probe for {client_id}.")
            vector = value.detach().to(device="cpu", dtype=torch.float64)
            if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
                raise ValueError(f"Invalid gradient-vector probe for {client_id}.")
            vectors.append(vector)
        dimensions = {int(vector.numel()) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("DivFL gradient vectors must have equal dimensions.")

        matrix = torch.stack(vectors)
        distances = torch.cdist(matrix, matrix, p=2)
        baseline = distances.max(dim=1).values
        current = baseline.clone()
        selected_indices: list[int] = []
        steps = []
        singleton_scores = {}
        for position in range(context.budget):
            gains = {}
            for candidate_index, client_id in enumerate(self._pending_clients):
                if candidate_index in selected_indices:
                    continue
                updated = torch.minimum(current, distances[:, candidate_index])
                gain = float((current - updated).sum())
                gains[client_id] = gain
                if position == 0:
                    singleton_scores[client_id] = gain
            maximum_gain = max(gains.values())
            tied = sorted(
                client_id
                for client_id, gain in gains.items()
                if math.isclose(
                    gain,
                    maximum_gain,
                    rel_tol=self.tie_tolerance,
                    abs_tol=self.tie_tolerance,
                )
            )
            chosen_id = self._rng.choice(tied)
            chosen_index = self._pending_clients.index(chosen_id)
            selected_indices.append(chosen_index)
            current = torch.minimum(current, distances[:, chosen_index])
            steps.append(
                {
                    "position": position,
                    "candidate_marginal_gains": gains,
                    "maximum_marginal_gain": maximum_gain,
                    "tied_maximisers": tied,
                    "selected_client": chosen_id,
                    "facility_value_after_selection": float(
                        (baseline - current).sum()
                    ),
                }
            )
        selected = tuple(
            self._pending_clients[index] for index in selected_indices
        )
        self._pending_selection = selected
        distance_records = {
            left_id: {
                right_id: float(distances[left_index, right_index])
                for right_index, right_id in enumerate(self._pending_clients)
            }
            for left_index, left_id in enumerate(self._pending_clients)
        }
        return SelectionPlan(
            round_index=context.round_index,
            selected_client_ids=selected,
            scores=singleton_scores,
            metadata={
                "selector": "divfl_gradient_facility_location",
                "variant": (
                    "full_greedy_full_local_gradient"
                    if self.max_probe_batches is None
                    else "full_greedy_minibatch_gradient"
                ),
                "distance": "euclidean_full_parameter_gradient",
                "probe_clients": list(self._pending_clients),
                "gradient_dimension": next(iter(dimensions)),
                "gradient_norms": {
                    client_id: float(torch.linalg.vector_norm(vectors[index]))
                    for index, client_id in enumerate(self._pending_clients)
                },
                "pairwise_distances": distance_records,
                "greedy_steps": steps,
                "final_facility_value": float((baseline - current).sum()),
                "aggregation_note": (
                    "selection-only adaptation; common component-wise "
                    "FedAvg aggregation is retained"
                ),
                "tie_breaking": "uniform_over_numerical_maximisers",
                "seed": self.seed,
            },
        )

    def observe_round(
        self,
        round_index: int,
        *,
        selected_client_ids: Iterable[str],
        successful_client_ids: Iterable[str],
    ) -> Mapping[str, Any]:
        del successful_client_ids
        selected = tuple(selected_client_ids)
        if self._pending_round != int(round_index):
            raise ValueError("Observed round does not match DivFL state.")
        if selected != self._pending_selection:
            raise ValueError("Observed selection does not match DivFL plan.")
        probe_clients = self._pending_clients
        self._pending_round = None
        self._pending_clients = None
        self._pending_selection = None
        return {
            "round": int(round_index),
            "probe_clients": list(probe_clients or ()),
        }

    def state_dict(self) -> Mapping[str, Any]:
        if self._pending_round is not None:
            raise RuntimeError("Cannot checkpoint a pending DivFL round.")
        return {
            "schema_version": 1,
            "seed": self.seed,
            "max_probe_batches": self.max_probe_batches,
            "tie_tolerance": self.tie_tolerance,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported DivFL state schema.")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("DivFL checkpoint seed differs.")
        if state.get("max_probe_batches") != self.max_probe_batches:
            raise ValueError("DivFL probe batch limit differs.")
        if float(state.get("tie_tolerance", -1.0)) != self.tie_tolerance:
            raise ValueError("DivFL tie tolerance differs.")
        self._rng.setstate(state["rng_state"])
