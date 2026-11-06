"""Policy-neutral client-selection and communication diagnostics."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import torch

from ..selection import SelectionContext, SelectionPlan


def validate_selection_plan(
    context: SelectionContext,
    plan: SelectionPlan,
) -> None:
    """Reject selector output that violates its declared round context."""

    if plan.round_index != context.round_index:
        raise ValueError("Selection plan round does not match its context.")
    if len(plan.selected_client_ids) != context.budget:
        raise ValueError("Selection plan does not satisfy the client budget.")
    eligible = {client.client_id for client in context.eligible_clients}
    ineligible = set(plan.selected_client_ids).difference(eligible)
    if ineligible:
        raise ValueError(
            f"Selection plan contains ineligible clients: {sorted(ineligible)}"
        )


def jain_fairness_index(counts: Mapping[str, int]) -> float:
    """Return Jain's index over non-negative client selection counts."""

    values = [int(value) for value in counts.values()]
    if not values:
        raise ValueError("At least one client count is required.")
    if any(value < 0 for value in values):
        raise ValueError("Selection counts cannot be negative.")
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0:
        return 1.0
    return float(sum(values) ** 2 / denominator)


def _pairwise_modality_redundancy(
    modality_sets: tuple[frozenset[str], ...],
) -> float:
    if len(modality_sets) < 2:
        return 0.0
    similarities = []
    for left, right in itertools.combinations(modality_sets, 2):
        union = left.union(right)
        similarities.append(
            0.0 if not union else len(left.intersection(right)) / len(union)
        )
    return float(sum(similarities) / len(similarities))


def selection_round_diagnostics(
    context: SelectionContext,
    plan: SelectionPlan,
    *,
    selection_counts: Mapping[str, int],
    modality_last_updated_round: Mapping[str, int | None],
) -> dict[str, Any]:
    """Compute pre-update diagnostics for one auditable selection decision."""

    validate_selection_plan(context, plan)
    clients = {client.client_id: client for client in context.clients}
    if set(selection_counts) != set(clients):
        raise ValueError(
            "selection_counts must contain every context client exactly once."
        )

    client_modalities = {
        modality
        for client in context.clients
        for modality in client.modalities
    }
    configured_modalities = set(modality_last_updated_round)
    unknown_client_modalities = client_modalities.difference(configured_modalities)
    if unknown_client_modalities:
        raise ValueError(
            "modality_last_updated_round is missing client modalities: "
            f"{sorted(unknown_client_modalities)}"
        )
    modalities = tuple(sorted(configured_modalities))
    selected_clients = tuple(
        clients[client_id] for client_id in plan.selected_client_ids
    )
    coverage_counts = {
        modality: sum(modality in client.modalities for client in selected_clients)
        for modality in modalities
    }
    coverage = {
        modality: int(count > 0)
        for modality, count in coverage_counts.items()
    }
    selected_staleness = {
        client.client_id: (
            context.round_index + 1
            if client.last_selected_round is None
            else context.round_index - client.last_selected_round
        )
        for client in selected_clients
    }
    modality_staleness = {
        modality: (
            context.round_index + 1
            if modality_last_updated_round[modality] is None
            else context.round_index - int(modality_last_updated_round[modality])
        )
        for modality in modalities
    }
    counts_after = {
        client_id: int(count) + int(client_id in plan.selected_client_ids)
        for client_id, count in selection_counts.items()
    }

    impact_weighted_coverage: float | None = None
    if context.component_importance:
        unknown = set(context.component_importance).difference(modalities)
        if unknown:
            raise ValueError(
                f"Component importance has unknown modalities: {sorted(unknown)}"
            )
        weights = {
            modality: float(context.component_importance.get(modality, 0.0))
            for modality in modalities
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError("Component importance cannot be negative.")
        total_weight = sum(weights.values())
        if total_weight > 0:
            impact_weighted_coverage = sum(
                weights[modality] * coverage[modality]
                for modality in modalities
            ) / total_weight

    return {
        "eligible_client_count": len(context.eligible_clients),
        "selected_client_count": len(selected_clients),
        "selected_modalities": sorted(
            {
                modality
                for client in selected_clients
                for modality in client.modalities
            }
        ),
        "modality_coverage": coverage,
        "selected_clients_per_modality": coverage_counts,
        "impact_weighted_modality_coverage": impact_weighted_coverage,
        "pairwise_modality_jaccard_redundancy": (
            _pairwise_modality_redundancy(
                tuple(client.modalities for client in selected_clients)
            )
        ),
        "selected_client_staleness_before_round": selected_staleness,
        "modality_staleness_before_round": modality_staleness,
        "selection_counts_after_round": counts_after,
        "jain_selection_fairness_after_round": jain_fairness_index(counts_after),
    }


def component_state_nbytes(state: Mapping[str, torch.Tensor]) -> int:
    """Count exact tensor storage represented by one component state."""

    total = 0
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Component state {name!r} is not a tensor.")
        total += tensor.numel() * tensor.element_size()
    return int(total)
