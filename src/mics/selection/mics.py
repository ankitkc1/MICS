"""MICS component-service queues and cardinality-greedy selection."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .base import ClientSelector, ClientState, SelectionContext, SelectionPlan

PURE_COVERAGE = "pure_coverage"
TRUNCATED_LINEAR = "truncated_linear"
MICS_SERVICE_TYPES = (PURE_COVERAGE, TRUNCATED_LINEAR)


def _finite_nonnegative_mapping(
    values: Mapping[str, float],
    *,
    name: str,
    require_nonempty: bool = True,
) -> dict[str, float]:
    result = {str(key): float(value) for key, value in values.items()}
    if require_nonempty and not result:
        raise ValueError(f"{name} cannot be empty.")
    if any(not key for key in result):
        raise ValueError(f"{name} keys cannot be empty.")
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError(f"{name} values must be finite and non-negative.")
    return result


def _validate_probability(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


def expected_truncated_linear_service(
    completion_probabilities: Iterable[float],
    saturation_level: int,
) -> float:
    """Return E[min(H, L)] / L for independent Bernoulli completions."""

    level = int(saturation_level)
    if level <= 0:
        raise ValueError("saturation_level must be positive.")
    probabilities = tuple(
        _validate_probability(value, name="completion probability")
        for value in completion_probabilities
    )
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for successes, mass in enumerate(distribution):
            updated[successes] += mass * (1.0 - probability)
            updated[successes + 1] += mass * probability
        distribution = updated
    expected_credit = sum(
        min(successes, level) * mass
        for successes, mass in enumerate(distribution)
    )
    return float(expected_credit / level)


def expected_component_service(
    completion_probabilities: Iterable[float],
    *,
    service_type: str,
    saturation_level: int,
) -> float:
    """Expected normalized service for one component and candidate set."""

    probabilities = tuple(
        _validate_probability(value, name="completion probability")
        for value in completion_probabilities
    )
    if service_type == PURE_COVERAGE:
        return float(1.0 - math.prod(1.0 - value for value in probabilities))
    if service_type == TRUNCATED_LINEAR:
        return expected_truncated_linear_service(
            probabilities,
            saturation_level,
        )
    raise ValueError(f"Unknown MICS service_type: {service_type!r}.")


def realised_component_service(
    successful_contributors: int,
    *,
    service_type: str,
    saturation_level: int,
) -> float:
    """Normalized service delivered by a realised contributor count."""

    count = int(successful_contributors)
    level = int(saturation_level)
    if count < 0:
        raise ValueError("successful_contributors cannot be negative.")
    if level <= 0:
        raise ValueError("saturation_level must be positive.")
    if service_type == PURE_COVERAGE:
        return float(count > 0)
    if service_type == TRUNCATED_LINEAR:
        return float(min(count, level) / level)
    raise ValueError(f"Unknown MICS service_type: {service_type!r}.")


@dataclass(frozen=True)
class StaticServiceCalibration:
    """Exact finite-state LP result for a static available-client set."""

    boundary_scale: float
    operational_targets: Mapping[str, float]
    priorities: Mapping[str, float]
    oracle_factor: float
    safety_margin: float
    support: tuple[Mapping[str, Any], ...]
    feasible_set_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "static_exact_service_mixture_lp",
            "boundary_scale": self.boundary_scale,
            "operational_targets": dict(self.operational_targets),
            "priorities": dict(self.priorities),
            "oracle_factor": self.oracle_factor,
            "safety_margin": self.safety_margin,
            "support": [dict(item) for item in self.support],
            "feasible_set_count": self.feasible_set_count,
        }


def calibrate_static_service_targets(
    client_modalities: Mapping[str, Iterable[str]],
    *,
    budget: int,
    priorities: Mapping[str, float],
    completion_probabilities: Mapping[str, float] | None = None,
    service_type: str = PURE_COVERAGE,
    saturation_levels: Mapping[str, int] | None = None,
    oracle_factor: float = 1.0 - 1.0 / math.e,
    safety_margin: float = 0.1,
) -> StaticServiceCalibration:
    """Calibrate oracle-aware targets for one static availability state.

    The linear program enumerates all sets under the cardinality budget and
    finds the largest common scale in the declared priority direction.  It is
    intended for small cross-silo studies, not large client populations.
    """

    try:
        from scipy.optimize import linprog
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError("Static target calibration requires SciPy.") from error

    manifests = {
        str(client_id): frozenset(str(name) for name in modalities)
        for client_id, modalities in client_modalities.items()
    }
    if not manifests or any(not values for values in manifests.values()):
        raise ValueError("Every calibrated client must expose a modality.")
    if len(manifests) > 24:
        raise ValueError(
            "Exact static calibration is restricted to at most 24 clients."
        )
    cardinality = int(budget)
    if cardinality <= 0 or cardinality > len(manifests):
        raise ValueError("budget must lie between one and the client count.")
    raw_priorities = _finite_nonnegative_mapping(priorities, name="priorities")
    if any(value <= 0.0 for value in raw_priorities.values()):
        raise ValueError("Every service priority must be positive.")
    maximum_priority = max(raw_priorities.values())
    normalized_priorities = {
        name: value / maximum_priority
        for name, value in raw_priorities.items()
    }
    modalities = tuple(normalized_priorities)
    unknown = {
        modality
        for values in manifests.values()
        for modality in values
        if modality not in normalized_priorities
    }
    if unknown:
        raise ValueError(
            f"Client manifests contain unknown modalities: {sorted(unknown)}"
        )
    missing = [
        modality
        for modality in modalities
        if not any(modality in values for values in manifests.values())
    ]
    if missing:
        raise ValueError(f"No client can serve modalities: {missing}")
    if service_type not in MICS_SERVICE_TYPES:
        raise ValueError(f"service_type must be one of {MICS_SERVICE_TYPES}.")
    if saturation_levels is not None and set(saturation_levels) != set(modalities):
        raise ValueError("saturation_levels must define every calibrated modality.")
    levels = {
        modality: int(
            1 if saturation_levels is None else saturation_levels[modality]
        )
        for modality in modalities
    }
    if any(value <= 0 for value in levels.values()):
        raise ValueError("Saturation levels must be positive integers.")
    if service_type == PURE_COVERAGE and any(value != 1 for value in levels.values()):
        raise ValueError("Pure coverage requires saturation level one.")
    if completion_probabilities is not None and set(
        completion_probabilities
    ) != set(manifests):
        raise ValueError(
            "completion_probabilities must define every calibrated client."
        )
    probabilities = {
        client_id: _validate_probability(
            1.0
            if completion_probabilities is None
            else completion_probabilities[client_id],
            name=f"completion probability for {client_id}",
        )
        for client_id in manifests
    }
    factor = float(oracle_factor)
    margin = float(safety_margin)
    if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("oracle_factor must lie in (0, 1].")
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("safety_margin must lie in (0, 1).")

    client_ids = tuple(sorted(manifests))
    feasible_set_count = sum(
        math.comb(len(client_ids), size)
        for size in range(cardinality + 1)
    )
    if feasible_set_count > 250_000:
        raise ValueError(
            "Exact calibration would enumerate more than 250,000 sets; "
            "use configuration types or a restricted master problem."
        )
    feasible_sets = tuple(
        selection
        for size in range(cardinality + 1)
        for selection in itertools.combinations(client_ids, size)
    )
    service_matrix = [
        [
            expected_component_service(
                (
                    probabilities[client_id]
                    for client_id in selection
                    if modality in manifests[client_id]
                ),
                service_type=service_type,
                saturation_level=levels[modality],
            )
            for selection in feasible_sets
        ]
        for modality in modalities
    ]
    variable_count = len(feasible_sets) + 1
    objective = [0.0] * variable_count
    objective[-1] = -1.0
    inequalities = []
    bounds = []
    for modality_index, modality in enumerate(modalities):
        row = [-value for value in service_matrix[modality_index]]
        row.append(normalized_priorities[modality])
        inequalities.append(row)
        bounds.append(0.0)
    equalities = [[1.0] * len(feasible_sets) + [0.0]]
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=bounds,
        A_eq=equalities,
        b_eq=[1.0],
        bounds=[(0.0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Static service calibration failed: {result.message}")
    boundary_scale = float(result.x[-1])
    operational_scale = (1.0 - margin) * factor * boundary_scale
    targets = {
        modality: operational_scale * normalized_priorities[modality]
        for modality in modalities
    }
    support = []
    for index, probability in enumerate(result.x[:-1]):
        if probability <= 1e-10:
            continue
        support.append(
            {
                "clients": list(feasible_sets[index]),
                "probability": float(probability),
                "expected_service": {
                    modality: float(service_matrix[modality_index][index])
                    for modality_index, modality in enumerate(modalities)
                },
            }
        )
    return StaticServiceCalibration(
        boundary_scale=boundary_scale,
        operational_targets=targets,
        priorities=normalized_priorities,
        oracle_factor=factor,
        safety_margin=margin,
        support=tuple(support),
        feasible_set_count=len(feasible_sets),
    )


class MICSSelector(ClientSelector):
    """Completion-aware MICS-Core/Impact cardinality-greedy selector."""

    def __init__(
        self,
        service_targets: Mapping[str, float],
        *,
        seed: int,
        service_type: str = PURE_COVERAGE,
        saturation_levels: Mapping[str, int] | None = None,
        impact_weight: float = 0.0,
        client_targets: Mapping[str, float] | None = None,
        reliability_aware: bool = True,
        beta_prior_success: float = 1.0,
        beta_prior_failure: float = 1.0,
        tie_tolerance: float = 1e-12,
    ) -> None:
        self.service_targets = _finite_nonnegative_mapping(
            service_targets,
            name="service_targets",
        )
        if any(value > 1.0 for value in self.service_targets.values()):
            raise ValueError("service_targets cannot exceed one.")
        self.modalities = tuple(self.service_targets)
        if service_type not in MICS_SERVICE_TYPES:
            raise ValueError(f"service_type must be one of {MICS_SERVICE_TYPES}.")
        self.service_type = service_type
        if saturation_levels is not None and set(saturation_levels) != set(
            self.modalities
        ):
            raise ValueError(
                "saturation_levels must define every configured modality."
            )
        self.saturation_levels = {
            modality: int(
                1 if saturation_levels is None else saturation_levels[modality]
            )
            for modality in self.modalities
        }
        if any(value <= 0 for value in self.saturation_levels.values()):
            raise ValueError("Saturation levels must be positive integers.")
        if service_type == PURE_COVERAGE and any(
            value != 1 for value in self.saturation_levels.values()
        ):
            raise ValueError("Pure coverage requires saturation level one.")
        self.impact_weight = float(impact_weight)
        if not math.isfinite(self.impact_weight) or self.impact_weight < 0.0:
            raise ValueError("impact_weight must be finite and non-negative.")
        self.client_targets = _finite_nonnegative_mapping(
            client_targets or {},
            name="client_targets",
            require_nonempty=False,
        )
        if any(value > 1.0 for value in self.client_targets.values()):
            raise ValueError("client_targets cannot exceed one.")
        self.reliability_aware = bool(reliability_aware)
        self.beta_prior_success = float(beta_prior_success)
        self.beta_prior_failure = float(beta_prior_failure)
        if (
            not math.isfinite(self.beta_prior_success)
            or not math.isfinite(self.beta_prior_failure)
            or self.beta_prior_success <= 0.0
            or self.beta_prior_failure <= 0.0
        ):
            raise ValueError("Beta-prior parameters must be finite and positive.")
        self.tie_tolerance = float(tie_tolerance)
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be finite and non-negative.")

        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._component_queues = {modality: 0.0 for modality in self.modalities}
        self._client_queues = {client_id: 0.0 for client_id in self.client_targets}
        self._attempts: dict[str, int] = {}
        self._successes: dict[str, int] = {}
        self._last_observed_round = -1
        self._pending_plan: SelectionPlan | None = None
        self._pending_clients: dict[str, ClientState] | None = None

    @property
    def component_queues(self) -> Mapping[str, float]:
        return dict(self._component_queues)

    @property
    def client_queues(self) -> Mapping[str, float]:
        return dict(self._client_queues)

    def _completion_probability(self, client_id: str) -> float:
        if not self.reliability_aware:
            return 1.0
        attempts = self._attempts.get(client_id, 0)
        successes = self._successes.get(client_id, 0)
        return float(
            (successes + self.beta_prior_success)
            / (
                attempts
                + self.beta_prior_success
                + self.beta_prior_failure
            )
        )

    def _impact(self, context: SelectionContext) -> dict[str, float]:
        if self.impact_weight == 0.0:
            return {modality: 0.0 for modality in self.modalities}
        supplied = _finite_nonnegative_mapping(
            context.component_importance,
            name="component_importance",
        )
        if set(supplied) != set(self.modalities):
            raise ValueError(
                "component_importance must define every configured modality."
            )
        total = sum(supplied.values())
        if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("component_importance must sum to one.")
        return supplied

    def _validate_context(self, context: SelectionContext) -> dict[str, ClientState]:
        if self._pending_plan is not None:
            raise RuntimeError("The previous MICS plan has not been observed.")
        if context.round_index != self._last_observed_round + 1:
            raise ValueError(
                "MICS rounds must be planned and observed consecutively."
            )
        clients = {client.client_id: client for client in context.clients}
        unknown_modalities = {
            modality
            for client in context.clients
            for modality in client.modalities
            if modality not in self.service_targets
        }
        if unknown_modalities:
            raise ValueError(
                f"Context contains unknown modalities: {sorted(unknown_modalities)}"
            )
        missing_target_clients = set(self.client_targets).difference(clients)
        if missing_target_clients:
            raise ValueError(
                "Context omits clients with participation targets: "
                f"{sorted(missing_target_clients)}"
            )
        return clients

    def _objective(
        self,
        context: SelectionContext,
        selected_client_ids: Sequence[str],
        clients: Mapping[str, ClientState],
        impact: Mapping[str, float],
    ) -> dict[str, Any]:
        selected = tuple(selected_client_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("Objective sets cannot contain duplicate clients.")
        unknown = set(selected).difference(clients)
        if unknown:
            raise ValueError(
                f"Objective set contains unknown clients: {sorted(unknown)}"
            )
        probabilities = {
            client_id: self._completion_probability(client_id)
            for client_id in selected
        }
        expected_service = {}
        effective_weights = {}
        component_terms = {}
        for modality in self.modalities:
            contributors = (
                probabilities[client_id]
                for client_id in selected
                if modality in clients[client_id].modalities
            )
            service = expected_component_service(
                contributors,
                service_type=self.service_type,
                saturation_level=self.saturation_levels[modality],
            )
            weight = (
                self._component_queues[modality]
                + self.impact_weight * impact[modality]
            )
            expected_service[modality] = service
            effective_weights[modality] = weight
            component_terms[modality] = weight * service
        client_terms = {
            client_id: self._client_queues.get(client_id, 0.0)
            * probabilities[client_id]
            for client_id in selected
            if client_id in self._client_queues
        }
        component_total = sum(component_terms.values())
        client_total = sum(client_terms.values())
        return {
            "total": float(component_total + client_total),
            "component_total": float(component_total),
            "client_queue_total": float(client_total),
            "expected_component_service": expected_service,
            "effective_component_weights": effective_weights,
            "component_terms": component_terms,
            "client_queue_terms": client_terms,
            "completion_probabilities": probabilities,
        }

    def objective_value(
        self,
        context: SelectionContext,
        selected_client_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        """Expose the fixed-round value oracle for audit and theorem tests."""

        clients = {client.client_id: client for client in context.clients}
        ineligible = {
            client_id
            for client_id in selected_client_ids
            if client_id in clients and not clients[client_id].available
        }
        if ineligible:
            raise ValueError(
                f"Objective set contains unavailable clients: {sorted(ineligible)}"
            )
        impact = self._impact(context)
        return self._objective(
            context,
            selected_client_ids,
            clients,
            impact,
        )

    def plan_round(self, context: SelectionContext) -> SelectionPlan:
        clients = self._validate_context(context)
        impact = self._impact(context)
        eligible = tuple(
            sorted(client.client_id for client in context.eligible_clients)
        )
        singleton_decompositions = {
            client_id: self._objective(
                context,
                (client_id,),
                clients,
                impact,
            )
            for client_id in eligible
        }
        singleton_scores = {
            client_id: float(value["total"])
            for client_id, value in singleton_decompositions.items()
        }
        selected: list[str] = []
        steps: list[dict[str, Any]] = []
        current_value = 0.0
        final_decomposition: Mapping[str, Any] | None = None
        for position in range(context.budget):
            candidate_marginals = {}
            candidate_values = {}
            candidate_decompositions = {}
            for client_id in eligible:
                if client_id in selected:
                    continue
                decomposition = (
                    singleton_decompositions[client_id]
                    if position == 0
                    else self._objective(
                        context,
                        (*selected, client_id),
                        clients,
                        impact,
                    )
                )
                value = float(decomposition["total"])
                marginal = value - current_value
                if marginal < -1e-9:
                    raise RuntimeError("MICS objective violated monotonicity.")
                candidate_values[client_id] = value
                candidate_decompositions[client_id] = decomposition
                candidate_marginals[client_id] = max(marginal, 0.0)
            best_marginal = max(candidate_marginals.values())
            tied = sorted(
                client_id
                for client_id, marginal in candidate_marginals.items()
                if math.isclose(
                    marginal,
                    best_marginal,
                    rel_tol=self.tie_tolerance,
                    abs_tol=self.tie_tolerance,
                )
            )
            chosen = self._rng.choice(tied)
            selected.append(chosen)
            current_value = candidate_values[chosen]
            final_decomposition = candidate_decompositions[chosen]
            steps.append(
                {
                    "position": position,
                    "candidate_marginal_gains": candidate_marginals,
                    "maximum_marginal_gain": best_marginal,
                    "tied_maximisers": tied,
                    "selected_client": chosen,
                    "objective_after_selection": current_value,
                }
            )
        if final_decomposition is None:  # pragma: no cover - budget is positive
            raise RuntimeError("MICS greedy selection produced no decomposition.")
        plan = SelectionPlan(
            round_index=context.round_index,
            selected_client_ids=tuple(selected),
            scores=singleton_scores,
            metadata={
                "selector": "mics_cardinality_greedy",
                "service_type": self.service_type,
                "service_targets": dict(self.service_targets),
                "saturation_levels": dict(self.saturation_levels),
                "impact_weight": self.impact_weight,
                "component_importance": impact,
                "component_queues_before_round": dict(self._component_queues),
                "client_queues_before_round": dict(self._client_queues),
                "reliability_aware": self.reliability_aware,
                "completion_probabilities": {
                    client_id: self._completion_probability(client_id)
                    for client_id in eligible
                },
                "greedy_steps": steps,
                "final_objective": final_decomposition,
                "greedy_candidate_value_oracle_calls": len(eligible)
                * context.budget
                - context.budget * (context.budget - 1) // 2,
                "tie_breaking": "uniform_over_numerical_maximisers",
                "seed": self.seed,
            },
        )
        self._pending_plan = plan
        self._pending_clients = clients
        return plan

    def observe_round(
        self,
        round_index: int,
        *,
        selected_client_ids: Iterable[str],
        successful_client_ids: Iterable[str],
    ) -> Mapping[str, Any]:
        """Update virtual queues from realised, never predicted, service."""

        if self._pending_plan is None or self._pending_clients is None:
            raise RuntimeError("No pending MICS plan exists for observation.")
        if int(round_index) != self._pending_plan.round_index:
            raise ValueError("Observed round does not match the pending plan.")
        selected = tuple(selected_client_ids)
        if selected != self._pending_plan.selected_client_ids:
            raise ValueError("Observed selected clients do not match the MICS plan.")
        successful = frozenset(str(value) for value in successful_client_ids)
        if not successful.issubset(selected):
            raise ValueError("Successful clients must be a subset of selected clients.")

        for client_id in selected:
            self._attempts[client_id] = self._attempts.get(client_id, 0) + 1
            self._successes.setdefault(client_id, 0)
            if client_id in successful:
                self._successes[client_id] += 1
        realised_service = {}
        contributors = {}
        queues_before = dict(self._component_queues)
        for modality in self.modalities:
            count = sum(
                modality in self._pending_clients[client_id].modalities
                for client_id in successful
            )
            service = realised_component_service(
                count,
                service_type=self.service_type,
                saturation_level=self.saturation_levels[modality],
            )
            contributors[modality] = count
            realised_service[modality] = service
            self._component_queues[modality] = max(
                queues_before[modality]
                + self.service_targets[modality]
                - service,
                0.0,
            )
        client_queues_before = dict(self._client_queues)
        realised_client_service = {}
        for client_id, target in self.client_targets.items():
            service = float(client_id in successful)
            realised_client_service[client_id] = service
            self._client_queues[client_id] = max(
                client_queues_before[client_id] + target - service,
                0.0,
            )
        self._last_observed_round = int(round_index)
        self._pending_plan = None
        self._pending_clients = None
        return {
            "round": int(round_index),
            "selected_clients": list(selected),
            "successful_clients": sorted(successful),
            "realised_component_contributors": contributors,
            "realised_component_service": realised_service,
            "component_queues_before_round": queues_before,
            "component_queues_after_round": dict(self._component_queues),
            "realised_client_service": realised_client_service,
            "client_queues_before_round": client_queues_before,
            "client_queues_after_round": dict(self._client_queues),
            "completion_attempts": dict(self._attempts),
            "completion_successes": dict(self._successes),
        }

    def state_dict(self) -> Mapping[str, Any]:
        if self._pending_plan is not None:
            raise RuntimeError("Cannot checkpoint MICS before observing its plan.")
        return {
            "schema_version": 1,
            "seed": self.seed,
            "rng_state": self._rng.getstate(),
            "component_queues": dict(self._component_queues),
            "client_queues": dict(self._client_queues),
            "attempts": dict(self._attempts),
            "successes": dict(self._successes),
            "last_observed_round": self._last_observed_round,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_keys = {
            "schema_version",
            "seed",
            "rng_state",
            "component_queues",
            "client_queues",
            "attempts",
            "successes",
            "last_observed_round",
        }
        if set(state) != expected_keys:
            raise ValueError("MICS selector state has unexpected keys.")
        if int(state["schema_version"]) != 1:
            raise ValueError("Unsupported MICS selector-state schema.")
        if int(state["seed"]) != self.seed:
            raise ValueError("MICS selector-state seed does not match.")
        component_queues = _finite_nonnegative_mapping(
            state["component_queues"],
            name="component_queues",
        )
        client_queues = _finite_nonnegative_mapping(
            state["client_queues"],
            name="client_queues",
            require_nonempty=False,
        )
        if set(component_queues) != set(self.service_targets):
            raise ValueError("MICS component queues do not match configuration.")
        if set(client_queues) != set(self.client_targets):
            raise ValueError("MICS client queues do not match configuration.")
        attempts = {str(key): int(value) for key, value in state["attempts"].items()}
        successes = {
            str(key): int(value) for key, value in state["successes"].items()
        }
        if set(attempts) != set(successes):
            raise ValueError("MICS completion histories have different clients.")
        if any(value < 0 for value in attempts.values()) or any(
            value < 0 for value in successes.values()
        ):
            raise ValueError("MICS completion histories cannot be negative.")
        if any(successes[key] > attempts[key] for key in attempts):
            raise ValueError("MICS successes cannot exceed attempts.")
        self._component_queues = component_queues
        self._client_queues = client_queues
        self._attempts = attempts
        self._successes = successes
        self._last_observed_round = int(state["last_observed_round"])
        if self._last_observed_round < -1:
            raise ValueError("MICS last_observed_round is invalid.")
        self._rng.setstate(state["rng_state"])
        self._pending_plan = None
        self._pending_clients = None
