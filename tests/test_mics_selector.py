"""Tests for MICS queues, calibration, and the greedy value oracle."""

from __future__ import annotations

import itertools
import math
import unittest

from mics import ClientState, MICSSelector, SelectionContext
from mics.selection import (
    MODULAR_ADDITIVE,
    PURE_COVERAGE,
    SUBMODULAR_SERVICE,
    TRUNCATED_LINEAR,
    calibrate_availability_aware_client_targets,
    calibrate_availability_aware_service_targets,
    calibrate_static_client_targets,
    calibrate_static_service_targets,
    expected_component_service,
)


def context(
    round_index: int,
    *,
    budget: int = 1,
    importance: dict[str, float] | None = None,
) -> SelectionContext:
    return SelectionContext(
        round_index=round_index,
        budget=budget,
        clients=(
            ClientState("A", {"m1"}, 10),
            ClientState("B", {"m1", "m2"}, 10),
            ClientState("C", {"m2"}, 10),
        ),
        component_importance=importance or {},
    )


class MICSSelectorTests(unittest.TestCase):
    def test_expected_service_matches_closed_form_examples(self) -> None:
        self.assertAlmostEqual(
            expected_component_service(
                (0.5, 0.5),
                service_type=PURE_COVERAGE,
                saturation_level=1,
            ),
            0.75,
        )
        self.assertAlmostEqual(
            expected_component_service(
                (0.5, 0.5),
                service_type=TRUNCATED_LINEAR,
                saturation_level=2,
            ),
            0.5,
        )

    def test_static_calibration_finds_symmetric_boundary(self) -> None:
        result = calibrate_static_service_targets(
            {"A": {"m1"}, "C": {"m2"}},
            budget=1,
            priorities={"m1": 1.0, "m2": 1.0},
            oracle_factor=1.0,
            safety_margin=0.2,
        )

        self.assertAlmostEqual(result.boundary_scale, 0.5)
        self.assertAlmostEqual(result.operational_targets["m1"], 0.4)
        self.assertAlmostEqual(result.operational_targets["m2"], 0.4)
        self.assertAlmostEqual(
            sum(float(item["probability"]) for item in result.support),
            1.0,
        )

    def test_client_target_calibration_is_symmetric_and_feasible(self) -> None:
        static = calibrate_static_client_targets(
            ("A", "B", "C", "D"),
            budget=2,
            completion_probabilities={name: 0.9 for name in "ABCD"},
            oracle_factor=1.0,
            safety_margin=0.2,
        )
        availability = calibrate_availability_aware_client_targets(
            ("A", "B", "C", "D"),
            budget=2,
            availability_probabilities={name: 0.8 for name in "ABCD"},
            minimum_available_clients=2,
            completion_probabilities={name: 0.9 for name in "ABCD"},
            oracle_factor=1.0,
            safety_margin=0.2,
        )

        self.assertAlmostEqual(static.boundary_scale, 0.45)
        self.assertEqual(set(static.operational_targets.values()), {0.36})
        self.assertAlmostEqual(availability.boundary_scale, 0.45)
        for target in availability.operational_targets.values():
            self.assertAlmostEqual(target, 0.36)

    def test_availability_calibration_respects_rare_component_opportunity(
        self,
    ) -> None:
        result = calibrate_availability_aware_service_targets(
            {"A": {"m1"}, "C": {"m2"}},
            budget=1,
            priorities={"m1": 1.0, "m2": 1.0},
            availability_probabilities={"A": 1.0, "C": 0.2},
            minimum_available_clients=1,
            oracle_factor=1.0,
            safety_margin=0.2,
        )

        self.assertAlmostEqual(result.conditioning_probability, 1.0)
        self.assertEqual(result.availability_state_count, 2)
        self.assertAlmostEqual(result.boundary_scale, 0.2)
        self.assertAlmostEqual(result.operational_targets["m1"], 0.16)
        self.assertAlmostEqual(result.operational_targets["m2"], 0.16)

    def test_full_availability_calibration_matches_static_capacity(self) -> None:
        arguments = {
            "client_modalities": {"A": {"m1"}, "C": {"m2"}},
            "budget": 1,
            "priorities": {"m1": 1.0, "m2": 1.0},
            "oracle_factor": 1.0,
            "safety_margin": 0.2,
        }
        static = calibrate_static_service_targets(**arguments)
        availability_aware = calibrate_availability_aware_service_targets(
            **arguments,
            availability_probabilities={"A": 1.0, "C": 1.0},
            minimum_available_clients=1,
        )

        self.assertAlmostEqual(
            availability_aware.boundary_scale,
            static.boundary_scale,
        )
        self.assertEqual(availability_aware.availability_state_count, 1)
        self.assertEqual(
            availability_aware.operational_targets,
            static.operational_targets,
        )

    def test_availability_calibration_conditions_on_feasible_rounds(self) -> None:
        result = calibrate_availability_aware_service_targets(
            {"A": {"m1"}, "C": {"m2"}},
            budget=1,
            priorities={"m1": 1.0, "m2": 1.0},
            availability_probabilities={"A": 0.5, "C": 0.5},
            minimum_available_clients=1,
            oracle_factor=1.0,
            safety_margin=0.2,
        )

        self.assertAlmostEqual(result.conditioning_probability, 0.75)
        self.assertEqual(result.availability_state_count, 3)
        self.assertAlmostEqual(result.boundary_scale, 0.5)
        state_probabilities = {
            tuple(item["available_clients"]): item[
                "availability_state_probability"
            ]
            for item in result.support
        }
        self.assertAlmostEqual(state_probabilities[("A",)], 1.0 / 3.0)
        self.assertAlmostEqual(state_probabilities[("C",)], 1.0 / 3.0)

    def test_impact_prefers_client_covering_both_modalities(self) -> None:
        selector = MICSSelector(
            {"m1": 0.4, "m2": 0.4},
            seed=7,
            impact_weight=1.0,
            reliability_aware=False,
        )
        plan = selector.plan_round(
            context(0, importance={"m1": 0.8, "m2": 0.2})
        )

        self.assertEqual(plan.selected_client_ids, ("B",))
        self.assertAlmostEqual(
            plan.metadata["final_objective"]["total"],
            1.0,
        )

    def test_fixed_round_objective_is_monotone_submodular(self) -> None:
        selector = MICSSelector(
            {"m1": 0.4, "m2": 0.4},
            seed=7,
            service_type=TRUNCATED_LINEAR,
            saturation_levels={"m1": 2, "m2": 2},
            impact_weight=1.0,
            reliability_aware=True,
        )
        round_context = SelectionContext(
            round_index=0,
            budget=2,
            clients=(
                ClientState("A", {"m1"}, 10),
                ClientState("B", {"m1"}, 10),
                ClientState("C", {"m2"}, 10),
            ),
            component_importance={"m1": 0.6, "m2": 0.4},
        )
        client_ids = ("A", "B", "C")
        values = {
            subset: float(selector.objective_value(round_context, subset)["total"])
            for size in range(4)
            for subset in itertools.combinations(client_ids, size)
        }
        for smaller_size in range(4):
            for smaller in itertools.combinations(client_ids, smaller_size):
                for larger_size in range(smaller_size, 4):
                    for larger in itertools.combinations(client_ids, larger_size):
                        if not set(smaller).issubset(larger):
                            continue
                        self.assertLessEqual(values[smaller], values[larger] + 1e-12)
                        for candidate in set(client_ids).difference(larger):
                            left = tuple(sorted((*smaller, candidate)))
                            right = tuple(sorted((*larger, candidate)))
                            marginal_small = values[left] - values[smaller]
                            marginal_large = values[right] - values[larger]
                            self.assertGreaterEqual(
                                marginal_small + 1e-12,
                                marginal_large,
                            )

    def test_greedy_plan_meets_exhaustive_cardinality_bound(self) -> None:
        selector = MICSSelector(
            {"m1": 0.4, "m2": 0.4},
            seed=23,
            service_type=TRUNCATED_LINEAR,
            saturation_levels={"m1": 2, "m2": 2},
            impact_weight=1.0,
            reliability_aware=False,
        )
        round_context = context(
            0,
            budget=2,
            importance={"m1": 0.7, "m2": 0.3},
        )
        optimum = max(
            float(selector.objective_value(round_context, subset)["total"])
            for subset in itertools.combinations(("A", "B", "C"), 2)
        )
        plan = selector.plan_round(round_context)
        achieved = float(plan.metadata["final_objective"]["total"])

        self.assertGreaterEqual(achieved, (1.0 - 1.0 / math.e) * optimum)
        self.assertEqual(
            plan.metadata["greedy_candidate_value_oracle_calls"],
            5,
        )

    def test_realised_service_not_prediction_updates_queues(self) -> None:
        selector = MICSSelector(
            {"m1": 0.5, "m2": 0.5},
            seed=11,
            reliability_aware=False,
        )
        first_context = SelectionContext(
            round_index=0,
            budget=1,
            clients=(
                ClientState("A", {"m1"}, 10),
                ClientState("C", {"m2"}, 10),
            ),
        )
        first = selector.plan_round(first_context)
        first_selected = first.selected_client_ids[0]
        first_modality = "m1" if first_selected == "A" else "m2"
        other_modality = "m2" if first_modality == "m1" else "m1"
        outcome = selector.observe_round(
            0,
            selected_client_ids=first.selected_client_ids,
            successful_client_ids=first.selected_client_ids,
        )

        self.assertEqual(outcome["realised_component_service"][first_modality], 1.0)
        self.assertEqual(selector.component_queues[first_modality], 0.0)
        self.assertEqual(selector.component_queues[other_modality], 0.5)

        second_context = SelectionContext(
            round_index=1,
            budget=1,
            clients=first_context.clients,
        )
        second = selector.plan_round(second_context)
        expected_client = "C" if other_modality == "m2" else "A"
        self.assertEqual(second.selected_client_ids, (expected_client,))

    def test_modular_ablation_counts_overlapping_contributions_twice(
        self,
    ) -> None:
        round_context = context(
            0,
            budget=2,
            importance={"m1": 0.6, "m2": 0.4},
        )
        submodular = MICSSelector(
            {"m1": 0.0, "m2": 0.0},
            seed=31,
            impact_weight=1.0,
            component_queue_weight=0.0,
            subset_utility=SUBMODULAR_SERVICE,
            reliability_aware=False,
        )
        modular = MICSSelector(
            {"m1": 0.0, "m2": 0.0},
            seed=31,
            impact_weight=1.0,
            component_queue_weight=0.0,
            subset_utility=MODULAR_ADDITIVE,
            reliability_aware=False,
        )

        submodular_plan = submodular.plan_round(round_context)
        modular_plan = modular.plan_round(round_context)

        self.assertIn("C", submodular_plan.selected_client_ids)
        self.assertEqual(
            len({"A", "B"}.intersection(submodular_plan.selected_client_ids)),
            1,
        )
        self.assertEqual(set(modular_plan.selected_client_ids), {"A", "B"})
        self.assertEqual(
            modular_plan.metadata["subset_utility"],
            MODULAR_ADDITIVE,
        )

    def test_no_queue_ablation_ignores_accumulated_component_deficit(
        self,
    ) -> None:
        full = MICSSelector(
            {"m1": 1.0, "m2": 1.0},
            seed=13,
            impact_weight=1.0,
            component_queue_weight=1.0,
            reliability_aware=False,
        )
        no_queue = MICSSelector(
            {"m1": 1.0, "m2": 1.0},
            seed=13,
            impact_weight=1.0,
            component_queue_weight=0.0,
            reliability_aware=False,
        )
        clients = (
            ClientState("A", {"m1"}, 10),
            ClientState("C", {"m2"}, 10),
        )
        first_context = SelectionContext(
            round_index=0,
            budget=1,
            clients=clients,
            component_importance={"m1": 0.9, "m2": 0.1},
        )
        for selector in (full, no_queue):
            plan = selector.plan_round(first_context)
            self.assertEqual(plan.selected_client_ids, ("A",))
            selector.observe_round(
                0,
                selected_client_ids=plan.selected_client_ids,
                successful_client_ids=plan.selected_client_ids,
            )

        second_context = SelectionContext(
            round_index=1,
            budget=1,
            clients=clients,
            component_importance={"m1": 0.9, "m2": 0.1},
        )
        self.assertEqual(full.plan_round(second_context).selected_client_ids, ("C",))
        self.assertEqual(
            no_queue.plan_round(second_context).selected_client_ids,
            ("A",),
        )

    def test_client_queues_prevent_complete_client_monopolisation(self) -> None:
        clients = (
            ClientState("A", {"m1", "m2"}, 10),
            ClientState("B", {"m1", "m2"}, 10),
            ClientState("C", {"m1"}, 10),
            ClientState("D", {"m2"}, 10),
        )
        baseline = MICSSelector(
            {"m1": 0.4, "m2": 0.4},
            seed=37,
            impact_weight=1.0,
            reliability_aware=False,
        )
        v2 = MICSSelector(
            {"m1": 0.4, "m2": 0.4},
            seed=37,
            impact_weight=1.0,
            client_targets={client.client_id: 0.2 for client in clients},
            reliability_aware=False,
        )
        counts = {
            "baseline": {client.client_id: 0 for client in clients},
            "v2": {client.client_id: 0 for client in clients},
        }
        for round_index in range(80):
            round_context = SelectionContext(
                round_index=round_index,
                budget=2,
                clients=clients,
                component_importance={"m1": 0.5, "m2": 0.5},
            )
            for name, selector in (("baseline", baseline), ("v2", v2)):
                plan = selector.plan_round(round_context)
                for client_id in plan.selected_client_ids:
                    counts[name][client_id] += 1
                selector.observe_round(
                    round_index,
                    selected_client_ids=plan.selected_client_ids,
                    successful_client_ids=plan.selected_client_ids,
                )

        def fairness(values: dict[str, int]) -> float:
            total = sum(values.values())
            return total * total / (len(values) * sum(v * v for v in values.values()))

        self.assertGreater(fairness(counts["v2"]), 0.98)
        self.assertGreater(
            fairness(counts["v2"]),
            fairness(counts["baseline"]) + 0.05,
        )
        self.assertGreater(counts["v2"]["C"], counts["baseline"]["C"])
        self.assertGreater(counts["v2"]["D"], counts["baseline"]["D"])

    def test_empty_client_targets_are_exactly_v1_selection_mechanics(self) -> None:
        clients = (
            ClientState("A", {"m1", "m2"}, 10),
            ClientState("B", {"m1"}, 10),
            ClientState("C", {"m2"}, 10),
        )
        v1 = MICSSelector(
            {"m1": 0.25, "m2": 0.25},
            seed=71,
            impact_weight=1.0,
            reliability_aware=False,
        )
        no_client_queue = MICSSelector(
            {"m1": 0.25, "m2": 0.25},
            seed=71,
            impact_weight=1.0,
            client_targets={},
            reliability_aware=False,
        )
        for round_index in range(12):
            context = SelectionContext(
                round_index=round_index,
                budget=2,
                clients=clients,
                component_importance={"m1": 0.7, "m2": 0.3},
            )
            left = v1.plan_round(context)
            right = no_client_queue.plan_round(context)
            self.assertEqual(
                left.selected_client_ids,
                right.selected_client_ids,
            )
            self.assertEqual(
                left.metadata["greedy_steps"],
                right.metadata["greedy_steps"],
            )
            for selector, plan in ((v1, left), (no_client_queue, right)):
                selector.observe_round(
                    round_index,
                    selected_client_ids=plan.selected_client_ids,
                    successful_client_ids=plan.selected_client_ids,
                )
        self.assertEqual(v1.component_queues, no_client_queue.component_queues)
        self.assertEqual(no_client_queue.client_queues, {})

    def test_state_round_trip_preserves_next_tie_break(self) -> None:
        original = MICSSelector(
            {"m1": 0.0, "m2": 0.0},
            seed=19,
            reliability_aware=False,
        )
        first = original.plan_round(context(0))
        original.observe_round(
            0,
            selected_client_ids=first.selected_client_ids,
            successful_client_ids=first.selected_client_ids,
        )
        state = original.state_dict()
        expected = original.plan_round(context(1)).selected_client_ids

        resumed = MICSSelector(
            {"m1": 0.0, "m2": 0.0},
            seed=19,
            reliability_aware=False,
        )
        resumed.load_state_dict(state)
        actual = resumed.plan_round(context(1)).selected_client_ids

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
