"""Tests for MICS queues, calibration, and the greedy value oracle."""

from __future__ import annotations

import itertools
import math
import unittest

from mics import ClientState, MICSSelector, SelectionContext
from mics.selection import (
    PURE_COVERAGE,
    TRUNCATED_LINEAR,
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
        round_context = context(
            0,
            budget=2,
            importance={"m1": 0.6, "m2": 0.4},
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
