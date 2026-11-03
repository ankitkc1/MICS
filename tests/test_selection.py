"""Tests for the selector contract and uniform baseline."""

from __future__ import annotations

import unittest

from mics import ClientState, SelectionContext, UniformRandomSelector


def make_context(round_index: int, budget: int = 2) -> SelectionContext:
    return SelectionContext(
        round_index=round_index,
        budget=budget,
        clients=(
            ClientState("C1", {"t1", "flair"}, 12),
            ClientState("C2", {"t1ce"}, 10),
            ClientState("C3", {"t2"}, 8),
            ClientState("C4", {"flair"}, 7, available=False),
        ),
    )


class UniformRandomSelectorTests(unittest.TestCase):
    def test_same_seed_produces_same_sequence(self) -> None:
        left = UniformRandomSelector(seed=42)
        right = UniformRandomSelector(seed=42)

        left_plans = [
            left.plan_round(make_context(round_index)).selected_client_ids
            for round_index in range(5)
        ]
        right_plans = [
            right.plan_round(make_context(round_index)).selected_client_ids
            for round_index in range(5)
        ]

        self.assertEqual(left_plans, right_plans)

    def test_unavailable_clients_are_never_selected(self) -> None:
        selector = UniformRandomSelector(seed=9)
        for round_index in range(20):
            plan = selector.plan_round(make_context(round_index))
            self.assertEqual(len(plan.selected_client_ids), 2)
            self.assertNotIn("C4", plan.selected_client_ids)

    def test_state_round_trip_preserves_resume_sequence(self) -> None:
        original = UniformRandomSelector(seed=3)
        original.plan_round(make_context(0))
        state = original.state_dict()
        expected = original.plan_round(make_context(1)).selected_client_ids

        resumed = UniformRandomSelector(seed=3)
        resumed.load_state_dict(state)
        actual = resumed.plan_round(make_context(1)).selected_client_ids

        self.assertEqual(actual, expected)

    def test_budget_cannot_exceed_eligible_clients(self) -> None:
        with self.assertRaisesRegex(ValueError, "budget exceeds"):
            make_context(0, budget=4)


if __name__ == "__main__":
    unittest.main()
