"""Tests for policy-neutral selection and communication diagnostics."""

from __future__ import annotations

import unittest

import torch

from mics import ClientState, SelectionContext, SelectionPlan
from mics.experiments import (
    component_state_nbytes,
    jain_fairness_index,
    selection_round_diagnostics,
    validate_selection_plan,
)


class ExperimentMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = SelectionContext(
            round_index=2,
            budget=2,
            clients=(
                ClientState(
                    "C1",
                    {"flair", "t1"},
                    28,
                    last_selected_round=0,
                ),
                ClientState("C2", {"t1ce", "t1"}, 28),
                ClientState("C3", {"t2"}, 27, last_selected_round=1),
            ),
            component_importance={
                "flair": 0.4,
                "t1ce": 0.3,
                "t1": 0.2,
                "t2": 0.1,
            },
        )
        self.plan = SelectionPlan(2, ("C1", "C2"))

    def test_diagnostics_capture_coverage_staleness_and_fairness(self) -> None:
        diagnostics = selection_round_diagnostics(
            self.context,
            self.plan,
            selection_counts={"C1": 1, "C2": 0, "C3": 1},
            modality_last_updated_round={
                "flair": 0,
                "t1": 1,
                "t1ce": None,
                "t2": 1,
            },
        )

        self.assertEqual(
            diagnostics["modality_coverage"],
            {"flair": 1, "t1": 1, "t1ce": 1, "t2": 0},
        )
        self.assertEqual(
            diagnostics["selected_clients_per_modality"],
            {"flair": 1, "t1": 2, "t1ce": 1, "t2": 0},
        )
        self.assertAlmostEqual(
            diagnostics["impact_weighted_modality_coverage"],
            0.9,
        )
        self.assertEqual(
            diagnostics["selected_client_staleness_before_round"],
            {"C1": 2, "C2": 3},
        )
        self.assertEqual(
            diagnostics["selection_counts_after_round"],
            {"C1": 2, "C2": 1, "C3": 1},
        )
        self.assertAlmostEqual(
            diagnostics["jain_selection_fairness_after_round"],
            16 / 18,
        )
        self.assertAlmostEqual(
            diagnostics["pairwise_modality_jaccard_redundancy"],
            1 / 3,
        )

    def test_plan_validation_rejects_wrong_budget_or_client(self) -> None:
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_selection_plan(self.context, SelectionPlan(2, ("C1",)))
        with self.assertRaisesRegex(ValueError, "ineligible"):
            validate_selection_plan(
                self.context,
                SelectionPlan(2, ("C1", "C9")),
            )

    def test_jain_index_and_tensor_byte_count(self) -> None:
        self.assertEqual(jain_fairness_index({"C1": 0, "C2": 0}), 1.0)
        self.assertEqual(jain_fairness_index({"C1": 2, "C2": 2}), 1.0)
        state = {
            "a": torch.zeros(3, dtype=torch.float32),
            "b": torch.zeros(2, dtype=torch.int64),
        }
        self.assertEqual(component_state_nbytes(state), 28)


if __name__ == "__main__":
    unittest.main()
