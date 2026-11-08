"""Tests for probe-based Power-of-Choice and DivFL baselines."""

from __future__ import annotations

import copy
import unittest
from collections import OrderedDict

import torch

from mics.experiments import component_state_nbytes, run_selection_probes
from mics.models import MultiEncoderUNet3D
from mics.selection import (
    ClientState,
    DivFLSelector,
    PowerOfChoiceSelector,
    SelectionContext,
    SelectionProbeRequest,
)


def _context(
    round_index: int,
    *,
    statistics: dict[str, dict[str, object]] | None = None,
) -> SelectionContext:
    values = statistics or {}
    return SelectionContext(
        round_index=round_index,
        budget=2,
        clients=tuple(
            ClientState(
                client_id,
                modalities,
                samples,
                statistics=values.get(client_id, {}),
            )
            for client_id, modalities, samples in (
                ("C1", {"flair", "t1ce"}, 12),
                ("C2", {"t1", "t2"}, 10),
                ("C3", {"flair", "t2"}, 8),
            )
        ),
    )


class PowerOfChoiceSelectorTests(unittest.TestCase):
    def test_selects_highest_current_losses_from_candidate_pool(self) -> None:
        selector = PowerOfChoiceSelector(candidate_size=3, seed=7)
        request = selector.prepare_round(_context(0))

        self.assertEqual(request.kind, "local_loss")
        self.assertEqual(set(request.client_ids), {"C1", "C2", "C3"})
        plan = selector.plan_round(
            _context(
                0,
                statistics={
                    "C1": {"local_loss": 1.0},
                    "C2": {"local_loss": 3.0},
                    "C3": {"local_loss": 2.0},
                },
            )
        )

        self.assertEqual(plan.selected_client_ids, ("C2", "C3"))
        selector.observe_round(
            0,
            selected_client_ids=plan.selected_client_ids,
            successful_client_ids=plan.selected_client_ids,
        )

    def test_checkpoint_round_trip_preserves_candidate_sampling(self) -> None:
        original = PowerOfChoiceSelector(candidate_size=2, seed=11)
        first_request = original.prepare_round(_context(0))
        first_statistics = {
            client_id: {"local_loss": float(index)}
            for index, client_id in enumerate(first_request.client_ids)
        }
        first_plan = original.plan_round(
            _context(0, statistics=first_statistics)
        )
        original.observe_round(
            0,
            selected_client_ids=first_plan.selected_client_ids,
            successful_client_ids=first_plan.selected_client_ids,
        )
        state = original.state_dict()
        expected = original.prepare_round(_context(1)).client_ids

        resumed = PowerOfChoiceSelector(candidate_size=2, seed=11)
        resumed.load_state_dict(state)
        actual = resumed.prepare_round(_context(1)).client_ids

        self.assertEqual(actual, expected)


class DivFLSelectorTests(unittest.TestCase):
    def test_greedy_facility_location_selects_representatives(self) -> None:
        selector = DivFLSelector(seed=3)
        request = selector.prepare_round(_context(0))
        self.assertEqual(request.kind, "gradient_vector")
        context = _context(
            0,
            statistics={
                "C1": {"gradient_vector": torch.tensor([0.0, 0.0])},
                "C2": {"gradient_vector": torch.tensor([1.0, 0.0])},
                "C3": {"gradient_vector": torch.tensor([10.0, 0.0])},
            },
        )

        plan = selector.plan_round(context)

        self.assertEqual(plan.selected_client_ids, ("C2", "C3"))
        self.assertGreater(plan.metadata["final_facility_value"], 0.0)
        selector.observe_round(
            0,
            selected_client_ids=plan.selected_client_ids,
            successful_client_ids=plan.selected_client_ids,
        )


class SelectionProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.model = MultiEncoderUNet3D(
            modalities=OrderedDict((name, 1) for name in ("flair", "t1")),
            num_classes=4,
            base_channels=2,
            num_levels=2,
            fusion="mean",
            modality_dropout=0.0,
        )
        self.component_bytes = {
            name: component_state_nbytes(state)
            for name, state in self.model.component_state_dict(
                clone=False
            ).items()
        }

    @staticmethod
    def _batch(modality: str) -> dict[str, object]:
        return {
            "inputs": {modality: torch.randn(1, 1, 4, 4, 4)},
            "target": torch.zeros(1, 4, 4, 4, dtype=torch.long),
        }

    def test_loss_probe_is_train_only_and_does_not_change_model(self) -> None:
        initial = copy.deepcopy(self.model.state_dict())
        outcome = run_selection_probes(
            self.model,
            SelectionProbeRequest(
                round_index=0,
                client_ids=("C1",),
                kind="local_loss",
                max_batches=1,
            ),
            loaders={"C1": [self._batch("flair")]},
            client_modalities={"C1": ("flair",)},
            component_bytes=self.component_bytes,
            device="cpu",
        )

        self.assertEqual(outcome.diagnostics["partition"], "train")
        self.assertFalse(outcome.diagnostics["model_updated"])
        self.assertEqual(outcome.upload_bytes_by_client["C1"], 8)
        self.assertGreater(outcome.statistics["C1"]["local_loss"], 0.0)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, initial[name]))

    def test_gradient_probe_returns_aligned_finite_vector(self) -> None:
        outcome = run_selection_probes(
            self.model,
            SelectionProbeRequest(
                round_index=0,
                client_ids=("C1", "C2"),
                kind="gradient_vector",
                max_batches=1,
            ),
            loaders={
                "C1": [self._batch("flair")],
                "C2": [self._batch("t1")],
            },
            client_modalities={"C1": ("flair",), "C2": ("t1",)},
            component_bytes=self.component_bytes,
            device="cpu",
        )

        expected_dimension = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        vectors = [
            outcome.statistics[client_id]["gradient_vector"]
            for client_id in ("C1", "C2")
        ]
        self.assertTrue(all(vector.numel() == expected_dimension for vector in vectors))
        self.assertTrue(all(bool(torch.isfinite(vector).all()) for vector in vectors))
        self.assertTrue(
            all(value > 0 for value in outcome.upload_bytes_by_client.values())
        )


if __name__ == "__main__":
    unittest.main()
