"""Regression tests for low-loss probes and checkpointed Oort feedback."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from mics.selection import ClientState, MFedMCSelector, OortSelector, SelectionContext
from mics.selection.oort import OORT_PARAMETERS
from mics.selection.vendor.oort_training import _training_selector


def _context(round_index, available=None, losses=None):
    available = {"C1", "C2", "C3"} if available is None else set(available)
    losses = losses or {"C1": 1.0, "C2": 3.0, "C3": 2.0}
    return SelectionContext(
        round_index=round_index,
        budget=2,
        clients=tuple(
            ClientState(
                client_id,
                {"flair", "t1"},
                10,
                available=client_id in available,
                statistics={"modality_losses": {"flair": loss, "t1": loss}},
            )
            for client_id, loss in losses.items()
        ),
    )


class ExtendedSelectorTests(unittest.TestCase):
    def test_mfedmc_selects_lower_mean_modality_losses(self):
        selector = MFedMCSelector(seed=7)
        self.assertEqual(selector.prepare_round(_context(0)).kind, "modality_loss")
        plan = selector.plan_round(_context(0))
        self.assertEqual(plan.selected_client_ids, ("C1", "C3"))
        self.assertEqual(plan.metadata["loss_direction"], "lower")

    def test_mfedmc_resume_preserves_ties(self):
        selector = MFedMCSelector(seed=11)
        context = _context(0, losses={"C1": 1.0, "C2": 1.0, "C3": 1.0})
        selector.prepare_round(context)
        plan = selector.plan_round(context)
        selector.observe_round(
            0,
            selected_client_ids=plan.selected_client_ids,
            successful_client_ids=plan.selected_client_ids,
        )
        resumed = MFedMCSelector(seed=11)
        resumed.load_state_dict(selector.state_dict())
        next_context = _context(1, losses={"C1": 1.0, "C2": 1.0, "C3": 1.0})
        for item in (selector, resumed):
            item.prepare_round(next_context)
        self.assertEqual(
            selector.plan_round(next_context), resumed.plan_round(next_context)
        )

    @staticmethod
    def _observe(selector, plan, successes=None):
        successes = plan.selected_client_ids if successes is None else successes
        selector.record_training_feedback(
            plan.round_index,
            {
                client_id: {
                    "loss_rms": 0.5 + int(client_id[-1]) / 10,
                    "elapsed_seconds": 1.0,
                }
                for client_id in successes
            },
        )
        return selector.observe_round(
            plan.round_index,
            selected_client_ids=plan.selected_client_ids,
            successful_client_ids=successes,
        )

    def test_oort_bootstrap_and_small_available_pools(self):
        selector = OortSelector(seed=19)
        for round_index in range(50):
            available = ("C1", "C2") if round_index % 2 else ("C2", "C3")
            plan = selector.plan_round(_context(round_index, available=available))
            self.assertEqual(set(plan.selected_client_ids), set(available))
            self._observe(selector, plan)

    def test_oort_uses_rms_utility_and_replays_checkpoint(self):
        selector = OortSelector(seed=23)
        plan = selector.plan_round(_context(0))
        feedback = self._observe(selector, plan)
        for client_id in plan.selected_client_ids:
            expected = 10 * (0.5 + int(client_id[-1]) / 10)
            self.assertAlmostEqual(
                feedback["client_feedback"][client_id]["statistical_utility"], expected
            )
        resumed = OortSelector(seed=23)
        resumed.load_state_dict(copy.deepcopy(selector.state_dict()))
        for round_index in range(1, 40):
            expected = selector.plan_round(_context(round_index))
            actual = resumed.plan_round(_context(round_index))
            self.assertEqual(expected.selected_client_ids, actual.selected_client_ids)
            self._observe(selector, expected)
            self._observe(resumed, actual)

    def test_oort_accepts_failed_attempts_and_zero_losses(self):
        selector = OortSelector(seed=29)
        for round_index in range(12):
            plan = selector.plan_round(_context(round_index))
            successes = () if round_index % 2 else plan.selected_client_ids
            selector.record_training_feedback(
                round_index,
                {
                    client_id: {"loss_rms": 0.0, "elapsed_seconds": 1.0}
                    for client_id in successes
                },
            )
            selector.observe_round(
                round_index,
                selected_client_ids=plan.selected_client_ids,
                successful_client_ids=successes,
            )

    def test_oort_rng_is_isolated_from_numpy_global_state(self):
        np.random.seed(31)
        expected = np.random.random(5)
        np.random.seed(31)
        selector = OortSelector(seed=37)
        plan = selector.plan_round(_context(0))
        self._observe(selector, plan)
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_oort_fills_budget_after_cutoff_leaves_one_positive_candidate(self):
        backend = _training_selector(
            SimpleNamespace(**OORT_PARAMETERS), sample_seed=1009
        )
        for client, reward in enumerate((0.0, 0.0, 20.0, 0.0)):
            backend.register_client(client, {"reward": 10.0, "duration": 1.0})
            backend.update_client_util(
                client,
                {
                    "reward": reward,
                    "duration": 1.0,
                    "time_stamp": 57,
                    "status": reward > 0,
                },
            )
        backend.training_round = 57
        backend.exploration = backend.exploration_min = 0.0
        selected = backend.select_participant(2, feasible_clients={0, 1, 2, 3})
        self.assertEqual(len(set(selected)), 2)
        self.assertIn(2, selected)
        self.assertEqual(selected[0], 2)

    def test_oort_mixed_failures_varying_availability_remain_checkpointable(self):
        selector = OortSelector(seed=1009)
        rng = np.random.RandomState(3001)
        for round_index in range(150):
            available = {f"C{i}" for i in range(8) if rng.random_sample() < 0.8}
            available.update({"C0", "C1"})
            context = SelectionContext(
                round_index,
                2,
                tuple(
                    ClientState(f"C{i}", {"m"}, 28, available=f"C{i}" in available)
                    for i in range(8)
                ),
            )
            plan = selector.plan_round(context)
            self.assertEqual(len(set(plan.selected_client_ids)), 2)
            self.assertTrue(set(plan.selected_client_ids).issubset(available))
            successes = [
                client
                for client in plan.selected_client_ids
                if rng.random_sample() < 0.6
            ]
            selector.record_training_feedback(
                round_index,
                {
                    client: {
                        "loss_rms": float(rng.uniform(0.1, 3)),
                        "elapsed_seconds": float(rng.uniform(2, 8)),
                    }
                    for client in successes
                },
            )
            selector.observe_round(
                round_index,
                selected_client_ids=plan.selected_client_ids,
                successful_client_ids=successes,
            )
            restored = OortSelector(seed=1009)
            restored.load_state_dict(selector.state_dict())
            selector = restored


if __name__ == "__main__":
    unittest.main()
