"""Regression tests for low-loss probes and checkpointed Oort feedback."""

import copy
import unittest

import numpy as np

from mics.selection import ClientState, MFedMCSelector, OortSelector, SelectionContext


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


if __name__ == "__main__":
    unittest.main()
