"""Reproducible uniform-random client selection."""

from __future__ import annotations

import random
from typing import Any, Mapping

from .base import ClientSelector, SelectionContext, SelectionPlan


class UniformRandomSelector(ClientSelector):
    """Sample eligible clients uniformly without replacement.

    The selector owns an independent pseudo-random stream. Data loading,
    augmentation, model initialization, and client selection must never share
    this stream because different policies consume different numbers of draws.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def plan_round(self, context: SelectionContext) -> SelectionPlan:
        eligible_ids = tuple(
            sorted(client.client_id for client in context.eligible_clients)
        )
        selected = tuple(self._rng.sample(eligible_ids, context.budget))
        return SelectionPlan(
            round_index=context.round_index,
            selected_client_ids=selected,
            scores={client_id: 1.0 for client_id in eligible_ids},
            metadata={
                "selector": "uniform_random",
                "without_replacement": True,
                "seed": self.seed,
            },
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "seed": self.seed,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"seed", "rng_state"}:
            raise ValueError("UniformRandomSelector state has unexpected keys.")
        if int(state["seed"]) != self.seed:
            raise ValueError("Selector state seed does not match this selector.")
        self._rng.setstate(state["rng_state"])
