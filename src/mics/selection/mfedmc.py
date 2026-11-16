"""Shared-model adaptation of MFedMC's paper-defined low-loss client rule."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from typing import Any

from .base import ClientSelector, SelectionContext, SelectionPlan, SelectionProbeRequest


class MFedMCSelector(ClientSelector):
    """Select the smallest mean single-modality loss using a fixed client budget.

    Single-modality predictions use the shared decoder. Probes precede local
    optimization, and all components used by a selected client are aggregated.
    This isolates the paper's low-loss client rule from its separate encoder
    uploads, post-training selection and personalized fusion architecture.
    """

    def __init__(self, *, seed: int, max_probe_batches: int | None = None) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if max_probe_batches is not None and max_probe_batches <= 0:
            raise ValueError("max_probe_batches must be positive when provided.")
        self.seed = int(seed)
        self.max_probe_batches = max_probe_batches
        self._rng = random.Random(self.seed)
        self._last_round = -1
        self._pending_round: int | None = None
        self._pending_selection: tuple[str, ...] | None = None

    def prepare_round(self, context: SelectionContext) -> SelectionProbeRequest:
        if self._pending_round is not None:
            raise RuntimeError("The previous MFedMC round has not been observed.")
        if context.round_index != self._last_round + 1:
            raise ValueError("MFedMC rounds must be consecutive.")
        self._pending_round = context.round_index
        return SelectionProbeRequest(
            round_index=context.round_index,
            client_ids=tuple(
                sorted(client.client_id for client in context.eligible_clients)
            ),
            kind="modality_loss",
            max_batches=self.max_probe_batches,
        )

    def plan_round(self, context: SelectionContext) -> SelectionPlan:
        if (
            self._pending_round != context.round_index
            or self._pending_selection is not None
        ):
            raise RuntimeError("MFedMC has no unconsumed prepared round.")
        losses = {}
        for client in context.eligible_clients:
            supplied = client.statistics.get("modality_losses")
            if not isinstance(supplied, Mapping) or not supplied:
                raise ValueError(
                    f"Missing modality-loss probes for {client.client_id}."
                )
            if not set(supplied).issubset(client.modalities):
                raise ValueError(
                    "Modality-loss probes contain an unknown client modality."
                )
            values = [float(value) for value in supplied.values()]
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("Modality losses must be finite and non-negative.")
            losses[client.client_id] = sum(values) / len(values)
        groups: dict[float, list[str]] = {}
        for client_id, loss in losses.items():
            groups.setdefault(loss, []).append(client_id)
        ranked = []
        for loss in sorted(groups):
            tied = sorted(groups[loss])
            self._rng.shuffle(tied)
            ranked.extend(tied)
        selected = tuple(ranked[: context.budget])
        self._pending_selection = selected
        return SelectionPlan(
            round_index=context.round_index,
            selected_client_ids=selected,
            scores={client_id: -loss for client_id, loss in losses.items()},
            metadata={
                "selector": "mfedmc_client_low_loss",
                "variant": "paper_low_loss_shared_decoder_probe",
                "mean_modality_losses": losses,
                "loss_direction": "lower",
                "probe_timing": "pre_training",
                "tie_breaking": "seeded_uniform_within_equal_loss",
                "max_probe_batches": self.max_probe_batches,
                "reference": "https://arxiv.org/abs/2401.16685v2",
                "released_code_revision": "bb9907254683818dfe9e4784f798a25eca6a1d58",
                "released_code_default_loss_direction": "higher",
            },
        )

    def observe_round(
        self,
        round_index: int,
        *,
        selected_client_ids: Iterable[str],
        successful_client_ids: Iterable[str],
    ) -> Mapping[str, Any]:
        selected = tuple(selected_client_ids)
        if round_index != self._pending_round or selected != self._pending_selection:
            raise ValueError("MFedMC observation does not match its pending plan.")
        if not set(successful_client_ids).issubset(selected):
            raise ValueError("Successful clients must be selected clients.")
        self._last_round = round_index
        self._pending_round = None
        self._pending_selection = None
        return {"round": round_index}

    def state_dict(self) -> Mapping[str, Any]:
        if self._pending_round is not None:
            raise RuntimeError("Cannot checkpoint a pending MFedMC round.")
        return {
            "schema_version": 1,
            "seed": self.seed,
            "max_probe_batches": self.max_probe_batches,
            "last_round": self._last_round,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or state.get("seed") != self.seed:
            raise ValueError("MFedMC checkpoint schema or seed differs.")
        if state.get("max_probe_batches") != self.max_probe_batches:
            raise ValueError("MFedMC checkpoint probe limit differs.")
        if self._pending_round is not None:
            raise RuntimeError("Cannot restore over a pending MFedMC round.")
        self._rng.setstate(state["rng_state"])
        self._last_round = int(state["last_round"])
