"""Stable interfaces for client-selection methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ClientState:
    """Round-time information that a selector may use about one client.

    The training system, not the selector, owns patient data and model
    parameters. Selection methods receive only explicitly reported metadata.
    """

    client_id: str
    modalities: frozenset[str]
    num_samples: int
    available: bool = True
    last_selected_round: int | None = None
    statistics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", frozenset(self.modalities))
        if not self.client_id:
            raise ValueError("client_id must be nonempty.")
        if not self.modalities:
            raise ValueError(f"Client {self.client_id!r} must expose at least one modality.")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.last_selected_round is not None and self.last_selected_round < 0:
            raise ValueError("last_selected_round cannot be negative.")


@dataclass(frozen=True)
class SelectionContext:
    """Immutable information supplied to a selector for one FL round."""

    round_index: int
    budget: int
    clients: tuple[ClientState, ...]
    component_importance: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clients", tuple(self.clients))
        if self.round_index < 0:
            raise ValueError("round_index cannot be negative.")
        if self.budget <= 0:
            raise ValueError("budget must be positive.")
        identifiers = [client.client_id for client in self.clients]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Client identifiers must be unique.")
        if self.budget > len(self.eligible_clients):
            raise ValueError("budget exceeds the number of eligible clients.")

    @property
    def eligible_clients(self) -> tuple[ClientState, ...]:
        return tuple(client for client in self.clients if client.available)


@dataclass(frozen=True)
class SelectionPlan:
    """Auditable output of a client selector."""

    round_index: int
    selected_client_ids: tuple[str, ...]
    scores: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_client_ids", tuple(self.selected_client_ids))
        if self.round_index < 0:
            raise ValueError("round_index cannot be negative.")
        if not self.selected_client_ids:
            raise ValueError("A selection plan cannot be empty.")
        if len(self.selected_client_ids) != len(set(self.selected_client_ids)):
            raise ValueError("A selection plan cannot contain duplicate clients.")


class ClientSelector(ABC):
    """Common selector API used by training and offline selector studies."""

    @abstractmethod
    def plan_round(self, context: SelectionContext) -> SelectionPlan:
        """Choose clients for one communication round."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return resumable selector state."""

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore selector state produced by state_dict."""

        if state:
            raise ValueError(f"{type(self).__name__} does not accept state.")
