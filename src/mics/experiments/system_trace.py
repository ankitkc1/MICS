"""Deterministic client availability and completion-potential traces."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import sha256_file


def _stream_seed(seed: int, name: str) -> int:
    material = f"{int(seed)}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _probabilities(
    clients: tuple[str, ...],
    values: dict[str, float],
    *,
    name: str,
) -> dict[str, float]:
    if set(values) != set(clients):
        raise ValueError(f"{name} must define every trace client.")
    result = {client_id: float(values[client_id]) for client_id in clients}
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in result.values()
    ):
        raise ValueError(f"{name} values must lie in [0, 1].")
    return result


@dataclass(frozen=True)
class SystemTraceRound:
    round_index: int
    available_clients: tuple[str, ...]
    completion_potential: dict[str, bool]


@dataclass(frozen=True)
class SystemTrace:
    trace_id: str
    seed: int
    clients: tuple[str, ...]
    availability_probabilities: dict[str, float]
    completion_probabilities: dict[str, float]
    minimum_available_clients: int
    rounds: tuple[SystemTraceRound, ...]
    path: Path | None = None
    file_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id cannot be empty.")
        if self.seed < 0:
            raise ValueError("trace seed must be non-negative.")
        if not self.clients or len(self.clients) != len(set(self.clients)):
            raise ValueError("trace clients must be non-empty and unique.")
        _probabilities(
            self.clients,
            self.availability_probabilities,
            name="availability_probabilities",
        )
        _probabilities(
            self.clients,
            self.completion_probabilities,
            name="completion_probabilities",
        )
        if not 1 <= self.minimum_available_clients <= len(self.clients):
            raise ValueError("minimum_available_clients is invalid.")
        if not self.rounds:
            raise ValueError("system trace must contain at least one round.")
        for expected_round, record in enumerate(self.rounds):
            if record.round_index != expected_round:
                raise ValueError("system trace rounds must be consecutive from zero.")
            if len(record.available_clients) < self.minimum_available_clients:
                raise ValueError("trace round violates minimum availability.")
            if len(record.available_clients) != len(set(record.available_clients)):
                raise ValueError("trace round contains duplicate available clients.")
            if not set(record.available_clients).issubset(self.clients):
                raise ValueError("trace round contains an unknown available client.")
            if set(record.completion_potential) != set(self.clients):
                raise ValueError("completion potential must define every client.")
            if any(
                not isinstance(value, bool)
                for value in record.completion_potential.values()
            ):
                raise ValueError("completion potentials must be Boolean.")
            available = set(record.available_clients)
            for client_id in self.clients:
                availability = self.availability_probabilities[client_id]
                completion = self.completion_probabilities[client_id]
                if availability == 1.0 and client_id not in available:
                    raise ValueError(
                        "certainly available client is absent from a round."
                    )
                if availability == 0.0 and client_id in available:
                    raise ValueError("certainly unavailable client appears in a round.")
                if completion == 1.0 and not record.completion_potential[client_id]:
                    raise ValueError("certain completion is false in a trace round.")
                if completion == 0.0 and record.completion_potential[client_id]:
                    raise ValueError("impossible completion is true in a trace round.")

    def round(self, round_index: int) -> SystemTraceRound:
        index = int(round_index)
        if not 0 <= index < len(self.rounds):
            raise IndexError(f"System trace has no round {index}.")
        return self.rounds[index]

    def successful_selected_clients(
        self,
        round_index: int,
        selected_client_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        record = self.round(round_index)
        available = set(record.available_clients)
        selected = tuple(selected_client_ids)
        if not set(selected).issubset(available):
            raise ValueError("Selected clients must be available in the trace.")
        return tuple(
            client_id
            for client_id in selected
            if record.completion_potential[client_id]
        )

    def source_record(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "path": None if self.path is None else str(self.path),
            "sha256": self.file_sha256,
            "seed": self.seed,
            "round_count": len(self.rounds),
            "minimum_available_clients": self.minimum_available_clients,
            "availability_probabilities": dict(self.availability_probabilities),
            "completion_probabilities": dict(self.completion_probabilities),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "seed": self.seed,
            "clients": list(self.clients),
            "generation": {
                "availability_model": (
                    "independent_bernoulli_conditioned_on_minimum_count"
                ),
                "completion_model": (
                    "selection_independent_client_bernoulli_potential_outcomes"
                ),
                "minimum_available_clients": self.minimum_available_clients,
                "availability_probabilities": dict(
                    self.availability_probabilities
                ),
                "completion_probabilities": dict(
                    self.completion_probabilities
                ),
            },
            "rounds": [
                {
                    "round": record.round_index,
                    "available_clients": list(record.available_clients),
                    "completion_potential": dict(record.completion_potential),
                }
                for record in self.rounds
            ],
        }


def generate_system_trace(
    clients: tuple[str, ...],
    *,
    rounds: int,
    seed: int,
    availability_probabilities: dict[str, float],
    completion_probabilities: dict[str, float],
    minimum_available_clients: int,
    trace_id: str,
) -> SystemTrace:
    """Generate independent potential outcomes with separate RNG streams."""

    client_ids = tuple(str(client_id) for client_id in clients)
    availability = _probabilities(
        client_ids,
        availability_probabilities,
        name="availability_probabilities",
    )
    completion = _probabilities(
        client_ids,
        completion_probabilities,
        name="completion_probabilities",
    )
    round_count = int(rounds)
    if round_count <= 0:
        raise ValueError("rounds must be positive.")
    if not 1 <= minimum_available_clients <= len(client_ids):
        raise ValueError("minimum_available_clients is invalid.")
    if sum(value > 0.0 for value in availability.values()) < minimum_available_clients:
        raise ValueError("Availability model cannot meet the minimum client count.")
    availability_rng = random.Random(_stream_seed(seed, "availability"))
    completion_rng = random.Random(_stream_seed(seed, "completion"))
    records = []
    for round_index in range(round_count):
        for _ in range(100_000):
            available_clients = tuple(
                client_id
                for client_id in client_ids
                if availability_rng.random() < availability[client_id]
            )
            if len(available_clients) >= minimum_available_clients:
                break
        else:  # pragma: no cover - protected by feasibility and generous cap
            raise RuntimeError("Could not sample a feasible availability state.")
        potential = {
            client_id: completion_rng.random() < completion[client_id]
            for client_id in client_ids
        }
        records.append(
            SystemTraceRound(
                round_index=round_index,
                available_clients=available_clients,
                completion_potential=potential,
            )
        )
    return SystemTrace(
        trace_id=str(trace_id),
        seed=int(seed),
        clients=client_ids,
        availability_probabilities=availability,
        completion_probabilities=completion,
        minimum_available_clients=int(minimum_available_clients),
        rounds=tuple(records),
    )


def load_system_trace(path: str | Path) -> SystemTrace:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"System trace does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported system-trace schema.")
    generation = payload.get("generation")
    raw_rounds = payload.get("rounds")
    if not isinstance(generation, dict) or not isinstance(raw_rounds, list):
        raise ValueError("System trace has invalid generation or round records.")
    if generation.get("availability_model") != (
        "independent_bernoulli_conditioned_on_minimum_count"
    ):
        raise ValueError("Unsupported system-trace availability model.")
    if generation.get("completion_model") != (
        "selection_independent_client_bernoulli_potential_outcomes"
    ):
        raise ValueError("Unsupported system-trace completion model.")
    raw_clients = payload.get("clients")
    if not isinstance(raw_clients, list):
        raise ValueError("System trace clients must be a list.")
    clients = tuple(str(value) for value in raw_clients)
    records = []
    for record in raw_rounds:
        if not isinstance(record, dict):
            raise ValueError("System-trace rounds must be objects.")
        available = record.get("available_clients")
        potential = record.get("completion_potential")
        if not isinstance(available, list) or not isinstance(potential, dict):
            raise ValueError("System-trace round fields have invalid types.")
        if any(not isinstance(value, bool) for value in potential.values()):
            raise ValueError("System-trace completion values must be Boolean.")
        records.append(
            SystemTraceRound(
                round_index=int(record["round"]),
                available_clients=tuple(str(value) for value in available),
                completion_potential={
                    str(key): value for key, value in potential.items()
                },
            )
        )
    return SystemTrace(
        trace_id=str(payload["trace_id"]),
        seed=int(payload["seed"]),
        clients=clients,
        availability_probabilities={
            str(key): float(value)
            for key, value in generation["availability_probabilities"].items()
        },
        completion_probabilities={
            str(key): float(value)
            for key, value in generation["completion_probabilities"].items()
        },
        minimum_available_clients=int(
            generation["minimum_available_clients"]
        ),
        rounds=tuple(records),
        path=resolved,
        file_sha256=sha256_file(resolved),
    )
