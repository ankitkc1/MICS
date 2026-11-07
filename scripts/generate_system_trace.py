#!/usr/bin/env python3
"""Generate a reusable client availability/completion trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mics.data import load_brats_scenario
from mics.experiments import atomic_write_json, generate_system_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MICS_DATA_ROOT"),
    )
    parser.add_argument("--scenario-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--minimum-available-clients", type=int, default=2)
    parser.add_argument("--availability-probability", type=float, default=1.0)
    parser.add_argument("--completion-probability", type=float, default=1.0)
    parser.add_argument(
        "--client-availability",
        action="append",
        default=[],
        metavar="CLIENT=PROBABILITY",
    )
    parser.add_argument(
        "--client-completion",
        action="append",
        default=[],
        metavar="CLIENT=PROBABILITY",
    )
    return parser


def _overrides(values: list[str], clients: tuple[str, ...]) -> dict[str, float]:
    result = {}
    for declaration in values:
        if "=" not in declaration:
            raise SystemExit(f"Expected CLIENT=PROBABILITY, found {declaration!r}.")
        client_id, raw_probability = declaration.split("=", 1)
        client_id = client_id.strip().upper()
        if client_id not in clients:
            raise SystemExit(f"Unknown trace client: {client_id}")
        if client_id in result:
            raise SystemExit(f"Duplicate trace override for {client_id}.")
        result[client_id] = float(raw_probability)
    return result


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.rounds <= 0 or arguments.seed < 0:
        raise SystemExit("Trace rounds must be positive and seed non-negative.")
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite system trace: {output}")
    scenario = load_brats_scenario(
        arguments.scenario_config,
        dataset_root=arguments.dataset_root,
    )
    clients = tuple(client.client_id for client in scenario.clients)
    availability = {
        client_id: float(arguments.availability_probability)
        for client_id in clients
    }
    completion = {
        client_id: float(arguments.completion_probability)
        for client_id in clients
    }
    availability.update(_overrides(arguments.client_availability, clients))
    completion.update(_overrides(arguments.client_completion, clients))
    trace = generate_system_trace(
        clients,
        rounds=arguments.rounds,
        seed=arguments.seed,
        availability_probabilities=availability,
        completion_probabilities=completion,
        minimum_available_clients=arguments.minimum_available_clients,
        trace_id=arguments.trace_id,
    )
    payload = trace.as_dict()
    payload["scenario"] = {
        "id": scenario.scenario_id,
        "config": str(scenario.config_path),
    }
    atomic_write_json(output, payload)
    summary = {
        "output": str(output),
        "trace_id": trace.trace_id,
        "round_count": len(trace.rounds),
        "available_client_counts": [
            len(record.available_clients) for record in trace.rounds
        ],
        "potential_completion_counts": [
            sum(record.completion_potential.values()) for record in trace.rounds
        ],
        "availability_probabilities": trace.availability_probabilities,
        "completion_probabilities": trace.completion_probabilities,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("MICS SYSTEM TRACE GENERATION PASSED")


if __name__ == "__main__":
    main()
