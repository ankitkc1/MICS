#!/usr/bin/env python3
"""Run a model-free matched selector preflight for MICS-v2 client queues."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from mics import ClientState, MICSSelector, SelectionContext
from mics.data import BRATS_MODALITIES, load_brats_scenario
from mics.experiments import (
    atomic_write_json,
    load_modality_importance_record,
    load_system_trace,
    sha256_file,
)
from mics.selection import (
    PURE_COVERAGE,
    calibrate_availability_aware_client_targets,
    calibrate_availability_aware_service_targets,
)


def _jain(values: dict[str, int]) -> float:
    numbers = tuple(float(value) for value in values.values())
    total = sum(numbers)
    squared = sum(value * value for value in numbers)
    return 1.0 if squared == 0.0 else total * total / (len(numbers) * squared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MICS_DATA_ROOT"),
    )
    parser.add_argument("--scenario-config", required=True, type=Path)
    parser.add_argument("--system-trace", required=True, type=Path)
    parser.add_argument("--importance-output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=1009)
    parser.add_argument("--safety-margin", type=float, default=0.1)
    parser.add_argument("--source-revision")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selector preflight: {output}")

    scenario = load_brats_scenario(
        arguments.scenario_config,
        dataset_root=arguments.dataset_root,
    )
    trace = load_system_trace(arguments.system_trace)
    importance = load_modality_importance_record(
        arguments.importance_output,
        modalities=BRATS_MODALITIES,
    )
    client_ids = tuple(client.client_id for client in scenario.clients)
    if trace.clients != client_ids:
        raise ValueError("System trace clients do not match the scenario.")
    manifests = {
        client.client_id: client.modalities for client in scenario.clients
    }
    common = {
        "budget": arguments.budget,
        "completion_probabilities": trace.completion_probabilities,
        "oracle_factor": 1.0 - 1.0 / math.e,
        "safety_margin": arguments.safety_margin,
    }
    component_calibration = calibrate_availability_aware_service_targets(
        manifests,
        priorities={modality: 1.0 for modality in BRATS_MODALITIES},
        availability_probabilities=trace.availability_probabilities,
        minimum_available_clients=trace.minimum_available_clients,
        service_type=PURE_COVERAGE,
        saturation_levels={modality: 1 for modality in BRATS_MODALITIES},
        **common,
    )
    client_calibration = calibrate_availability_aware_client_targets(
        client_ids,
        availability_probabilities=trace.availability_probabilities,
        minimum_available_clients=trace.minimum_available_clients,
        **common,
    )
    specifications = {
        "mics_v2_impact": {
            "impact_weight": 1.0,
            "importance": importance.weights,
            "client_targets": client_calibration.operational_targets,
        },
        "mics_v2_core": {
            "impact_weight": 0.0,
            "importance": {},
            "client_targets": client_calibration.operational_targets,
        },
        "mics_v2_impact_no_client_queue": {
            "impact_weight": 1.0,
            "importance": importance.weights,
            "client_targets": {},
        },
    }
    summaries = {}
    for name, specification in specifications.items():
        selector = MICSSelector(
            component_calibration.operational_targets,
            seed=arguments.selection_seed,
            impact_weight=specification["impact_weight"],
            client_targets=specification["client_targets"],
            reliability_aware=True,
        )
        counts = {client_id: 0 for client_id in client_ids}
        last_selected = {client_id: None for client_id in client_ids}
        for round_index, trace_round in enumerate(trace.rounds):
            context = SelectionContext(
                round_index=round_index,
                budget=arguments.budget,
                clients=tuple(
                    ClientState(
                        client_id=client.client_id,
                        modalities=frozenset(client.modalities),
                        num_samples=len(client.train_case_ids),
                        available=(
                            client.client_id in trace_round.available_clients
                        ),
                        last_selected_round=last_selected[client.client_id],
                    )
                    for client in scenario.clients
                ),
                component_importance=specification["importance"],
            )
            plan = selector.plan_round(context)
            selected = plan.selected_client_ids
            successful = trace.successful_selected_clients(
                round_index,
                selected,
            )
            for client_id in selected:
                counts[client_id] += 1
                last_selected[client_id] = round_index
            selector.observe_round(
                round_index,
                selected_client_ids=selected,
                successful_client_ids=successful,
            )
        summaries[name] = {
            "selection_counts": counts,
            "jain_selection_fairness": _jain(counts),
            "minimum_selection_count": min(counts.values()),
            "maximum_selection_count": max(counts.values()),
            "final_component_queues": selector.component_queues,
            "final_client_queues": selector.client_queues,
        }

    payload = {
        "schema_version": 1,
        "purpose": "model_free_mics_v2_selector_preflight",
        "source_revision": arguments.source_revision,
        "inputs": {
            "scenario_config": {
                "path": str(scenario.config_path),
                "sha256": sha256_file(scenario.config_path),
            },
            "system_trace": trace.source_record(),
            "modality_importance": importance.source_record(),
        },
        "round_count": len(trace.rounds),
        "calibration": {
            "components": component_calibration.as_dict(),
            "clients": client_calibration.as_dict(),
        },
        "client_targets": dict(client_calibration.operational_targets),
        "component_targets": dict(component_calibration.operational_targets),
        "methods": summaries,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("MICS-V2 SELECTOR PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
