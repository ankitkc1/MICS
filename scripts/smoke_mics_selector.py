#!/usr/bin/env python3
"""Run MICS selection only on a materialized scenario and importance record."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from mics import ClientState, MICSSelector, SelectionContext
from mics.data import BRATS_MODALITIES, load_brats_scenario
from mics.experiments import (
    atomic_write_json,
    jain_fairness_index,
    load_modality_importance_record,
    selection_round_diagnostics,
    validate_selection_plan,
)
from mics.selection import PURE_COVERAGE, calibrate_static_service_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MICS_DATA_ROOT"),
    )
    parser.add_argument("--scenario-config", required=True, type=Path)
    parser.add_argument("--importance-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=2)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--impact-weight", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.1)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.rounds <= 0 or arguments.clients_per_round <= 0:
        raise SystemExit("Round count and client budget must be positive.")
    if arguments.seed < 0:
        raise SystemExit("Selection seed must be non-negative.")
    output_path = arguments.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite selector smoke: {output_path}")

    scenario = load_brats_scenario(
        arguments.scenario_config,
        dataset_root=arguments.dataset_root,
    )
    if arguments.clients_per_round > len(scenario.clients):
        raise SystemExit("Client budget exceeds the scenario client count.")
    loaded_importance = load_modality_importance_record(
        arguments.importance_output,
        modalities=BRATS_MODALITIES,
    )
    importance_payload = loaded_importance.payload
    importance_weights = loaded_importance.weights
    manifests = {
        client.client_id: client.modalities for client in scenario.clients
    }
    calibration = calibrate_static_service_targets(
        manifests,
        budget=arguments.clients_per_round,
        priorities={modality: 1.0 for modality in BRATS_MODALITIES},
        service_type=PURE_COVERAGE,
        oracle_factor=1.0 - 1.0 / math.e,
        safety_margin=arguments.safety_margin,
    )
    selector = MICSSelector(
        calibration.operational_targets,
        seed=arguments.seed,
        service_type=PURE_COVERAGE,
        impact_weight=arguments.impact_weight,
        reliability_aware=False,
    )
    selection_counts = {client.client_id: 0 for client in scenario.clients}
    last_selected_round: dict[str, int | None] = {
        client.client_id: None for client in scenario.clients
    }
    modality_last_updated_round: dict[str, int | None] = {
        modality: None for modality in BRATS_MODALITIES
    }
    service_totals = {modality: 0.0 for modality in BRATS_MODALITIES}
    history = []
    total_selection_seconds = 0.0

    for round_index in range(arguments.rounds):
        context = SelectionContext(
            round_index=round_index,
            budget=arguments.clients_per_round,
            clients=tuple(
                ClientState(
                    client_id=client.client_id,
                    modalities=frozenset(client.modalities),
                    num_samples=len(client.train_case_ids),
                    last_selected_round=last_selected_round[client.client_id],
                )
                for client in scenario.clients
            ),
            component_importance=importance_weights,
            metadata={"scenario_id": scenario.scenario_id},
        )
        started = time.perf_counter()
        plan = selector.plan_round(context)
        selection_seconds = time.perf_counter() - started
        total_selection_seconds += selection_seconds
        validate_selection_plan(context, plan)
        diagnostics = selection_round_diagnostics(
            context,
            plan,
            selection_counts=selection_counts,
            modality_last_updated_round=modality_last_updated_round,
        )
        outcome = selector.observe_round(
            round_index,
            selected_client_ids=plan.selected_client_ids,
            successful_client_ids=plan.selected_client_ids,
        )
        for client_id in plan.selected_client_ids:
            selection_counts[client_id] += 1
            last_selected_round[client_id] = round_index
        for modality, service in outcome[
            "realised_component_service"
        ].items():
            service_totals[modality] += float(service)
            if service > 0.0:
                modality_last_updated_round[modality] = round_index
        history.append(
            {
                "round": round_index,
                "selected_clients": list(plan.selected_client_ids),
                "selection_seconds": selection_seconds,
                "plan": dict(plan.metadata),
                "diagnostics_before_outcome": diagnostics,
                "realised_outcome": dict(outcome),
            }
        )

    average_service = {
        modality: service_totals[modality] / arguments.rounds
        for modality in BRATS_MODALITIES
    }
    result = {
        "schema_version": 1,
        "purpose": "selector_integration_smoke_not_model_quality",
        "method": "mics_impact_cardinality_greedy",
        "test_partition_used_for_selection": False,
        "scenario": {
            "id": scenario.scenario_id,
            "config": str(scenario.config_path),
            "modality_pool_sizes": scenario.modality_pool_sizes,
        },
        "importance_source": {
            "path": str(arguments.importance_output.resolve()),
            "purpose": importance_payload.get("purpose"),
            "weights": importance_weights,
        },
        "calibration": calibration.as_dict(),
        "configuration": {
            "rounds": arguments.rounds,
            "clients_per_round": arguments.clients_per_round,
            "seed": arguments.seed,
            "impact_weight": arguments.impact_weight,
            "service_type": PURE_COVERAGE,
            "reliability_aware": False,
            "completion_outcomes": "all_successful_integration_fixture",
        },
        "summary": {
            "selection_counts": selection_counts,
            "jain_selection_fairness": jain_fairness_index(selection_counts),
            "average_realised_component_service": average_service,
            "service_targets": dict(calibration.operational_targets),
            "final_component_queues": dict(selector.component_queues),
            "maximum_final_component_queue": max(
                selector.component_queues.values()
            ),
            "total_selection_seconds": total_selection_seconds,
        },
        "rounds": history,
    }
    atomic_write_json(output_path, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Selector smoke output: {output_path}")
    print("MICS BALANCED-SCENARIO SELECTOR SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
