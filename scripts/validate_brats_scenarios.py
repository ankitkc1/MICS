#!/usr/bin/env python3
"""Validate publication BraTS split bundles before an experiment starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mics.data import load_brats_scenario, validate_matched_brats_scenarios


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--scenario-config",
        required=True,
        action="append",
        type=Path,
        help=(
            "Path to materialized_config.json; repeat only for an optional "
            "controlled comparison."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    scenarios = [
        load_brats_scenario(path, dataset_root=arguments.dataset_root)
        for path in arguments.scenario_config
    ]

    summaries = []
    for scenario in scenarios:
        summaries.append(
            {
                "scenario_id": scenario.scenario_id,
                "data_seed": scenario.data_seed,
                "config_path": str(scenario.config_path),
                "client_count": len(scenario.clients),
                "dataset_case_count": len(scenario.dataset_case_ids),
                "partition_counts": scenario.partition_counts,
                "global_test_count": len(scenario.global_test_case_ids),
                "covered_case_count": scenario.covered_case_count,
                "modality_pool_sizes": scenario.modality_pool_sizes,
                "clients": {
                    client.client_id: {
                        "modalities": list(client.modalities),
                        "train": len(client.train_case_ids),
                        "validation": len(client.validation_case_ids),
                        "test": len(client.test_case_ids),
                    }
                    for client in scenario.clients
                },
            }
        )

    comparisons = []
    reference = scenarios[0]
    for comparison in scenarios[1:]:
        changed_clients = validate_matched_brats_scenarios(reference, comparison)
        comparisons.append(
            {
                "reference": reference.scenario_id,
                "comparison": comparison.scenario_id,
                "case_assignments_identical": True,
                "changed_modality_clients": list(changed_clients),
            }
        )

    print(json.dumps({"scenarios": summaries, "comparisons": comparisons}, indent=2))
    print("MICS BRATS SCENARIO VALIDATION PASSED")


if __name__ == "__main__":
    main()
