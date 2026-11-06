#!/usr/bin/env python3
"""Create the single symmetric BraTS modality scenario used by MICS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mics.data import (
    BRATS_BALANCED_PROFILE_SEED,
    materialize_balanced_brats_scenario,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MICS_DATA_ROOT"),
    )
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=BRATS_BALANCED_PROFILE_SEED,
    )
    arguments = parser.parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")

    scenario = materialize_balanced_brats_scenario(
        arguments.source_config,
        arguments.output_directory,
        dataset_root=arguments.dataset_root,
        profile_seed=arguments.profile_seed,
    )
    summary = {
        "scenario_id": scenario.scenario_id,
        "config_path": str(scenario.config_path),
        "data_seed": scenario.data_seed,
        "partition_counts": scenario.partition_counts,
        "global_test_count": len(scenario.global_test_case_ids),
        "covered_case_count": scenario.covered_case_count,
        "modality_pool_sizes": scenario.modality_pool_sizes,
        "clients": {
            client.client_id: list(client.modalities)
            for client in scenario.clients
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("MICS BALANCED BRATS SCENARIO MATERIALIZATION PASSED")


if __name__ == "__main__":
    main()
