#!/usr/bin/env python3
"""Create a BraTS scenario with disjoint calibration and validation cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mics.data import (
    BRATS_CALIBRATION_SPLIT_SEED,
    materialize_calibration_brats_scenario,
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
        "--split-seed",
        type=int,
        default=BRATS_CALIBRATION_SPLIT_SEED,
    )
    arguments = parser.parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")

    scenario = materialize_calibration_brats_scenario(
        arguments.source_config,
        arguments.output_directory,
        dataset_root=arguments.dataset_root,
        split_seed=arguments.split_seed,
    )
    summary = {
        "scenario_id": scenario.scenario_id,
        "schema_version": scenario.schema_version,
        "config_path": str(scenario.config_path),
        "data_seed": scenario.data_seed,
        "partition_counts": scenario.partition_counts,
        "global_test_count": len(scenario.global_test_case_ids),
        "covered_case_count": scenario.covered_case_count,
        "modality_pool_sizes": scenario.modality_pool_sizes,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("MICS CALIBRATION-SCENARIO MATERIALIZATION PASSED")


if __name__ == "__main__":
    main()
