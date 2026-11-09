#!/usr/bin/env python3
"""Estimate non-test modality importance from a tracked checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mics.evaluation import BRATS_REGIONS, estimate_leave_one_modality_out_importance
from mics.experiments import (
    atomic_write_json,
    collect_runtime_metadata,
    load_federated_checkpoint,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MICS_DATA_ROOT"),
    )
    parser.add_argument("--scenario-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--client", action="append")
    parser.add_argument(
        "--partition",
        choices=("calibration", "validation"),
        default="validation",
    )
    parser.add_argument("--max-cases-per-client", type=int)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--region-weights",
        type=float,
        nargs=3,
        metavar=("WT", "TC", "ET"),
        default=(1.0, 1.0, 1.0),
        help="Non-negative WT, TC, and ET weights (default: equal).",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.max_cases_per_client is not None:
        if arguments.max_cases_per_client <= 0:
            raise SystemExit("--max-cases-per-client must be positive.")
    if not 0.0 <= arguments.overlap < 1.0:
        raise SystemExit("--overlap must lie in [0, 1).")

    loaded = load_federated_checkpoint(
        arguments.run_dir,
        arguments.dataset_root,
        scenario_config=arguments.scenario_config,
        checkpoint=arguments.checkpoint,
    )
    importance = estimate_leave_one_modality_out_importance(
        loaded.model,
        loaded.scenario,
        arguments.dataset_root,
        partition=arguments.partition,
        client_ids=arguments.client,
        max_cases_per_client=arguments.max_cases_per_client,
        region_weights=dict(zip(BRATS_REGIONS, arguments.region_weights)),
        epsilon=arguments.epsilon,
        roi_size=tuple(arguments.roi_size),
        overlap=arguments.overlap,
        device=arguments.device,
        amp=not arguments.no_amp,
    )

    subset = not importance["complete_partition_probe"]
    purpose = (
        "integration_smoke_not_modality_importance"
        if subset
        else (
            "selector_calibration_modality_importance"
            if arguments.partition == "calibration"
            else "validation_derived_modality_importance"
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "purpose": purpose,
        "run_id": loaded.run_dir.name,
        "completed_rounds": loaded.completed_rounds,
        "study": loaded.resolved_config.get("study"),
        "checkpoint": {
            "path": str(loaded.checkpoint_path),
            "sha256": sha256_file(loaded.checkpoint_path),
            "config_fingerprint": loaded.config_fingerprint,
        },
        "scenario": {
            "id": loaded.scenario.scenario_id,
            "path": str(loaded.scenario_path),
            "sha256": sha256_file(loaded.scenario_path),
        },
        "training_source": loaded.training_metadata,
        "estimation_source": collect_runtime_metadata(project_root),
        "importance": importance,
    }
    output_path = arguments.output
    if output_path is None:
        suffix = "_subset" if subset else ""
        output_path = loaded.run_dir / "evaluations" / (
            f"modality_importance{suffix}_after_round_"
            f"{loaded.completed_rounds:04d}.json"
        )
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite importance output: {output_path}")
    atomic_write_json(output_path, output)

    summary = {
        "purpose": purpose,
        "output": str(output_path),
        "completed_rounds": loaded.completed_rounds,
        "normalised_selection_weights": importance[
            "normalised_selection_weights"
        ],
        "normalisation_fallback_uniform": importance[
            "normalisation_fallback_uniform"
        ],
        "probe_cost": importance["probe_cost"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if subset:
        print(
            "NOTE: Subset estimates verify integration only and must not be "
            "used as scientific modality-importance results."
        )
    print("MICS VALIDATION-ONLY MODALITY IMPORTANCE ESTIMATION PASSED")


if __name__ == "__main__":
    main()
