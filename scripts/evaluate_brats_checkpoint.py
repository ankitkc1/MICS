#!/usr/bin/env python3
"""Evaluate a federated checkpoint on held-out whole BraTS client volumes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mics.evaluation import evaluate_brats_client_partition
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
    parser.add_argument(
        "--partition",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--client", action="append")
    parser.add_argument("--max-cases-per-client", type=int)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-test-evaluation", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.dataset_root is None:
        raise SystemExit("Set MICS_DATA_ROOT or pass --dataset-root.")
    if arguments.partition == "test" and not arguments.confirm_test_evaluation:
        raise SystemExit(
            "Local-test evaluation is locked. Pass --confirm-test-evaluation "
            "only after the complete study protocol is frozen."
        )
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
    run_dir = loaded.run_dir
    resolved_config = loaded.resolved_config
    if (
        arguments.partition == "test"
        and resolved_config.get("purpose")
        != "full_training_checkpoint_for_held_out_evaluation"
    ):
        raise SystemExit("Test evaluation is forbidden for a truncated smoke run.")

    evaluation = evaluate_brats_client_partition(
        loaded.model,
        loaded.scenario,
        arguments.dataset_root,
        partition=arguments.partition,
        client_ids=arguments.client,
        max_cases_per_client=arguments.max_cases_per_client,
        roi_size=tuple(arguments.roi_size),
        overlap=arguments.overlap,
        device=arguments.device,
        amp=not arguments.no_amp,
    )

    subset = not evaluation["complete_partition"]
    purpose = (
        "integration_smoke_not_model_quality"
        if subset
        else (
            "held_out_validation"
            if arguments.partition == "validation"
            else "final_local_test"
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "purpose": purpose,
        "run_id": run_dir.name,
        "completed_rounds": loaded.completed_rounds,
        "checkpoint": {
            "path": str(loaded.checkpoint_path),
            "sha256": sha256_file(loaded.checkpoint_path),
            "config_fingerprint": loaded.config_fingerprint,
        },
        "training_source": loaded.training_metadata,
        "evaluation_source": collect_runtime_metadata(project_root),
        "evaluation": evaluation,
    }
    output_path = arguments.output
    if output_path is None:
        suffix = "_subset" if subset else ""
        output_path = run_dir / "evaluations" / (
            f"{arguments.partition}{suffix}_after_round_"
            f"{loaded.completed_rounds:04d}.json"
        )
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_path}")
    atomic_write_json(output_path, output)

    summary = {
        "purpose": purpose,
        "output": str(output_path),
        "completed_rounds": loaded.completed_rounds,
        "client_count": evaluation["client_count"],
        "case_count": evaluation["case_count"],
        "macro_client": evaluation["macro_client"],
        "sample_weighted": evaluation["sample_weighted"],
        "client_dispersion": evaluation["client_dispersion"],
        "total_evaluation_seconds": evaluation["total_evaluation_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if subset:
        print(
            "NOTE: This is a validation integration subset and must not be "
            "reported as model quality."
        )
    print("MICS CHECKPOINT WHOLE-VOLUME EVALUATION PASSED")


if __name__ == "__main__":
    main()
