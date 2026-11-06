#!/usr/bin/env python3
"""Evaluate a federated checkpoint on held-out whole BraTS client volumes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from mics import MultiEncoderUNet3D
from mics.data import load_brats_scenario
from mics.evaluation import evaluate_brats_client_partition
from mics.experiments import (
    atomic_write_json,
    collect_runtime_metadata,
    configuration_fingerprint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required run file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


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


def _build_model(config: dict[str, Any]) -> MultiEncoderUNet3D:
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Resolved run configuration has no model object.")
    modalities = model_config.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        raise ValueError("Resolved model modalities must be a non-empty list.")
    return MultiEncoderUNet3D(
        modalities=OrderedDict((str(name), 1) for name in modalities),
        num_classes=int(model_config["num_classes"]),
        base_channels=int(model_config["base_channels"]),
        num_levels=int(model_config["num_levels"]),
        fusion=str(model_config["fusion"]),
        modality_dropout=float(model_config["modality_dropout"]),
    )


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

    run_dir = arguments.run_dir.resolve()
    resolved_config = _json_object(run_dir / "resolved_config.json")
    training_metadata = _json_object(run_dir / "run_metadata.json")
    if (
        arguments.partition == "test"
        and resolved_config.get("purpose")
        != "full_training_checkpoint_for_held_out_evaluation"
    ):
        raise SystemExit("Test evaluation is forbidden for a truncated smoke run.")

    checkpoint_path = (
        arguments.checkpoint.resolve()
        if arguments.checkpoint is not None
        else run_dir / "checkpoints" / "last.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a mapping.")
    expected_fingerprint = configuration_fingerprint(resolved_config)
    if checkpoint.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("Checkpoint does not match the resolved run configuration.")
    completed_rounds = int(checkpoint.get("next_round", -1))
    if completed_rounds <= 0:
        raise ValueError("Checkpoint does not represent a completed round.")

    recorded_scenario = resolved_config.get("scenario")
    if not isinstance(recorded_scenario, dict):
        raise ValueError("Resolved run configuration has no scenario object.")
    scenario_path = (
        arguments.scenario_config.resolve()
        if arguments.scenario_config is not None
        else Path(recorded_scenario["config"])
    )
    expected_scenario_sha = recorded_scenario.get("config_sha256")
    if expected_scenario_sha and _sha256(scenario_path) != expected_scenario_sha:
        raise ValueError("Scenario configuration checksum does not match the run.")
    scenario = load_brats_scenario(
        scenario_path,
        dataset_root=arguments.dataset_root,
    )
    if scenario.scenario_id != recorded_scenario.get("id"):
        raise ValueError("Scenario ID does not match the training run.")

    model = _build_model(resolved_config)
    model.load_component_state_dict(checkpoint["global_components"])
    evaluation = evaluate_brats_client_partition(
        model,
        scenario,
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
        "completed_rounds": completed_rounds,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "config_fingerprint": expected_fingerprint,
        },
        "training_source": training_metadata,
        "evaluation_source": collect_runtime_metadata(project_root),
        "evaluation": evaluation,
    }
    output_path = arguments.output
    if output_path is None:
        suffix = "_subset" if subset else ""
        output_path = run_dir / "evaluations" / (
            f"{arguments.partition}{suffix}_after_round_{completed_rounds:04d}.json"
        )
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_path}")
    atomic_write_json(output_path, output)

    summary = {
        "purpose": purpose,
        "output": str(output_path),
        "completed_rounds": completed_rounds,
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
