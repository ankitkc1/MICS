#!/usr/bin/env python3
"""Validate a frozen MICS ablation protocol and resolve one seed task."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from mics.experiments import sha256_file

EXPECTED_METHODS = (
    "mics_impact",
    "mics_core",
    "mics_impact_no_queue",
    "mics_impact_modular",
    "mics_impact_uniform",
)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported ablation-study protocol schema.")
    if payload.get("protocol_status") != "frozen":
        raise ValueError("The ablation protocol must be frozen before execution.")
    study_id = str(payload.get("study_id", ""))
    if not study_id or not study_id.replace("_", "").isalnum():
        raise ValueError("study_id must be a safe non-empty identifier.")
    if tuple(payload.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("The registered MICS ablation methods changed.")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("At least three preregistered seeds are required.")
    seed_ids = []
    seed_tuples = []
    for index, raw_seed in enumerate(seeds):
        seed = _object(raw_seed, f"seeds[{index}]")
        seed_id = str(seed.get("seed_id", ""))
        values = tuple(
            int(seed[name]) for name in ("model", "selection", "data", "trace")
        )
        if not seed_id or any(value < 0 for value in values):
            raise ValueError("Seed identifiers and values must be valid.")
        seed_ids.append(seed_id)
        seed_tuples.append(values)
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("Seed identifiers must be unique.")
    if len(seed_tuples) != len(set(seed_tuples)):
        raise ValueError("Scientific seed tuples must be unique.")

    trace = _object(payload.get("system_trace"), "system_trace")
    training = _object(payload.get("training"), "training")
    evaluation = _object(payload.get("evaluation"), "evaluation")
    _positive_int(trace.get("rounds"), "system_trace.rounds")
    _positive_int(training.get("clients_per_round"), "clients_per_round")
    _positive_int(training.get("checkpoint_every"), "checkpoint_every")
    _probability(
        trace.get("availability_probability"),
        "availability_probability",
    )
    _probability(
        trace.get("completion_probability"),
        "completion_probability",
    )
    if trace["minimum_available_clients"] < training["clients_per_round"]:
        raise ValueError("System trace cannot satisfy the client budget.")
    if training.get("reliability_aware") is not True:
        raise ValueError("The frozen MICS study requires reliability awareness.")
    if trace["rounds"] % training["checkpoint_every"] != 0:
        raise ValueError("Round count must be divisible by checkpoint interval.")
    if evaluation.get("partition") != "validation":
        raise ValueError("The scientific ablation must evaluate validation only.")
    if evaluation.get("evaluate_every_archived_checkpoint") is not True:
        raise ValueError("Every archived checkpoint must be evaluated.")
    return payload


def _shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return shlex.quote(str(value))


def _resolved_values(
    manifest: Path,
    payload: dict[str, Any],
    task_index: int,
) -> dict[str, Any]:
    seeds = payload["seeds"]
    if task_index < 0 or task_index >= len(seeds):
        raise ValueError(
            f"task-index must be in [0, {len(seeds) - 1}], found {task_index}."
        )
    seed = seeds[task_index]
    trace = payload["system_trace"]
    training = payload["training"]
    evaluation = payload["evaluation"]
    return {
        "STUDY_ID": payload["study_id"],
        "PROTOCOL_PATH": str(manifest.resolve()),
        "PROTOCOL_SHA256": sha256_file(manifest),
        "SEED_ID": seed["seed_id"],
        "MODEL_SEED": seed["model"],
        "SELECTION_SEED": seed["selection"],
        "DATA_SEED": seed["data"],
        "TRACE_SEED": seed["trace"],
        "ROUNDS": trace["rounds"],
        "MINIMUM_AVAILABLE_CLIENTS": trace["minimum_available_clients"],
        "AVAILABILITY_PROBABILITY": trace["availability_probability"],
        "COMPLETION_PROBABILITY": trace["completion_probability"],
        "CLIENTS_PER_ROUND": training["clients_per_round"],
        "CHECKPOINT_EVERY": training["checkpoint_every"],
        "BASE_CHANNELS": training["base_channels"],
        "NUM_LEVELS": training["num_levels"],
        "FUSION": training["fusion"],
        "MODALITY_DROPOUT": training["modality_dropout"],
        "PATCH_SIZE": training["patch_size"],
        "FOREGROUND_PROBABILITY": training["foreground_probability"],
        "BATCH_SIZE": training["batch_size"],
        "NUM_WORKERS": training["num_workers"],
        "LOCAL_EPOCHS": training["local_epochs"],
        "LEARNING_RATE": training["learning_rate"],
        "WEIGHT_DECAY": training["weight_decay"],
        "GRADIENT_CLIP_NORM": training["gradient_clip_norm"],
        "AMP_INITIAL_SCALE": training["amp_initial_scale"],
        "AMP_MAX_RETRIES": training["amp_max_retries"],
        "SERVICE_TYPE": training["service_type"],
        "SATURATION_LEVEL": training["saturation_level"],
        "SAFETY_MARGIN": training["safety_margin"],
        "IMPACT_WEIGHT": training["impact_weight"],
        "RELIABILITY_AWARE": training["reliability_aware"],
        "EVALUATION_PARTITION": evaluation["partition"],
        "ROI_SIZE": evaluation["roi_size"],
        "OVERLAP": evaluation["overlap"],
    }


def _calibration_values(
    manifest: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    calibration = _object(
        payload.get("importance_calibration"),
        "importance_calibration",
    )
    training = payload["training"]
    scenario = _object(payload.get("scenario"), "scenario")
    if calibration.get("pilot_selector") != "uniform_random":
        raise ValueError("The importance pilot must use uniform random selection.")
    if calibration.get("partition") != "calibration":
        raise ValueError("Importance must use the disjoint calibration partition.")
    _probability(
        calibration.get("availability_probability"),
        "calibration availability_probability",
    )
    _probability(
        calibration.get("completion_probability"),
        "calibration completion_probability",
    )
    return {
        "STUDY_ID": payload["study_id"],
        "PROTOCOL_PATH": str(manifest.resolve()),
        "PROTOCOL_SHA256": sha256_file(manifest),
        "CALIBRATION_SPLIT_SEED": scenario["calibration_split_seed"],
        "ROUNDS": _positive_int(calibration["pilot_rounds"], "pilot_rounds"),
        "MODEL_SEED": int(calibration["model_seed"]),
        "SELECTION_SEED": int(calibration["selection_seed"]),
        "DATA_SEED": int(calibration["data_seed"]),
        "TRACE_SEED": int(calibration["trace_seed"]),
        "CALIBRATION_AVAILABILITY_PROBABILITY": calibration[
            "availability_probability"
        ],
        "CALIBRATION_COMPLETION_PROBABILITY": calibration[
            "completion_probability"
        ],
        "CLIENTS_PER_ROUND": training["clients_per_round"],
        "CHECKPOINT_EVERY": training["checkpoint_every"],
        "BASE_CHANNELS": training["base_channels"],
        "NUM_LEVELS": training["num_levels"],
        "FUSION": training["fusion"],
        "MODALITY_DROPOUT": training["modality_dropout"],
        "PATCH_SIZE": training["patch_size"],
        "FOREGROUND_PROBABILITY": training["foreground_probability"],
        "BATCH_SIZE": training["batch_size"],
        "NUM_WORKERS": training["num_workers"],
        "LOCAL_EPOCHS": training["local_epochs"],
        "LEARNING_RATE": training["learning_rate"],
        "WEIGHT_DECAY": training["weight_decay"],
        "GRADIENT_CLIP_NORM": training["gradient_clip_norm"],
        "AMP_INITIAL_SCALE": training["amp_initial_scale"],
        "AMP_MAX_RETRIES": training["amp_max_retries"],
        "ROI_SIZE": payload["evaluation"]["roi_size"],
        "OVERLAP": payload["evaluation"]["overlap"],
        "REGION_WEIGHTS": calibration["region_weights"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task-index", type=int)
    task.add_argument("--calibration", action="store_true")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    arguments = parser.parse_args()
    payload = load_protocol(arguments.manifest)
    values = (
        _calibration_values(arguments.manifest, payload)
        if arguments.calibration
        else _resolved_values(arguments.manifest, payload, arguments.task_index)
    )
    if arguments.format == "json":
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for name, value in values.items():
            print(f"export {name}={_shell_value(value)}")


if __name__ == "__main__":
    main()
