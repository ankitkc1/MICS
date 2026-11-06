"""Validated loading of tracked federated checkpoints for analysis."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..data import BraTSScenario, load_brats_scenario
from ..models import MultiEncoderUNet3D
from .tracking import configuration_fingerprint


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required run file does not exist: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {resolved}.")
    return value


def build_model_from_resolved_config(
    config: dict[str, Any],
) -> MultiEncoderUNet3D:
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


@dataclass(frozen=True)
class LoadedFederatedCheckpoint:
    run_dir: Path
    resolved_config: dict[str, Any]
    training_metadata: dict[str, Any]
    checkpoint_path: Path
    checkpoint_state: dict[str, Any]
    config_fingerprint: str
    completed_rounds: int
    scenario_path: Path
    scenario: BraTSScenario
    model: MultiEncoderUNet3D


def load_federated_checkpoint(
    run_dir: str | Path,
    dataset_root: str | Path,
    *,
    scenario_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> LoadedFederatedCheckpoint:
    """Load a tracked model only after provenance compatibility checks."""

    resolved_run_dir = Path(run_dir).resolve()
    resolved_config = read_json_object(resolved_run_dir / "resolved_config.json")
    training_metadata = read_json_object(resolved_run_dir / "run_metadata.json")
    checkpoint_path = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else resolved_run_dir / "checkpoints" / "last.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint_state, dict):
        raise ValueError("Checkpoint must contain a mapping.")
    expected_fingerprint = configuration_fingerprint(resolved_config)
    if checkpoint_state.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("Checkpoint does not match the resolved run configuration.")
    completed_rounds = int(checkpoint_state.get("next_round", -1))
    if completed_rounds <= 0:
        raise ValueError("Checkpoint does not represent a completed round.")

    recorded_scenario = resolved_config.get("scenario")
    if not isinstance(recorded_scenario, dict):
        raise ValueError("Resolved run configuration has no scenario object.")
    scenario_path = (
        Path(scenario_config).resolve()
        if scenario_config is not None
        else Path(recorded_scenario["config"])
    )
    expected_scenario_sha = recorded_scenario.get("config_sha256")
    if expected_scenario_sha and sha256_file(scenario_path) != expected_scenario_sha:
        raise ValueError("Scenario configuration checksum does not match the run.")
    scenario = load_brats_scenario(scenario_path, dataset_root=dataset_root)
    if scenario.scenario_id != recorded_scenario.get("id"):
        raise ValueError("Scenario ID does not match the training run.")

    model = build_model_from_resolved_config(resolved_config)
    model.load_component_state_dict(checkpoint_state["global_components"])
    return LoadedFederatedCheckpoint(
        run_dir=resolved_run_dir,
        resolved_config=resolved_config,
        training_metadata=training_metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_state=checkpoint_state,
        config_fingerprint=expected_fingerprint,
        completed_rounds=completed_rounds,
        scenario_path=scenario_path,
        scenario=scenario,
        model=model,
    )
