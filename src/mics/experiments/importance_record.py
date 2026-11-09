"""Validated loading of modality-importance records for selectors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import read_json_object, sha256_file


@dataclass(frozen=True)
class LoadedModalityImportance:
    path: Path
    sha256: str
    purpose: str
    payload: dict[str, Any]
    weights: dict[str, float]
    partition: str
    complete_probe: bool

    @property
    def complete_validation_probe(self) -> bool:
        return self.partition == "validation" and self.complete_probe

    @property
    def complete_calibration_probe(self) -> bool:
        return self.partition == "calibration" and self.complete_probe

    def source_record(self) -> dict[str, Any]:
        importance = self.payload.get("importance", {})
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "purpose": self.purpose,
            "weights": dict(self.weights),
            "partition": self.partition,
            "complete_probe": self.complete_probe,
            "complete_validation_probe": self.complete_validation_probe,
            "complete_calibration_probe": self.complete_calibration_probe,
            "checkpoint": self.payload.get("checkpoint"),
            "study": self.payload.get("study"),
            "scenario": self.payload.get("scenario"),
            "probe_cost": importance.get("probe_cost"),
            "test_data_accessed": importance.get("test_data_accessed"),
        }


def load_modality_importance_record(
    path: str | Path,
    *,
    modalities: Sequence[str],
) -> LoadedModalityImportance:
    """Load normalized selector weights from a non-test calibration record."""

    resolved = Path(path).resolve()
    payload = read_json_object(resolved)
    importance = payload.get("importance")
    if not isinstance(importance, dict):
        raise ValueError("Importance record has no importance object.")
    if importance.get("test_data_accessed") is not False:
        raise ValueError("Importance record does not certify non-test use.")
    raw_weights = importance.get("normalised_selection_weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("Importance record has no normalized selection weights.")
    weights = {str(key): float(value) for key, value in raw_weights.items()}
    expected = tuple(str(modality) for modality in modalities)
    if set(weights) != set(expected):
        raise ValueError("Importance weights do not match configured modalities.")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in weights.values()
    ):
        raise ValueError("Importance weights must be finite and non-negative.")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Importance weights must sum to one.")
    purpose = str(payload.get("purpose", ""))
    partition = str(importance.get("partition", "validation")).strip().lower()
    if partition not in ("calibration", "validation"):
        raise ValueError("Importance record has an invalid source partition.")
    complete_probe = bool(
        importance.get(
            "complete_partition_probe",
            importance.get("complete_validation_probe", False),
        )
    )
    return LoadedModalityImportance(
        path=resolved,
        sha256=sha256_file(resolved),
        purpose=purpose,
        payload=payload,
        weights={modality: weights[modality] for modality in expected},
        partition=partition,
        complete_probe=complete_probe,
    )
