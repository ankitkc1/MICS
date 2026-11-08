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
    complete_validation_probe: bool

    def source_record(self) -> dict[str, Any]:
        importance = self.payload.get("importance", {})
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "purpose": self.purpose,
            "weights": dict(self.weights),
            "complete_validation_probe": self.complete_validation_probe,
            "checkpoint": self.payload.get("checkpoint"),
            "probe_cost": importance.get("probe_cost"),
            "test_data_accessed": importance.get("test_data_accessed"),
        }


def load_modality_importance_record(
    path: str | Path,
    *,
    modalities: Sequence[str],
) -> LoadedModalityImportance:
    """Load normalized selector weights only from a validation-only record."""

    resolved = Path(path).resolve()
    payload = read_json_object(resolved)
    importance = payload.get("importance")
    if not isinstance(importance, dict):
        raise ValueError("Importance record has no importance object.")
    if importance.get("test_data_accessed") is not False:
        raise ValueError("Importance record does not certify validation-only use.")
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
    return LoadedModalityImportance(
        path=resolved,
        sha256=sha256_file(resolved),
        purpose=purpose,
        payload=payload,
        weights={modality: weights[modality] for modality in expected},
        complete_validation_probe=bool(
            importance.get("complete_validation_probe", False)
        ),
    )
