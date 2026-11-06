"""Validation-only leave-one-modality-out importance estimation."""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..data import BRATS_MODALITIES, BraTSScenario
from .brats import BRATS_REGIONS
from .federated import evaluate_brats_client_partition


def _mean(values: Iterable[float]) -> float:
    collected = tuple(float(value) for value in values)
    if not collected:
        raise ValueError("Cannot average an empty collection.")
    return float(sum(collected) / len(collected))


def _normalise_region_weights(
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    supplied = (
        {region: 1.0 for region in BRATS_REGIONS}
        if weights is None
        else {str(region): float(value) for region, value in weights.items()}
    )
    if set(supplied) != set(BRATS_REGIONS):
        raise ValueError(
            f"region_weights must define exactly {tuple(BRATS_REGIONS)}."
        )
    if any(not math.isfinite(value) or value < 0.0 for value in supplied.values()):
        raise ValueError("region_weights must be finite and non-negative.")
    total = sum(supplied.values())
    if total <= 0.0:
        raise ValueError("At least one region weight must be positive.")
    return {region: supplied[region] / total for region in BRATS_REGIONS}


def normalise_nonnegative_importance(
    signed_importance: Mapping[str, float],
    *,
    epsilon: float = 1e-6,
) -> tuple[dict[str, float], bool]:
    """Clip signed effects for selection and normalise them to a simplex.

    Signed estimates remain available to callers.  This transformation is only
    the non-negative representation required by the MICS coverage objective.
    """

    if not signed_importance:
        raise ValueError("signed_importance cannot be empty.")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    values = {str(name): float(value) for name, value in signed_importance.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Importance values must be finite.")
    positive = {name: max(value, 0.0) for name, value in values.items()}
    fallback = all(value == 0.0 for value in positive.values())
    if fallback:
        uniform = 1.0 / len(positive)
        return {name: uniform for name in positive}, True
    floored = {name: max(value, epsilon) for name, value in positive.items()}
    total = sum(floored.values())
    return {name: value / total for name, value in floored.items()}, False


def _record_index(evaluation: Mapping[str, Any]) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for record in evaluation["cases"]:
        key = (str(record["client_id"]), str(record["case_id"]))
        if key in result:
            raise ValueError(f"Duplicate evaluation case record: {key}.")
        result[key] = record
    return result


def _uncertainty(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        raise ValueError("At least one value is required for uncertainty.")
    if len(values) == 1:
        return {
            "sample_standard_deviation": None,
            "standard_error": None,
            "normal_95_percent_interval": None,
        }
    deviation = float(statistics.stdev(values))
    standard_error = deviation / math.sqrt(len(values))
    centre = _mean(values)
    return {
        "sample_standard_deviation": deviation,
        "standard_error": standard_error,
        "normal_95_percent_interval": [
            centre - 1.96 * standard_error,
            centre + 1.96 * standard_error,
        ],
    }


def estimate_leave_one_modality_out_importance(
    model: nn.Module,
    scenario: BraTSScenario,
    dataset_root: str | Path,
    *,
    client_ids: Iterable[str] | None = None,
    max_cases_per_client: int | None = None,
    region_weights: Mapping[str, float] | None = None,
    epsilon: float = 1e-6,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    device: str | torch.device = "cuda",
    amp: bool = True,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    """Estimate modality effects from paired validation counterfactuals.

    For every eligible client and validation case, the method subtracts the
    Dice obtained after removing one available modality from the Dice obtained
    with the client's native modality set.  The primary effect is a macro-client
    mean, preventing clients with larger validation partitions from dominating.
    Test partitions are not accepted or opened by this API.
    """

    weights = _normalise_region_weights(region_weights)
    selected_client_ids = (
        tuple(client.client_id for client in scenario.clients)
        if client_ids is None
        else tuple(scenario.client(client_id).client_id for client_id in client_ids)
    )
    if not selected_client_ids:
        raise ValueError("At least one client is required.")
    if len(selected_client_ids) != len(set(selected_client_ids)):
        raise ValueError("client_ids cannot contain duplicates.")

    started = time.perf_counter()
    baseline = evaluate_brats_client_partition(
        model,
        scenario,
        dataset_root,
        partition="validation",
        client_ids=None if client_ids is None else selected_client_ids,
        max_cases_per_client=max_cases_per_client,
        roi_size=roi_size,
        overlap=overlap,
        device=device,
        amp=amp,
        spacing=spacing,
    )
    baseline_index = _record_index(baseline)
    modality_results: dict[str, Any] = {}
    ablation_evaluations: dict[str, Any] = {}
    signed_primary: dict[str, float] = {}

    for modality in BRATS_MODALITIES:
        possessing = [
            client_id
            for client_id in selected_client_ids
            if modality in scenario.client(client_id).modalities
        ]
        eligible = [
            client_id
            for client_id in possessing
            if len(scenario.client(client_id).modalities) > 1
        ]
        excluded_single_modality = [
            client_id for client_id in possessing if client_id not in eligible
        ]
        if not eligible:
            raise ValueError(
                f"Cannot estimate {modality!r}: no selected client can be "
                "evaluated after removing that modality."
            )
        overrides = {
            client_id: tuple(
                name
                for name in scenario.client(client_id).modalities
                if name != modality
            )
            for client_id in eligible
        }
        ablated = evaluate_brats_client_partition(
            model,
            scenario,
            dataset_root,
            partition="validation",
            client_ids=eligible,
            max_cases_per_client=max_cases_per_client,
            modality_overrides=overrides,
            roi_size=roi_size,
            overlap=overlap,
            device=device,
            amp=amp,
            spacing=spacing,
        )
        ablation_evaluations[modality] = ablated

        case_deltas: list[dict[str, Any]] = []
        client_values: dict[str, list[dict[str, float]]] = defaultdict(list)
        for ablated_record in ablated["cases"]:
            key = (
                str(ablated_record["client_id"]),
                str(ablated_record["case_id"]),
            )
            if key not in baseline_index:
                raise ValueError(f"No paired baseline record for {key}.")
            baseline_record = baseline_index[key]
            region_delta = {
                region: float(baseline_record["dice"][region])
                - float(ablated_record["dice"][region])
                for region in BRATS_REGIONS
            }
            weighted_delta = sum(
                weights[region] * region_delta[region]
                for region in BRATS_REGIONS
            )
            client_values[key[0]].append(
                {**region_delta, "weighted_mean": weighted_delta}
            )
            case_deltas.append(
                {
                    "client_id": key[0],
                    "case_id": key[1],
                    "native_modalities": baseline_record["modalities"],
                    "ablated_modalities": ablated_record["modalities"],
                    "baseline_dice": {
                        region: float(baseline_record["dice"][region])
                        for region in BRATS_REGIONS
                    },
                    "ablated_dice": {
                        region: float(ablated_record["dice"][region])
                        for region in BRATS_REGIONS
                    },
                    "signed_dice_delta": {
                        **region_delta,
                        "weighted_mean": weighted_delta,
                    },
                }
            )

        per_client = {
            client_id: {
                "case_count": len(values),
                "signed_dice_delta": {
                    region: _mean(value[region] for value in values)
                    for region in (*BRATS_REGIONS, "weighted_mean")
                },
            }
            for client_id, values in client_values.items()
        }
        client_weighted_values = [
            float(value["signed_dice_delta"]["weighted_mean"])
            for value in per_client.values()
        ]
        macro_region = {
            region: _mean(
                value["signed_dice_delta"][region]
                for value in per_client.values()
            )
            for region in (*BRATS_REGIONS, "weighted_mean")
        }
        sample_region = {
            region: _mean(
                value[region]
                for values in client_values.values()
                for value in values
            )
            for region in (*BRATS_REGIONS, "weighted_mean")
        }
        signed_primary[modality] = macro_region["weighted_mean"]
        modality_results[modality] = {
            "possessing_clients": possessing,
            "eligible_clients": eligible,
            "excluded_single_modality_clients": excluded_single_modality,
            "client_count": len(eligible),
            "case_count": len(case_deltas),
            "primary_signed_importance": macro_region["weighted_mean"],
            "positive_clipped_importance": max(
                macro_region["weighted_mean"], 0.0
            ),
            "macro_client_signed_dice_delta": macro_region,
            "sample_weighted_signed_dice_delta": sample_region,
            "uncertainty_across_client_means": _uncertainty(
                client_weighted_values
            ),
            "per_client": per_client,
            "case_deltas": case_deltas,
        }

    normalised, fallback = normalise_nonnegative_importance(
        signed_primary,
        epsilon=epsilon,
    )
    for modality, value in normalised.items():
        modality_results[modality]["normalised_selection_weight"] = value

    evaluations = [baseline, *ablation_evaluations.values()]
    ablation_case_count = sum(
        int(value["case_count"]) for value in ablation_evaluations.values()
    )
    return {
        "schema_version": 1,
        "method": "paired_leave_one_modality_out_validation_dice",
        "partition": "validation",
        "test_data_accessed": False,
        "complete_validation_probe": (
            client_ids is None and max_cases_per_client is None
        ),
        "primary_reduction": "macro_client",
        "region_weights": weights,
        "normalisation_epsilon": float(epsilon),
        "normalisation_fallback_uniform": fallback,
        "normalised_selection_weights": normalised,
        "modalities": modality_results,
        "probe_cost": {
            "native_baseline_case_evaluations": int(baseline["case_count"]),
            "additional_ablation_case_evaluations": ablation_case_count,
            "total_case_evaluations": int(baseline["case_count"])
            + ablation_case_count,
            "total_sliding_windows": sum(
                int(value["sample_weighted"]["window_count"])
                for value in evaluations
            ),
            "total_inference_seconds": sum(
                float(value["sample_weighted"]["inference_seconds"])
                for value in evaluations
            ),
            "summed_evaluation_seconds": sum(
                float(value["total_evaluation_seconds"])
                for value in evaluations
            ),
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": max(
                int(value["peak_cuda_allocated_bytes"])
                for value in evaluations
            ),
            "peak_cuda_reserved_bytes": max(
                int(value["peak_cuda_reserved_bytes"])
                for value in evaluations
            ),
        },
        "baseline_evaluation": baseline,
        "ablation_evaluations": ablation_evaluations,
    }
