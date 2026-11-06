"""Whole-volume federated evaluation with client and global reductions."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data import BRATS_MODALITIES, BraTSNpyDataset, BraTSScenario
from .brats import (
    BRATS_HD95_PENALTY_MM,
    BRATS_REGIONS,
    brats_region_dice,
    brats_region_hd95,
)
from .sliding_window import sliding_window_inference


def _mean(values: Iterable[float]) -> float:
    collected = tuple(float(value) for value in values)
    if not collected:
        raise ValueError("Cannot average an empty metric collection.")
    return float(sum(collected) / len(collected))


def _aggregate_cases(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one evaluated case is required.")
    dice = {
        region: _mean(record["dice"][region] for record in records)
        for region in BRATS_REGIONS
    }
    hd95 = {
        region: _mean(record["hd95_mm"][region] for record in records)
        for region in BRATS_REGIONS
    }
    return {
        "case_count": len(records),
        "dice": {**dice, "mean": _mean(dice.values())},
        "hd95_mm": {**hd95, "mean": _mean(hd95.values())},
        "inference_seconds": sum(
            float(record["inference_seconds"]) for record in records
        ),
        "window_count": sum(int(record["window_count"]) for record in records),
    }


def _normalise_client_ids(
    scenario: BraTSScenario,
    client_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    if client_ids is None:
        return tuple(client.client_id for client in scenario.clients)
    result = tuple(scenario.client(client_id).client_id for client_id in client_ids)
    if not result:
        raise ValueError("client_ids cannot be empty when supplied.")
    if len(result) != len(set(result)):
        raise ValueError("client_ids cannot contain duplicates.")
    return result


def _evaluation_modalities(
    scenario: BraTSScenario,
    client_id: str,
    overrides: Mapping[str, Iterable[str]] | None,
) -> tuple[str, ...]:
    client = scenario.client(client_id)
    if overrides is None or client_id not in overrides:
        return client.modalities
    requested = set(str(value).lower() for value in overrides[client_id])
    if not requested:
        raise ValueError(f"Evaluation modalities for {client_id} cannot be empty.")
    unknown = requested.difference(client.modalities)
    if unknown:
        raise ValueError(
            f"Evaluation override for {client_id} contains unavailable modalities: "
            f"{sorted(unknown)}"
        )
    return tuple(modality for modality in BRATS_MODALITIES if modality in requested)


def evaluate_brats_client_partition(
    model: nn.Module,
    scenario: BraTSScenario,
    dataset_root: str | Path,
    *,
    partition: str = "validation",
    client_ids: Iterable[str] | None = None,
    max_cases_per_client: int | None = None,
    modality_overrides: Mapping[str, Iterable[str]] | None = None,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    device: str | torch.device = "cuda",
    amp: bool = True,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    """Evaluate held-out whole cases under each client's modality mask."""

    partition_name = str(partition).strip().lower()
    if partition_name not in {"validation", "test"}:
        raise ValueError(
            "Whole-volume evaluation partition must be validation or test."
        )
    if max_cases_per_client is not None and max_cases_per_client <= 0:
        raise ValueError("max_cases_per_client must be positive when supplied.")
    selected_client_ids = _normalise_client_ids(scenario, client_ids)
    if modality_overrides is not None:
        unknown_override_clients = set(modality_overrides).difference(
            selected_client_ids
        )
        if unknown_override_clients:
            raise ValueError(
                "Modality overrides contain clients outside this evaluation: "
                f"{sorted(unknown_override_clients)}"
            )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable.")
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    all_records: list[dict[str, Any]] = []
    per_client: dict[str, Any] = {}
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        for client_id in selected_client_ids:
            client = scenario.client(client_id)
            modalities = _evaluation_modalities(
                scenario,
                client_id,
                modality_overrides,
            )
            dataset = BraTSNpyDataset(
                dataset_root,
                split_file=client.split_file(partition_name),
                available_modalities=modalities,
                patch_size=None,
                training=False,
                validate_files=True,
            )
            case_limit = (
                len(dataset)
                if max_cases_per_client is None
                else min(len(dataset), max_cases_per_client)
            )
            client_records = []
            for case_index in range(case_limit):
                sample = dataset[case_index]
                inputs = {
                    name: tensor.unsqueeze(0)
                    for name, tensor in sample["inputs"].items()
                }
                target = sample["target"].unsqueeze(0)
                availability = sample["availability"].unsqueeze(0)
                case_started = time.perf_counter()
                inference = sliding_window_inference(
                    model,
                    inputs,
                    availability,
                    roi_size=roi_size,
                    overlap=overlap,
                    device=target_device,
                    amp=amp,
                )
                prediction = inference.logits.argmax(dim=1)
                dice_tensors = brats_region_dice(prediction, target)
                hd95_tensors = brats_region_hd95(
                    prediction,
                    target,
                    spacing=spacing,
                    one_empty_penalty=BRATS_HD95_PENALTY_MM,
                )
                dice = {
                    region: float(value)
                    for region, value in dice_tensors.items()
                }
                hd95 = {
                    region: float(value)
                    for region, value in hd95_tensors.items()
                }
                record = {
                    "client_id": client_id,
                    "case_id": sample["case_id"],
                    "modalities": list(modalities),
                    "dice": {**dice, "mean": _mean(dice.values())},
                    "hd95_mm": {**hd95, "mean": _mean(hd95.values())},
                    "window_count": inference.window_count,
                    "inference_seconds": inference.elapsed_seconds,
                    "evaluation_seconds": time.perf_counter() - case_started,
                }
                client_records.append(record)
                all_records.append(record)
                del inference, prediction, inputs, target, availability

            client_summary = _aggregate_cases(client_records)
            client_summary["modalities"] = list(modalities)
            client_summary["available_case_count"] = len(dataset)
            client_summary["complete_client_partition"] = (
                case_limit == len(dataset)
            )
            per_client[client_id] = client_summary
    finally:
        model.to("cpu")
        model.train(was_training)

    sample_weighted = _aggregate_cases(all_records)
    macro_dice = {
        region: _mean(
            summary["dice"][region] for summary in per_client.values()
        )
        for region in (*BRATS_REGIONS, "mean")
    }
    macro_hd95 = {
        region: _mean(
            summary["hd95_mm"][region] for summary in per_client.values()
        )
        for region in (*BRATS_REGIONS, "mean")
    }
    client_mean_dice = [
        float(summary["dice"]["mean"])
        for summary in per_client.values()
    ]
    complete_partition = (
        client_ids is None
        and max_cases_per_client is None
        and modality_overrides is None
    )
    return {
        "partition": partition_name,
        "complete_partition": complete_partition,
        "client_count": len(per_client),
        "case_count": len(all_records),
        "per_client": per_client,
        "macro_client": {
            "dice": macro_dice,
            "hd95_mm": macro_hd95,
        },
        "sample_weighted": sample_weighted,
        "client_dispersion": {
            "mean_dice_population_std": (
                statistics.pstdev(client_mean_dice)
                if len(client_mean_dice) > 1
                else 0.0
            ),
            "mean_dice_min": min(client_mean_dice),
            "mean_dice_25th_percentile": float(
                np.percentile(client_mean_dice, 25)
            ),
            "mean_dice_max": max(client_mean_dice),
        },
        "cases": all_records,
        "total_evaluation_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(target_device))
            if target_device.type == "cuda"
            else 0
        ),
        "roi_size": [int(value) for value in roi_size],
        "overlap": float(overlap),
        "spacing_mm": [float(value) for value in spacing],
        "empty_region_policy": {
            "both_empty_dice": 1.0,
            "both_empty_hd95_mm": 0.0,
            "one_empty_dice": 0.0,
            "one_empty_hd95_mm": BRATS_HD95_PENALTY_MM,
        },
    }
