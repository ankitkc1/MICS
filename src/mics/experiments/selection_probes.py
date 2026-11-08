"""Deterministic train-partition probes requested by client selectors."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ..models import MultiEncoderUNet3D
from ..selection import SelectionProbeRequest
from ..training import DiceCrossEntropyLoss


@dataclass(frozen=True)
class SelectionProbeOutcome:
    statistics: Mapping[str, Mapping[str, Any]]
    diagnostics: Mapping[str, Any]
    download_bytes_by_client: Mapping[str, int]
    upload_bytes_by_client: Mapping[str, int]


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    raw_inputs = batch.get("inputs")
    target = batch.get("target")
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise ValueError("Probe batch requires a nonempty input mapping.")
    if not isinstance(target, torch.Tensor):
        raise TypeError("Probe batch target must be a tensor.")
    inputs = {
        str(name): value.to(device, non_blocking=True)
        for name, value in raw_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    if not inputs:
        raise ValueError("Probe batch contains no modality tensor.")
    availability = batch.get("availability")
    if availability is not None:
        if not isinstance(availability, torch.Tensor):
            raise TypeError("Probe availability must be a tensor when provided.")
        availability = availability.to(
            device,
            non_blocking=True,
            dtype=torch.bool,
        )
    return (
        inputs,
        target.to(device, non_blocking=True, dtype=torch.long),
        availability,
    )


def _relevant_components(
    modalities: Iterable[str],
) -> tuple[str, ...]:
    return (
        *(f"encoder:{modality}" for modality in modalities),
        "fusion",
        "decoder",
        "head",
    )


def run_selection_probes(
    model: MultiEncoderUNet3D,
    request: SelectionProbeRequest,
    *,
    loaders: Mapping[str, Iterable[Mapping[str, Any]]],
    client_modalities: Mapping[str, tuple[str, ...]],
    component_bytes: Mapping[str, int],
    device: str | torch.device,
) -> SelectionProbeOutcome:
    """Evaluate requested loss or gradient statistics without model updates."""

    if set(loaders) != set(request.client_ids):
        raise ValueError("Probe loaders do not match requested clients.")
    if not set(request.client_ids).issubset(client_modalities):
        raise ValueError("Probe request contains an unknown client.")
    target_device = torch.device(device)
    original_training = model.training
    model.to(target_device)
    model.eval()
    criterion = DiceCrossEntropyLoss().to(target_device)
    parameter_items = tuple(model.named_parameters())
    statistics: dict[str, dict[str, Any]] = {}
    download_bytes = {}
    upload_bytes = {}
    client_diagnostics = {}
    started = time.perf_counter()

    try:
        for client_id in request.client_ids:
            modalities = tuple(client_modalities[client_id])
            model.configure_client_modalities(modalities)
            model.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            examples = 0
            batches = 0
            client_started = time.perf_counter()
            for batch in loaders[client_id]:
                inputs, target, availability = _move_batch(batch, target_device)
                batch_size = int(target.shape[0])
                if batch_size <= 0:
                    raise ValueError("Probe batch cannot be empty.")
                if request.kind == "local_loss":
                    with torch.no_grad():
                        loss = criterion(model(inputs, availability), target)
                else:
                    loss = criterion(model(inputs, availability), target)
                    (loss * batch_size).backward()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        f"Selection probe for {client_id} produced non-finite loss."
                    )
                weighted_loss += float(loss.detach()) * batch_size
                examples += batch_size
                batches += 1
                if (
                    request.max_batches is not None
                    and batches >= request.max_batches
                ):
                    break
            if batches == 0 or examples == 0:
                raise RuntimeError(f"Selection probe for {client_id} saw no data.")

            mean_loss = weighted_loss / examples
            client_statistics: dict[str, Any] = {"local_loss": mean_loss}
            logical_gradient_bytes = 0
            gradient_dimension = None
            gradient_norm = None
            if request.kind == "gradient_vector":
                vector_parts = []
                for _, parameter in parameter_items:
                    if parameter.grad is None:
                        part = torch.zeros(
                            parameter.numel(),
                            dtype=torch.float32,
                        )
                    else:
                        part = (
                            parameter.grad.detach()
                            .reshape(-1)
                            .to(device="cpu", dtype=torch.float32)
                            / examples
                        )
                        logical_gradient_bytes += (
                            parameter.numel() * parameter.element_size()
                        )
                    vector_parts.append(part)
                gradient_vector = torch.cat(vector_parts)
                gradient_dimension = int(gradient_vector.numel())
                gradient_norm = float(torch.linalg.vector_norm(gradient_vector))
                client_statistics.update(
                    {
                        "gradient_vector": gradient_vector,
                        "gradient_norm": gradient_norm,
                    }
                )
                upload_bytes[client_id] = logical_gradient_bytes
            else:
                upload_bytes[client_id] = 8
            statistics[client_id] = client_statistics
            components = _relevant_components(modalities)
            download_bytes[client_id] = sum(
                int(component_bytes[component]) for component in components
            )
            client_diagnostics[client_id] = {
                "modalities": list(modalities),
                "examples": examples,
                "batches": batches,
                "mean_loss": mean_loss,
                "gradient_dimension": gradient_dimension,
                "gradient_norm": gradient_norm,
                "elapsed_seconds": time.perf_counter() - client_started,
                "download_bytes": download_bytes[client_id],
                "upload_bytes": upload_bytes[client_id],
            }
            model.zero_grad(set_to_none=True)
    finally:
        model.configure_client_modalities(model.modality_names)
        model.train(original_training)
        model.zero_grad(set_to_none=True)
        model.to("cpu")
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    diagnostics = {
        "kind": request.kind,
        "client_ids": list(request.client_ids),
        "max_batches": request.max_batches,
        "clients": client_diagnostics,
        "total_examples": sum(
            int(record["examples"]) for record in client_diagnostics.values()
        ),
        "total_batches": sum(
            int(record["batches"]) for record in client_diagnostics.values()
        ),
        "total_seconds": time.perf_counter() - started,
        "total_download_bytes": sum(download_bytes.values()),
        "total_upload_bytes": sum(upload_bytes.values()),
        "partition": "train",
        "model_updated": False,
    }
    return SelectionProbeOutcome(
        statistics=statistics,
        diagnostics=diagnostics,
        download_bytes_by_client=download_bytes,
        upload_bytes_by_client=upload_bytes,
    )
