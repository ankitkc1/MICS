"""Reusable local-client optimization for synchronous federated experiments."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..evaluation import BRATS_REGIONS, brats_dice_from_logits
from ..federated.aggregation import ComponentUpdate
from ..models import MultiEncoderUNet3D
from .losses import DiceCrossEntropyLoss


@dataclass(frozen=True)
class LocalTrainingConfig:
    """Hyperparameters that must remain identical across selector methods."""

    local_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    amp: bool = True
    amp_initial_scale: float = 4096.0
    amp_max_retries: int = 8
    max_batches_per_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if self.amp_initial_scale <= 0:
            raise ValueError("amp_initial_scale must be positive.")
        if self.amp_max_retries < 0:
            raise ValueError("amp_max_retries cannot be negative.")
        if (
            self.max_batches_per_epoch is not None
            and self.max_batches_per_epoch <= 0
        ):
            raise ValueError("max_batches_per_epoch must be positive when set.")


@dataclass(frozen=True)
class LocalTrainingResult:
    """Metrics and component states produced by one selected client."""

    client_id: str
    random_seed: int | None
    examples_seen: int
    aggregation_weight: float
    batches_processed: int
    optimizer_steps: int
    amp_overflow_retries: int
    mean_loss: float
    mean_gradient_norm: float
    max_gradient_norm: float
    region_dice: Mapping[str, float]
    elapsed_seconds: float
    amp_enabled: bool
    final_amp_scale: float
    trainable_components: tuple[str, ...]
    update: ComponentUpdate


def _gradient_l2_norm(
    parameters: list[nn.Parameter],
) -> torch.Tensor:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        raise RuntimeError("No gradients were produced for trainable parameters.")
    return torch.stack(norms).norm(2)


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    if "inputs" not in batch or "target" not in batch:
        raise KeyError("Every training batch requires 'inputs' and 'target'.")
    raw_inputs = batch["inputs"]
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise ValueError("batch['inputs'] must be a nonempty modality mapping.")

    inputs: dict[str, torch.Tensor] = {}
    for name, value in raw_inputs.items():
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Input modality {name!r} is not a tensor.")
        inputs[str(name)] = value.to(device, non_blocking=True)
    if not inputs:
        raise ValueError("The batch does not contain an available modality tensor.")

    target = batch["target"]
    if not isinstance(target, torch.Tensor):
        raise TypeError("batch['target'] must be a tensor.")
    target = target.to(device, non_blocking=True, dtype=torch.long)

    availability = batch.get("availability")
    if availability is not None:
        if not isinstance(availability, torch.Tensor):
            raise TypeError("batch['availability'] must be a tensor when provided.")
        availability = availability.to(
            device,
            non_blocking=True,
            dtype=torch.bool,
        )
    return inputs, target, availability


def _aggregation_weight(
    batches: Iterable[Mapping[str, Any]],
    explicit_weight: float | None,
) -> float | None:
    if explicit_weight is not None:
        weight = float(explicit_weight)
        if weight <= 0:
            raise ValueError("aggregation_weight must be positive.")
        return weight

    dataset = getattr(batches, "dataset", None)
    if dataset is None:
        return None
    try:
        weight = float(len(dataset))
    except TypeError:
        return None
    if weight <= 0:
        raise ValueError("The local dataset must contain at least one example.")
    return weight


def _set_dataset_epoch(
    batches: Iterable[Mapping[str, Any]],
    epoch: int,
) -> None:
    dataset = getattr(batches, "dataset", None)
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(epoch)


def _cpu_component_states(
    model: MultiEncoderUNet3D,
    component_names: tuple[str, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    all_states = model.component_state_dict(clone=False)
    return {
        component: {
            name: tensor.detach().cpu().clone()
            for name, tensor in all_states[component].items()
        }
        for component in component_names
    }


def train_local_client(
    model: MultiEncoderUNet3D,
    batches: Iterable[Mapping[str, Any]],
    *,
    client_id: str,
    available_modalities: Iterable[str],
    device: str | torch.device,
    config: LocalTrainingConfig | None = None,
    criterion: nn.Module | None = None,
    aggregation_weight: float | None = None,
    epoch_offset: int = 0,
    random_seed: int | None = None,
) -> LocalTrainingResult:
    """Train one logical client and return only components it updated.

    The default aggregation weight is the local dataset size, not the number
    of examples processed across local epochs. Callers using a generic
    iterable should pass aggregation_weight explicitly.
    """

    if not str(client_id):
        raise ValueError("client_id cannot be empty.")
    if epoch_offset < 0:
        raise ValueError("epoch_offset cannot be negative.")
    if random_seed is not None and random_seed < 0:
        raise ValueError("random_seed cannot be negative.")
    settings = config or LocalTrainingConfig()
    modality_names = tuple(str(name) for name in available_modalities)
    if not modality_names:
        raise ValueError("available_modalities cannot be empty.")

    target_device = torch.device(device)
    use_amp = settings.amp and target_device.type == "cuda"
    model.to(target_device)
    model.configure_client_modalities(modality_names)
    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("The client model has no trainable parameters.")

    loss_function = criterion or DiceCrossEntropyLoss()
    loss_function.to(target_device)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=settings.amp_initial_scale,
    )
    local_generator = (
        None
        if random_seed is None
        else torch.Generator(device=target_device).manual_seed(random_seed)
    )

    declared_weight = _aggregation_weight(batches, aggregation_weight)
    trainable_components = model.trainable_component_names()
    examples_seen = 0
    batches_processed = 0
    optimizer_steps = 0
    overflow_retries = 0
    weighted_loss = 0.0
    gradient_norm_sum = 0.0
    maximum_gradient_norm = 0.0
    dice_sums = {name: 0.0 for name in BRATS_REGIONS}
    dice_examples = 0
    started = time.perf_counter()

    for local_epoch in range(settings.local_epochs):
        _set_dataset_epoch(batches, epoch_offset + local_epoch)
        epoch_batches = 0
        for batch in batches:
            inputs, target, availability = _move_batch(batch, target_device)
            batch_size = int(target.shape[0])
            if batch_size <= 0:
                raise ValueError("Training batches cannot be empty.")

            maximum_attempts = 1 + (
                settings.amp_max_retries if use_amp else 0
            )
            batch_generator_state = (
                None
                if local_generator is None
                else local_generator.get_state()
            )
            for attempt in range(1, maximum_attempts + 1):
                if attempt > 1 and batch_generator_state is not None:
                    local_generator.set_state(batch_generator_state)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(
                        inputs,
                        availability,
                        modality_dropout_generator=local_generator,
                    )
                    loss = loss_function(logits, target)

                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        f"Client {client_id} produced a non-finite forward loss."
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = _gradient_l2_norm(trainable_parameters)
                gradients_are_finite = bool(torch.isfinite(gradient_norm))

                if gradients_are_finite:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=settings.gradient_clip_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_steps += 1
                    break

                if not use_amp:
                    raise RuntimeError(
                        f"Client {client_id} produced non-finite gradients."
                    )

                scaler.step(optimizer)
                scaler.update()
                overflow_retries += 1
                if attempt == maximum_attempts:
                    raise RuntimeError(
                        f"Client {client_id} exceeded "
                        f"{settings.amp_max_retries} AMP retries."
                    )

            batch_scores = brats_dice_from_logits(
                logits.detach(),
                target,
                reduction="none",
            )
            for name, values in batch_scores.items():
                dice_sums[name] += float(values.sum().detach())
            dice_examples += batch_size
            weighted_loss += float(loss.detach()) * batch_size
            gradient_value = float(gradient_norm.detach())
            gradient_norm_sum += gradient_value
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_value)
            examples_seen += batch_size
            batches_processed += 1
            epoch_batches += 1

            if (
                settings.max_batches_per_epoch is not None
                and epoch_batches >= settings.max_batches_per_epoch
            ):
                break

    if batches_processed == 0 or examples_seen == 0:
        raise ValueError("The local training iterable produced no batches.")

    final_weight = (
        declared_weight
        if declared_weight is not None
        else float(examples_seen) / settings.local_epochs
    )
    states = _cpu_component_states(model, trainable_components)
    update = ComponentUpdate(
        client_id=str(client_id),
        states=states,
        weight=final_weight,
        successful=True,
    )
    elapsed_seconds = time.perf_counter() - started

    return LocalTrainingResult(
        client_id=str(client_id),
        random_seed=random_seed,
        examples_seen=examples_seen,
        aggregation_weight=final_weight,
        batches_processed=batches_processed,
        optimizer_steps=optimizer_steps,
        amp_overflow_retries=overflow_retries,
        mean_loss=weighted_loss / examples_seen,
        mean_gradient_norm=gradient_norm_sum / optimizer_steps,
        max_gradient_norm=maximum_gradient_norm,
        region_dice={
            name: total / dice_examples
            for name, total in dice_sums.items()
        },
        elapsed_seconds=elapsed_seconds,
        amp_enabled=use_amp,
        final_amp_scale=float(scaler.get_scale()),
        trainable_components=trainable_components,
        update=update,
    )
