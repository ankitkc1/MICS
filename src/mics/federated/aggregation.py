"""Component-wise federated averaging for MICS-compatible models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

ComponentState = Mapping[str, torch.Tensor]
ModelComponentState = Mapping[str, ComponentState]


@dataclass(frozen=True)
class ComponentUpdate:
    """One client's successfully trained component states.

    A client should omit components it did not train. The weight may be a
    single sample count or a mapping of component-specific sample counts.
    """

    client_id: str
    states: ModelComponentState
    weight: float | Mapping[str, float] = 1.0
    successful: bool = True

    def weight_for(self, component: str) -> float:
        if isinstance(self.weight, Mapping):
            return float(self.weight.get(component, 0.0))
        return float(self.weight)


def _validate_component_state(
    component: str,
    reference: ComponentState,
    candidate: ComponentState,
) -> None:
    if set(candidate) != set(reference):
        missing = sorted(set(reference).difference(candidate))
        extra = sorted(set(candidate).difference(reference))
        raise ValueError(
            f"State keys for component {component!r} do not match; "
            f"missing={missing}, extra={extra}."
        )
    for name, reference_tensor in reference.items():
        candidate_tensor = candidate[name]
        if candidate_tensor.shape != reference_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {component!r}/{name!r}: "
                f"{tuple(candidate_tensor.shape)} != {tuple(reference_tensor.shape)}."
            )


def aggregate_components(
    global_state: ModelComponentState,
    updates: Sequence[ComponentUpdate],
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    """Perform sample-weighted FedAvg independently for every model component.

    Components receiving no successful positive-weight update retain their
    previous global value. Non-floating buffers are also retained, avoiding
    undefined averaging of counters or boolean state.
    """

    aggregated: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
    unknown_components = {
        component
        for update in updates
        for component in update.states
        if component not in global_state
    }
    if unknown_components:
        raise KeyError(f"Updates contain unknown components: {sorted(unknown_components)}")

    for component, reference in global_state.items():
        eligible: list[tuple[float, ComponentState]] = []
        for update in updates:
            if not update.successful or component not in update.states:
                continue
            weight = update.weight_for(component)
            if weight < 0:
                raise ValueError("Aggregation weights cannot be negative.")
            if weight == 0:
                continue
            candidate = update.states[component]
            _validate_component_state(component, reference, candidate)
            eligible.append((weight, candidate))

        component_result: dict[str, torch.Tensor] = {}
        total_weight = sum(weight for weight, _ in eligible)
        for name, reference_tensor in reference.items():
            if not eligible or not (reference_tensor.is_floating_point() or reference_tensor.is_complex()):
                component_result[name] = reference_tensor.detach().clone()
                continue

            accumulator_dtype = (
                torch.float64
                if reference_tensor.dtype == torch.float64
                else torch.float32
            )
            accumulator = torch.zeros_like(
                reference_tensor,
                dtype=accumulator_dtype,
                device=reference_tensor.device,
            )
            for weight, candidate in eligible:
                value = candidate[name].detach().to(
                    device=reference_tensor.device,
                    dtype=accumulator_dtype,
                )
                accumulator.add_(value, alpha=weight / total_weight)
            component_result[name] = accumulator.to(dtype=reference_tensor.dtype)
        aggregated[component] = component_result

    return aggregated

