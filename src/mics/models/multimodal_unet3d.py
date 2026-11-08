"""A general multi-encoder 3D U-Net with explicit federated components."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .blocks import Decoder3D, Encoder3D
from .fusion import build_fusion


class MultiEncoderUNet3D(nn.Module):
    """Modular multimodal 3D U-Net for heterogeneous modality availability.

    Every modality owns an independent encoder. Features at matching U-Net
    resolutions are fused using only modalities marked available for each
    sample. A shared decoder and prediction head then produce task logits.

    The model exposes named components so a federated server can aggregate
    each modality encoder over the clients that actually trained it while
    aggregating the fusion, decoder, and head over all successful clients.
    """

    def __init__(
        self,
        modalities: Mapping[str, int],
        num_classes: int,
        *,
        base_channels: int = 16,
        num_levels: int = 4,
        feature_channels: Sequence[int] | None = None,
        fusion: str = "mean_var",
        norm: str = "group",
        dropout: float = 0.0,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not modalities:
            raise ValueError("At least one modality must be configured.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if num_levels < 2:
            raise ValueError("num_levels must be at least two.")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must lie in [0, 1).")

        modality_channels = OrderedDict((str(name), int(channels)) for name, channels in modalities.items())
        for name, channels in modality_channels.items():
            if not name or "." in name:
                raise ValueError("Modality names must be nonempty and cannot contain '.'.")
            if channels <= 0:
                raise ValueError(f"Input channels for modality {name!r} must be positive.")

        if feature_channels is None:
            channels = tuple(base_channels * (2**level) for level in range(num_levels))
        else:
            channels = tuple(int(value) for value in feature_channels)
            if len(channels) != num_levels:
                raise ValueError("feature_channels length must equal num_levels.")
            if any(value <= 0 for value in channels):
                raise ValueError("All feature channel values must be positive.")

        self.modality_channels = modality_channels
        self.modality_names = tuple(modality_channels)
        self.num_classes = int(num_classes)
        self.feature_channels = channels
        self.fusion_kind = fusion
        self.modality_dropout = float(modality_dropout)

        self.encoders = nn.ModuleDict(
            {
                name: Encoder3D(
                    in_channels,
                    channels,
                    norm=norm,
                    dropout=dropout,
                )
                for name, in_channels in modality_channels.items()
            }
        )
        self.fusions = nn.ModuleList(
            [
                build_fusion(
                    fusion,
                    level_channels,
                    len(self.modality_names),
                    norm=norm,
                )
                for level_channels in channels
            ]
        )
        self.decoder = Decoder3D(channels, norm=norm, dropout=dropout)
        self.head = nn.Conv3d(channels[0], num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.01,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.InstanceNorm3d, nn.BatchNorm3d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def shared_component_names(self) -> tuple[str, ...]:
        return ("fusion", "decoder", "head")

    @property
    def encoder_component_names(self) -> tuple[str, ...]:
        return tuple(f"encoder:{name}" for name in self.modality_names)

    def component_modules(self) -> OrderedDict[str, nn.Module]:
        """Return the exact model blocks used for component-wise aggregation."""

        components: OrderedDict[str, nn.Module] = OrderedDict()
        for name in self.modality_names:
            components[f"encoder:{name}"] = self.encoders[name]
        components["fusion"] = self.fusions
        components["decoder"] = self.decoder
        components["head"] = self.head
        return components

    def component_state_dict(self, *, clone: bool = True) -> OrderedDict[str, dict[str, torch.Tensor]]:
        """Return state dictionaries keyed by stable federated component IDs."""

        result: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        for component, module in self.component_modules().items():
            state = module.state_dict()
            result[component] = {
                name: tensor.detach().clone() if clone else tensor
                for name, tensor in state.items()
            }
        return result

    def load_component_state_dict(
        self,
        states: Mapping[str, Mapping[str, torch.Tensor]],
        *,
        strict: bool = True,
    ) -> None:
        """Load one or more component state dictionaries."""

        modules = self.component_modules()
        unknown = set(states).difference(modules)
        if unknown:
            raise KeyError(f"Unknown model components: {sorted(unknown)}")
        for component, state in states.items():
            modules[component].load_state_dict(state, strict=strict)

    def parameter_component_map(self) -> dict[str, str]:
        """Map every fully qualified parameter name to its MICS component."""

        owner_by_id: dict[int, str] = {}
        for component, module in self.component_modules().items():
            for parameter in module.parameters():
                owner_by_id[id(parameter)] = component
        return {
            name: owner_by_id[id(parameter)]
            for name, parameter in self.named_parameters()
        }

    def configure_client_modalities(
        self,
        available_modalities: Iterable[str],
        *,
        train_shared: bool = True,
    ) -> None:
        """Freeze unavailable encoders before a client's local optimization."""

        available = set(available_modalities)
        unknown = available.difference(self.modality_names)
        if unknown:
            raise KeyError(f"Unknown modalities: {sorted(unknown)}")
        if not available:
            raise ValueError("A client must have at least one available modality.")

        for name, encoder in self.encoders.items():
            enabled = name in available
            for parameter in encoder.parameters():
                parameter.requires_grad_(enabled)
        for component in self.shared_component_names:
            for parameter in self.component_modules()[component].parameters():
                parameter.requires_grad_(train_shared)

    def trainable_component_names(self) -> tuple[str, ...]:
        """Report components containing at least one trainable parameter."""

        return tuple(
            name
            for name, module in self.component_modules().items()
            if any(parameter.requires_grad for parameter in module.parameters())
        )

    def _validate_inputs(
        self,
        inputs: Mapping[str, torch.Tensor | None],
        availability_mask: torch.Tensor | Mapping[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        unknown = set(inputs).difference(self.modality_names)
        if unknown:
            raise KeyError(f"Unknown input modalities: {sorted(unknown)}")

        present = {
            name: tensor
            for name, tensor in inputs.items()
            if tensor is not None
        }
        if not present:
            raise ValueError("At least one modality tensor must be provided.")

        reference = next(iter(present.values()))
        if reference.ndim != 5:
            raise ValueError("Every modality tensor must have shape [B, C, D, H, W].")
        batch = reference.shape[0]
        spatial = reference.shape[2:]
        device = reference.device

        for name, tensor in present.items():
            if tensor.ndim != 5:
                raise ValueError(f"Input {name!r} must have shape [B, C, D, H, W].")
            expected_channels = self.modality_channels[name]
            if tensor.shape[1] != expected_channels:
                raise ValueError(
                    f"Input {name!r} has {tensor.shape[1]} channels; "
                    f"expected {expected_channels}."
                )
            if tensor.shape[0] != batch or tensor.shape[2:] != spatial:
                raise ValueError("All modality tensors must share batch and spatial dimensions.")
            if tensor.device != device:
                raise ValueError("All modality tensors must be on the same device.")

        if availability_mask is None:
            mask = torch.zeros(batch, len(self.modality_names), dtype=torch.bool, device=device)
            for index, name in enumerate(self.modality_names):
                if name in present:
                    mask[:, index] = True
        elif isinstance(availability_mask, torch.Tensor):
            if availability_mask.shape != (batch, len(self.modality_names)):
                raise ValueError(
                    "availability_mask must have shape "
                    f"[{batch}, {len(self.modality_names)}]."
                )
            mask = availability_mask.to(device=device, dtype=torch.bool)
        else:
            unknown_mask_keys = set(availability_mask).difference(self.modality_names)
            if unknown_mask_keys:
                raise KeyError(f"Unknown mask modalities: {sorted(unknown_mask_keys)}")
            mask = torch.zeros(batch, len(self.modality_names), dtype=torch.bool, device=device)
            for index, name in enumerate(self.modality_names):
                value = availability_mask.get(name)
                if value is not None:
                    value = value.to(device=device, dtype=torch.bool)
                    if value.shape != (batch,):
                        raise ValueError(f"Mask for {name!r} must have shape [{batch}].")
                    mask[:, index] = value

        for index, name in enumerate(self.modality_names):
            if name not in present and torch.any(mask[:, index]):
                raise ValueError(
                    f"Availability mask marks {name!r} present, but no tensor was provided."
                )
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every sample must have at least one available modality.")
        return present, mask

    def _apply_modality_dropout(
        self,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not self.training or self.modality_dropout == 0.0:
            return mask

        keep = (
            torch.rand(
                mask.shape,
                device=mask.device,
                generator=generator,
            )
            >= self.modality_dropout
        )
        dropped = mask & keep
        empty = ~dropped.any(dim=1)
        if torch.any(empty):
            random_scores = torch.rand(
                mask.shape,
                device=mask.device,
                generator=generator,
            )
            random_scores = random_scores.masked_fill(~mask, -1.0)
            fallback = random_scores.argmax(dim=1)
            rows = torch.arange(mask.shape[0], device=mask.device)[empty]
            dropped[rows, fallback[empty]] = True
        return dropped

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor | None],
        availability_mask: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        *,
        return_aux: bool = False,
        modality_dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        present, mask = self._validate_inputs(inputs, availability_mask)
        effective_mask = self._apply_modality_dropout(
            mask,
            generator=modality_dropout_generator,
        )

        encoded: dict[str, tuple[torch.Tensor, ...]] = {}
        for index, name in enumerate(self.modality_names):
            if name in present and torch.any(effective_mask[:, index]):
                encoded[name] = self.encoders[name](present[name])

        if not encoded:
            raise RuntimeError("No encoder was executed; check the modality availability mask.")
        reference_features = next(iter(encoded.values()))

        fused_levels: list[torch.Tensor] = []
        fusion_weights: list[torch.Tensor] = []
        for level, fusion_module in enumerate(self.fusions):
            level_features = []
            for name in self.modality_names:
                if name in encoded:
                    level_features.append(encoded[name][level])
                else:
                    level_features.append(torch.zeros_like(reference_features[level]))
            stacked = torch.stack(level_features, dim=1)
            fused, weights = fusion_module(stacked, effective_mask)
            fused_levels.append(fused)
            fusion_weights.append(weights)

        decoded = self.decoder(fused_levels)
        logits = self.head(decoded)
        if not return_aux:
            return logits
        return {
            "logits": logits,
            "effective_mask": effective_mask,
            "fused_features": tuple(fused_levels),
            "fusion_weights": tuple(fusion_weights),
        }

