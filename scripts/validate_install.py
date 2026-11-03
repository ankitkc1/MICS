"""Fast installation and runtime validation for local or HPC environments."""

from __future__ import annotations

import platform

import torch

from mics import (
    ClientState,
    MultiEncoderUNet3D,
    SelectionContext,
    UniformRandomSelector,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiEncoderUNet3D(
        modalities={"a": 1, "b": 1},
        num_classes=3,
        base_channels=2,
        num_levels=3,
        fusion="mean_var",
    ).to(device)
    model.eval()

    with torch.inference_mode():
        output = model({"a": torch.randn(1, 1, 8, 8, 8, device=device)})
    if output.shape != (1, 3, 8, 8, 8):
        raise RuntimeError(f"Unexpected smoke-test shape: {tuple(output.shape)}")

    selector = UniformRandomSelector(seed=42)
    plan = selector.plan_round(
        SelectionContext(
            round_index=0,
            budget=1,
            clients=(
                ClientState("C1", {"a"}, 4),
                ClientState("C2", {"b"}, 4),
            ),
        )
    )
    if len(plan.selected_client_ids) != 1:
        raise RuntimeError("Selector smoke test did not return one client.")

    print("MICS INSTALLATION VALID")
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
    print("Output shape:", tuple(output.shape))
    print("Selected client:", plan.selected_client_ids[0])


if __name__ == "__main__":
    main()
