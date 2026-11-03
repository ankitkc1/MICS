"""Minimal forward/backward example with incomplete BraTS modalities."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch.nn import functional as F

from mics import MultiEncoderUNet3D


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiEncoderUNet3D(
        modalities=OrderedDict(
            flair=1,
            t1ce=1,
            t1=1,
            t2=1,
        ),
        num_classes=4,
        base_channels=8,
        num_levels=3,
        fusion="mean_var",
        modality_dropout=0.2,
    ).to(device)

    # This example client owns only T1 and FLAIR.
    model.configure_client_modalities({"t1", "flair"})
    inputs = {
        "t1": torch.randn(1, 1, 32, 32, 32, device=device),
        "flair": torch.randn(1, 1, 32, 32, 32, device=device),
    }
    target = torch.randint(0, 4, (1, 32, 32, 32), device=device)

    logits = model(inputs)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    print("device:", device)
    print("logit shape:", tuple(logits.shape))
    print("loss:", float(loss.detach()))
    print("trained components:", model.trainable_component_names())


if __name__ == "__main__":
    main()
