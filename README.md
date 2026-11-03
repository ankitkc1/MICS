# MICS modular multimodal 3D U-Net

This project provides a general segmentation backbone for experiments with
heterogeneous modality availability in federated learning. It is not tied to
BraTS: modality names, input-channel counts, the number of classes, network
depth, and fusion rule are all configurable.

The implementation deliberately separates two concerns:

1. the neural architecture learns from the modalities available at a client;
2. a client-selection policy such as MICS decides which clients participate.

This makes it possible to compare client-selection rules while keeping the
training architecture fixed.

## Architecture

Each modality has an independent residual 3D encoder. Corresponding feature
scales are fused with an availability mask, after which a shared 3D U-Net
decoder produces voxel-level logits.

    image modality a -> encoder:a --\
    image modality b -> encoder:b ----> masked multi-scale fusion
    image modality c -> encoder:c --/             |
                                                    v
                              shared decoder -> prediction head -> logits

The default mean--variance fusion is permutation invariant over the available
modality representations. It computes masked first- and second-order feature
statistics at every scale and learns a 1x1x1 projection. Parameter-free masked
mean and learned spatial attention are also available for ablation studies.

Group normalization is the default because 3D segmentation commonly uses small
local batch sizes and because it avoids aggregating client-specific batch
statistics.

## Installation

From this directory:

    python -m pip install -e .

Only PyTorch is required. Python 3.10 or later is supported.

## Basic use

    from collections import OrderedDict

    import torch
    from mics import MultiEncoderUNet3D

    model = MultiEncoderUNet3D(
        modalities=OrderedDict([
            ("t1", 1),
            ("t1ce", 1),
            ("t2", 1),
            ("flair", 1),
        ]),
        num_classes=4,
        base_channels=16,
        num_levels=4,
        fusion="mean_var",
        modality_dropout=0.2,
    )

    inputs = {
        "t1": torch.randn(2, 1, 96, 96, 96),
        "flair": torch.randn(2, 1, 96, 96, 96),
    }
    logits = model(inputs)

An input is a mapping from modality name to a tensor with shape
[batch, channels, depth, height, width]. Omitted keys are treated as missing.
The result contains unnormalised logits with shape
[batch, classes, depth, height, width].

For a mixed batch, pass a Boolean availability mask of shape
[batch, number_of_modalities]. Its columns follow model.modality_names.
Every sample must retain at least one modality.

    availability = torch.tensor([
        [True, False, False, False],
        [False, False, False, True],
    ])
    logits = model(inputs, availability)

## Federated component boundary

The model exposes stable component identifiers:

- encoder:modality for each modality-specific encoder;
- fusion for all scale-wise fusion modules;
- decoder for the shared decoder;
- head for the voxel classifier.

Before local training, freeze encoders that a client cannot update:

    model.configure_client_modalities({"t1", "flair"})
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=1e-4,
    )

The client can upload only the components that it actually trained:

    states = model.component_state_dict()
    upload = {
        name: states[name]
        for name in model.trainable_component_names()
    }

At the server, component-wise averaging prevents a missing modality from being
interpreted as a zero update:

    from mics import ComponentUpdate, aggregate_components

    new_global_state = aggregate_components(
        global_state=model.component_state_dict(),
        updates=[
            ComponentUpdate(
                client_id="hospital-01",
                states=upload,
                weight=number_of_training_cases,
            )
        ],
    )
    model.load_component_state_dict(new_global_state)

An update may use a mapping of component-specific weights when the number of
effective examples differs across modality encoders. Components receiving no
valid update retain their previous global state.

## Training notes

- Use the returned logits directly with cross-entropy or a Dice--cross-entropy
  loss; do not apply softmax before a loss that expects logits.
- modality_dropout randomly removes available modalities during training while
  guaranteeing that at least one remains. It should be reported as a training
  hyperparameter and ablated independently of client selection.
- All inputs at a client must be spatially registered before feature fusion.
- Patch dimensions should normally be divisible by 2 ** (num_levels - 1).
  Odd dimensions are supported by decoder interpolation, but predictable patch
  dimensions make experiments easier to reproduce.
- The attention fusion option is more expressive but can encode modality
  identity through learned biases. Treat it as an architectural ablation rather
  than as the MICS modality-importance score.

The reference BraTS-style settings are in configs/brats_unet3d.yaml. They are
documentation rather than a hard dependency on a configuration framework.

## Validation

Run the dependency-free test suite from this directory:

    $env:PYTHONPATH = "src"
    python -m unittest discover -s tests -v

The tests cover output shape, arbitrary modality sets, per-sample availability,
modality dropout, unavailable-encoder freezing, component state transfer, and
component-wise aggregation.
