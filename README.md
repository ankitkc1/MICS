# MICS: multimodal federated client selection

MICS is a research codebase for studying client selection under heterogeneous
modality availability. The repository is organised for journal reproducibility,
local development, and synchronous cross-silo experiments on a Slurm-managed
HPC cluster.

The central experimental question is separated from the segmentation
architecture: a selector decides which clients participate, while the
multimodal model learns only from the modalities available at each selected
client. This allows MICS and competing selectors to be compared using the same
training, aggregation, data partitions, and evaluation protocol.

## Current implementation status

Implemented and tested:

- a configurable multi-encoder residual 3D U-Net;
- masked multi-scale mean, mean--variance, and attention fusion;
- missing-modality support at client and sample levels;
- modality dropout for robustness experiments;
- stable federated component boundaries;
- modality-specific component-wise aggregation;
- a BraTS2020 NumPy loader for the existing HPC arrays and client splits;
- deterministic foreground-aware 3D patch sampling;
- Dice--cross-entropy loss and WT, TC, and ET metrics;
- a real-data CUDA forward/backward smoke test with peak-memory reporting;
- reusable AMP-aware local-client optimization and component update export;
- sequential synchronous federated rounds with component-wise aggregation;
- whole-volume Gaussian sliding-window inference and BraTS Dice/HD95 metrics;
- a selector interface with a reproducible uniform-random baseline;
- validation-only paired leave-one-modality-out importance estimation;
- exact small-silo component-service target calibration;
- checkpointable MICS-Core/Impact component queues and cardinality greedy;
- a shared tracked FL runner for uniform and MICS selector comparisons;
- immutable matched availability/completion-potential traces for controlled
  system-heterogeneity experiments;
- immutable run metadata, selector-state checkpoints, and resume support;
- guarded paired-run audits for selection, service, communication, and metrics;
- CPU tests and an HPC GPU smoke-job template.

Planned but not yet claimed as experimental evidence:

- cached-update MICS-Diversity and knapsack-oracle extensions;
- PoC, DivFL, Oort, and modality-aware comparison methods;
- complete multi-seed journal experiments.

## Model architecture

Each modality is processed by an independent 3D encoder. Features at matching
resolutions are fused using only the modalities marked as available. A shared
3D U-Net decoder and prediction head produce voxel-level logits.

    modality a -> encoder:a --\
    modality b -> encoder:b ----> masked multi-scale fusion
    modality c -> encoder:c --/             |
                                                 v
                           shared decoder -> segmentation head -> logits

The default mean--variance fusion is permutation invariant over the available
encoded representations. Group normalization is used by default because 3D
segmentation commonly requires small local batches and client-specific batch
statistics complicate federated aggregation.

## Repository layout

    MICS/
    +-- configs/             reviewed experiment configuration
    +-- docs/                HPC and reproducibility documentation
    +-- environments/        local and cluster environments
    +-- examples/            minimal executable examples
    +-- experiments/         manifests and study definitions
    +-- hpc/slurm/           scheduler scripts
    +-- scripts/             validation and experiment entry points
    +-- src/mics/
    |   +-- data/            dataset and client-split adapters
    |   +-- evaluation/      task metrics
    |   +-- federated/       component aggregation
    |   +-- models/          multimodal 3D U-Net
    |   +-- selection/       selector contracts and methods
    |   +-- training/        losses and local-training utilities
    +-- tests/               deterministic unit tests

See docs/repository_layout.md for the ownership rules that prevent experimental
logic from becoming entangled with model code.

## Installation and local validation

Create an isolated Python environment, then run:

    python -m pip install -e .
    python scripts/validate_install.py

For development directly from the checkout:

    $env:PYTHONPATH = "src"
    python -m unittest discover -s tests -v
    python examples/brats_forward.py

## BraTS-style model example

    from collections import OrderedDict
    import torch
    from mics import MultiEncoderUNet3D

    model = MultiEncoderUNet3D(
        modalities=OrderedDict([
            ("flair", 1),
            ("t1ce", 1),
            ("t1", 1),
            ("t2", 1),
        ]),
        num_classes=4,
        base_channels=16,
        num_levels=4,
        fusion="mean_var",
        modality_dropout=0.2,
    )

    logits = model({
        "t1": torch.randn(1, 1, 96, 96, 96),
        "flair": torch.randn(1, 1, 96, 96, 96),
    })

The same model supports any named modality set and is therefore not restricted
to BraTS. Inputs use the shape batch, channels, depth, height, width.

## Federated component contract

The model exposes:

- encoder:modality for each modality-specific encoder;
- fusion for scale-wise fusion modules;
- decoder for the shared decoder;
- head for the prediction layer.

An unavailable encoder is frozen locally and omitted from that client's upload.
Server aggregation averages each component over only the successful clients that
updated it. A component receiving no update retains its previous global state.

## Intended simulation model

The planned main experiments use a custom, centralised, synchronous cross-silo
simulation in PyTorch. Logical clients maintain separate data partitions,
modality masks, optimizer states, and local metadata on a Slurm-allocated node.
Communication is represented by parameter exchange; raw patient data are never
placed in server aggregation objects.

Selector-only dry runs must be labelled offline simulations. They do not load
MRI data, train a neural network, or constitute FL accuracy experiments.

## Reproducibility policy

Source code, reviewed configurations, client manifests, scheduler scripts, and
small aggregate result tables belong in version control. Patient data, model
checkpoints, raw logs, scheduler output, credentials, and large generated
artifacts do not.

Every reported experiment must record its source revision, full configuration,
environment, seeds, client partition, modality mask, selected clients per
round, checkpoint lineage, and metric definition. The required fields are
listed in docs/reproducibility.md.

## Public release checklist

Before public journal release, add the final author list, institutional
acknowledgements, dataset access statement, software licence, archival DOI, and
the exact commands that reproduce every manuscript table and figure.
