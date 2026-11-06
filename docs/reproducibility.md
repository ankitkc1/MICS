# Reproducibility requirements

No value should appear in a manuscript table unless its complete provenance can
be reconstructed from an immutable run directory.

## Required run metadata

Each run must record:

- unique run and study identifiers;
- UTC start and completion times;
- source revision and dirty-worktree status;
- fully resolved configuration and its checksum;
- Python, PyTorch, CUDA, cuDNN, driver, and package versions;
- scheduler job ID, array task ID, hostname, GPU model, and GPU count;
- model, data-partition, augmentation, worker, and selection seeds;
- client identifiers, sample counts, modality masks, and partition checksums;
- selected and successfully completed clients at every round;
- components trained and uploaded by every selected client;
- component-specific aggregation weights;
- communication and optional probe-cost accounting;
- metric definitions, reduction rules, and evaluation masks;
- checkpoint parent and resume history.

## Independent random streams

At minimum, use separate deterministic streams for:

1. data partitioning;
2. model initialization;
3. client selection;
4. data shuffling;
5. augmentation and modality dropout;
6. data-loader workers.

Policies consume different numbers of selection draws. Sharing that generator
with data shuffling silently changes local minibatch order and invalidates a
controlled policy comparison.

## Client scenario

The materialized client scenario binds client IDs, modality availability,
train/validation/local-test files, and the global-test file. It does not copy or
relocate the dataset stored on the HPC. Absolute paths recorded by the split
generator are resolved portably against the scenario bundle.

All compared selection methods must use the same client mapping and case split.
The current Scenario A and Scenario B are matched modality-availability
conditions: their case assignments are identical and their modality masks
differ. They must not be described as independent data partitions or averaged
as independent repetitions. Independent model/selection seeds remain necessary
for uncertainty estimates.

## Checkpoint contract

A resumable checkpoint must contain:

- next communication round;
- global component states;
- required client-local and optimizer states;
- aggregation history needed by the method;
- selector state, including its random-generator state;
- all other random-generator states needed for deterministic continuation;
- best metrics and evaluation history;
- resolved configuration and source revision.

The uniform baseline runner resets AdamW for every client and round, following
the current FedAvg contract, so no persistent client optimizer state exists to
checkpoint. It does checkpoint global components, selector and PyTorch random
states, participation history, modality staleness, and cumulative communication
counters.

The loader must validate the checkpoint schema and configuration compatibility.
The coordinator must never restart at a completed round or overwrite a previous
run without an explicit, recorded action.

## Evaluation contract

The complete metric hierarchy and comparison rules are defined in
docs/evaluation_metrics.md.

For every evaluation point, report:

- macro-average client metric;
- sample-weighted federated metric;
- per-client metric;
- class- or region-wise metric;
- the exact modality mask used for each evaluation.

For BraTS-style segmentation, define Whole Tumour, Tumour Core, and Enhancing
Tumour explicitly. State whether absent classes are ignored, assigned a perfect
score, or assigned zero; use one convention for every method.

The server need not own patient data. The global model can be broadcast to
clients for evaluation on held-out local data, after which only aggregate
metrics and sample counts are returned.

## Fair selector comparison

Hold constant:

- model initialization and training hyperparameters;
- client data and modality manifests;
- local epochs and minibatch order;
- number of selected clients or a clearly defined resource budget;
- evaluation schedule and checkpoint rule;
- number and list of experimental seeds.

Report the extra inference, probing, and communication cost of methods that
need losses, gradients, embeddings, or modality ablations before selection.
Selection-only dry runs must never be presented as segmentation performance.

## Suggested generated run layout

    results/study_name/run_id/
    +-- resolved_config.yaml
    +-- run_metadata.json
    +-- client_manifest.csv
    +-- selection_history.jsonl
    +-- metrics.json
    +-- checkpoints/
    +-- logs/
    +-- environment/

This directory is ignored by default because it may be large. Publication
scripts should transform reviewed runs into compact aggregate tables and
figures without modifying the raw records.
