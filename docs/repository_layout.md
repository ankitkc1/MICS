# Repository ownership and architecture

The repository is divided by scientific responsibility. A module should not
cross these boundaries merely to make a single experiment convenient.

## Source package

src/mics/models owns neural-network architecture only. It must not know which
client was selected, which communication round is running, or where data are
stored.

src/mics/federated owns parameter exchange and aggregation. Aggregators operate
on explicit component states and weights; they never read patient data.

src/mics/selection owns participation decisions. Selectors consume
SelectionContext metadata and return a SelectionPlan. They must not train the
model or mutate data-loader state.

Future modules should follow the same rule:

- data will own preprocessing, sample loading, and deterministic partitions;
- training will own local optimization and the round coordinator;
- evaluation will own metric definitions and federated evaluation;
- experiment will own resolved configuration and run metadata.

## Thin entry points

Files in scripts should parse configuration and call package code. Algorithms
must remain importable and testable inside src/mics. Slurm files should invoke
scripts; they must not contain scientific logic.

## Configuration layering

The configuration tree separates data, model, federation, selection, and
complete experiment definitions. One study configuration must resolve to one
immutable run configuration. Personal filesystem paths are supplied by
environment variables or ignored local overrides.

## Experiments and results

experiments contains committed study definitions and client-manifest schemas.
It does not contain raw checkpoints or ad hoc training code.

Generated run directories belong under ignored results or artifacts folders.
Only reviewed, compact aggregate tables may be force-included for a public
release. Every included result must identify the command, source
revision, configuration, and raw run IDs from which it was derived.

## Tests

Unit tests verify model blocks, selector behavior, and aggregation independently.
Integration tests will later cover a tiny deterministic federated round using
synthetic fixtures.

## Paper-to-code mapping

The final manuscript should map its sections to the following code surfaces:

- System model: client manifest, component boundary, and aggregation contract;
- Problem formulation: selection context, budget, and selector objective;
- MICS framework: the future MICS selector implementation;
- Convergence analysis: assumptions and theorem, not runtime code;
- Experiments: immutable configurations, manifests, runner, and evaluation;
- Reproducibility: environment, Slurm jobs, checkpoints, logs, and result tools.

This mapping makes it possible for reviewers to distinguish the proposed
selection contribution from the segmentation backbone and execution system.
