# Probe-based client-selection baselines

MICS compares selector policies under one shared model, data scenario, local
training procedure, component-wise sample-weighted aggregation rule, system
trace, and evaluation protocol. A baseline that requires current model losses
or gradients must pay for those quantities in both elapsed time and logical
communication.

## Power-of-Choice

The implementation follows the Power-of-Choice client-selection rule: at each
round it samples `d` currently available candidates without replacement using
weights proportional to local training-set size, evaluates their current local
training loss, and selects the `m` highest-loss candidates. Equal losses are
resolved by a seeded random tie break.

With no probe batch limit, the recorded variant is `pow_d_full_local_loss` and
uses the complete local training partition. With a positive probe batch limit,
the recorded variant is `cpow_d_minibatch_loss`; this is a computation-reduced
loss estimate and must not be reported as the full-loss variant. The primary
reference is Cho, Wang, and Joshi, *Towards Understanding Biased Client
Selection in Federated Learning* (AISTATS 2022):
<https://proceedings.mlr.press/v151/jee-cho22a.html>.

## DivFL-style gradient facility location

The DivFL-style selector obtains a current, aligned full-parameter gradient
vector from every currently available client. With no probe batch limit, each
vector estimates the full local training objective; with a positive limit, the
run is explicitly recorded as a minibatch-gradient variant. For a client
lacking a modality, the coordinates of that modality encoder are zero. Those
known zero blocks need not be transmitted. The selector computes pairwise
Euclidean gradient distances and applies cardinality-constrained greedy
maximisation to the corresponding monotone facility-location objective. This
selects a subset whose gradients represent the available client population.

This repository deliberately labels the method
`divfl_gradient_facility`. It adapts the selection objective from DivFL while
retaining the same component-wise sample-weighted FedAvg aggregation used by
all MICS comparisons. It is therefore a selection-only controlled adaptation,
not a claim of reproducing every end-to-end detail of the original method.
The primary reference is Balakrishnan et al., *Diverse Client Selection for
Federated Learning via Submodular Maximization* (ICLR 2022):
<https://openreview.net/forum?id=nwKXyFvaUm>.

## Probe and communication contract

Selection probes can access only the local training partition. They never use
validation or test cases and never update the global model. Each round records:

- probed client IDs, probe kind, batch limit, examples, batches, and losses;
- gradient dimension and norm for a gradient probe;
- wall-clock probe time separately from server-side selection time;
- logical download bytes for the model components required by each client;
- an eight-byte scalar upload for a loss probe, or the trainable-component
  gradient bytes for a gradient probe; and
- training download and update-upload bytes separately from probe traffic.

If a probed client is selected in the same round, its already downloaded model
is reused and is not charged a second time. The update upload remains separate.
This accounting is a logical communication model, not a measurement of Python
object copies or transport-protocol overhead.

Use `scripts/run_reference_baseline.py` for both methods. Any run with a local
training batch cap is an integration smoke test and cannot be reported as model
quality or selector superiority.

MFedMC's client-policy adaptation is available through `--method mfedmc`, with
the result identifier `mfedmc_client_low_loss`. Each available client's
modalities are evaluated separately through the shared segmentation decoder,
using only its training partition. The arithmetic mean of its available
single-modality losses determines its score, and the lowest-loss clients fill
the fixed client budget. Equal scores use a checkpointed random tie break.
Probe downloads, one scalar upload per evaluated modality, and elapsed probe
time are charged to the method.

This adaptation implements the low-loss client rule in Equations 17–19 of the
paper. The original framework independently trains modality predictors,
personalizes fusion, chooses encoder uploads, and selects clients after local
training. This experiment uses pre-training probes and the common shared
model and aggregation contract. The released ActionSense code at commit
`bb9907254683818dfe9e4784f798a25eca6a1d58` defaults to higher-loss priority;
the paper's lower-loss direction is declared explicitly in result metadata.
[MFedMC paper](https://arxiv.org/abs/2401.16685v2),
[released client-priority code](https://github.com/liangqiyuan/MFedMC/blob/bb9907254683818dfe9e4784f798a25eca6a1d58/ActionSense/federated.py)

Oort is available through `--method oort`, with result identifier
`oort_training`. The Apache-2.0 training selector is pinned to upstream commit
`05a3aa1677a10f8e621055b1626ef82e73d09759`. Its exploration/exploitation,
temporal uncertainty and pacer are retained. The adapter isolates and
checkpoints NumPy and Python RNG state, handles empty or small exploitation
pools and zero utilities, and disables blacklisting through the declared
default configuration. The upstream license is distributed beside the
vendored training selector.
[Pinned upstream implementation](https://github.com/SymbioticLab/Oort/blob/05a3aa1677a10f8e621055b1626ef82e73d09759/oort/oort.py)

Oort receives feedback only after selected clients complete. Statistical
utility is training-set size times RMS batch training loss. For the configured
batch size one, each batch corresponds to one sampled patch. Duration is the
observed local optimization time. Registration starts with sample-count
utility and a declared duration of one until an observation exists. Failed
simulated attempts receive zero utility and retain their last known duration.
The simulator's failure and logical-network cost model applies to this method
as well. Each successful client sends two eight-byte feedback scalars for
RMS loss and duration, charged separately from its model upload. Oort's use
of observed timing means its decisions depend on measured
execution history in addition to the declared seed.

The complete single-seed workflow, including historical v2 provenance and
full validation, is described in `docs/single_seed_baselines.md`.

The Oort exploitation sampler also caps its quota at the number of positive
cutoff-qualified candidates. Failure-derived zero utilities can otherwise
leave an undersized pool after normalization. The existing exploration and
eligible-client fallback fill the remaining budget. The isolated repair
study and completed-baseline reuse checks are described in
`docs/oort_recovery.md`.
