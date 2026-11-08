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
