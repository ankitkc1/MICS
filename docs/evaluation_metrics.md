# Evaluation metrics

MICS uses a conservative core evaluation suite established in federated
client-selection research. Method-specific modality diagnostics supplement,
but do not replace, these common outcomes.

## Primary model and convergence outcomes

- held-out whole-volume WT, TC, ET, and mean Dice;
- held-out WT, TC, and ET HD95;
- final-round and best validation performance;
- rounds to a preregistered target performance;
- area under the round-versus-performance curve;
- mean and standard deviation over matched experimental seeds.

Power-of-Choice evaluates convergence speed and test performance. Oort
emphasizes time-to-accuracy and final accuracy. DivFL evaluates learning
efficiency, convergence, and uniformity across clients. MICS will report the
corresponding segmentation outcomes rather than classification accuracy.

## Efficiency and resource outcomes

- wall-clock training time and selector-only overhead;
- time to the preregistered target under controlled system conditions;
- cumulative uploaded and downloaded parameter bytes;
- communication rounds to target;
- successful, failed, and unavailable selected clients.

Raw HPC timing is reported as an implementation measurement, not as a general
systems claim. Comparisons under system heterogeneity require identical,
controlled client-latency and availability traces.

Communication is counted from the exact tensor sizes of the modality encoders
and shared components logically transferred by the federated protocol. It is
not inferred from Python object size or in-process simulator memory copies.
Training-patch Dice is retained only as an optimization diagnostic; all model
quality claims use held-out whole-volume reconstruction.

Validation metrics are computed at preregistered archived checkpoints, every
ten communication rounds by default. The evaluator reports per-client,
macro-client, and sample-weighted reductions. Development never evaluates the
local-test or global-test cases; those partitions are opened only after method
and hyperparameter choices are frozen.

## Validation-derived modality importance

MICS estimates modality importance with paired leave-one-modality-out probes.
For a validation case `i` held by client `k`, let `D[k,i,r](A[k])` be the Dice
for region `r` under the client's native modality set `A[k]`. For every
available modality `m` that can be removed without leaving an empty input,
the signed case effect is

    delta[k,i,m,r] = D[k,i,r](A[k]) - D[k,i,r](A[k] without m).

WT, TC, and ET receive equal weights in the primary analysis. Region weights
are configurable but must be fixed before the comparative study; no BraTS
region or modality is favoured by default. Case effects are first averaged
within client, then across eligible clients, so a client with a larger probe
partition cannot dominate the estimate. The estimator retains region-wise,
sample-weighted, per-client, and signed effects together with the standard
error across client means.

The subset objective requires non-negative modality weights. Therefore only
the selector-facing copy is clipped at zero and given a small positive floor
before normalisation. The signed estimate is never discarded. If all signed
effects are non-positive, the implementation records an explicit fallback and
uses uniform weights rather than inventing a ranking.

These probes consume validation data as a selector/calibration input. Once
used for this purpose, the same scores are not described as an independent
generalisation estimate. Local-test and global-test partitions remain locked
until the complete protocol, hyperparameters, checkpoint rule, importance
refresh schedule, and target thresholds are frozen. If independent validation
model selection is required, a disjoint selection-probe partition must be
materialised before the main experiments.

Probe cost is reported separately: native baseline case evaluations,
additional modality-ablation evaluations, sliding-window count, inference and
wall time, and peak GPU memory. A truncated one-case-per-client run validates
the pipeline only and cannot establish scientific modality weights.

## Client-level outcomes

- macro-client and sample-weighted performance;
- standard deviation and lower-tail client performance;
- client selection counts and Jain selection-fairness index.

These complement the global model score and expose selection policies that
improve the average while repeatedly excluding particular clients.

## MICS explanatory diagnostics

- per-round modality coverage;
- signed and normalised validation-derived modality importance;
- impact-weighted modality coverage;
- component update counts;
- client and modality staleness;
- maximum opportunity-conditioned modality stall;
- redundancy among selected modality sets;
- greedy marginal gain and objective-term decomposition.

These are mechanism diagnostics specific to the proposed contribution. They
must not be presented as substitutes for the primary accuracy, convergence,
fairness, time, and communication metrics above.

BraTS 2020 cases use one-millimetre isotropic spacing. For a region absent in
both prediction and reference, Dice is one and HD95 is zero. If exactly one is
empty, Dice is zero and HD95 receives the BraTS 2020 penalty of 373.13 mm.
Metrics are computed per whole subject after Gaussian sliding-window
reconstruction, never by averaging training-patch scores.

## Comparison rules

All methods use identical model initialization, client splits, modality
manifest, training batches, evaluation cases, selected-client budget, and
matched seeds. Target thresholds are fixed before comparing methods. Test data
never supplies client utility or modality importance to a selector.

The paired-run audit rejects comparisons when the scenario checksums, client
partitions, model, training configuration, model/data seeds, client budget, or
whole-volume evaluation cases and modality masks differ. Smoke-run differences
are recorded for pipeline verification only and are explicitly marked as
ineligible for model-quality or selector-superiority claims.

Primary references:

- Oort: https://www.usenix.org/conference/osdi21/presentation/lai
- Power-of-Choice: https://proceedings.mlr.press/v151/jee-cho22a.html
- DivFL: https://openreview.net/forum?id=nwKXyFvaUm
- FedScale: https://proceedings.mlr.press/v162/lai22a.html
- q-FFL: https://openreview.net/forum?id=ByexElSYDr
