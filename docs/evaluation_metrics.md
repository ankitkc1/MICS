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

## Client-level outcomes

- macro-client and sample-weighted performance;
- standard deviation and lower-tail client performance;
- client selection counts and Jain selection-fairness index.

These complement the global model score and expose selection policies that
improve the average while repeatedly excluding particular clients.

## MICS explanatory diagnostics

- per-round modality coverage;
- impact-weighted modality coverage;
- component update counts;
- client and modality staleness;
- maximum opportunity-conditioned modality stall;
- redundancy among selected modality sets;
- greedy marginal gain and objective-term decomposition.

These are mechanism diagnostics specific to the proposed contribution. They
must not be presented as substitutes for the primary accuracy, convergence,
fairness, time, and communication metrics above.

## Comparison rules

All methods use identical model initialization, client splits, modality
manifest, training batches, evaluation cases, selected-client budget, and
matched seeds. Target thresholds are fixed before comparing methods. Test data
never supplies client utility or modality importance to a selector.

Primary references:

- Oort: https://www.usenix.org/conference/osdi21/presentation/lai
- Power-of-Choice: https://proceedings.mlr.press/v151/jee-cho22a.html
- DivFL: https://openreview.net/forum?id=nwKXyFvaUm
- FedScale: https://proceedings.mlr.press/v162/lai22a.html
- q-FFL: https://openreview.net/forum?id=ByexElSYDr
