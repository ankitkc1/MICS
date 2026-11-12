# MICS selector contract

The implemented selector follows the cardinality-constrained MICS-Core and
MICS-Impact formulation in the manuscript. Version 1 uses component-service
queues only. Version 2 additionally protects calibrated successful-client
participation through client-specific virtual queues. It does not yet implement
cached-update facility location or the knapsack extension.

## Fixed-round score

For every modality `m`, the selector maintains a non-negative service-deficit
queue `D[m]`. With an optional calibration-derived importance vector `psi` and
impact coefficient `V_I`, the effective component weight is

    W[m] = D[m] + V_I * psi[m].

For MICS-v2, client `k` also has a non-negative participation-deficit queue
`Z[k]`. The fixed-round value of candidate set `S` is

    F_t(S) = sum_m W_t[m] g_m(S) + sum_{k in S} Z_t[k] p_t[k].

Here, `p_t[k]` is the pre-decision completion estimate and `g_m(S)` is expected
component service. Pure coverage uses `1 - product(1 - p_t[k])` over selected
clients possessing the modality. Truncated-linear service computes
`E[min(H[m], L[m])] / L[m]` exactly under independent Bernoulli completion
probabilities. The client-queue term is non-negative and modular. Consequently,
its addition preserves monotonicity and submodularity for the fixed
pre-decision state, so cardinality greedy retains its standard `1 - 1/e`
approximation guarantee for that round's objective.

The cardinality oracle starts with the empty set and repeatedly adds the client
with largest marginal gain. Ties within a declared numerical tolerance are
broken uniformly using the selector's independent, checkpointed random stream.
Every greedy step records all candidate marginal gains, tied maximisers, the
chosen client, the resulting objective value, and its term decomposition.

## Queue and reliability updates

After training, queues use realised completion outcomes:

    D[m] <- max(D[m] + r[m] - Y[m], 0).

MICS-v2 applies the analogous update to every calibrated client:

    Z[k] <- max(Z[k] + rho[k] - I[k completed the round], 0).

Thus, a failed selected client does not repay participation debt, and an
unavailable or unselected client accrues debt according to its unconditional
calibrated target. The target is feasible under the declared availability and
completion distribution; it is not a requirement that every client be selected
in every finite window.

Predicted completion probabilities never repay debt. When reliability-aware
selection is enabled, each client's probability is the beta-binomial posterior
mean `(successes + a) / (attempts + a + b)` based only on earlier selections.
The tracked simulator can instead reveal outcomes from a pre-generated,
selection-independent completion trace. Reliability-aware experiments estimate
client probabilities only from preceding selected attempts and must report the
estimator state and its additional assumptions.

## Service-target calibration

Small static cross-silo scenarios use an exact linear program over all
cardinality-feasible client sets. Under stochastic availability, the
availability-aware program enumerates every nonzero availability state and all
feasible client subsets in that state. State probabilities follow the declared
independent Bernoulli model conditioned on the configured minimum available
client count, which cannot be smaller than the selection budget. Its variables
form a state-contingent selection mixture, and its objective finds the maximum
common expected-service scale in the declared priority direction. It uses the
declared distribution rather than realised future trace rounds, preventing
look-ahead leakage.

Operational targets multiply the capacity boundary by the greedy oracle factor
and a safety margin. Component and client targets are calibrated separately but
against the same budget, availability model, completion model, oracle factor,
and safety margin. Client calibration encodes each client as a distinct
unit-service item and solves the same availability-conditioned max--min linear
program with equal client priorities. The certificate records the conditioning
probability, number of availability states, number of state--action pairs, and
sparse optimal mixture. It applies only to the supplied stationary availability
and completion model; distribution-shift experiments require separate analysis.

Equal component priorities are the default. Validation-derived modality impact
does not alter the long-run target direction, preventing importance from being
counted twice and preventing a currently low-impact component from losing its
service protection.

## Versioned ablations

`mics_v2_impact` and `mics_v2_core` use the identical calibrated client targets
and identical client-queue dynamics. Their only intended difference is whether
the validation-derived component-impact term is present. This makes the impact
comparison structurally matched.

`mics_v2_impact_no_client_queue` passes no client targets to the selector. It is
therefore mechanically equivalent to v1 MICS-Impact for matched seeds, traces,
importance weights, and tie breaking. It is retained under a distinct method
identifier so the new client-participation mechanism can be isolated without
rewriting or overwriting v1 evidence.

## Current scope

- implemented: pure and truncated-linear service, exact static and
  availability-conditioned calibration, component and optional client queues,
  completion-aware expectations, MICS-Core/Impact cardinality greedy,
  selection-independent availability/completion traces, full state
  checkpointing, and the shared tracked federated training loop used by the
  uniform baseline;
- not yet implemented: cached-update facility location, the knapsack-specific
  partial-enumeration oracle or IPW learning correction;
- selector-only smoke results are integration evidence, not segmentation or
  convergence results.
