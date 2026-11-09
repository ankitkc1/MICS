# MICS selector contract

The implemented selector follows the cardinality-constrained MICS-Core and
MICS-Impact formulation in the manuscript. It does not yet implement cached
update facility location or the knapsack extension.

## Fixed-round score

For every modality `m`, the selector maintains a non-negative service-deficit
queue `D[m]`. With an optional calibration-derived importance vector `psi` and
impact coefficient `V_I`, the effective component weight is

    W[m] = D[m] + V_I * psi[m].

The value of a candidate client set is the sum of `W[m]` times the expected
service delivered to each component, plus optional client-queue terms. Pure
coverage uses `1 - product(1 - p[k])` over selected clients possessing the
modality. Truncated-linear service computes `E[min(H[m], L[m])] / L[m]` exactly
under independent Bernoulli completion probabilities. Both yield monotone
submodular set functions for the fixed pre-decision state.

The cardinality oracle starts with the empty set and repeatedly adds the client
with largest marginal gain. Ties within a declared numerical tolerance are
broken uniformly using the selector's independent, checkpointed random stream.
Every greedy step records all candidate marginal gains, tied maximisers, the
chosen client, the resulting objective value, and its term decomposition.

## Queue and reliability updates

After training, queues use realised completion outcomes:

    D[m] <- max(D[m] + r[m] - Y[m], 0).

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
and a safety margin. The certificate records the conditioning probability,
number of availability states, number of state--action pairs, and sparse
optimal mixture. It applies only to the supplied stationary availability and
completion model; distribution-shift experiments require separate analysis.

Equal component priorities are the default. Validation-derived modality impact
does not alter the long-run target direction, preventing importance from being
counted twice and preventing a currently low-impact component from losing its
service protection.

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
