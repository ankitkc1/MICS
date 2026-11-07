# MICS selector contract

The implemented selector follows the cardinality-constrained MICS-Core and
MICS-Impact formulation in the manuscript. It does not yet implement cached
update facility location or the knapsack extension.

## Fixed-round score

For every modality `m`, the selector maintains a non-negative service-deficit
queue `D[m]`. With an optional validation-derived importance vector `psi` and
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
The current deterministic simulator uses the reliability-free value `p=1`;
dropout experiments must enable and evaluate the completion model explicitly.

## Service-target calibration

Small static cross-silo scenarios use an exact linear program over all
cardinality-feasible client sets. It finds the maximum common service scale in
the declared priority direction. Operational targets multiply that boundary by
the greedy oracle factor and a safety margin. The resulting certificate is for
the supplied static availability and completion model; it is not a guarantee
under an unmodelled nonstationary process.

Equal component priorities are the default. Validation-derived modality impact
does not alter the long-run target direction, preventing importance from being
counted twice and preventing a currently low-impact component from losing its
service protection.

## Current scope

- implemented: pure and truncated-linear service, exact static calibration,
  component and optional client queues, completion-aware expectations,
  MICS-Core/Impact cardinality greedy, full state checkpointing, and the
  shared tracked federated training loop used by the uniform baseline;
- not yet implemented: cached-update facility location, the knapsack-specific
  partial-enumeration oracle, stochastic availability generation, or IPW
  learning correction;
- selector-only smoke results are integration evidence, not segmentation or
  convergence results.
