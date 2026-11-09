# MICS ablation study

The primary ablation study changes one selector mechanism at a time while
holding the model, client budget, local training, component-wise aggregation,
system trace, seeds, and evaluation cases fixed.

## Registered variants

`mics_impact` is the complete method. For modality `m`, its fixed-round weight
is the sum of the component-service queue and the scaled calibration-derived
importance. The weighted service utility saturates with represented
contributors, so a second client covering an already represented modality has
a smaller marginal gain.

`mics_core` sets the impact coefficient to zero. Selection is driven by
component-service deficits, subset coverage, and the shared completion model.
This isolates the contribution of calibration-derived modality importance.

`mics_impact_no_queue` sets the component-queue coefficient to zero. Deficit
queues continue to be updated and recorded as counterfactual diagnostics, but
they cannot influence selection. This isolates adaptive participation history
from impact-weighted subset coverage.

`mics_impact_modular` replaces saturating subset service with the sum of
individual contributor probabilities. It retains the same queue and impact
weights, but overlapping clients receive repeated credit within a round. This
isolates the diminishing-returns/subset-interaction mechanism. The variant is
modular and is not presented as the proposed overlap-aware MICS objective.
Mathematically, a modular function is a special case of a submodular function;
the removed mechanism is overlap-sensitive diminishing returns, not membership
in the broad submodular function class.

`mics_impact_uniform` replaces calibration-derived modality weights with
`1 / |M|` while retaining queues and saturating subset service. It isolates
whether learned modality impact adds value beyond symmetric component
coverage.

## Interpretation rules

The no-queue and modular variants are mechanism ablations, not competing
published algorithms. They must be labelled by their full names in tables and
figures. MICS-Core must not be described as removing submodularity: it removes
only the impact term. Uniform importance has no modality-importance probe cost;
the full, no-queue, and modular variants share the same immutable importance
record and disclose its one-time cost.

The smoke job uses four rounds, one local batch, and one validation case per
client. It validates plumbing only. The scientific protocol instead uses
complete local training, five preregistered seed tuples, full validation at
every archived checkpoint, and a disjoint selector-calibration partition.

Run `hpc/slurm/mics_ablation_matrix_smoke.slurm` after setting
`MICS_IMPORTANCE_OUTPUT`. The job creates one shared neutral system trace, five
tracked runs, five evaluation records, and one guarded matrix report.

## Frozen full-study protocol

`configs/mics_ablation_study_v1.json` is the immutable protocol for the first
complete ablation study. The original 48 validation cases are divided by a
label-independent hash rule into 24 selector-calibration cases and 24 model
validation cases, three of each per client. Training, local-test, and global-
test assignments are unchanged. Modality importance is estimated once from a
separately tracked uniform-selection pilot and the complete calibration
partition. All five ablation variants then use fresh, matched model
initializations; the three learned-importance variants consume the same fixed
importance record.

Each seed task trains the five variants against the same pre-generated
availability/completion trace and evaluates all 24 validation cases every ten
rounds. `scripts/aggregate_ablation_study.py` accepts only completed,
scientifically interpretable matrices with identical protocol hashes,
scenarios, importance records, and source revisions. It reports means, sample
standard deviations, two-sided 95% Student-t intervals, and within-seed paired
differences from full MICS-Impact. These intervals quantify variability across
the five declared runs; they do not replace a power analysis or justify claims
from non-significant differences.
