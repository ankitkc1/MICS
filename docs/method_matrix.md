# Matched selector method matrix

The method-matrix pipeline validates Uniform Random, Power-of-Choice,
DivFL-style gradient facility location, and MICS-Impact under one controlled
federated experiment contract. Its smoke configuration verifies mechanics; it
does not provide evidence that one selector is more accurate than another.

## Matched invariants

The comparison report refuses to run unless every method has:

- the same scenario and checksummed client partition files;
- the same model architecture and initialization seed;
- the same round count, selected-client budget, and aggregation rule;
- the same local optimizer, batch cap, patch sampling, and data seed;
- the same checksummed availability/completion-potential trace;
- the same clean source revision and software/GPU environment; and
- the same checkpoint round, evaluation cases, modality masks, ROI, overlap,
  spacing, and held-out partition.

Data-loader and local-training randomness are derived independently for each
client and round. In particular, modality-dropout masks use a client-round
seed rather than the process-wide PyTorch stream. Thus, a client selected in
the same round by two methods receives the same stochastic local-training
stream regardless of which other clients either method selected.

## Cost accounting

The per-round method summaries include local training time, server selection
time, train-loss or gradient probe time, model downloads, probe uploads, and
training-update uploads. Power-of-Choice and DivFL therefore pay for the
statistics used to choose clients. A model downloaded for a probe and reused
for training in the same round is charged once.

Scientific MICS runs use a calibration-derived modality-importance record
created before the matched run. Its source checkpoint, checksum, cases, and one-time probe cost
are disclosed as an external selector input. That cost is not silently folded
into per-round communication or training time. A final paper should report
both the one-time calibration cost and the amortized per-round cost.

## Smoke interpretation

The Slurm smoke job truncates local training to one batch and evaluates only
one validation case per client. The report therefore sets
`interpretation_allowed` to false. Moreover, a subset importance record may
overlap the validation subset during integration testing. Before a scientific
run, materialize disjoint selector-calibration and model-validation roles and
freeze all thresholds, seeds, checkpoints, and evaluation schedules.

Run `hpc/slurm/four_method_matrix_smoke.slurm` after setting
`MICS_IMPORTANCE_OUTPUT` to a completed non-test importance JSON. The
job writes four immutable run directories, four evaluation records, the
shared system trace, and one guarded method-matrix report under `results/`.
