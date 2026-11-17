# Oort sampler repair and completed-baseline reuse

Array 53102 completed Uniform Random, PoC, DivFL-style and MFedMC adaptation.
Oort completed rounds 0–57 and failed while planning round 58 because its
cutoff left fewer exploitation candidates than the requested sample size.
Zero-utility failed clients can have negative normalized scores, which makes
that small-pool case possible after every client has been explored.

The repaired sampler retains positive exploitation weights and caps the
exploitation quota at that pool's size. Oort's existing exploration and
eligible-client fallback fill the remaining budget without replacement.
Replaying the uploaded history reproduces the original error, preserves all
58 prefix selections and RNG states under the repair, and completes the
failing selection. All 247 availability subsets with at least two clients
were checked at that failure state.

Use `configs/mics_v2_single_seed_baselines_oort_fix_v1.json` for recovery.
It starts a new Oort run from round zero, so the corrected 150-round result
has one source revision. The old failed Oort directory is retained. The four
successful baselines remain in their original directory and are read during
comparison; they are not trained or evaluated again.

Before reuse, the runner verifies the old study binding, complete training
and validation contracts, method identities and original source revision
`07d8d867dc85572f462262c7b0cb961a24ee5d61`. Git ancestry and changed execution
paths must show that only the Oort sampler and comparison orchestration
changed under `src/` and `scripts/`. Shared training, model, data, other
selectors and evaluation source changes are rejected. The new study binding
freezes hashes of each reused run's configuration, metadata, status, round
metrics and final validation artifact. Default studies retain their original
same-baseline-revision binding behavior.

The new comparison keeps the original source revision on the four reused
rows and the corrected revision on Oort. It records baseline revision
differences and includes the reuse checks and artifact hashes in its
provenance. The historical MICS-v2 reference is still reused at its original
revision. No old metadata or study-binding file is rewritten.

## Submit only the repaired Oort task

Run in the university session after pulling the committed repair. Keep the
existing environment and data paths used by array 53102.

```bash
cd "$HOME/MICS" &&
git pull --ff-only origin development &&
conda activate mics &&
export MICS_BASELINE_MANIFEST="$HOME/MICS/configs/mics_v2_single_seed_baselines_oort_fix_v1.json" &&
if oort_job=$(command sbatch --parsable \
    --gres=gpu:rtxa4000:1 --array=4 \
    hpc/slurm/mics_v2_baseline_comparison.slurm); then
    oort_job=${oort_job%%;*}
    printf 'Oort repair array: %s\n' "$oort_job"
    command sbatch --kill-on-invalid-dep=yes \
        --dependency="afterok:$oort_job" \
        hpc/slurm/mics_v2_baseline_aggregate.slurm
fi
```

Task index 4 remains Oort. No specific node is pinned, allowing any GPU of
the original `rtxa4000` type. If a newly started repaired Oort run is
interrupted, resubmit index 4 at the same clean revision and manifest to use
the existing checkpoint recovery.

```bash
squeue -u "$USER"
tail -n 30 "mics-v2-baselines-${oort_job}_4.out"
tail -n 60 "mics-v2-baselines-${oort_job}_4.err"
```

The new result directory is
`results/studies/mics_v2_single_seed_baselines_oort_fix_v1/`.
The summary writes `single_seed_method_matrix.json` and
`single_seed_method_summary.csv` there, with the usual eight method rows.
The repaired study rejects missing or changed reused artifacts, incompatible
source, incomplete validation, or drift in seeds, trace, training or runtime.
