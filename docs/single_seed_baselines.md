Full single-seed comparison against the completed MICS-v2 result

The manifest `configs/mics_v2_single_seed_baselines_v1.json` fixes five external
comparators to the completed v2 experiment. It records model seed 101,
selection seed 1009, data seed 2003 and trace seed 3001. Every method uses 150
rounds, two clients per round, complete local epochs and all 24 validation
cases at the final checkpoint. Model, optimization, patch sampling and
inference arguments are derived from the historical matrix's recorded
configuration. All selection probes use the complete training partition.

| Array index | Method |
| --- | --- |
| 0 | Uniform Random |
| 1 | Power-of-Choice, four candidates |
| 2 | DivFL-style gradient facility location |
| 3 | MFedMC paper-defined low-loss client-policy adaptation |
| 4 | Oort training selection |

The reference matrix is
`results/studies/mics_v2_client_queue_pilot_v1/mics_v2_pilot_matrix.json`.
Its full-v2, Core-v2 and no-client-queue results are reused from revision
`3fb331ff1fe64cd65272e33c372e1d17f03ad9b4`. The new baseline code has its own
recorded source revision. Comparison output explicitly records that these
source revisions differ. It verifies the declared model, training, data,
trace, software/GPU runtime, evaluation cases and modality-mask contracts.
The existing same-revision comparison script retains its original checks.

The historical model was trained using Python 3.10.22, PyTorch 2.5.1, CUDA
12.1 and an NVIDIA RTX A4000. The full baseline job checks those recorded
runtime fields before training. Use the university's resource syntax to
request the appropriate GPU, and adapt the partition/account directives to
the site. The aggregation job uses a CPU task and no GPU resource, with the
example `gpu` partition retained for consistency with the other templates.

Install the updated package in the existing environment, validate it and
commit the reviewed code before submitting the frozen study. The runner
records the clean baseline revision, manifest hash and reference-file hash
in `study_binding.json`. A resume with changed source or inputs is rejected
by the study binding. The repository's existing tracking workflow also uses
clean source revisions for scientific provenance.

```bash
cd "$HOME/MICS"
conda activate mics
python -m pip install -e . --no-deps
python scripts/validate_install.py
python -m unittest discover -s tests -v
git status --short
```

The commands below assume the updated files have been committed in the
cluster checkout. The existing BraTS arrays, scientific scenario, reference
matrix and trace stay at their recorded cluster paths. Export
`MICS_DATA_ROOT` or `MICS_SCENARIO_CONFIG` when using an alternate data root
or scenario path. The latter still needs to match the recorded scenario
hashes and partition-file contract.

```bash
export MICS_PROJECT_ROOT="$HOME/MICS"
export MICS_CONDA_ENV=mics
export MICS_DATA_ROOT="$HOME/BraTS2020_npy"

bash -n hpc/slurm/mics_v2_baseline_comparison.slurm
bash -n hpc/slurm/mics_v2_baseline_aggregate.slurm

array_job=$(sbatch --parsable hpc/slurm/mics_v2_baseline_comparison.slurm)
array_job=${array_job%%;*}
summary_job=$(sbatch --parsable --dependency="afterok:$array_job" \
  hpc/slurm/mics_v2_baseline_aggregate.slurm)
summary_job=${summary_job%%;*}

printf 'Baseline array %s\nSummary %s\n' "$array_job" "$summary_job"
```

The default `--array=0-4%1` permits one baseline task to run at a time. Each
task holds a method-specific operating-system lock for training and
evaluation. Completed training is reused after validating its configuration
and source, incomplete runs resume through their existing checkpoints, and
existing evaluations are validated before reuse. The shared study binding
and final comparison have separate process locks.

```bash
squeue -u "$USER"
sacct -j "$array_job" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
tail -n 30 "mics-v2-baselines-${array_job}_0.out"
cat "mics-v2-baselines-${array_job}_0.err"
```

Per-method outputs are under
`results/studies/mics_v2_single_seed_baselines_v1/<method>`.
Each successful task ends with
`MICS-V2 FULL SINGLE-SEED BASELINE PASSED: <method>`.
The final job writes `single_seed_method_matrix.json` and
`single_seed_method_summary.csv` beside those run directories, with eight
method rows covering the five external baselines and three historical v2
variants. It reports both selection-count and successful-completion Jain
indices. The final success marker is
`MICS-V2 FULL SINGLE-SEED BASELINE COMPARISON PASSED`.

An individual failed task can be resubmitted with its original index after
its previous writer exits. For example, `sbatch --array=3
hpc/slurm/mics_v2_baseline_comparison.slurm` reruns the MFedMC task. If a
dependency job failed, submit a new summary dependency against the successful
replacement task or submit the summary job after all five methods pass.

The result scope is one training seed and complete held-out validation.
Test partitions remain outside this workflow. Report MFedMC and DivFL with
their declared shared-model adaptations and Oort with its loss/timing proxy.
Include the historical importance probe and calibration-model training cost
when making end-to-end efficiency comparisons. Method definitions and source
references are in `docs/reference_baselines.md`.
