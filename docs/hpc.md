# HPC workflow

The intended execution environment is one Slurm-managed node containing one or
more GPUs. Multiple logical clients may be emulated by processes on that node;
this is a federated learning simulation, not a deployment across physically
separate hospitals.

## Initial setup

Copy or clone the repository to the cluster. Adapt
environments/environment-hpc.example.yml to the installed driver and CUDA
toolchain, then create the environment:

    conda env create -f environments/environment-hpc.example.yml
    conda activate mics
    python scripts/validate_install.py

Point the configuration to the dataset already present on the cluster:

    export MICS_DATA_ROOT=$HOME/BraTS2020_npy
    export MICS_SPLIT_ROOT=$HOME/pmcs/split/20_c8_heter_modalnum

No dataset copy is required.

`MICS_SPLIT_ROOT` above is retained only for the completed legacy integration
smoke tests. Publication experiments use one symmetric, balanced scenario.
Materialize it once from the validated SplitA case assignments; the SplitA
modality masks are explicitly discarded:

    export MICS_SPLIT_BUNDLE_ROOT=$HOME/pmcs/split/generated/data_seed_20260905
    python scripts/materialize_balanced_brats_scenario.py \
      --dataset-root "$MICS_DATA_ROOT" \
      --source-config "$MICS_SPLIT_BUNDLE_ROOT/splitA/materialized_config.json" \
      --output-directory "$MICS_SPLIT_BUNDLE_ROOT/balanced"

Then validate the resulting immutable bundle:

    python scripts/validate_brats_scenarios.py \
      --dataset-root "$MICS_DATA_ROOT" \
      --scenario-config "$MICS_SPLIT_BUNDLE_ROOT/balanced/materialized_config.json"

The command must report 368 covered cases, 222 training cases, 48 validation
cases, 48 local-test cases, and 50 global-test cases. Each modality pool must
contain five clients. The materializer assigns two complete-modality templates
and all six unique modality-pair templates using the fixed profile seed 1701,
independently of patient data. Validation must finish with
`MICS BRATS SCENARIO VALIDATION PASSED`.

Validate the real arrays and one existing client split before training:

    python scripts/smoke_brats_data.py \
      --client C1 \
      --modalities flair,t1ce,t1,t2 \
      --samples 2

Record the final resolved environment before a reported study:

    conda env export --no-builds > environment.resolved.yml
    python -m pip freeze > requirements.resolved.txt

These generated files should be archived with the study metadata.

## Scheduler validation

First edit only the cluster-specific scheduler directives in
hpc/slurm/model_smoke.slurm. Then:

    export MICS_PROJECT_ROOT=$HOME/MICS
    export MICS_CONDA_ENV=mics
    bash -n hpc/slurm/model_smoke.slurm
    sbatch hpc/slurm/model_smoke.slurm

The output must end with MICS HPC SMOKE TEST PASSED before a training job is
created.

After the synthetic model check, run one forward/backward optimizer step on a
real foreground-containing BraTS patch:

    bash -n hpc/slurm/brats_gpu_smoke.slurm
    sbatch hpc/slurm/brats_gpu_smoke.slurm

The first run intentionally uses four base channels so architecture, loss,
gradient, and memory failures can be diagnosed safely. If it passes with
adequate memory headroom, repeat with eight and then sixteen base channels:

    MICS_BASE_CHANNELS=8 sbatch hpc/slurm/brats_gpu_smoke.slurm
    MICS_BASE_CHANNELS=16 sbatch hpc/slurm/brats_gpu_smoke.slurm

The Dice values printed by this job come from one patch and a randomly
initialized model. They are diagnostic only and must not be reported as
validation or test performance. Reported BraTS accuracy must be computed on
held-out whole cases after sliding-window reconstruction.

Once the reference channel width passes, validate reusable local optimization
for both complete and single-modality clients:

    bash -n hpc/slurm/local_training_smoke.slurm
    sbatch hpc/slurm/local_training_smoke.slurm

This job trains C1 and C6 for two patches each. It must show that C1 exports all
four encoders while C6 exports only encoder:t1ce plus the shared fusion,
decoder, and head components.

After local training passes, validate two complete communication rounds:

    bash -n hpc/slurm/federated_rounds_smoke.slurm
    sbatch hpc/slurm/federated_rounds_smoke.slurm

Both clients must start each round from the same pre-round global model. The
T1ce encoder and shared components receive two updates; the other modality
encoders receive one update. Every global component must change in both rounds.

Install the SciPy metric dependency once after pulling the whole-volume
evaluation update:

    conda install -n mics -c conda-forge "scipy>=1.10" -y

Then validate complete-case reconstruction:

    bash -n hpc/slurm/whole_volume_eval_smoke.slurm
    sbatch hpc/slurm/whole_volume_eval_smoke.slurm

This diagnostic uses one C1 training case only to validate shape restoration,
window coverage, blending, Dice, and HD95. Journal results must instead use the
fixed held-out validation and test case lists.

After the balanced scenario passes validation, run the uniform-random training
pipeline:

    bash -n hpc/slurm/uniform_baseline_smoke.slurm
    sbatch hpc/slurm/uniform_baseline_smoke.slurm

This job performs two rounds with two selected clients and two patches per
client. It validates scenario-driven modality masks, independent uniform
selection, component-wise aggregation, communication accounting, JSONL metric
records, and a resumable checkpoint. Because local training is truncated, its
loss and patch Dice values are integration diagnostics and not manuscript
performance.

Validate checkpoint-based whole-volume evaluation using the completed balanced
smoke run. Supply its absolute run directory:

    export MICS_RUN_DIR=$HOME/MICS/results/uniform-balanced-smoke-<job-id>
    bash -n hpc/slurm/checkpoint_validation_smoke.slurm
    sbatch hpc/slurm/checkpoint_validation_smoke.slurm

This job evaluates one validation case from each of the eight clients using
that client's recorded modality mask. It verifies checkpoint reconstruction,
sliding-window inference, WT/TC/ET Dice and HD95, per-client reductions,
macro-client and sample-weighted reductions, timing, and memory accounting. The
eight-case subset is an integration check, not a performance result.

After the individual selector jobs pass, validate the complete matched
four-method path. Point the job to an existing validation-only modality
importance record:

    export MICS_IMPORTANCE_OUTPUT=/absolute/path/to/modality_importance.json
    bash -n hpc/slurm/four_method_matrix_smoke.slurm
    JOB_ID=$(sbatch --parsable hpc/slurm/four_method_matrix_smoke.slurm)
    echo "JOB_ID=$JOB_ID"

Inspect the complete job by its scheduler ID:

    sacct -j "$JOB_ID" --allocations \
      --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
    cat "mics-four-way-${JOB_ID}.out"
    cat "mics-four-way-${JOB_ID}.err"
    cat "results/four-way-method-matrix-smoke-${JOB_ID}.json"

The job generates one neutral availability/completion trace; runs Uniform
Random, Power-of-Choice, DivFL-style, and MICS-Impact from the same model seed;
evaluates the same validation case for every client; and produces one guarded
matrix report. The report must remain marked as ineligible for scientific
interpretation because training and validation are truncated.

Once the four-method path passes, validate the registered MICS ablations:

    export MICS_IMPORTANCE_OUTPUT=/absolute/path/to/modality_importance.json
    JOB_ID=$(sbatch --parsable hpc/slurm/mics_ablation_matrix_smoke.slurm)
    echo "JOB_ID=$JOB_ID"

Inspect the ablation job and its guarded report:

    sacct -j "$JOB_ID" --allocations \
      --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
    cat "mics-ablation-${JOB_ID}.out"
    cat "mics-ablation-${JOB_ID}.err"
    cat "results/mics-ablation-matrix-smoke-${JOB_ID}.json"

This job holds the trace, model, training, and cases fixed while separately
removing learned importance, queue feedback, or diminishing returns, and while
replacing learned importance with uniform component weights.

Full training retains `last.pt` every round and archives a checkpoint every ten
rounds, plus the final round. Full validation uses all 48 validation cases at
those preregistered checkpoints. Local-test and global-test evaluation remain
locked until the selection method and hyperparameters are frozen.

## Training and array studies

The future training entry point should accept a complete experiment
configuration, run ID, and output root. Array tasks should differ only in
declared factors such as selector, client budget, split, and seed.

Do not derive scientific seeds from scheduler IDs. Use a pre-registered table
of seeds so failed and resubmitted jobs reproduce the same trajectory.

Before local training starts, a job should print and save its resolved
configuration, source revision, device inventory, client manifest path, and
resume decision. After each communication round it should atomically write
metrics and a last checkpoint.

## Failure handling

A failed job must keep its logs and last valid checkpoint in the run directory.
Resume into the same run only after verifying configuration compatibility and
the stored next-round value. Code changes after failure require either a new
run ID or an explicit provenance record linking the continuation to both source
revisions.
