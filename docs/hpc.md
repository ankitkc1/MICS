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
