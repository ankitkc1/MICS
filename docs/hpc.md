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

    export MICS_DATA_ROOT=$HOME/FedMEPD/data/BraTS2020_npy

No dataset copy is required.

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
