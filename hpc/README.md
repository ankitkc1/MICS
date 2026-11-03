# HPC execution

The scheduler files are version-controlled templates. Inspect and adapt the
partition, GPU resource syntax, memory, wall time, modules, and CUDA environment
to the target cluster before submission.

The current model smoke job validates one GPU, imports the installed package,
runs the full unit suite, and performs a forward/backward pass:

    export MICS_PROJECT_ROOT=$HOME/MICS
    export MICS_CONDA_ENV=mics
    sbatch hpc/slurm/model_smoke.slurm

Do not create the full training job until the federated runner and checkpoint
schema are implemented. A training script submitted before those contracts are
stable would produce non-comparable results.

For future array studies, each task must receive an immutable experiment
configuration, seed, output directory, and run identifier. The task must refuse
to overwrite a completed run. Scheduler job ID and array task ID must be
recorded in run metadata, not used as scientific random seeds.

Long jobs must checkpoint enough information to reproduce an uninterrupted
trajectory: next round, global component states, client optimizer states,
selector state, random-generator states, best metrics, resolved configuration,
and source revision.
