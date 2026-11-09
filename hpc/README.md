# HPC execution

The scheduler files are version-controlled templates. Inspect and adapt the
partition, GPU resource syntax, memory, wall time, modules, and CUDA environment
to the target cluster before submission.

The current model smoke job validates one GPU, imports the installed package,
runs the full unit suite, and performs a forward/backward pass:

    export MICS_PROJECT_ROOT=$HOME/MICS
    export MICS_CONDA_ENV=mics
    sbatch hpc/slurm/model_smoke.slurm

The uniform-random baseline smoke job validates the federated runner and
checkpoint schema on a materialized scenario. Its deliberately truncated local
training is not a model-quality experiment. Full studies must remove the batch
limit and use the subsequently validated whole-volume evaluation schedule.

`checkpoint_validation_smoke.slurm` consumes a completed run selected through
`MICS_RUN_DIR`. It evaluates only a small validation subset and refuses to use
the locked test partition.

The full MICS ablation array is defined by
`configs/mics_ablation_study_v1.json`. Each task resolves one immutable seed
tuple, generates its matched potential-outcome trace, resumes incomplete runs,
skips completed artifacts, and refuses protocol drift. Scheduler job and array
IDs are recorded as runtime metadata but never used as scientific seeds.

Long jobs must checkpoint enough information to reproduce an uninterrupted
trajectory: next round, global component states, client optimizer states,
selector state, random-generator states, best metrics, resolved configuration,
and source revision.
