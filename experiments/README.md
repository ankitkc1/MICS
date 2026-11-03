# Experiment registry

This directory contains study definitions and simple client manifests. It is
not a location for raw results or exploratory scripts.

A journal experiment matrix should specify:

- dataset and heterogeneous client split;
- model and fusion method;
- selection policy and client budget;
- local training and communication rounds;
- evaluation interval;
- seeds;
- primary and secondary outcomes;
- required ablations;
- stopping and checkpoint rules.

At minimum, compare full participation, uniform random selection, and the final
MICS method under identical resource budgets. Additional baselines should be
implemented through the same ClientSelector interface and must disclose any
extra probing or communication.

The current BraTS manifest records the eight logical clients and their modality
availability. The future data loader will connect those clients to the dataset
already present on the HPC.
