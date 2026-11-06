# Client scenarios

The publication BraTS setting uses the generated `materialized_config.json` in
each scenario directory as its authoritative client manifest. That file binds
client IDs, train/validation/local-test files, modality masks, modality order,
the data seed, and the held-out global-test file. Keeping these fields together
prevents a modality CSV from silently drifting away from its data partition.

Scenario A and Scenario B intentionally use identical case assignments. They
are controlled modality-availability conditions: A has a large FLAIR pool and
a small T1ce pool (7 versus 2 clients), whereas B reverses those pool sizes
(2 versus 7 clients). T1 and T2 remain available at 5 and 4 clients,
respectively. They are not independent data splits and must not be treated as
repeated samples.

The MRI arrays remain at `MICS_DATA_ROOT`; no images or segmentations are copied
into this repository. Before a publication run, validate both materialized
scenarios with `scripts/validate_brats_scenarios.py` and archive their files and
checksums with the resolved run configuration.
