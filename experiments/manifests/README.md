# Client scenario

The publication BraTS setting uses one generated `materialized_config.json` as
its authoritative client manifest. That file binds
client IDs, train/validation/local-test files, modality masks, modality order,
the data seed, and the held-out global-test file. Keeping these fields together
prevents a modality CSV from silently drifting away from its data partition.

The primary profile is modality-symmetric. Two clients have all four
modalities, while the remaining six clients receive each possible modality
pair exactly once. Consequently, every modality is available at five clients
and every modality pair co-occurs at three clients. A fixed seed assigns these
templates to client IDs independently of patient data. This preserves
heterogeneous overlap without making a particular modality artificially rare.

The MRI arrays remain at `MICS_DATA_ROOT`; no images or segmentations are copied
into this repository. Before a publication run, validate the materialized
scenario with `scripts/validate_brats_scenarios.py` and archive its files and
checksums with the resolved run configuration.
