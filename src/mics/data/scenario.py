"""Validated BraTS client scenarios for publication experiments.

A scenario binds three things that must not drift independently: client case
partitions, client modality masks, and the held-out global test set.  The
``materialized_config.json`` file produced with the PMCS splits is the
authoritative record for that binding.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brats import BRATS_MODALITIES, BraTSNpyDataset, read_case_ids

BRATS_PARTITIONS = ("train", "validation", "test")


def _normalise_client_id(value: str | int) -> str:
    text = str(value).strip().lower()
    if text.startswith("c"):
        text = text[1:]
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"Invalid client ID: {value!r}")
    return f"C{int(text)}"


def _normalise_modality(value: Any) -> str:
    name = str(value).strip().lower()
    if name not in BRATS_MODALITIES:
        raise ValueError(f"Unknown BraTS modality in scenario: {value!r}")
    return name


def _resolve_materialized_file(
    declared_path: Any,
    *,
    local_directory: Path,
) -> Path:
    """Resolve a recorded file while allowing a bundle to move between hosts."""

    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ValueError("A materialized split file path must be a non-empty string.")
    declared = Path(declared_path)
    candidates = [local_directory / declared.name]
    if not declared.is_absolute():
        candidates.append(local_directory / declared)
    candidates.append(declared)

    checked: list[Path] = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            return candidate.resolve()

    rendered = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Materialized split file was not found; checked: {rendered}"
    )


def _mask_values(raw_mask: Any, *, client_id: str) -> tuple[bool, ...]:
    if not isinstance(raw_mask, list) or len(raw_mask) != len(BRATS_MODALITIES):
        raise ValueError(
            f"Modality mask for {client_id} must contain "
            f"{len(BRATS_MODALITIES)} binary values."
        )
    if any(value not in (0, 1, False, True) for value in raw_mask):
        raise ValueError(f"Modality mask for {client_id} must be binary.")
    mask = tuple(bool(value) for value in raw_mask)
    if not any(mask):
        raise ValueError(f"Client {client_id} must have at least one modality.")
    return mask


def _duplicate_assignments(case_ids: list[str]) -> tuple[str, ...]:
    counts = Counter(case_ids)
    return tuple(sorted(case_id for case_id, count in counts.items() if count > 1))


def _preview(case_ids: set[str] | tuple[str, ...]) -> str:
    values = sorted(case_ids)
    preview = ", ".join(values[:5])
    if len(values) > 5:
        preview += f", and {len(values) - 5} more"
    return preview


@dataclass(frozen=True)
class BraTSClientScenario:
    """One client''s immutable case partitions and available modalities."""

    client_id: str
    modalities: tuple[str, ...]
    train_file: Path
    validation_file: Path
    test_file: Path
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    def split_file(self, partition: str) -> Path:
        name = str(partition).strip().lower()
        if name not in BRATS_PARTITIONS:
            raise ValueError(
                f"partition must be one of {BRATS_PARTITIONS}, found {partition!r}."
            )
        return getattr(self, f"{name}_file")

    def case_ids(self, partition: str) -> tuple[str, ...]:
        name = str(partition).strip().lower()
        if name not in BRATS_PARTITIONS:
            raise ValueError(
                f"partition must be one of {BRATS_PARTITIONS}, found {partition!r}."
            )
        return getattr(self, f"{name}_case_ids")


@dataclass(frozen=True)
class BraTSScenario:
    """A leakage-checked multimodal federated data scenario."""

    scenario_id: str
    data_seed: int
    schema_version: int
    config_path: Path
    clients: tuple[BraTSClientScenario, ...]
    global_test_file: Path
    global_test_case_ids: tuple[str, ...]
    dataset_case_ids: tuple[str, ...]

    def client(self, client_id: str | int) -> BraTSClientScenario:
        requested = _normalise_client_id(client_id)
        for client in self.clients:
            if client.client_id == requested:
                return client
        raise KeyError(f"Unknown client in scenario {self.scenario_id}: {requested}")

    @property
    def partition_counts(self) -> dict[str, int]:
        return {
            partition: sum(len(client.case_ids(partition)) for client in self.clients)
            for partition in BRATS_PARTITIONS
        }

    @property
    def modality_pool_sizes(self) -> dict[str, int]:
        return {
            modality: sum(modality in client.modalities for client in self.clients)
            for modality in BRATS_MODALITIES
        }

    @property
    def covered_case_count(self) -> int:
        covered = set(self.global_test_case_ids)
        for client in self.clients:
            for partition in BRATS_PARTITIONS:
                covered.update(client.case_ids(partition))
        return len(covered)


def _validate_case_assignments(
    clients: tuple[BraTSClientScenario, ...],
    global_test_case_ids: tuple[str, ...],
) -> set[str]:
    partition_sets: dict[str, set[str]] = {}
    for partition in BRATS_PARTITIONS:
        assignments = [
            case_id
            for client in clients
            for case_id in client.case_ids(partition)
        ]
        duplicates = _duplicate_assignments(assignments)
        if duplicates:
            raise ValueError(
                f"The {partition} partition assigns cases to multiple clients: "
                f"{_preview(duplicates)}"
            )
        partition_sets[partition] = set(assignments)

    global_set = set(global_test_case_ids)
    all_sets = {**partition_sets, "global_test": global_set}
    names = tuple(all_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = all_sets[left_name].intersection(all_sets[right_name])
            if overlap:
                raise ValueError(
                    f"Data leakage between {left_name} and {right_name}: "
                    f"{_preview(overlap)}"
                )
    return set().union(*all_sets.values())


def _dataset_case_ids(dataset_root: Path) -> set[str]:
    volume_root = dataset_root / "vol"
    segmentation_root = dataset_root / "seg"
    if not volume_root.is_dir():
        raise FileNotFoundError(f"Volume directory does not exist: {volume_root}")
    if not segmentation_root.is_dir():
        raise FileNotFoundError(
            f"Segmentation directory does not exist: {segmentation_root}"
        )
    volume_ids = {path.stem for path in volume_root.glob("*.npy")}
    segmentation_ids = {path.stem for path in segmentation_root.glob("*.npy")}
    missing_segmentations = volume_ids.difference(segmentation_ids)
    missing_volumes = segmentation_ids.difference(volume_ids)
    if missing_segmentations or missing_volumes:
        raise ValueError(
            "BraTS volume/segmentation IDs differ: "
            f"missing segmentations [{_preview(missing_segmentations)}]; "
            f"missing volumes [{_preview(missing_volumes)}]."
        )
    if not volume_ids:
        raise ValueError(f"No .npy cases were found under {dataset_root}.")
    return volume_ids


def load_brats_scenario(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> BraTSScenario:
    """Load and strictly validate a materialized BraTS client scenario.

    When ``dataset_root`` is supplied, the split bundle must cover the dataset
    exactly and every volume must have a matching segmentation.  Test cases
    are never exposed through a training dataset implicitly.
    """

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario configuration does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid scenario JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Scenario configuration must contain a JSON object.")

    schema_version = int(payload.get("schema_version", -1))
    if schema_version != 1:
        raise ValueError(f"Unsupported scenario schema_version: {schema_version}")
    scenario_id = str(payload.get("split_id", "")).strip()
    if not scenario_id:
        raise ValueError("Scenario configuration requires a non-empty split_id.")
    data_seed = int(payload.get("data_seed", -1))
    if data_seed < 0:
        raise ValueError("Scenario data_seed must be non-negative.")

    raw_order = payload.get("modality_order")
    if not isinstance(raw_order, list):
        raise ValueError("Scenario modality_order must be a list.")
    modality_order = tuple(_normalise_modality(value) for value in raw_order)
    if modality_order != BRATS_MODALITIES:
        raise ValueError(
            "Scenario modality_order must match the verified volume channel order "
            f"{BRATS_MODALITIES}, found {modality_order}."
        )

    raw_clients = payload.get("clients")
    raw_masks = payload.get("modality_masks")
    if not isinstance(raw_clients, dict) or not raw_clients:
        raise ValueError("Scenario clients must be a non-empty object.")
    if not isinstance(raw_masks, dict):
        raise ValueError("Scenario modality_masks must be an object.")

    normalised_masks: dict[str, Any] = {}
    for raw_client_id, raw_mask in raw_masks.items():
        client_id = _normalise_client_id(raw_client_id)
        if client_id in normalised_masks:
            raise ValueError(f"Duplicate modality mask for {client_id}.")
        normalised_masks[client_id] = raw_mask

    clients: list[BraTSClientScenario] = []
    seen_clients: set[str] = set()
    for raw_client_id, raw_client in raw_clients.items():
        client_id = _normalise_client_id(raw_client_id)
        if client_id in seen_clients:
            raise ValueError(f"Duplicate client definition for {client_id}.")
        seen_clients.add(client_id)
        if not isinstance(raw_client, dict) or not isinstance(
            raw_client.get("files"), dict
        ):
            raise ValueError(f"Client {client_id} requires a files object.")
        if client_id not in normalised_masks:
            raise ValueError(f"Client {client_id} has no modality mask.")

        mask = _mask_values(normalised_masks[client_id], client_id=client_id)
        modalities = tuple(
            modality
            for modality, available in zip(modality_order, mask, strict=True)
            if available
        )
        files: dict[str, Path] = {}
        cases: dict[str, tuple[str, ...]] = {}
        for partition in BRATS_PARTITIONS:
            if partition not in raw_client["files"]:
                raise ValueError(f"Client {client_id} has no {partition} file.")
            split_file = _resolve_materialized_file(
                raw_client["files"][partition],
                local_directory=path.parent,
            )
            files[partition] = split_file
            cases[partition] = read_case_ids(split_file)

        clients.append(
            BraTSClientScenario(
                client_id=client_id,
                modalities=modalities,
                train_file=files["train"],
                validation_file=files["validation"],
                test_file=files["test"],
                train_case_ids=cases["train"],
                validation_case_ids=cases["validation"],
                test_case_ids=cases["test"],
            )
        )

    extra_masks = set(normalised_masks).difference(seen_clients)
    if extra_masks:
        raise ValueError(f"Modality masks have unknown clients: {sorted(extra_masks)}")
    clients.sort(key=lambda client: int(client.client_id[1:]))
    client_tuple = tuple(clients)

    if "global_test_file" not in payload:
        raise ValueError("Scenario configuration requires global_test_file.")
    global_test_file = _resolve_materialized_file(
        payload["global_test_file"],
        local_directory=path.parent.parent,
    )
    global_test_case_ids = read_case_ids(global_test_file)
    covered = _validate_case_assignments(client_tuple, global_test_case_ids)

    expected_pools_raw = payload.get("expected_modality_pool_sizes")
    if not isinstance(expected_pools_raw, dict):
        raise ValueError("Scenario expected_modality_pool_sizes must be an object.")
    expected_pools = {
        _normalise_modality(name): int(size)
        for name, size in expected_pools_raw.items()
    }
    actual_pools = {
        modality: sum(modality in client.modalities for client in client_tuple)
        for modality in BRATS_MODALITIES
    }
    if expected_pools != actual_pools:
        raise ValueError(
            "Modality pool sizes do not match the materialized masks: "
            f"expected {expected_pools}, found {actual_pools}."
        )

    dataset_ids: set[str] = set()
    if dataset_root is not None:
        dataset_ids = _dataset_case_ids(Path(dataset_root))
        missing = dataset_ids.difference(covered)
        unknown = covered.difference(dataset_ids)
        if missing or unknown:
            raise ValueError(
                "Scenario does not exactly cover the dataset: "
                f"missing [{_preview(missing)}]; unknown [{_preview(unknown)}]."
            )

    return BraTSScenario(
        scenario_id=scenario_id,
        data_seed=data_seed,
        schema_version=schema_version,
        config_path=path.resolve(),
        clients=client_tuple,
        global_test_file=global_test_file,
        global_test_case_ids=global_test_case_ids,
        dataset_case_ids=tuple(sorted(dataset_ids)),
    )


def validate_matched_brats_scenarios(
    reference: BraTSScenario,
    comparison: BraTSScenario,
) -> tuple[str, ...]:
    """Validate a controlled comparison that changes only modality masks."""

    if reference.data_seed != comparison.data_seed:
        raise ValueError("Matched scenarios must use the same data_seed.")
    if reference.global_test_case_ids != comparison.global_test_case_ids:
        raise ValueError("Matched scenarios have different global-test cases.")
    if tuple(client.client_id for client in reference.clients) != tuple(
        client.client_id for client in comparison.clients
    ):
        raise ValueError("Matched scenarios have different client IDs.")

    changed_clients: list[str] = []
    for left, right in zip(reference.clients, comparison.clients, strict=True):
        for partition in BRATS_PARTITIONS:
            if left.case_ids(partition) != right.case_ids(partition):
                raise ValueError(
                    f"Matched scenarios differ in {left.client_id} {partition} cases."
                )
        if left.modalities != right.modalities:
            changed_clients.append(left.client_id)
    if not changed_clients:
        raise ValueError("Matched scenarios have identical modality masks.")
    return tuple(changed_clients)


def build_brats_scenario_dataset(
    dataset_root: str | Path,
    scenario: BraTSScenario,
    client_id: str | int,
    *,
    partition: str,
    patch_size: tuple[int, int, int] | None = None,
    training: bool | None = None,
    foreground_probability: float = 0.5,
    seed: int = 0,
) -> BraTSNpyDataset:
    """Build a client dataset without manually repeating its modality mask."""

    client = scenario.client(client_id)
    partition_name = str(partition).strip().lower()
    if partition_name not in BRATS_PARTITIONS:
        raise ValueError(
            f"partition must be one of {BRATS_PARTITIONS}, found {partition!r}."
        )
    is_training = partition_name == "train" if training is None else bool(training)
    if is_training and partition_name != "train":
        raise ValueError(
            "Validation and test partitions cannot be opened in training mode."
        )
    return BraTSNpyDataset(
        dataset_root,
        split_file=client.split_file(partition_name),
        available_modalities=client.modalities,
        patch_size=patch_size,
        training=is_training,
        foreground_probability=foreground_probability,
        seed=seed,
    )
