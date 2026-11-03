"""BraTS NumPy datasets with named modalities and deterministic patch sampling."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

BRATS_MODALITIES = ("flair", "t1ce", "t1", "t2")
BRATS_LABELS = frozenset({0, 1, 2, 3})


def read_case_ids(path: str | Path) -> tuple[str, ...]:
    """Read a one-column case list, with or without a case_id header."""

    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Client split file does not exist: {split_path}")

    case_ids: list[str] = []
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 1:
                raise ValueError(
                    f"{split_path}:{row_number} must contain exactly one case ID."
                )
            case_id = row[0].strip()
            if not case_ids and case_id.lower() in {"case", "case_id", "subject_id"}:
                continue
            if not case_id or case_id in {".", ".."}:
                raise ValueError(f"Invalid case ID at {split_path}:{row_number}.")
            if any(separator in case_id for separator in ("/", "\\")):
                raise ValueError(
                    f"Case ID {case_id!r} must not contain a path separator."
                )
            case_ids.append(case_id)

    if not case_ids:
        raise ValueError(f"Client split file is empty: {split_path}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Client split contains duplicate case IDs: {split_path}")
    return tuple(case_ids)


def resolve_client_split_file(
    split_root: str | Path,
    client_id: str | int,
) -> Path:
    """Resolve C1, c1, or 1 to the existing c1.csv convention."""

    value = str(client_id).strip().lower()
    if value.startswith("c"):
        value = value[1:]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"Invalid client ID: {client_id!r}")
    path = Path(split_root) / f"c{int(value)}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Client split file does not exist: {path}")
    return path


def _normalise_patch_size(
    patch_size: Sequence[int] | None,
) -> tuple[int, int, int] | None:
    if patch_size is None:
        return None
    result = tuple(int(value) for value in patch_size)
    if len(result) != 3 or any(value <= 0 for value in result):
        raise ValueError("patch_size must contain three positive integers.")
    return result


def _pad_spatial(
    volume: np.ndarray,
    target: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    spatial = volume.shape[:3]
    padding: list[tuple[int, int]] = []
    for current, required in zip(spatial, patch_size, strict=True):
        total = max(required - current, 0)
        before = total // 2
        padding.append((before, total - before))
    if any(before or after for before, after in padding):
        volume = np.pad(volume, (*padding, (0, 0)), mode="constant")
        target = np.pad(target, padding, mode="constant")
    return volume, target


def _extract_patch(
    volume: np.ndarray,
    target: np.ndarray,
    patch_size: tuple[int, int, int],
    *,
    training: bool,
    foreground_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    volume, target = _pad_spatial(volume, target, patch_size)
    spatial = volume.shape[:3]
    maximum = np.asarray(spatial) - np.asarray(patch_size)

    if training:
        foreground = np.argwhere(target > 0)
        use_foreground = (
            foreground.size > 0 and rng.random() < foreground_probability
        )
        if use_foreground:
            center = foreground[int(rng.integers(len(foreground)))]
            start = np.clip(
                center - np.asarray(patch_size) // 2,
                0,
                maximum,
            )
        else:
            start = np.asarray(
                [
                    int(rng.integers(limit + 1)) if limit > 0 else 0
                    for limit in maximum
                ]
            )
    else:
        start = maximum // 2

    stop = start + np.asarray(patch_size)
    slices = tuple(
        slice(int(begin), int(end))
        for begin, end in zip(start, stop, strict=True)
    )
    return volume[slices + (slice(None),)], target[slices]


class BraTSNpyDataset(Dataset[dict[str, Any]]):
    """Load preprocessed BraTS arrays from vol and seg subdirectories.

    The verified storage layout is [H, W, D, M] with modality order
    [FLAIR, T1ce, T1, T2]. Intensities are already z-score normalised, so this
    dataset performs no second intensity normalisation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        case_ids: Iterable[str] | None = None,
        split_file: str | Path | None = None,
        available_modalities: Iterable[str] = BRATS_MODALITIES,
        patch_size: Sequence[int] | None = None,
        training: bool = False,
        foreground_probability: float = 0.5,
        seed: int = 0,
        validate_files: bool = True,
    ) -> None:
        if (case_ids is None) == (split_file is None):
            raise ValueError("Provide exactly one of case_ids or split_file.")
        self.root = Path(root)
        self.volume_root = self.root / "vol"
        self.segmentation_root = self.root / "seg"
        self.case_ids = (
            tuple(str(value) for value in case_ids)
            if case_ids is not None
            else read_case_ids(split_file)
        )
        if not self.case_ids:
            raise ValueError("case_ids cannot be empty.")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("case_ids cannot contain duplicates.")

        requested = set(available_modalities)
        unknown = requested.difference(BRATS_MODALITIES)
        if unknown:
            raise ValueError(f"Unknown BraTS modalities: {sorted(unknown)}")
        if not requested:
            raise ValueError("At least one modality must be available.")
        self.available_modalities = tuple(
            name for name in BRATS_MODALITIES if name in requested
        )
        self.patch_size = _normalise_patch_size(patch_size)
        self.training = bool(training)
        self.foreground_probability = float(foreground_probability)
        self.seed = int(seed)
        self.epoch = 0

        if not 0.0 <= self.foreground_probability <= 1.0:
            raise ValueError("foreground_probability must lie in [0, 1].")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if validate_files:
            self._validate_paths()

    def _validate_paths(self) -> None:
        if not self.volume_root.is_dir():
            raise FileNotFoundError(f"Volume directory does not exist: {self.volume_root}")
        if not self.segmentation_root.is_dir():
            raise FileNotFoundError(
                f"Segmentation directory does not exist: {self.segmentation_root}"
            )
        missing: list[str] = []
        for case_id in self.case_ids:
            if not self._volume_path(case_id).is_file():
                missing.append(str(self._volume_path(case_id)))
            if not self._segmentation_path(case_id).is_file():
                missing.append(str(self._segmentation_path(case_id)))
        if missing:
            preview = ", ".join(missing[:4])
            suffix = "" if len(missing) <= 4 else f" and {len(missing) - 4} more"
            raise FileNotFoundError(f"Missing BraTS arrays: {preview}{suffix}")

    def _volume_path(self, case_id: str) -> Path:
        return self.volume_root / f"{case_id}.npy"

    def _segmentation_path(self, case_id: str) -> Path:
        return self.segmentation_root / f"{case_id}.npy"

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic training-patch sampling."""

        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id = self.case_ids[index]
        volume = np.load(self._volume_path(case_id), mmap_mode="r")
        target = np.load(self._segmentation_path(case_id), mmap_mode="r")

        if volume.ndim != 4 or volume.shape[-1] != len(BRATS_MODALITIES):
            raise ValueError(
                f"{case_id} volume must have shape [H, W, D, 4], "
                f"found {volume.shape}."
            )
        if target.shape != volume.shape[:3]:
            raise ValueError(
                f"{case_id} volume/segmentation mismatch: "
                f"{volume.shape[:3]} versus {target.shape}."
            )
        if volume.dtype != np.float32:
            raise ValueError(f"{case_id} volume must be float32, found {volume.dtype}.")
        if target.dtype != np.uint8:
            raise ValueError(
                f"{case_id} segmentation must be uint8, found {target.dtype}."
            )

        volume_array = np.asarray(volume)
        target_array = np.asarray(target)
        if target_array.size and int(target_array.max()) > max(BRATS_LABELS):
            raise ValueError(
                f"{case_id} contains a label greater than {max(BRATS_LABELS)}."
            )

        if self.patch_size is not None:
            seed_sequence = np.random.SeedSequence((self.seed, self.epoch, index))
            rng = np.random.default_rng(seed_sequence)
            volume_array, target_array = _extract_patch(
                volume_array,
                target_array,
                self.patch_size,
                training=self.training,
                foreground_probability=self.foreground_probability,
                rng=rng,
            )

        inputs: dict[str, torch.Tensor] = {}
        for modality_index, name in enumerate(BRATS_MODALITIES):
            if name not in self.available_modalities:
                continue
            channel = np.ascontiguousarray(
                volume_array[..., modality_index].transpose(2, 0, 1)
            )
            inputs[name] = torch.from_numpy(channel).unsqueeze(0)

        segmentation = np.ascontiguousarray(target_array.transpose(2, 0, 1))
        availability = torch.tensor(
            [name in self.available_modalities for name in BRATS_MODALITIES],
            dtype=torch.bool,
        )
        return {
            "inputs": inputs,
            "target": torch.from_numpy(segmentation.astype(np.int64, copy=False)),
            "availability": availability,
            "case_id": case_id,
        }


def build_brats_client_dataset(
    dataset_root: str | Path,
    split_root: str | Path,
    client_id: str | int,
    *,
    available_modalities: Iterable[str],
    patch_size: Sequence[int] | None = None,
    training: bool = False,
    foreground_probability: float = 0.5,
    seed: int = 0,
) -> BraTSNpyDataset:
    """Create a client dataset using the existing c1.csv, ..., c8.csv files."""

    return BraTSNpyDataset(
        dataset_root,
        split_file=resolve_client_split_file(split_root, client_id),
        available_modalities=available_modalities,
        patch_size=patch_size,
        training=training,
        foreground_probability=foreground_probability,
        seed=seed,
    )
