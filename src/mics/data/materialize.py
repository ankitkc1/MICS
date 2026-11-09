"""Materialize portable BraTS scenarios from validated case assignments."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .brats import BRATS_MODALITIES
from .profiles import (
    BRATS_BALANCED_PROFILE_SEED,
    balanced_brats_modalities,
    validate_balanced_brats_modalities,
)
from .scenario import BRATS_PARTITIONS, BraTSScenario, load_brats_scenario

BRATS_CALIBRATION_SPLIT_SEED = 2701


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_fingerprint(
    source: BraTSScenario,
    profile: dict[str, tuple[str, ...]],
) -> str:
    payload = {
        "data_seed": source.data_seed,
        "clients": {
            client.client_id: {
                "modalities": list(profile[client.client_id]),
                **{
                    partition: list(client.case_ids(partition))
                    for partition in source.partitions
                },
            }
            for client in source.clients
        },
        "global_test": list(source.global_test_case_ids),
    }
    canonical = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _case_order_key(case_id: str, *, client_id: str, seed: int) -> str:
    value = f"{seed}:{client_id}:{case_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _calibration_split(
    case_ids: tuple[str, ...],
    *,
    client_id: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(case_ids) < 2:
        raise ValueError(
            f"Client {client_id} needs at least two validation cases to "
            "materialize disjoint calibration and validation partitions."
        )
    ordered = tuple(
        sorted(
            case_ids,
            key=lambda case_id: _case_order_key(
                case_id,
                client_id=client_id,
                seed=seed,
            ),
        )
    )
    calibration_count = len(ordered) // 2
    return ordered[:calibration_count], ordered[calibration_count:]


def _write_case_ids(path: Path, case_ids: tuple[str, ...]) -> None:
    path.write_text("".join(f"{case_id}\n" for case_id in case_ids), encoding="utf-8")


def _materialized_payload(
    source_config: Path,
    source: BraTSScenario,
    profile: dict[str, tuple[str, ...]],
    *,
    profile_seed: int,
) -> dict[str, Any]:
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    result = copy.deepcopy(payload)
    raw_order = result["modality_order"]
    normalised_order = tuple(str(name).strip().lower() for name in raw_order)
    if normalised_order != BRATS_MODALITIES:
        raise ValueError("Source modality order does not match BraTS channels.")

    result["split_id"] = "balanced"
    result["masks"] = {
        client_id: [int(modality in modalities) for modality in normalised_order]
        for client_id, modalities in profile.items()
    }
    result.pop("modality_masks", None)
    result["expected_modality_pool_sizes"] = {
        raw_name: 5 for raw_name in raw_order
    }
    result["global_test_file"] = source.global_test_file.name
    result["source_scenario"] = {
        "split_id": source.scenario_id,
        "config_sha256": _sha256(source_config),
        "mapping_fingerprint": payload.get("mapping_fingerprint"),
        "modality_masks_used": False,
    }
    result["modality_profile"] = {
        "name": "symmetric_balanced_v1",
        "assignment_seed": profile_seed,
        "design": (
            "two complete clients plus each of the six modality pairs once"
        ),
        "modality_prevalence_per_client_pool": 5,
        "pair_cooccurrence_per_client_pool": 3,
        "derived_independently_of_patient_data": True,
    }
    result["mapping_fingerprint"] = _mapping_fingerprint(source, profile)
    return result


def materialize_balanced_brats_scenario(
    source_config: str | Path,
    output_directory: str | Path,
    *,
    dataset_root: str | Path,
    profile_seed: int = BRATS_BALANCED_PROFILE_SEED,
) -> BraTSScenario:
    """Create one immutable, portable, modality-symmetric scenario bundle."""

    source_path = Path(source_config).resolve()
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Output scenario already exists; refusing overwrite: {output_path}"
        )
    source = load_brats_scenario(source_path, dataset_root=dataset_root)
    profile = balanced_brats_modalities(seed=profile_seed)
    validate_balanced_brats_modalities(profile)
    if set(profile) != {client.client_id for client in source.clients}:
        raise ValueError("Source scenario clients do not match the balanced profile.")

    payload = _materialized_payload(
        source_path,
        source,
        profile,
        profile_seed=profile_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}-",
            dir=output_path.parent,
        )
    )
    try:
        for client in source.clients:
            raw_client = payload["clients"][client.client_id]
            raw_client["files"] = {}
            for partition in BRATS_PARTITIONS:
                source_file = client.split_file(partition)
                destination = temporary / source_file.name
                shutil.copy2(source_file, destination)
                raw_client["files"][partition] = destination.name

        shutil.copy2(
            source.global_test_file,
            temporary / source.global_test_file.name,
        )
        config_path = temporary / "materialized_config.json"
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_brats_scenario(config_path, dataset_root=dataset_root)
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return load_brats_scenario(
        output_path / "materialized_config.json",
        dataset_root=dataset_root,
    )


def materialize_calibration_brats_scenario(
    source_config: str | Path,
    output_directory: str | Path,
    *,
    dataset_root: str | Path,
    split_seed: int = BRATS_CALIBRATION_SPLIT_SEED,
) -> BraTSScenario:
    """Split validation cases into disjoint selector and evaluation roles.

    Assignment is deterministic and depends only on case identifiers, client
    identifiers, and the declared seed. Labels, images, and modality masks do
    not influence the split.
    """

    if split_seed < 0:
        raise ValueError("Calibration split seed must be non-negative.")
    source_path = Path(source_config).resolve()
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Output scenario already exists; refusing overwrite: {output_path}"
        )
    source = load_brats_scenario(source_path, dataset_root=dataset_root)
    if source.schema_version != 1 or "calibration" in source.partitions:
        raise ValueError(
            "Calibration materialization requires a schema-v1 scenario with "
            "an unsplit validation partition."
        )

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["schema_version"] = 2
    payload["split_id"] = f"{source.scenario_id}_calibration"
    payload["source_scenario"] = {
        "split_id": source.scenario_id,
        "config_sha256": _sha256(source_path),
        "mapping_fingerprint": payload.get("mapping_fingerprint"),
    }
    payload["calibration_partition"] = {
        "source_partition": "validation",
        "assignment_seed": split_seed,
        "assignment_rule": "sha256(seed, client_id, case_id)",
        "derived_without_images_labels_or_modality_masks": True,
        "role": "selector_calibration_only",
    }
    payload["global_test_file"] = source.global_test_file.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}-",
            dir=output_path.parent,
        )
    )
    try:
        for client in source.clients:
            raw_client = payload["clients"][client.client_id]
            raw_client["files"] = {}
            for partition in ("train", "test"):
                source_file = client.split_file(partition)
                destination = temporary / source_file.name
                shutil.copy2(source_file, destination)
                raw_client["files"][partition] = destination.name

            calibration_ids, validation_ids = _calibration_split(
                client.validation_case_ids,
                client_id=client.client_id,
                seed=split_seed,
            )
            calibration_name = f"{client.client_id}_calibration.csv"
            validation_name = f"{client.client_id}_validation.csv"
            _write_case_ids(temporary / calibration_name, calibration_ids)
            _write_case_ids(temporary / validation_name, validation_ids)
            raw_client["files"]["calibration"] = calibration_name
            raw_client["files"]["validation"] = validation_name

        shutil.copy2(
            source.global_test_file,
            temporary / source.global_test_file.name,
        )
        provisional = temporary / "materialized_config.json"
        provisional.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        scenario = load_brats_scenario(provisional, dataset_root=dataset_root)
        payload["mapping_fingerprint"] = _mapping_fingerprint(
            scenario,
            {
                client.client_id: client.modalities
                for client in scenario.clients
            },
        )
        provisional.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_brats_scenario(provisional, dataset_root=dataset_root)
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return load_brats_scenario(
        output_path / "materialized_config.json",
        dataset_root=dataset_root,
    )
