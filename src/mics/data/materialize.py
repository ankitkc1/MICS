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
                    for partition in BRATS_PARTITIONS
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
