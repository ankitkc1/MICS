#!/usr/bin/env python3
"""Run complete single-seed baselines against an explicit historical v2 matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import importlib
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from compare_method_matrix import (
    EVALUATION_INVARIANTS,
    RUNTIME_INVARIANTS,
    TRAINING_INVARIANTS,
    _differences_from_reference,
    _load_method,
)
from compare_tracked_runs import (
    _evaluation_case_key,
    _evaluation_summary,
    _json_object,
    _nested,
    _selection_summary,
)
from mics.data import load_brats_scenario
from mics.experiments import (
    atomic_write_json,
    collect_runtime_metadata,
    configuration_fingerprint,
    load_system_trace,
    sha256_file,
)
from mics.experiments.metrics import jain_fairness_index

RUNNER_METHODS = {
    "uniform_random": "uniform_random",
    "power_of_choice": "power_of_choice",
    "divfl_gradient_facility": "divfl",
    "mfedmc_client_low_loss": "mfedmc",
    "oort_training": "oort",
}


@contextmanager
def _method_lock(path: Path):
    """Hold an operating-system lock through training, evaluation or comparison."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_reference(manifest: dict[str, Any], reference: dict[str, Any]) -> None:
    if manifest.get("protocol_status") != "frozen_single_seed":
        raise ValueError("The baseline manifest must declare frozen_single_seed.")
    methods = manifest.get("methods", [])
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("The manifest needs unique baseline methods.")
    if set(methods).difference(RUNNER_METHODS):
        raise ValueError("The manifest contains an unsupported baseline method.")
    if reference.get("purpose") != "matched_completed_method_matrix_comparison":
        raise ValueError("The historical reference must be a completed method matrix.")
    if reference.get("interpretation_allowed") is not True:
        raise ValueError("The historical reference is marked as an integration smoke.")
    label = manifest["reference_label"]
    method = reference["methods"][label]
    selection = method["selection_configuration"]
    if selection.get("name") != "mics_v2_impact" or not selection.get(
        "client_queue_enabled"
    ):
        raise ValueError(
            "The historical reference must be full MICS-v2 with client queues."
        )
    invariants = reference["paired_invariants"]
    runtime = invariants["runtime"]
    if runtime["source_revision"] != manifest["reference_source_revision"]:
        raise ValueError("The reference source revision differs from the manifest.")
    if runtime.get("source_dirty") is not False:
        raise ValueError("The reference must come from a clean source tree.")
    training = invariants["training"]
    if training["training"]["max_batches_per_epoch"] is not None:
        raise ValueError("The reference uses truncated local training.")
    expected = manifest["reference_inputs"]
    actual = {
        "scenario_config_sha256": training["scenario.config_sha256"],
        "system_trace_sha256": training["system_trace"]["sha256"],
        "modality_importance_sha256": method["external_importance_source"]["sha256"],
    }
    if expected != actual:
        raise ValueError("The reference input hashes differ from the frozen manifest.")
    for name in ("model", "selection", "data"):
        if training[f"seeds.{name}"] != manifest["seeds"][name]:
            raise ValueError(f"The reference {name} seed differs.")
    if training["system_trace"]["seed"] != manifest["seeds"]["trace"]:
        raise ValueError("The reference trace seed differs.")
    if training["federation.rounds"] != manifest["rounds"]:
        raise ValueError("The reference round count differs.")
    if training["federation.clients_per_round"] != manifest["clients_per_round"]:
        raise ValueError("The reference client budget differs.")
    evaluation = invariants["evaluation"]
    if evaluation["partition"] != "validation" or not evaluation["complete_partition"]:
        raise ValueError(
            "The reference must evaluate the complete validation partition."
        )
    if evaluation["case_count"] != manifest["validation_case_count"]:
        raise ValueError("The reference validation case count differs.")
    if method["evaluation"]["completed_rounds"] != manifest["rounds"]:
        raise ValueError("The reference evaluation is not at the final round.")
    if manifest.get("probe_max_batches") is not None:
        raise ValueError("This full-probe study requires probe_max_batches=null.")


def training_command(
    manifest: dict[str, Any],
    reference: dict[str, Any],
    method: str,
    *,
    project_root: Path,
    output_root: Path,
    dataset_root: Path,
    scenario_config: Path,
) -> list[str]:
    """Derive training arguments from the completed reference's recorded contract."""

    invariants = reference["paired_invariants"]["training"]
    runner = (
        "run_uniform_baseline.py"
        if method == "uniform_random"
        else "run_reference_baseline.py"
    )
    command = [sys.executable, str(project_root / "scripts" / runner)]
    if method != "uniform_random":
        command.extend(("--method", RUNNER_METHODS[method]))
    values = {
        "dataset-root": dataset_root,
        "scenario-config": scenario_config,
        "system-trace": invariants["system_trace"]["path"],
        "output-root": output_root,
        "run-id": method,
        "rounds": invariants["federation.rounds"],
        "clients-per-round": invariants["federation.clients_per_round"],
        "checkpoint-every": invariants["federation.checkpoint_every"],
        "device": invariants["device"],
    }
    for name in ("base_channels", "num_levels", "fusion", "modality_dropout"):
        values[name.replace("_", "-")] = invariants["model"][name]
    for name in (
        "foreground_probability",
        "batch_size",
        "num_workers",
        "local_epochs",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "amp_initial_scale",
        "amp_max_retries",
    ):
        values[name.replace("_", "-")] = invariants["training"][name]
    for name in ("model", "selection", "data"):
        values[f"{name}-seed"] = invariants[f"seeds.{name}"]
    for name, value in values.items():
        command.extend((f"--{name}", str(value)))
    command.extend(
        (
            "--patch-size",
            *(str(value) for value in invariants["training"]["patch_size"]),
        )
    )
    if not invariants["training"]["amp"]:
        command.append("--no-amp")
    if method == "power_of_choice":
        command.extend(
            ("--candidate-size", str(manifest["power_of_choice_candidate_size"]))
        )
    return command


def _completion_diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(summary)
    counts = Counter({client_id: 0 for client_id in result["selection_counts"]})
    for successful in result["successful_clients_by_round"]:
        counts.update(successful)
    result["successful_completion_counts"] = dict(counts)
    result["jain_completion_fairness"] = jain_fairness_index(counts)
    return result


def validate_training_contract(reference: dict[str, Any], config, metadata) -> None:
    training = reference["paired_invariants"]["training"]
    for field in TRAINING_INVARIANTS:
        if _nested(config, field) != training[field]:
            raise ValueError(
                f"Historical-reference training invariant differs at {field}."
            )
    if config.get("system_trace") != training["system_trace"]:
        raise ValueError("Historical-reference system trace differs.")
    runtime = reference["paired_invariants"]["runtime"]
    for field in RUNTIME_INVARIANTS:
        if field != "source_revision" and metadata.get(field) != runtime[field]:
            raise ValueError(
                f"Historical-reference runtime invariant differs at {field}."
            )


def validate_candidate(reference: dict[str, Any], candidate) -> None:
    validate_training_contract(reference, candidate.config, candidate.metadata)
    if (
        candidate.evaluation_payload["evaluation_source"].get("source_dirty")
        is not False
    ):
        raise ValueError("The baseline evaluation must come from a clean source tree.")
    evaluation = candidate.evaluation_payload["evaluation"]
    for field in EVALUATION_INVARIANTS:
        if evaluation[field] != reference["paired_invariants"]["evaluation"][field]:
            raise ValueError(
                f"Historical-reference evaluation invariant differs at {field}."
            )
    if (
        _evaluation_case_key(evaluation)
        != reference["paired_invariants"]["evaluation_cases_and_masks"]
    ):
        raise ValueError(
            "Historical-reference evaluation cases or modality masks differ."
        )


def _binding(
    manifest_path: Path, reference_path: Path, reference: dict[str, Any], root: Path
) -> dict[str, Any]:
    metadata = collect_runtime_metadata(root)
    if metadata.get("source_dirty") is not False or not metadata.get("source_revision"):
        raise ValueError(
            "Commit the comparison code before executing the frozen HPC study."
        )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "reference_matrix_sha256": sha256_file(reference_path),
        "reference_source_revision": reference["paired_invariants"]["runtime"][
            "source_revision"
        ],
        "baseline_source_revision": metadata["source_revision"],
    }


def _bind_study(output_root: Path, binding: dict[str, Any]) -> None:
    with _method_lock(output_root / ".study-binding.lock"):
        path = output_root / "study_binding.json"
        if path.exists():
            if _json_object(path) != binding:
                raise ValueError(
                    "The existing study uses different source, manifest or reference inputs."
                )
        else:
            atomic_write_json(path, binding)


def _run_method(
    arguments, manifest, reference, project_root, output_root, binding
) -> None:
    if arguments.task_index is None or not 0 <= arguments.task_index < len(
        manifest["methods"]
    ):
        raise ValueError("run requires a valid zero-based --task-index.")
    method = manifest["methods"][arguments.task_index]
    invariants = reference["paired_invariants"]["training"]
    scenario = arguments.scenario_config
    if scenario is None:
        paths = [
            Path(path)
            for path in invariants["scenario.file_sha256"]
            if Path(path).name == "materialized_config.json"
        ]
        if len(paths) != 1:
            raise ValueError("Provide --scenario-config for this reference.")
        scenario = paths[0]
    if sha256_file(scenario) != invariants["scenario.config_sha256"]:
        raise ValueError("The current scenario file differs from the reference.")
    for path, expected in invariants["scenario.file_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"A scenario input changed: {path}")
    if (
        sha256_file(invariants["system_trace"]["path"])
        != invariants["system_trace"]["sha256"]
    ):
        raise ValueError("The system trace changed since the reference run.")
    runtime = collect_runtime_metadata(project_root)
    for field in RUNTIME_INVARIANTS:
        if (
            field != "source_revision"
            and runtime.get(field) != reference["paired_invariants"]["runtime"][field]
        ):
            raise ValueError(f"The allocated training runtime differs at {field}.")
    if not arguments.dataset_root.is_dir():
        raise FileNotFoundError(arguments.dataset_root)
    with _method_lock(output_root / f".{method}.lock"):
        _bind_study(output_root, binding)
        run_dir = output_root / method
        command = training_command(
            manifest,
            reference,
            method,
            project_root=project_root,
            output_root=output_root,
            dataset_root=arguments.dataset_root,
            scenario_config=scenario,
        )
        module_name = Path(command[1]).stem
        runner = importlib.import_module(module_name)
        parsed = runner.build_parser().parse_args(command[2:])
        expected_config = runner._resolved_config(
            parsed,
            load_brats_scenario(scenario, dataset_root=arguments.dataset_root),
            load_system_trace(invariants["system_trace"]["path"]),
        )
        validate_training_contract(reference, expected_config, runtime)
        completed = False
        if run_dir.exists():
            metadata = _json_object(run_dir / "run_metadata.json")
            if metadata.get("source_revision") != binding["baseline_source_revision"]:
                raise ValueError("A baseline run has a different source revision.")
            if configuration_fingerprint(
                _json_object(run_dir / "resolved_config.json")
            ) != configuration_fingerprint(expected_config):
                raise ValueError(
                    "An existing baseline run has a different configuration."
                )
            completed = (
                _json_object(run_dir / "status.json").get("state") == "completed"
            )
            if not completed:
                command.append("--resume")
        if completed:
            print(f"Training already completed: {run_dir}", flush=True)
        else:
            subprocess.run(command, cwd=project_root, check=True)
        evaluation_path = (
            run_dir
            / "evaluations"
            / f"validation_after_round_{manifest['rounds']:04d}.json"
        )
        if not evaluation_path.exists():
            evaluation = reference["paired_invariants"]["evaluation"]
            command = [
                sys.executable,
                str(project_root / "scripts" / "evaluate_brats_checkpoint.py"),
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(arguments.dataset_root),
                "--scenario-config",
                str(scenario),
                "--checkpoint",
                str(
                    run_dir / "checkpoints" / f"after_round_{manifest['rounds']:04d}.pt"
                ),
                "--partition",
                "validation",
                "--roi-size",
                *(str(value) for value in evaluation["roi_size"]),
                "--overlap",
                str(evaluation["overlap"]),
                "--device",
                invariants["device"],
                "--output",
                str(evaluation_path),
            ]
            if not invariants["training"]["amp"]:
                command.append("--no-amp")
            subprocess.run(command, cwd=project_root, check=True)
        candidate = _load_method((method, str(run_dir), str(evaluation_path)))
        validate_candidate(reference, candidate)
        if candidate.config["method"] != method:
            raise ValueError("The baseline method identity differs from the manifest.")
        print(f"MICS-V2 FULL SINGLE-SEED BASELINE PASSED: {method}", flush=True)


def _compare(manifest, reference, output_root, binding) -> None:
    _bind_study(output_root, binding)
    label = manifest["reference_label"]
    order = [label, *(name for name in reference["method_order"] if name != label)]
    methods = copy.deepcopy(reference["methods"])
    for name in methods:
        methods[name]["source_revision"] = binding["reference_source_revision"]
        methods[name]["selection_and_training"] = _completion_diagnostics(
            methods[name]["selection_and_training"]
        )
    for name in manifest["methods"]:
        run_dir = output_root / name
        path = (
            run_dir
            / "evaluations"
            / f"validation_after_round_{manifest['rounds']:04d}.json"
        )
        candidate = _load_method((name, str(run_dir), str(path)))
        validate_candidate(reference, candidate)
        if candidate.metadata["source_revision"] != binding["baseline_source_revision"]:
            raise ValueError("Baseline source revisions differ within the study.")
        if candidate.config["method"] != name:
            raise ValueError("A baseline method differs from its manifest identity.")
        methods[name] = {
            "method": candidate.config["method"],
            "run_dir": str(run_dir),
            "source_revision": candidate.metadata["source_revision"],
            "selection_configuration": candidate.config.get(
                "selection", {"name": name}
            ),
            "evaluation_path": str(path),
            "evaluation_sha256": sha256_file(path),
            "resolved_config_sha256": sha256_file(run_dir / "resolved_config.json"),
            "selection_and_training": _completion_diagnostics(
                _selection_summary(candidate.metrics, candidate.config)
            ),
            "evaluation": _evaluation_summary(candidate.evaluation_payload),
        }
        order.append(name)
    primary = methods[label]
    report = {
        "schema_version": 1,
        "purpose": "single_seed_baseline_comparison_with_explicit_historical_reference",
        "interpretation_allowed": True,
        "scientific_scope": "single_seed_complete_held_out_validation",
        "study_id": manifest["study_id"],
        "seeds": manifest["seeds"],
        "reference_label": label,
        "historical_reference_binding": binding,
        "source_revisions_match": binding["reference_source_revision"]
        == binding["baseline_source_revision"],
        "provenance_note": "Historical v2 results are reused across source revisions. Recorded model, training, data, trace, runtime and evaluation contracts are matched; both revisions are retained.",
        "paired_invariants": reference["paired_invariants"],
        "method_order": order,
        "methods": methods,
        "differences_from_reference": {
            name: _differences_from_reference(
                primary["selection_and_training"],
                methods[name]["selection_and_training"],
                primary["evaluation"],
                methods[name]["evaluation"],
            )
            for name in order
            if name != label
        },
    }
    with _method_lock(output_root / ".comparison.lock"):
        path = output_root / "single_seed_method_matrix.json"
        if path.exists() and _json_object(path) != report:
            raise FileExistsError(
                "The existing comparison differs from current artifacts."
            )
        if not path.exists():
            atomic_write_json(path, report)
        with (output_root / "single_seed_method_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "method",
                    "macro_dice",
                    "macro_hd95_mm",
                    "selection_jain",
                    "completion_jain",
                    "selection_pipeline_seconds",
                    "training_seconds",
                    "logical_bytes",
                    "source_revision",
                )
            )
            for name in order:
                item = methods[name]
                selection = item["selection_and_training"]
                writer.writerow(
                    (
                        name,
                        item["evaluation"]["macro_client"]["dice"]["mean"],
                        item["evaluation"]["macro_client"]["hd95_mm"]["mean"],
                        selection["jain_selection_fairness"],
                        selection["jain_completion_fairness"],
                        selection["total_selection_pipeline_seconds"],
                        selection["total_training_seconds"],
                        selection["cumulative_download_bytes"]
                        + selection["cumulative_upload_bytes"],
                        item["source_revision"],
                    )
                )
    print(f"Comparison: {output_root / 'single_seed_method_matrix.json'}")
    print("MICS-V2 FULL SINGLE-SEED BASELINE COMPARISON PASSED")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "compare"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "configs/mics_v2_single_seed_baselines_v1.json",
    )
    parser.add_argument("--reference-matrix", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get("MICS_DATA_ROOT", str(Path.home() / "BraTS2020_npy"))
        ),
    )
    parser.add_argument(
        "--scenario-config", type=Path, default=os.environ.get("MICS_SCENARIO_CONFIG")
    )
    arguments = parser.parse_args()
    manifest_path = arguments.manifest.resolve()
    manifest = _json_object(manifest_path)
    reference_path = arguments.reference_matrix or root / manifest["reference_matrix"]
    reference = _json_object(reference_path)
    validate_reference(manifest, reference)
    output_root = (
        arguments.output_root or root / "results" / "studies" / manifest["study_id"]
    )
    binding = _binding(manifest_path, reference_path, reference, root)
    if arguments.action == "run":
        _run_method(arguments, manifest, reference, root, output_root, binding)
    else:
        _compare(manifest, reference, output_root, binding)


if __name__ == "__main__":
    main()
