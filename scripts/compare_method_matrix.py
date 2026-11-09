#!/usr/bin/env python3
"""Create one guarded comparison report for multiple tracked FL methods."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compare_tracked_runs import (
    _assert_equal,
    _evaluation_case_key,
    _evaluation_summary,
    _json_object,
    _jsonl,
    _numeric_differences,
    _selection_summary,
)
from mics.experiments import (
    atomic_write_json,
    configuration_fingerprint,
    sha256_file,
)

TRAINING_INVARIANTS = (
    "purpose",
    "scenario.id",
    "scenario.config_sha256",
    "scenario.file_sha256",
    "scenario.clients",
    "model",
    "federation.rounds",
    "federation.clients_per_round",
    "federation.checkpoint_every",
    "federation.aggregation",
    "training",
    "seeds.model",
    "seeds.selection",
    "seeds.data",
    "device",
)
RUNTIME_INVARIANTS = (
    "source_revision",
    "source_dirty",
    "python",
    "pytorch",
    "cuda_build",
    "gpu_names",
)
EVALUATION_INVARIANTS = (
    "partition",
    "roi_size",
    "overlap",
    "spacing_mm",
    "case_count",
    "complete_partition",
)
SELECTION_DISCLOSURE_FIELDS = (
    "name",
    "variant",
    "ablation",
    "service_type",
    "saturation_level",
    "impact_weight",
    "component_queue_weight",
    "subset_utility",
    "importance_mode",
    "reliability_aware",
    "candidate_size",
    "probe_max_batches",
)


@dataclass(frozen=True)
class MethodArtifacts:
    label: str
    run_dir: Path
    evaluation_path: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    metrics: list[dict[str, Any]]
    evaluation_payload: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        nargs=3,
        required=True,
        metavar=("LABEL", "RUN_DIR", "EVALUATION_JSON"),
        help="Repeat once for every method in the matched comparison.",
    )
    parser.add_argument("--reference-label", default="uniform_random")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _load_method(values: Sequence[str]) -> MethodArtifacts:
    label, raw_run_dir, raw_evaluation = values
    if not label or any(character.isspace() for character in label):
        raise ValueError(f"Invalid method label: {label!r}.")
    run_dir = Path(raw_run_dir).resolve()
    evaluation_path = Path(raw_evaluation).resolve()
    config = _json_object(run_dir / "resolved_config.json")
    metadata = _json_object(run_dir / "run_metadata.json")
    status = _json_object(run_dir / "status.json")
    if status.get("state") != "completed":
        raise ValueError(f"Method {label!r} has not completed.")
    metrics = _jsonl(run_dir / "round_metrics.jsonl")
    expected_rounds = int(config["federation"]["rounds"])
    if len(metrics) != expected_rounds:
        raise ValueError(
            f"Method {label!r} has {len(metrics)} records for "
            f"{expected_rounds} configured rounds."
        )
    evaluation_payload = _json_object(evaluation_path)
    if evaluation_payload.get("run_id") != run_dir.name:
        raise ValueError(f"Evaluation run ID does not match {label!r}.")
    if int(evaluation_payload.get("completed_rounds", -1)) != expected_rounds:
        raise ValueError(f"Evaluation round does not match {label!r}.")
    evaluation_source = evaluation_payload.get("evaluation_source")
    if not isinstance(evaluation_source, Mapping):
        raise ValueError(f"Evaluation source is missing for {label!r}.")
    if evaluation_source.get("source_revision") != metadata.get(
        "source_revision"
    ):
        raise ValueError(
            f"Training and evaluation revisions differ for {label!r}."
        )
    return MethodArtifacts(
        label=label,
        run_dir=run_dir,
        evaluation_path=evaluation_path,
        config=config,
        metadata=metadata,
        metrics=metrics,
        evaluation_payload=evaluation_payload,
    )


def _differences_from_reference(
    reference_selection: Mapping[str, Any],
    candidate_selection: Mapping[str, Any],
    reference_evaluation: Mapping[str, Any],
    candidate_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        key.replace("baseline", "reference"): value
        for key, value in _numeric_differences(
            reference_evaluation,
            candidate_evaluation,
        ).items()
    }
    result.update(
        {
            "candidate_minus_reference_jain_selection_fairness": (
                candidate_selection["jain_selection_fairness"]
                - reference_selection["jain_selection_fairness"]
            ),
            "candidate_minus_reference_completion_rate": (
                candidate_selection["selected_completion_rate"]
                - reference_selection["selected_completion_rate"]
            ),
            "candidate_minus_reference_selection_pipeline_seconds": (
                candidate_selection["total_selection_pipeline_seconds"]
                - reference_selection["total_selection_pipeline_seconds"]
            ),
            "candidate_minus_reference_training_seconds": (
                candidate_selection["total_training_seconds"]
                - reference_selection["total_training_seconds"]
            ),
            "candidate_minus_reference_download_bytes": (
                candidate_selection["cumulative_download_bytes"]
                - reference_selection["cumulative_download_bytes"]
            ),
            "candidate_minus_reference_upload_bytes": (
                candidate_selection["cumulative_upload_bytes"]
                - reference_selection["cumulative_upload_bytes"]
            ),
            "candidate_minus_reference_modality_coverage_rate": {
                modality: (
                    candidate_selection["modality_coverage_rate"][modality]
                    - reference_selection["modality_coverage_rate"][modality]
                )
                for modality in reference_selection[
                    "modality_coverage_rate"
                ]
            },
        }
    )
    return result


def main() -> None:
    arguments = build_parser().parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    methods = [_load_method(values) for values in arguments.method]
    if len(methods) < 2:
        raise ValueError("A method-matrix comparison requires at least two methods.")
    labels = [method.label for method in methods]
    if len(labels) != len(set(labels)):
        raise ValueError("Method labels must be unique.")
    by_label = {method.label: method for method in methods}
    if arguments.reference_label not in by_label:
        raise ValueError("The requested reference label was not provided.")
    reference = by_label[arguments.reference_label]
    if reference.metadata.get("source_dirty") is not False:
        raise ValueError("Reference run must come from a clean source tree.")

    reference_evaluation = reference.evaluation_payload["evaluation"]
    evaluation_cases = _evaluation_case_key(reference_evaluation)
    matched_training: dict[str, Any] | None = None
    matched_runtime: dict[str, Any] | None = None
    matched_evaluation: dict[str, Any] | None = None
    for method in methods:
        if method.metadata.get("source_dirty") is not False:
            raise ValueError(
                f"Method {method.label!r} comes from a dirty source tree."
            )
        training = _assert_equal(
            reference.config,
            method.config,
            TRAINING_INVARIANTS,
        )
        runtime = _assert_equal(
            reference.metadata,
            method.metadata,
            RUNTIME_INVARIANTS,
        )
        if method.config.get("system_trace") != reference.config.get(
            "system_trace"
        ):
            raise ValueError(
                f"Method {method.label!r} uses a different system trace."
            )
        evaluation = method.evaluation_payload["evaluation"]
        evaluation_invariants = _assert_equal(
            reference_evaluation,
            evaluation,
            EVALUATION_INVARIANTS,
        )
        if _evaluation_case_key(evaluation) != evaluation_cases:
            raise ValueError(
                f"Evaluation cases or modality masks differ for {method.label!r}."
            )
        matched_training = training
        matched_runtime = runtime
        matched_evaluation = evaluation_invariants

    summaries = {}
    for method in methods:
        selection = _selection_summary(method.metrics, method.config)
        evaluation = _evaluation_summary(method.evaluation_payload)
        selection_config = method.config.get("selection", {})
        summaries[method.label] = {
            "method": method.config["method"],
            "run_dir": str(method.run_dir),
            "evaluation_path": str(method.evaluation_path),
            "resolved_config_sha256": sha256_file(
                method.run_dir / "resolved_config.json"
            ),
            "selection_config_fingerprint": configuration_fingerprint(
                selection_config
            ),
            "selection_configuration": {
                field: selection_config[field]
                for field in SELECTION_DISCLOSURE_FIELDS
                if field in selection_config
            },
            "external_importance_source": (
                selection_config.get("importance_source")
                if isinstance(selection_config, Mapping)
                else None
            ),
            "selection_and_training": selection,
            "evaluation": evaluation,
        }

    reference_summary = summaries[arguments.reference_label]
    differences = {
        method.label: _differences_from_reference(
            reference_summary["selection_and_training"],
            summaries[method.label]["selection_and_training"],
            reference_summary["evaluation"],
            summaries[method.label]["evaluation"],
        )
        for method in methods
        if method.label != arguments.reference_label
    }
    integration_only = any(
        method.config["purpose"]
        != "full_training_checkpoint_for_held_out_evaluation"
        or method.evaluation_payload["purpose"]
        == "integration_smoke_not_model_quality"
        for method in methods
    )
    report = {
        "schema_version": 1,
        "purpose": (
            "matched_method_matrix_integration_audit_not_model_quality"
            if integration_only
            else "matched_completed_method_matrix_comparison"
        ),
        "interpretation_allowed": not integration_only,
        "warning": (
            "This report verifies matched mechanics only. It is not evidence "
            "of model quality or selector superiority."
            if integration_only
            else None
        ),
        "reference_label": arguments.reference_label,
        "method_order": labels,
        "cost_accounting": {
            "per_round": (
                "Training, selection probes, and logical communication are "
                "included in each method summary."
            ),
            "external_modality_importance": (
                "MICS importance-estimation cost is disclosed under "
                "external_importance_source but is not added to per-round "
                "training time or communication."
            ),
        },
        "methods": summaries,
        "paired_invariants": {
            "training": {
                **(matched_training or {}),
                "system_trace": reference.config.get("system_trace"),
            },
            "runtime": matched_runtime,
            "evaluation": matched_evaluation,
            "evaluation_cases_and_masks": evaluation_cases,
        },
        "differences_from_reference": differences,
    }
    atomic_write_json(output, report)
    printed = {
        "purpose": report["purpose"],
        "output": str(output),
        "reference_label": arguments.reference_label,
        "methods": {
            label: {
                "method": value["method"],
                "macro_mean_dice": value["evaluation"]["macro_client"][
                    "dice"
                ]["mean"],
                "selection_pipeline_seconds": value[
                    "selection_and_training"
                ]["total_selection_pipeline_seconds"],
                "download_bytes": value["selection_and_training"][
                    "cumulative_download_bytes"
                ],
                "upload_bytes": value["selection_and_training"][
                    "cumulative_upload_bytes"
                ],
            }
            for label, value in summaries.items()
        },
    }
    print(json.dumps(printed, indent=2, sort_keys=True))
    print("MICS MATCHED METHOD-MATRIX COMPARISON PASSED")


if __name__ == "__main__":
    main()
