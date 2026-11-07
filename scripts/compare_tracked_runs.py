#!/usr/bin/env python3
"""Create a guarded paired comparison of two tracked federated runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mics.experiments import atomic_write_json, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--candidate-run-dir", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", required=True, type=Path)
    parser.add_argument("--candidate-evaluation", required=True, type=Path)
    parser.add_argument("--baseline-label", default="uniform_random")
    parser.add_argument("--candidate-label", default="mics_impact")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL file does not exist: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}.")
        records.append(value)
    if not records:
        raise ValueError(f"Tracked record is empty: {path}")
    return records


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"Missing required comparison field: {path}")
        current = current[key]
    return current


def _assert_equal(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    paths: Iterable[str],
) -> dict[str, Any]:
    matched = {}
    for path in paths:
        left = _nested(baseline, path)
        right = _nested(candidate, path)
        if left != right:
            raise ValueError(
                f"Paired-run invariant differs at {path}: "
                f"baseline={left!r}, candidate={right!r}."
            )
        matched[path] = left
    return matched


def _mean(values: Iterable[float]) -> float:
    collected = tuple(float(value) for value in values)
    if not collected:
        raise ValueError("Cannot average an empty collection.")
    return float(sum(collected) / len(collected))


def _longest_zero_stall(values: Sequence[int]) -> int:
    longest = 0
    current = 0
    for value in values:
        if int(value) == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _selection_summary(
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    diagnostics = [record["selection_diagnostics"] for record in metrics]
    modalities = tuple(sorted(diagnostics[0]["modality_coverage"]))
    coverage_sequences = {
        modality: [
            int(record["modality_coverage"][modality])
            for record in diagnostics
        ]
        for modality in modalities
    }
    component_counts: Counter[str] = Counter()
    for record in metrics:
        component_counts.update(
            {
                str(name): int(value)
                for name, value in record["component_update_counts"].items()
            }
        )
    final_diagnostics = diagnostics[-1]
    selector_outcomes = [record.get("selector_outcome", {}) for record in metrics]
    final_queues = None
    maximum_queue_over_run = None
    queue_records = [
        outcome.get("component_queues_after_round")
        for outcome in selector_outcomes
        if isinstance(outcome, Mapping)
        and isinstance(outcome.get("component_queues_after_round"), Mapping)
    ]
    if queue_records:
        final_queues = dict(queue_records[-1])
        maximum_queue_over_run = max(
            float(value)
            for record in queue_records
            for value in record.values()
        )
    impact_coverage = [
        record.get("impact_weighted_modality_coverage")
        for record in diagnostics
    ]
    finite_impact_coverage = [
        float(value) for value in impact_coverage if value is not None
    ]
    communication = metrics[-1]["communication"]
    examples_seen = sum(int(record["examples_seen"]) for record in metrics)
    weighted_loss = (
        sum(
            float(record["sample_weighted_training_loss"])
            * int(record["examples_seen"])
            for record in metrics
            if record["sample_weighted_training_loss"] is not None
        )
        / examples_seen
        if examples_seen > 0
        else None
    )
    selected_count = sum(len(record["selected_clients"]) for record in metrics)
    successful_count = sum(
        len(record.get("successful_clients", record["selected_clients"]))
        for record in metrics
    )
    return {
        "round_count": len(metrics),
        "selected_clients_by_round": [
            list(record["selected_clients"]) for record in metrics
        ],
        "successful_clients_by_round": [
            list(record.get("successful_clients", record["selected_clients"]))
            for record in metrics
        ],
        "failed_selected_clients_by_round": [
            list(record.get("failed_selected_clients", []))
            for record in metrics
        ],
        "available_client_count_by_round": [
            len(record.get("available_clients", []))
            if "available_clients" in record
            else int(record["selection_diagnostics"]["eligible_client_count"])
            for record in metrics
        ],
        "selected_attempt_count": selected_count,
        "successful_completion_count": successful_count,
        "selected_completion_rate": successful_count / selected_count,
        "selection_counts": dict(
            final_diagnostics["selection_counts_after_round"]
        ),
        "jain_selection_fairness": float(
            final_diagnostics["jain_selection_fairness_after_round"]
        ),
        "modality_coverage_rate": {
            modality: _mean(coverage_sequences[modality])
            for modality in modalities
        },
        "maximum_consecutive_uncovered_rounds": {
            modality: _longest_zero_stall(coverage_sequences[modality])
            for modality in modalities
        },
        "mean_selected_clients_per_modality": {
            modality: _mean(
                record["selected_clients_per_modality"][modality]
                for record in diagnostics
            )
            for modality in modalities
        },
        "mean_pairwise_modality_jaccard_redundancy": _mean(
            record["pairwise_modality_jaccard_redundancy"]
            for record in diagnostics
        ),
        "mean_impact_weighted_modality_coverage": (
            _mean(finite_impact_coverage) if finite_impact_coverage else None
        ),
        "cumulative_component_update_counts": dict(component_counts),
        "final_component_queues": final_queues,
        "maximum_component_queue_over_run": maximum_queue_over_run,
        "total_selection_seconds": sum(
            float(record["selection_seconds"]) for record in metrics
        ),
        "total_training_seconds": sum(
            float(record["training_seconds"]) for record in metrics
        ),
        "examples_seen": examples_seen,
        "example_weighted_training_loss": weighted_loss,
        "cumulative_download_bytes": int(
            communication["cumulative_download_bytes"]
        ),
        "cumulative_upload_bytes": int(
            communication["cumulative_upload_bytes"]
        ),
    }


def _evaluation_case_key(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "client_id": str(record["client_id"]),
            "case_id": str(record["case_id"]),
            "modalities": list(record["modalities"]),
        }
        for record in evaluation["cases"]
    ]


def _evaluation_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = payload["evaluation"]
    return {
        "purpose": payload["purpose"],
        "completed_rounds": int(payload["completed_rounds"]),
        "case_count": int(evaluation["case_count"]),
        "client_count": int(evaluation["client_count"]),
        "macro_client": evaluation["macro_client"],
        "sample_weighted": {
            key: evaluation["sample_weighted"][key]
            for key in ("case_count", "dice", "hd95_mm")
        },
        "client_dispersion": evaluation["client_dispersion"],
        "total_evaluation_seconds": float(
            evaluation["total_evaluation_seconds"]
        ),
        "window_count": int(evaluation["sample_weighted"]["window_count"]),
    }


def _numeric_differences(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_minus_baseline_macro_dice": {
            region: float(candidate["macro_client"]["dice"][region])
            - float(baseline["macro_client"]["dice"][region])
            for region in ("WT", "TC", "ET", "mean")
        },
        "candidate_minus_baseline_macro_hd95_mm": {
            region: float(candidate["macro_client"]["hd95_mm"][region])
            - float(baseline["macro_client"]["hd95_mm"][region])
            for region in ("WT", "TC", "ET", "mean")
        },
    }


def main() -> None:
    arguments = build_parser().parse_args()
    output_path = arguments.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output_path}")
    baseline_dir = arguments.baseline_run_dir.resolve()
    candidate_dir = arguments.candidate_run_dir.resolve()
    baseline_config = _json_object(baseline_dir / "resolved_config.json")
    candidate_config = _json_object(candidate_dir / "resolved_config.json")
    baseline_status = _json_object(baseline_dir / "status.json")
    candidate_status = _json_object(candidate_dir / "status.json")
    if baseline_status.get("state") != "completed":
        raise ValueError("Baseline run is not completed.")
    if candidate_status.get("state") != "completed":
        raise ValueError("Candidate run is not completed.")

    matched_training = _assert_equal(
        baseline_config,
        candidate_config,
        (
            "purpose",
            "scenario.id",
            "scenario.config_sha256",
            "scenario.file_sha256",
            "scenario.clients",
            "model",
            "federation.rounds",
            "federation.clients_per_round",
            "federation.aggregation",
            "training",
            "seeds.model",
            "seeds.data",
            "device",
        ),
    )
    baseline_trace = baseline_config.get("system_trace")
    candidate_trace = candidate_config.get("system_trace")
    if baseline_trace != candidate_trace:
        raise ValueError("Compared runs do not use the same system trace.")
    matched_training["system_trace"] = baseline_trace
    baseline_metrics = _jsonl(baseline_dir / "round_metrics.jsonl")
    candidate_metrics = _jsonl(candidate_dir / "round_metrics.jsonl")
    if len(baseline_metrics) != len(candidate_metrics):
        raise ValueError("Compared runs contain different round counts.")
    baseline_evaluation_payload = _json_object(
        arguments.baseline_evaluation.resolve()
    )
    candidate_evaluation_payload = _json_object(
        arguments.candidate_evaluation.resolve()
    )
    baseline_evaluation = baseline_evaluation_payload["evaluation"]
    candidate_evaluation = candidate_evaluation_payload["evaluation"]
    matched_evaluation = _assert_equal(
        baseline_evaluation,
        candidate_evaluation,
        ("partition", "roi_size", "overlap", "spacing_mm", "case_count"),
    )
    baseline_cases = _evaluation_case_key(baseline_evaluation)
    candidate_cases = _evaluation_case_key(candidate_evaluation)
    if baseline_cases != candidate_cases:
        raise ValueError("Evaluation cases or modality masks are not paired.")

    baseline_selection = _selection_summary(baseline_metrics)
    candidate_selection = _selection_summary(candidate_metrics)
    baseline_eval_summary = _evaluation_summary(baseline_evaluation_payload)
    candidate_eval_summary = _evaluation_summary(candidate_evaluation_payload)
    integration_only = (
        baseline_config["purpose"] != "full_training_checkpoint_for_held_out_evaluation"
        or baseline_evaluation_payload["purpose"]
        == "integration_smoke_not_model_quality"
        or candidate_evaluation_payload["purpose"]
        == "integration_smoke_not_model_quality"
    )
    result = {
        "schema_version": 1,
        "purpose": (
            "paired_integration_audit_not_model_quality"
            if integration_only
            else "paired_completed_run_comparison"
        ),
        "interpretation_allowed": not integration_only,
        "warning": (
            "This report verifies paired mechanics only. It is not statistical "
            "evidence of model quality or selector superiority."
            if integration_only
            else None
        ),
        "baseline": {
            "label": arguments.baseline_label,
            "run_dir": str(baseline_dir),
            "method": baseline_config["method"],
            "resolved_config_sha256": sha256_file(
                baseline_dir / "resolved_config.json"
            ),
            "selection_and_training": baseline_selection,
            "evaluation": baseline_eval_summary,
        },
        "candidate": {
            "label": arguments.candidate_label,
            "run_dir": str(candidate_dir),
            "method": candidate_config["method"],
            "resolved_config_sha256": sha256_file(
                candidate_dir / "resolved_config.json"
            ),
            "selection_and_training": candidate_selection,
            "evaluation": candidate_eval_summary,
        },
        "paired_invariants": {
            "training": matched_training,
            "evaluation": matched_evaluation,
            "evaluation_cases_and_masks": baseline_cases,
        },
        "exact_differences": {
            **_numeric_differences(
                baseline_eval_summary,
                candidate_eval_summary,
            ),
            "candidate_minus_baseline_jain_selection_fairness": (
                candidate_selection["jain_selection_fairness"]
                - baseline_selection["jain_selection_fairness"]
            ),
            "candidate_minus_baseline_selection_seconds": (
                candidate_selection["total_selection_seconds"]
                - baseline_selection["total_selection_seconds"]
            ),
            "candidate_minus_baseline_download_bytes": (
                candidate_selection["cumulative_download_bytes"]
                - baseline_selection["cumulative_download_bytes"]
            ),
            "candidate_minus_baseline_upload_bytes": (
                candidate_selection["cumulative_upload_bytes"]
                - baseline_selection["cumulative_upload_bytes"]
            ),
        },
    }
    atomic_write_json(output_path, result)
    printed = {
        "purpose": result["purpose"],
        "output": str(output_path),
        "baseline_selection": baseline_selection,
        "candidate_selection": candidate_selection,
        "exact_differences": result["exact_differences"],
    }
    print(json.dumps(printed, indent=2, sort_keys=True))
    print("MICS PAIRED TRACKED-RUN COMPARISON PASSED")


if __name__ == "__main__":
    main()
