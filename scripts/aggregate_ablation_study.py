#!/usr/bin/env python3
"""Aggregate completed matched MICS ablation reports across fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mics.experiments import atomic_write_json, sha256_file

METRICS = (
    ("macro_dice_mean", "evaluation.macro_client.dice.mean", "higher"),
    ("macro_dice_WT", "evaluation.macro_client.dice.WT", "higher"),
    ("macro_dice_TC", "evaluation.macro_client.dice.TC", "higher"),
    ("macro_dice_ET", "evaluation.macro_client.dice.ET", "higher"),
    ("macro_hd95_mm_mean", "evaluation.macro_client.hd95_mm.mean", "lower"),
    (
        "sample_weighted_dice_mean",
        "evaluation.sample_weighted.dice.mean",
        "higher",
    ),
    (
        "client_dice_population_std",
        "evaluation.client_dispersion.mean_dice_population_std",
        "lower",
    ),
    (
        "client_dice_25th_percentile",
        "evaluation.client_dispersion.mean_dice_25th_percentile",
        "higher",
    ),
    (
        "jain_selection_fairness",
        "selection_and_training.jain_selection_fairness",
        "higher",
    ),
    (
        "selected_completion_rate",
        "selection_and_training.selected_completion_rate",
        "higher",
    ),
    (
        "selection_pipeline_seconds",
        "selection_and_training.total_selection_pipeline_seconds",
        "lower",
    ),
    (
        "training_seconds",
        "selection_and_training.total_training_seconds",
        "lower",
    ),
    (
        "download_bytes",
        "selection_and_training.cumulative_download_bytes",
        "lower",
    ),
    (
        "upload_bytes",
        "selection_and_training.cumulative_upload_bytes",
        "lower",
    ),
)

T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for name in path.split("."):
        if not isinstance(current, Mapping) or name not in current:
            raise ValueError(f"Missing aggregate metric field: {path}.")
        current = current[name]
    return current


def _summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if len(numbers) < 3:
        raise ValueError("At least three seeds are required for aggregation.")
    mean = float(statistics.mean(numbers))
    deviation = float(statistics.stdev(numbers))
    standard_error = deviation / math.sqrt(len(numbers))
    critical = T_975.get(len(numbers) - 1, 1.96)
    margin = critical * standard_error
    return {
        "n": len(numbers),
        "values": numbers,
        "mean": mean,
        "sample_standard_deviation": deviation,
        "standard_error": standard_error,
        "student_t_95_percent_interval": [mean - margin, mean + margin],
    }


def _training_invariants(report: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(report["paired_invariants"]["training"])
    return {
        key: value
        for key, value in training.items()
        if not key.startswith("seeds.") and key != "system_trace"
    }


def _run_config(method: Mapping[str, Any]) -> dict[str, Any]:
    return _json_object(Path(method["run_dir"]) / "resolved_config.json")


def _validate_reports(
    seed_reports: Sequence[tuple[str, Path, dict[str, Any]]],
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[str], str, str]:
    if len(seed_reports) < 3:
        raise ValueError("At least three seed reports are required.")
    seed_ids = [seed_id for seed_id, _, _ in seed_reports]
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("Seed report identifiers must be unique.")
    reference_report = seed_reports[0][2]
    method_order = list(reference_report["method_order"])
    if manifest.get("protocol_status") != "frozen":
        raise ValueError("The study manifest is not frozen.")
    if list(manifest.get("methods", ())) != method_order:
        raise ValueError("Matrix methods differ from the frozen manifest.")
    expected_seeds = {
        str(seed["seed_id"]): seed for seed in manifest.get("seeds", ())
    }
    if set(expected_seeds) != {
        seed_id for seed_id, _, _ in seed_reports
    }:
        raise ValueError("Seed reports do not exactly match the frozen manifest.")
    manifest_sha = sha256_file(manifest_path)
    manifest_study_id = str(manifest.get("study_id", ""))
    reference_label = str(reference_report["reference_label"])
    reference_training = _training_invariants(reference_report)
    reference_evaluation = reference_report["paired_invariants"]["evaluation"]
    source_revision = reference_report["paired_invariants"]["runtime"][
        "source_revision"
    ]
    study_id = None
    protocol_sha = None
    scientific_seed_tuples = []
    importance_sha = None

    for seed_id, path, report in seed_reports:
        if report.get("purpose") != "matched_completed_method_matrix_comparison":
            raise ValueError(f"Seed report is not a completed comparison: {path}")
        if report.get("interpretation_allowed") is not True:
            raise ValueError(f"Seed report forbids scientific interpretation: {path}")
        if list(report.get("method_order", ())) != method_order:
            raise ValueError("Method order differs across seeds.")
        if report.get("reference_label") != reference_label:
            raise ValueError("Reference method differs across seeds.")
        if _training_invariants(report) != reference_training:
            raise ValueError("Training invariants differ across seeds.")
        if report["paired_invariants"]["evaluation"] != reference_evaluation:
            raise ValueError("Evaluation protocol differs across seeds.")
        if report["paired_invariants"]["runtime"]["source_revision"] != source_revision:
            raise ValueError("Source revision differs across seeds.")

        configurations = {
            label: _run_config(report["methods"][label])
            for label in method_order
        }
        seed_studies = {config["study"]["id"] for config in configurations.values()}
        seed_protocols = {
            config["study"]["protocol_sha256"]
            for config in configurations.values()
        }
        seed_values = {
            tuple(config["seeds"][name] for name in ("model", "selection", "data"))
            for config in configurations.values()
        }
        if len(seed_studies) != 1 or len(seed_protocols) != 1 or len(seed_values) != 1:
            raise ValueError(f"Study provenance is not matched within {seed_id}.")
        current_study = seed_studies.pop()
        current_protocol = seed_protocols.pop()
        expected_seed = expected_seeds[seed_id]
        expected_tuple = tuple(
            int(expected_seed[name])
            for name in ("model", "selection", "data")
        )
        observed_tuple = seed_values.pop()
        observed_trace_seed = int(
            configurations[reference_label]["system_trace"]["seed"]
        )
        if observed_tuple != expected_tuple:
            raise ValueError(f"Scientific seeds differ from manifest for {seed_id}.")
        if observed_trace_seed != int(expected_seed["trace"]):
            raise ValueError(f"Trace seed differs from manifest for {seed_id}.")
        if current_study != manifest_study_id or current_protocol != manifest_sha:
            raise ValueError("Run provenance differs from the supplied manifest.")
        if study_id is None:
            study_id = current_study
            protocol_sha = current_protocol
        elif current_study != study_id or current_protocol != protocol_sha:
            raise ValueError("Study manifest differs across seeds.")
        scientific_seed_tuples.append(observed_tuple)

        full_source = report["methods"][reference_label][
            "external_importance_source"
        ]
        if not isinstance(full_source, Mapping):
            raise ValueError("Full MICS has no external importance provenance.")
        if full_source.get("partition") != "calibration":
            raise ValueError("Full MICS importance did not use calibration data.")
        if full_source.get("complete_calibration_probe") is not True:
            raise ValueError("Full MICS importance calibration is incomplete.")
        if full_source.get("test_data_accessed") is not False:
            raise ValueError("Importance provenance does not exclude test data.")
        importance_study = full_source.get("study")
        if not isinstance(importance_study, Mapping):
            raise ValueError("Importance record has no frozen-study provenance.")
        if (
            importance_study.get("id") != manifest_study_id
            or importance_study.get("protocol_sha256") != manifest_sha
        ):
            raise ValueError("Importance record differs from the frozen study.")
        if importance_sha is None:
            importance_sha = full_source["sha256"]
        elif full_source["sha256"] != importance_sha:
            raise ValueError("Modality-importance weights differ across seeds.")

    if len(scientific_seed_tuples) != len(set(scientific_seed_tuples)):
        raise ValueError("Scientific seed tuples are not unique across reports.")
    return method_order, reference_label, str(study_id)


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate CSV: {path}")
    fields = (
        "method",
        "metric",
        "direction",
        "n",
        "mean",
        "sample_standard_deviation",
        "ci95_lower",
        "ci95_upper",
        "paired_delta_mean",
        "paired_delta_ci95_lower",
        "paired_delta_ci95_upper",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, method_report in report["methods"].items():
            for metric, metric_report in method_report["metrics"].items():
                interval = metric_report["across_seeds"][
                    "student_t_95_percent_interval"
                ]
                paired = metric_report.get("paired_candidate_minus_reference")
                paired_interval = (
                    None
                    if paired is None
                    else paired["student_t_95_percent_interval"]
                )
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "direction": metric_report["direction"],
                        "n": metric_report["across_seeds"]["n"],
                        "mean": metric_report["across_seeds"]["mean"],
                        "sample_standard_deviation": metric_report[
                            "across_seeds"
                        ]["sample_standard_deviation"],
                        "ci95_lower": interval[0],
                        "ci95_upper": interval[1],
                        "paired_delta_mean": None if paired is None else paired["mean"],
                        "paired_delta_ci95_lower": (
                            None if paired_interval is None else paired_interval[0]
                        ),
                        "paired_delta_ci95_upper": (
                            None if paired_interval is None else paired_interval[1]
                        ),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-report",
        action="append",
        nargs=2,
        required=True,
        metavar=("SEED_ID", "MATRIX_JSON"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate report: {output}")
    seed_reports = [
        (seed_id, Path(raw_path).resolve(), _json_object(Path(raw_path).resolve()))
        for seed_id, raw_path in arguments.seed_report
    ]
    manifest_path = arguments.manifest.resolve()
    manifest = _json_object(manifest_path)
    method_order, reference_label, study_id = _validate_reports(
        seed_reports,
        manifest_path,
        manifest,
    )

    methods: dict[str, Any] = {}
    for method in method_order:
        metrics = {}
        for metric_name, metric_path, direction in METRICS:
            values = [
                float(_nested(report["methods"][method], metric_path))
                for _, _, report in seed_reports
            ]
            metric_report: dict[str, Any] = {
                "direction": direction,
                "across_seeds": _summary(values),
            }
            if method != reference_label:
                reference_values = [
                    float(_nested(report["methods"][reference_label], metric_path))
                    for _, _, report in seed_reports
                ]
                metric_report["paired_candidate_minus_reference"] = _summary(
                    [
                        candidate - reference
                        for candidate, reference in zip(
                            values,
                            reference_values,
                            strict=True,
                        )
                    ]
                )
            metrics[metric_name] = metric_report
        methods[method] = {"metrics": metrics}

    report = {
        "schema_version": 1,
        "purpose": "matched_multiseed_mics_ablation_summary",
        "interpretation_allowed": True,
        "study_id": study_id,
        "protocol_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "reference_label": reference_label,
        "method_order": method_order,
        "seed_ids": [seed_id for seed_id, _, _ in seed_reports],
        "seed_reports": [
            {"seed_id": seed_id, "path": str(path), "sha256": sha256_file(path)}
            for seed_id, path, _ in seed_reports
        ],
        "uncertainty": (
            "Mean, sample standard deviation, and two-sided 95% Student-t "
            "interval across matched seeds; method effects are paired within seed."
        ),
        "methods": methods,
    }
    atomic_write_json(output, report)
    _write_csv(arguments.csv_output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(output),
                "csv_output": str(arguments.csv_output.resolve()),
                "study_id": study_id,
                "seed_count": len(seed_reports),
                "methods": method_order,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("MICS MULTI-SEED ABLATION AGGREGATION PASSED")


if __name__ == "__main__":
    main()
