"""Tests for guarded multi-method comparison reporting."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _dice(value: float) -> dict[str, float]:
    return {region: value for region in ("WT", "TC", "ET", "mean")}


def _hd95(value: float) -> dict[str, float]:
    return {region: value for region in ("WT", "TC", "ET", "mean")}


def _write_method(
    root: Path,
    *,
    label: str,
    method: str,
    score: float,
) -> tuple[Path, Path]:
    run_dir = root / label
    run_dir.mkdir()
    config = {
        "purpose": "integration_smoke_not_model_quality",
        "method": method,
        "scenario": {
            "id": "balanced",
            "config_sha256": "scenario-sha",
            "file_sha256": {"split.csv": "split-sha"},
            "clients": {
                "C1": {"modalities": ["flair"]},
                "C2": {"modalities": ["t1"]},
            },
        },
        "model": {
            "modalities": ["flair", "t1"],
            "num_classes": 4,
            "base_channels": 2,
            "num_levels": 2,
            "fusion": "mean",
            "modality_dropout": 0.0,
        },
        "federation": {
            "rounds": 1,
            "clients_per_round": 1,
            "checkpoint_every": 1,
            "aggregation": "component_weighted_mean",
        },
        "selection": {"name": method},
        "training": {"max_batches_per_epoch": 1},
        "seeds": {"model": 101, "selection": 202, "data": 303},
        "device": "cpu",
        "system_trace": {"sha256": "trace-sha"},
    }
    metadata = {
        "source_revision": "abc123",
        "source_dirty": False,
        "python": "3.10.0",
        "pytorch": "2.5.1",
        "cuda_build": None,
        "gpu_names": [],
    }
    metric = {
        "round": 0,
        "selected_clients": ["C1"],
        "successful_clients": ["C1"],
        "failed_selected_clients": [],
        "available_clients": ["C1", "C2"],
        "selection_seconds": 0.01,
        "selection_probe": None,
        "training_seconds": 0.1,
        "examples_seen": 1,
        "sample_weighted_training_loss": 1.0,
        "component_update_counts": {
            "encoder:flair": 1,
            "encoder:t1": 0,
            "fusion": 1,
            "decoder": 1,
            "head": 1,
        },
        "selection_diagnostics": {
            "eligible_client_count": 2,
            "modality_coverage": {"flair": 1, "t1": 0},
            "selected_clients_per_modality": {"flair": 1, "t1": 0},
            "pairwise_modality_jaccard_redundancy": 0.0,
            "impact_weighted_modality_coverage": None,
            "selection_counts_after_round": {"C1": 1, "C2": 0},
            "jain_selection_fairness_after_round": 0.5,
        },
        "selector_outcome": {},
        "communication": {
            "cumulative_download_bytes": 100,
            "cumulative_upload_bytes": 50,
            "selection_probe_download_bytes": 0,
            "selection_probe_upload_bytes": 0,
        },
    }
    evaluation = {
        "schema_version": 1,
        "purpose": "integration_smoke_not_model_quality",
        "run_id": label,
        "completed_rounds": 1,
        "evaluation_source": {"source_revision": "abc123"},
        "evaluation": {
            "partition": "validation",
            "roi_size": [4, 4, 4],
            "overlap": 0.5,
            "spacing_mm": [1.0, 1.0, 1.0],
            "case_count": 2,
            "client_count": 2,
            "complete_partition": False,
            "macro_client": {
                "dice": _dice(score),
                "hd95_mm": _hd95(10.0 - score),
            },
            "sample_weighted": {
                "case_count": 2,
                "dice": _dice(score),
                "hd95_mm": _hd95(10.0 - score),
                "window_count": 2,
            },
            "client_dispersion": {
                "mean_dice_population_std": 0.0,
            },
            "total_evaluation_seconds": 0.2,
            "cases": [
                {
                    "client_id": "C1",
                    "case_id": "Case001",
                    "modalities": ["flair"],
                },
                {
                    "client_id": "C2",
                    "case_id": "Case002",
                    "modalities": ["t1"],
                },
            ],
        },
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed"}),
        encoding="utf-8",
    )
    (run_dir / "round_metrics.jsonl").write_text(
        json.dumps(metric) + "\n",
        encoding="utf-8",
    )
    evaluation_path = run_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    return run_dir, evaluation_path


class MethodMatrixComparisonTests(unittest.TestCase):
    def test_four_matched_methods_produce_guarded_report(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            specifications = (
                ("uniform_random", "uniform_random", 0.10),
                ("power_of_choice", "power_of_choice", 0.20),
                (
                    "divfl_gradient_facility",
                    "divfl_gradient_facility",
                    0.30,
                ),
                ("mics_impact", "mics_impact", 0.40),
            )
            command = [
                sys.executable,
                str(project_root / "scripts" / "compare_method_matrix.py"),
            ]
            for label, method, score in specifications:
                run_dir, evaluation = _write_method(
                    root,
                    label=label,
                    method=method,
                    score=score,
                )
                command.extend(
                    ("--method", label, str(run_dir), str(evaluation))
                )
            output = root / "matrix.json"
            command.extend(("--output", str(output)))
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "src")

            result = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.fail(
                    "Method-matrix comparison failed:\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn(
                "MICS MATCHED METHOD-MATRIX COMPARISON PASSED",
                result.stdout,
            )
            self.assertEqual(
                report["purpose"],
                "matched_method_matrix_integration_audit_not_model_quality",
            )
            self.assertFalse(report["interpretation_allowed"])
            self.assertEqual(len(report["methods"]), 4)
            difference = report["differences_from_reference"]["mics_impact"]
            self.assertAlmostEqual(
                difference["candidate_minus_reference_macro_dice"]["mean"],
                0.30,
            )
            self.assertEqual(
                report["paired_invariants"]["runtime"]["source_revision"],
                "abc123",
            )

    def test_mismatched_system_trace_is_rejected(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            reference_dir, reference_evaluation = _write_method(
                root,
                label="uniform_random",
                method="uniform_random",
                score=0.1,
            )
            candidate_dir, candidate_evaluation = _write_method(
                root,
                label="mics_impact",
                method="mics_impact",
                score=0.2,
            )
            candidate_config_path = candidate_dir / "resolved_config.json"
            candidate_config = json.loads(
                candidate_config_path.read_text(encoding="utf-8")
            )
            candidate_config["system_trace"] = {"sha256": "other-trace"}
            candidate_config_path.write_text(
                json.dumps(candidate_config),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "src")
            result = subprocess.run(
                (
                    sys.executable,
                    str(
                        project_root
                        / "scripts"
                        / "compare_method_matrix.py"
                    ),
                    "--method",
                    "uniform_random",
                    str(reference_dir),
                    str(reference_evaluation),
                    "--method",
                    "mics_impact",
                    str(candidate_dir),
                    str(candidate_evaluation),
                    "--output",
                    str(root / "must_not_exist.json"),
                ),
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different system trace", result.stderr)
            self.assertFalse((root / "must_not_exist.json").exists())


if __name__ == "__main__":
    unittest.main()
