"""Tests for matched multi-seed MICS ablation aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


METHODS = (
    "mics_impact",
    "mics_core",
    "mics_impact_no_queue",
    "mics_impact_modular",
    "mics_impact_uniform",
)


def _summary(score: float) -> dict:
    regions = {name: score for name in ("WT", "TC", "ET", "mean")}
    return {
        "evaluation": {
            "macro_client": {
                "dice": regions,
                "hd95_mm": {
                    name: 100.0 - score for name in regions
                },
            },
            "sample_weighted": {"dice": regions},
            "client_dispersion": {
                "mean_dice_population_std": 0.1,
                "mean_dice_25th_percentile": score - 0.1,
            },
        },
        "selection_and_training": {
            "jain_selection_fairness": 0.8,
            "selected_completion_rate": 0.9,
            "total_selection_pipeline_seconds": 1.0,
            "total_training_seconds": 10.0,
            "cumulative_download_bytes": 100,
            "cumulative_upload_bytes": 50,
        },
    }


def _write_seed(root: Path, seed_index: int, protocol_sha: str) -> Path:
    seed_id = f"s{seed_index:02d}"
    methods = {}
    for method_index, method in enumerate(METHODS):
        run_dir = root / seed_id / method
        run_dir.mkdir(parents=True)
        config = {
            "study": {
                "id": "mics_ablation_balanced_v1",
                "protocol_sha256": protocol_sha,
            },
            "seeds": {
                "model": 100 + seed_index,
                "selection": 200 + seed_index,
                "data": 300 + seed_index,
            },
            "system_trace": {"seed": 400 + seed_index},
        }
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        score = 0.5 + 0.01 * seed_index - 0.02 * method_index
        methods[method] = {
            "run_dir": str(run_dir),
            **_summary(score),
            "external_importance_source": (
                {
                    "sha256": "importance-sha",
                    "partition": "calibration",
                    "complete_calibration_probe": True,
                    "test_data_accessed": False,
                    "study": {
                        "id": "mics_ablation_balanced_v1",
                        "protocol_sha256": protocol_sha,
                    },
                }
                if method == "mics_impact"
                else None
            ),
        }
    report = {
        "purpose": "matched_completed_method_matrix_comparison",
        "interpretation_allowed": True,
        "reference_label": "mics_impact",
        "method_order": list(METHODS),
        "methods": methods,
        "paired_invariants": {
            "training": {
                "model": {"base_channels": 16},
                "seeds.model": 100 + seed_index,
                "seeds.selection": 200 + seed_index,
                "seeds.data": 300 + seed_index,
                "system_trace": {"sha256": f"trace-{seed_index}"},
            },
            "runtime": {"source_revision": "abc123"},
            "evaluation": {"partition": "validation", "case_count": 24},
        },
    }
    path = root / seed_id / "ablation_matrix.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class AblationStudyAggregationTests(unittest.TestCase):
    def test_aggregates_paired_seed_effects_and_writes_csv(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "study_id": "mics_ablation_balanced_v1",
                        "protocol_status": "frozen",
                        "methods": list(METHODS),
                        "seeds": [
                            {
                                "seed_id": f"s{index:02d}",
                                "model": 100 + index,
                                "selection": 200 + index,
                                "data": 300 + index,
                                "trace": 400 + index,
                            }
                            for index in range(1, 4)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            protocol_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            paths = [
                _write_seed(root, index, protocol_sha)
                for index in range(1, 4)
            ]
            output = root / "aggregate.json"
            csv_output = root / "aggregate.csv"
            command = [
                sys.executable,
                str(project_root / "scripts" / "aggregate_ablation_study.py"),
                "--manifest",
                str(manifest),
            ]
            for index, path in enumerate(paths, start=1):
                command.extend(("--seed-report", f"s{index:02d}", str(path)))
            command.extend(
                ("--output", str(output), "--csv-output", str(csv_output))
            )
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
                    f"Aggregation failed:\n{result.stdout}\n{result.stderr}"
                )
            report = json.loads(output.read_text(encoding="utf-8"))
            paired = report["methods"]["mics_core"]["metrics"][
                "macro_dice_mean"
            ]["paired_candidate_minus_reference"]
            self.assertAlmostEqual(paired["mean"], -0.02)
            self.assertEqual(paired["n"], 3)
            self.assertTrue(csv_output.is_file())
            self.assertIn(
                "MICS MULTI-SEED ABLATION AGGREGATION PASSED",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
