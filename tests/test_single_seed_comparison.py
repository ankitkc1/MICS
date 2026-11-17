"""Historical-reference matching and full-study command regression checks."""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from test_method_matrix_comparison import _write_method

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_single_seed_comparison import (
    TRAINING_INVARIANTS,
    _bind_study,
    _compare,
    _evaluation_case_key,
    _evaluation_summary,
    _load_method,
    _method_location,
    _nested,
    _selection_summary,
    _reuse_binding,
    _verify_reuse_source,
    training_command,
    sha256_file,
    configuration_fingerprint,
    validate_candidate,
    validate_reference,
)


class SingleSeedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        run_dir, original_evaluation = _write_method(
            self.root,
            label="uniform_random",
            method="uniform_random",
            score=0.6,
        )
        config_path = run_dir / "resolved_config.json"
        config = json.loads(config_path.read_text())
        config.pop("selection")
        config["purpose"] = "full_training_checkpoint_for_held_out_evaluation"
        config["system_trace"].update({"seed": 3001, "path": "/trace.json"})
        config["training"] = {
            "max_batches_per_epoch": None,
            "amp": False,
            "patch_size": [4, 4, 4],
            "foreground_probability": 0.5,
            "batch_size": 1,
            "num_workers": 0,
            "local_epochs": 1,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "gradient_clip_norm": 5.0,
            "amp_initial_scale": 4096,
            "amp_max_retries": 8,
        }
        config_path.write_text(json.dumps(config))
        payload = json.loads(original_evaluation.read_text())
        payload["purpose"] = "held_out_validation"
        payload["evaluation"]["complete_partition"] = True
        payload["evaluation_source"]["source_dirty"] = False
        evaluation_path = run_dir / "evaluations" / "validation_after_round_0001.json"
        evaluation_path.parent.mkdir()
        evaluation_path.write_text(json.dumps(payload))
        self.candidate = _load_method(
            ("uniform_random", str(run_dir), str(evaluation_path))
        )
        runtime = copy.deepcopy(self.candidate.metadata)
        runtime["source_revision"] = "historical-ref"
        training = {field: _nested(config, field) for field in TRAINING_INVARIANTS}
        training["system_trace"] = config["system_trace"]
        self.reference = {
            "purpose": "matched_completed_method_matrix_comparison",
            "interpretation_allowed": True,
            "method_order": ["mics_v2_impact"],
            "paired_invariants": {
                "training": training,
                "runtime": runtime,
                "evaluation": {
                    field: payload["evaluation"][field]
                    for field in (
                        "partition",
                        "roi_size",
                        "overlap",
                        "spacing_mm",
                        "case_count",
                        "complete_partition",
                    )
                },
                "evaluation_cases_and_masks": _evaluation_case_key(
                    payload["evaluation"]
                ),
            },
            "methods": {
                "mics_v2_impact": {
                    "selection_configuration": {
                        "name": "mics_v2_impact",
                        "client_queue_enabled": True,
                    },
                    "external_importance_source": {"sha256": "importance-sha"},
                    "selection_and_training": _selection_summary(
                        self.candidate.metrics, config
                    ),
                    "evaluation": _evaluation_summary(payload),
                }
            },
        }
        self.manifest = {
            "protocol_status": "frozen_single_seed",
            "study_id": "test-study",
            "reference_label": "mics_v2_impact",
            "reference_source_revision": "historical-ref",
            "reference_inputs": {
                "scenario_config_sha256": "scenario-sha",
                "system_trace_sha256": "trace-sha",
                "modality_importance_sha256": "importance-sha",
            },
            "methods": ["uniform_random"],
            "rounds": 1,
            "clients_per_round": 1,
            "validation_case_count": 2,
            "probe_max_batches": None,
            "seeds": {"model": 101, "selection": 202, "data": 303, "trace": 3001},
            "power_of_choice_candidate_size": 2,
        }
        self.binding = {
            "reference_source_revision": "historical-ref",
            "baseline_source_revision": "abc123",
            "reference_matrix_sha256": "reference-sha",
            "manifest_sha256": "manifest-sha",
        }

    def test_reference_checks_seeds_hashes_and_complete_evaluation(self):
        validate_reference(self.manifest, self.reference)
        for key, value in (
            ("seeds", {**self.manifest["seeds"], "data": 999}),
            ("reference_source_revision", "wrong-revision"),
        ):
            changed = {**self.manifest, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reference(changed, self.reference)
        changed = copy.deepcopy(self.reference)
        changed["paired_invariants"]["evaluation"]["complete_partition"] = False
        with self.assertRaises(ValueError):
            validate_reference(self.manifest, changed)

    def test_candidate_allows_disclosed_revision_and_rejects_contract_drift(self):
        validate_candidate(self.reference, self.candidate)
        changed = copy.deepcopy(self.candidate)
        changed.config["seeds"]["selection"] = 999
        with self.assertRaises(ValueError):
            validate_candidate(self.reference, changed)
        changed = copy.deepcopy(self.candidate)
        changed.evaluation_payload["evaluation"]["cases"][0]["modalities"] = ["t1"]
        with self.assertRaises(ValueError):
            validate_candidate(self.reference, changed)
        changed = copy.deepcopy(self.candidate)
        changed.evaluation_payload["evaluation_source"]["source_dirty"] = True
        with self.assertRaises(ValueError):
            validate_candidate(self.reference, changed)

    def test_commands_copy_reference_seeds_and_omit_training_truncation(self):
        command = training_command(
            self.manifest,
            self.reference,
            "mfedmc_client_low_loss",
            project_root=PROJECT_ROOT,
            output_root=self.root,
            dataset_root=self.root / "data",
            scenario_config=self.root / "scenario.json",
        )
        self.assertEqual(command[command.index("--method") + 1], "mfedmc")
        self.assertEqual(command[command.index("--selection-seed") + 1], "202")
        self.assertEqual(command[command.index("--data-seed") + 1], "303")
        self.assertNotIn("--max-batches-per-epoch", command)
        self.assertNotIn("--probe-max-batches", command)

    def test_comparison_retains_both_revisions_and_completed_fairness(self):
        _compare(self.manifest, self.reference, self.root, self.binding)
        result = json.loads((self.root / "single_seed_method_matrix.json").read_text())
        self.assertFalse(result["source_revisions_match"])
        self.assertEqual(
            result["methods"]["mics_v2_impact"]["source_revision"], "historical-ref"
        )
        self.assertEqual(
            result["methods"]["uniform_random"]["source_revision"], "abc123"
        )
        selection = result["methods"]["uniform_random"]["selection_and_training"]
        self.assertEqual(selection["successful_completion_counts"], {"C1": 1, "C2": 0})
        self.assertEqual(selection["jain_completion_fairness"], 0.5)
        _compare(self.manifest, self.reference, self.root, self.binding)
        with self.assertRaises(ValueError):
            _bind_study(self.root, {**self.binding, "manifest_sha256": "changed"})

    def _repair_fixture(self):
        original = {**self.manifest, "study_id": "original-study"}
        original_path = self.root / "original-manifest.json"
        original_path.write_text(json.dumps(original))
        reference_path = self.root / "reference.json"
        reference_path.write_text(json.dumps(self.reference))
        old_root = self.root / "results" / "studies" / original["study_id"]
        old_root.mkdir(parents=True)
        shutil.copytree(self.candidate.run_dir, old_root / "uniform_random")
        evaluation_path = (
            old_root
            / "uniform_random"
            / "evaluations"
            / "validation_after_round_0001.json"
        )
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["checkpoint"] = {
            "config_fingerprint": configuration_fingerprint(self.candidate.config)
        }
        evaluation_path.write_text(json.dumps(evaluation))
        old_binding = {
            **self.binding,
            "manifest_sha256": sha256_file(original_path),
            "reference_matrix_sha256": sha256_file(reference_path),
        }
        (old_root / "study_binding.json").write_text(json.dumps(old_binding))
        repair = {
            **self.manifest,
            "completed_baseline_reuse": {
                "manifest": str(original_path),
                "source_revision": "abc123",
                "methods": ["uniform_random"],
            },
        }
        return repair, reference_path, old_root

    def test_repair_freezes_reused_artifacts_and_retains_their_source(self):
        repair, reference_path, old_root = self._repair_fixture()
        with patch(
            "run_single_seed_comparison._verify_reuse_source",
            return_value=["src/mics/selection/vendor/oort_training.py"],
        ) as verify:
            reuse = _reuse_binding(repair, self.reference, reference_path, self.root)
            verify.assert_called_once_with(self.root, "abc123")
            binding = {
                **self.binding,
                "baseline_source_revision": "fixed-source",
                "completed_baseline_reuse": reuse,
            }
            new_root = self.root / "repaired-study"
            _compare(repair, self.reference, new_root, binding)
            result = json.loads(
                (new_root / "single_seed_method_matrix.json").read_text()
            )
            self.assertEqual(
                result["methods"]["uniform_random"]["source_revision"], "abc123"
            )
            self.assertEqual(
                Path(result["methods"]["uniform_random"]["run_dir"]),
                old_root / "uniform_random",
            )
            self.assertEqual(
                _method_location("oort_training", new_root, binding),
                (new_root / "oort_training", "fixed-source"),
            )
            metrics = old_root / "uniform_random" / "round_metrics.jsonl"
            record = json.loads(metrics.read_text())
            record["training_seconds"] += 0.1
            metrics.write_text(json.dumps(record) + "\n")
            changed = _reuse_binding(repair, self.reference, reference_path, self.root)
            with self.assertRaises(ValueError):
                _bind_study(new_root, {**binding, "completed_baseline_reuse": changed})

    def test_repair_rejects_original_binding_and_protocol_drift(self):
        repair, reference_path, old_root = self._repair_fixture()
        with patch("run_single_seed_comparison._verify_reuse_source", return_value=[]):
            changed = {**repair, "power_of_choice_candidate_size": 999}
            with self.assertRaises(ValueError):
                _reuse_binding(changed, self.reference, reference_path, self.root)
            evaluation_path = (
                old_root
                / "uniform_random"
                / "evaluations"
                / "validation_after_round_0001.json"
            )
            evaluation = json.loads(evaluation_path.read_text())
            original_evaluation = copy.deepcopy(evaluation)
            evaluation["checkpoint"]["config_fingerprint"] = "wrong-fingerprint"
            evaluation_path.write_text(json.dumps(evaluation))
            with self.assertRaises(ValueError):
                _reuse_binding(repair, self.reference, reference_path, self.root)
            evaluation_path.write_text(json.dumps(original_evaluation))
            binding_path = old_root / "study_binding.json"
            original_binding = json.loads(binding_path.read_text())
            original_binding["baseline_source_revision"] = "different-source"
            binding_path.write_text(json.dumps(original_binding))
            with self.assertRaises(ValueError):
                _reuse_binding(repair, self.reference, reference_path, self.root)

    def test_repair_rejects_shared_training_source_changes(self):
        success = subprocess.CompletedProcess([], 0, stdout="")
        allowed = subprocess.CompletedProcess(
            [], 0, stdout="src/mics/selection/vendor/oort_training.py\n"
        )
        forbidden = subprocess.CompletedProcess(
            [], 0, stdout="src/mics/training/local.py\n"
        )
        with patch(
            "run_single_seed_comparison.subprocess.run", side_effect=[success, allowed]
        ):
            self.assertEqual(
                _verify_reuse_source(self.root, "abc123"),
                ["src/mics/selection/vendor/oort_training.py"],
            )
        with patch(
            "run_single_seed_comparison.subprocess.run",
            side_effect=[success, forbidden],
        ):
            with self.assertRaises(ValueError):
                _verify_reuse_source(self.root, "abc123")


if __name__ == "__main__":
    unittest.main()
