"""Tests for matched client availability and completion traces."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mics.experiments import generate_system_trace, load_system_trace


class SystemTraceTests(unittest.TestCase):
    def test_generation_is_deterministic_and_uses_independent_potentials(self) -> None:
        arguments = {
            "clients": ("C1", "C2", "C3"),
            "rounds": 12,
            "seed": 41,
            "availability_probabilities": {
                "C1": 0.7,
                "C2": 0.8,
                "C3": 0.9,
            },
            "completion_probabilities": {
                "C1": 0.0,
                "C2": 0.5,
                "C3": 1.0,
            },
            "minimum_available_clients": 2,
            "trace_id": "test-trace",
        }
        first = generate_system_trace(**arguments)
        second = generate_system_trace(**arguments)

        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertTrue(
            all(len(record.available_clients) >= 2 for record in first.rounds)
        )
        self.assertTrue(
            all(not record.completion_potential["C1"] for record in first.rounds)
        )
        self.assertTrue(
            all(record.completion_potential["C3"] for record in first.rounds)
        )

    def test_round_trip_and_successful_selected_order(self) -> None:
        trace = generate_system_trace(
            ("C1", "C2"),
            rounds=2,
            seed=9,
            availability_probabilities={"C1": 1.0, "C2": 1.0},
            completion_probabilities={"C1": 0.0, "C2": 1.0},
            minimum_available_clients=2,
            trace_id="round-trip",
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "trace.json"
            path.write_text(json.dumps(trace.as_dict()), encoding="utf-8")
            loaded = load_system_trace(path)

        self.assertIsNotNone(loaded.file_sha256)
        self.assertEqual(loaded.round(0).available_clients, ("C1", "C2"))
        self.assertEqual(
            loaded.successful_selected_clients(0, ("C2", "C1")),
            ("C2",),
        )

    def test_loader_rejects_non_boolean_completion_values(self) -> None:
        trace = generate_system_trace(
            ("C1", "C2"),
            rounds=1,
            seed=2,
            availability_probabilities={"C1": 1.0, "C2": 1.0},
            completion_probabilities={"C1": 0.5, "C2": 0.5},
            minimum_available_clients=1,
            trace_id="invalid-completion",
        ).as_dict()
        trace["rounds"][0]["completion_potential"]["C1"] = "false"
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "trace.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be Boolean"):
                load_system_trace(path)


if __name__ == "__main__":
    unittest.main()
