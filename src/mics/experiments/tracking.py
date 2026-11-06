"""Crash-conscious, append-only experiment provenance and state recording."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    data = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _config_fingerprint(config: Any) -> str:
    canonical = json.dumps(
        config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_runtime_metadata(project_root: str | Path) -> dict[str, Any]:
    """Collect non-secret runtime and source provenance for a run."""

    root = Path(project_root).resolve()
    dirty_output = _git_value(root, "status", "--porcelain")
    scheduler_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURMD_NODENAME",
    )
    return {
        "recorded_utc": _utc_now(),
        "command": list(sys.argv),
        "source_revision": _git_value(root, "rev-parse", "HEAD"),
        "source_dirty": None if dirty_output is None else bool(dirty_output),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        ),
        "scheduler": {
            key: os.environ[key]
            for key in scheduler_keys
            if key in os.environ
        },
    }


class RunRecorder:
    """Own one immutable run directory and its append-only round records."""

    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        *,
        resolved_config: dict[str, Any],
        runtime_metadata: dict[str, Any],
        resume: bool = False,
    ) -> None:
        if not run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be non-empty.")
        if any(separator in run_id for separator in ("/", "\\")):
            raise ValueError("run_id cannot contain path separators.")

        self.run_id = run_id
        self.run_dir = Path(output_root).resolve() / run_id
        self.config_path = self.run_dir / "resolved_config.json"
        self.metadata_path = self.run_dir / "run_metadata.json"
        self.selection_path = self.run_dir / "selection_history.jsonl"
        self.metrics_path = self.run_dir / "round_metrics.jsonl"
        self.status_path = self.run_dir / "status.json"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_path = self.checkpoint_dir / "last.pt"
        requested_fingerprint = _config_fingerprint(resolved_config)

        if self.run_dir.exists():
            if not resume:
                raise FileExistsError(
                    "Run directory already exists; refusing overwrite: "
                    f"{self.run_dir}"
                )
            if not self.config_path.is_file():
                raise FileNotFoundError("Existing run has no resolved_config.json.")
            existing_config = json.loads(
                self.config_path.read_text(encoding="utf-8")
            )
            existing_fingerprint = _config_fingerprint(existing_config)
            if existing_fingerprint != requested_fingerprint:
                raise ValueError(
                    "Resume configuration does not match the existing run."
                )
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError("Existing run has no resumable checkpoint.")
            status = self.read_status()
            if status.get("state") == "completed":
                raise ValueError("A completed run cannot be resumed.")
        else:
            if resume:
                raise FileNotFoundError(f"Run does not exist: {self.run_dir}")
            self.checkpoint_dir.mkdir(parents=True)
            _atomic_json(self.config_path, resolved_config)
            _atomic_json(self.metadata_path, runtime_metadata)
            self._write_status("initialized", next_round=0)

        self.config_fingerprint = requested_fingerprint

    def _write_status(
        self,
        state: str,
        *,
        next_round: int,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "state": state,
            "next_round": int(next_round),
            "updated_utc": _utc_now(),
        }
        if error is not None:
            payload["error"] = str(error)
        _atomic_json(self.status_path, payload)

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            raise FileNotFoundError(
                f"Run status does not exist: {self.status_path}"
            )
        value = json.loads(self.status_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Run status must contain a JSON object.")
        return value

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def mark_running(self, *, next_round: int) -> None:
        self._write_status("running", next_round=next_round)

    def record_selection(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.selection_path, record)

    def record_round(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.metrics_path, record)

    def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        temporary = self.checkpoint_dir / ".last.pt.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, self.checkpoint_path)

    def load_checkpoint(self) -> dict[str, Any]:
        value = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(value, dict):
            raise ValueError("Checkpoint must contain a mapping.")
        if value.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError("Checkpoint configuration fingerprint does not match.")
        return value

    def mark_round_complete(self, *, next_round: int) -> None:
        self._write_status("running", next_round=next_round)

    def mark_completed(self, *, next_round: int) -> None:
        self._write_status("completed", next_round=next_round)

    def mark_failed(self, *, next_round: int, error: str) -> None:
        self._write_status("failed", next_round=next_round, error=error)
