"""Safe artifact, heartbeat and run-mode helpers for E35 scripts."""

from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


LIMITATIONS = (
    "No parameter scan, fold, SNIC or Floquet certification in T1/V0.",
    "Smoke mode is too short to establish a valid recall cycle.",
    "B2 models periodic global translation only, not rotation or occlusion.",
    "TSVD cutoffs are supplied explicitly; they are not selected on hidden frames.",
    "V0 compares only JH_KH and a candidate pre-specified before target scoring.",
    "Exploration uses seed 42; any important contrast needs targeted multi-seed validation.",
    "Production horizons are provisional until a benchmark shows at least three post-transient tours.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any], lock=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock is None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")
        return
    with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


class RunArtifacts:
    """Manifest, append-only events, minute heartbeats and atomic checkpoints."""

    def __init__(
        self,
        root: Path,
        run_id: str,
        config: dict[str, Any],
        *,
        repository: Path,
        heartbeat_interval: float = 60.0,
    ) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.config = dict(config)
        self.repository = Path(repository)
        self.heartbeat_interval = float(heartbeat_interval)
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self.log_path = self.root / "logs" / f"{run_id}.jsonl"
        self.heartbeat_path = self.root / "logs" / f"{run_id}_heartbeat.jsonl"
        self.manifest_path = self.root / "manifests" / f"{run_id}.json"
        self.checkpoint_path = self.root / "checkpoints" / f"{run_id}.npz"
        self.data_path = self.root / "data" / f"{run_id}.npz"
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._progress: dict[str, Any] = {
            "stage": "initializing",
            "completed": 0,
            "total": 0,
            "nan_count": 0,
        }

    def __enter__(self) -> "RunArtifacts":
        for name in ("logs", "manifests", "checkpoints", "data", "figures"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": self.run_id,
            "created_at_utc": utc_now(),
            "pid": os.getpid(),
            "git_commit": git_commit(self.repository),
            "config": self.config,
            "limitations": LIMITATIONS,
        }
        atomic_json(self.manifest_path, manifest)
        self.event("run_started")
        self._emit_heartbeat()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self.run_id}-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.update(
            stage="failed" if exc_type is not None else "finished",
            nan_count=self._progress.get("nan_count", 0),
        )
        self._emit_heartbeat()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=min(2.0, self.heartbeat_interval + 0.1))
        self.event(
            "run_failed" if exc_type is not None else "run_finished",
            error=None if exc_value is None else repr(exc_value),
        )

    def event(self, event: str, **payload: Any) -> None:
        append_jsonl(
            self.log_path,
            {"timestamp_utc": utc_now(), "event": event, **payload},
            self._write_lock,
        )

    def update(self, **progress: Any) -> None:
        with self._state_lock:
            self._progress.update(progress)

    def checkpoint(
        self,
        *,
        hist: np.ndarray,
        dhist: np.ndarray,
        architecture: str,
        simulated_time: float,
        completed_steps: int,
    ) -> None:
        temporary = self.checkpoint_path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            hist=np.asarray(hist, dtype=np.float64),
            dhist=np.asarray(dhist, dtype=np.float64),
            architecture=np.array(architecture),
            simulated_time=np.array(simulated_time),
            completed_steps=np.array(completed_steps),
            config_json=np.array(json.dumps(json_safe(self.config), sort_keys=True)),
        )
        os.replace(temporary, self.checkpoint_path)
        self.event(
            "checkpoint",
            architecture=architecture,
            simulated_time=simulated_time,
            completed_steps=completed_steps,
        )

    def save_data(self, **arrays: Any) -> None:
        temporary = self.data_path.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, self.data_path)
        self.event("data_saved", path=str(self.data_path))

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval):
            self._emit_heartbeat()

    def _emit_heartbeat(self) -> None:
        with self._state_lock:
            progress = dict(self._progress)
        append_jsonl(
            self.heartbeat_path,
            {
                "timestamp_utc": utc_now(),
                "pid": os.getpid(),
                **progress,
            },
            self._write_lock,
        )


@contextmanager
def exclusive_compute_lock(root: Path, run_id: str) -> Iterator[None]:
    """Fail rather than overlap with another E35 heavy-capable process."""

    lock_path = Path(root) / ".e35_compute.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"E35 compute lock already exists at {lock_path}: {owner}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"run_id": run_id, "pid": os.getpid(), "created_at_utc": utc_now()}
                )
                + "\n"
            )
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def make_run_id(prefix: str, mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{mode}_{stamp}_{os.getpid()}"
