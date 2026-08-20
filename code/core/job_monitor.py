"""Read-only health monitor for long E34/E35/E36 jobs.

The monitor never starts, stops, or modifies a research process. It checks a PID,
the freshness and content of a JSONL heartbeat, and optionally appends its own
observations to a separate JSONL file. Use ``--once`` from an hourly scheduler;
the scheduler will only be enabled after the user authorizes a heavy run.
"""

from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Health:
    checked_at_utc: str
    status: str
    reasons: tuple[str, ...]
    pid: int
    pid_alive: bool
    heartbeat_age_seconds: float | None
    heartbeat: dict[str, Any] | None
    process: dict[str, Any] | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        start = max(0, end - 256 * 1024)
        stream.seek(start)
        chunk = stream.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def process_snapshot(pid: int) -> dict[str, Any] | None:
    if not pid_alive(pid):
        return None
    command = [
        "ps",
        "-o",
        "pid=,%cpu=,rss=,etime=,state=,command=",
        "-p",
        str(pid),
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
    except OSError as error:
        # Sandboxed monitors may be allowed to test the PID but not invoke `ps`.
        return {"snapshot_error": f"{type(error).__name__}: {error}"}
    line = result.stdout.strip()
    if result.returncode != 0 or not line:
        return None
    fields = line.split(maxsplit=5)
    if len(fields) < 6:
        return {"raw": line}
    return {
        "pid": int(fields[0]),
        "cpu_percent": float(fields[1]),
        "rss_mb": int(fields[2]) / 1024.0,
        "elapsed": fields[3],
        "state": fields[4],
        "command": fields[5],
    }


def inspect(pid: int, heartbeat_path: Path, max_age_seconds: float) -> Health:
    reasons: list[str] = []
    alive = pid_alive(pid)
    heartbeat = read_last_jsonl(heartbeat_path)
    age: float | None = None

    if not alive:
        reasons.append("process_missing")
    if heartbeat is None:
        reasons.append("heartbeat_missing_or_invalid")
    else:
        age = max(0.0, time.time() - heartbeat_path.stat().st_mtime)
        if age > max_age_seconds:
            reasons.append("heartbeat_stale")
        if heartbeat.get("nan_count", 0):
            reasons.append("nan_or_inf_reported")
        heartbeat_pid = heartbeat.get("pid")
        if heartbeat_pid is not None and int(heartbeat_pid) != pid:
            reasons.append("heartbeat_pid_mismatch")

    status = "healthy" if not reasons else "alert"
    return Health(
        checked_at_utc=_utc_now(),
        status=status,
        reasons=tuple(reasons),
        pid=pid,
        pid_alive=alive,
        heartbeat_age_seconds=age,
        heartbeat=heartbeat,
        process=process_snapshot(pid),
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--max-age-min", type=float, default=15.0)
    parser.add_argument("--monitor-log", type=Path)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Accepted for explicit scheduler commands; one check is always performed.",
    )
    args = parser.parse_args()

    report = inspect(
        pid=args.pid,
        heartbeat_path=args.heartbeat,
        max_age_seconds=60.0 * args.max_age_min,
    )
    payload = asdict(report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.monitor_log is not None:
        append_jsonl(args.monitor_log, payload)
    return 0 if report.status == "healthy" else 2


if __name__ == "__main__":
    raise SystemExit(main())
