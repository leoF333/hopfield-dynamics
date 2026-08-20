"""Restart-safe, resource-bounded execution of the requested 24-hour campaigns."""
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
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from v5_paths import ROOT, RUNS, write_json


LOGS = ROOT / "logs" / "NEXT24H"
STATE = RUNS / "NEXT24H" / "state.json"
SEEDS = [42, 43, 44, 45, 46]
N2_SEEDS = [42, 60, 47, 52, 51]


@dataclass(frozen=True)
class Task:
    name: str
    command: list[str]


def python_task(name: str, script: str, *arguments: str) -> Task:
    return Task(name, [sys.executable, str(ROOT / "src" / script), *arguments])


def phases() -> list[tuple[str, list[Task]]]:
    audit = [
        python_task(
            f"N1_AUDIT_seed{seed}",
            "n1_convergence_audit.py",
            "--mode", "dynamic",
            "--seeds", str(seed),
            "--dts", "0.005",
            "--tours", "5",
            "--bracket-tol", "5e-5",
            "--hysteresis",
        )
        for seed in [42, 44, 48, 61]
    ]
    n2 = [
        python_task(
            f"N2_seed{seed}", "n2_local_laws.py", "--seeds", str(seed)
        )
        for seed in N2_SEEDS
    ]
    n6 = [
        python_task(
            f"N6_seed{seed}", "n6_lyapunov_campaign.py", "--seeds", str(seed)
        )
        for seed in SEEDS
    ]
    n8 = [
        python_task(
            f"N8_seed{seed}", "n8_highload.py", "--seeds", str(seed)
        )
        for seed in SEEDS
    ]
    n5a = [
        python_task(
            f"N5A_boundary_seed{seed}",
            "n5a_delay_scan.py",
            "--mode", "boundary",
            "--seeds", str(seed),
        )
        for seed in SEEDS
    ]
    n5b_memory = [
        python_task(
            f"N5B_memory_seed{seed}",
            "n5b_long_delay.py",
            "--mode", "memory",
            "--seeds", str(seed),
        )
        for seed in SEEDS
    ]
    pinned_chunks = [
        list(range(42, 47)),
        list(range(47, 52)),
        list(range(52, 57)),
        list(range(57, 62)),
    ]
    n5b_pinned = [
        python_task(
            f"N5B_pinned_{chunk[0]}_{chunk[-1]}",
            "n5b_long_delay.py",
            "--mode", "pinned",
            "--pinned-seeds", *(str(seed) for seed in chunk),
        )
        for chunk in pinned_chunks
    ]
    finalize = [
        python_task(
            "N1_AUDIT_finalize",
            "n1_convergence_audit.py",
            "--mode", "dynamic",
            "--seeds", "42", "44", "48", "61",
            "--dts", "0.005",
            "--tours", "5",
            "--bracket-tol", "5e-5",
            "--hysteresis",
        ),
        python_task(
            "N2_finalize",
            "n2_local_laws.py",
            "--seeds", *(str(seed) for seed in N2_SEEDS),
        ),
        python_task(
            "N6_finalize",
            "n6_lyapunov_campaign.py",
            "--seeds", *(str(seed) for seed in SEEDS),
        ),
        python_task(
            "N8_finalize",
            "n8_highload.py",
            "--seeds", *(str(seed) for seed in SEEDS),
        ),
        python_task(
            "N5A_finalize",
            "n5a_delay_scan.py",
            "--mode", "all",
            "--seeds", *(str(seed) for seed in SEEDS),
        ),
        python_task("N5A_analyse", "n5a_analyse.py"),
        python_task(
            "N5B_finalize",
            "n5b_long_delay.py",
            "--mode", "all",
            "--seeds", *(str(seed) for seed in SEEDS),
            "--pinned-seeds", *(str(seed) for seed in range(42, 62)),
        ),
    ]
    return [
        ("N1_AUDIT", audit),
        ("N2", n2),
        ("N6", n6),
        ("N8", n8),
        ("N5A", n5a),
        ("N5B_MEMORY", n5b_memory),
        ("N5B_PINNED", n5b_pinned),
        ("FINALIZE", finalize),
    ]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {
        "created": now(),
        "updated": now(),
        "status": "not_started",
        "max_parallel": 4,
        "phases": {},
        "events": [],
    }


def save_state(state: dict) -> None:
    state["updated"] = now()
    write_json(STATE, state)


def run_batch(phase: str, tasks: list[Task], state: dict) -> None:
    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    LOGS.mkdir(parents=True, exist_ok=True)
    phase_state = state["phases"].setdefault(
        phase, {"status": "pending", "tasks": {}}
    )
    phase_state["status"] = "running"
    phase_state["started"] = phase_state.get("started", now())
    phase_state.pop("failed_task", None)
    phase_state.pop("ended", None)
    save_state(state)

    for start in range(0, len(tasks), 4):
        batch = tasks[start : start + 4]
        active = []
        for task in batch:
            task_state = phase_state["tasks"].setdefault(task.name, {})
            if task_state.get("status") == "completed":
                continue
            log_path = LOGS / f"{task.name}.log"
            stream = log_path.open("a", encoding="utf-8")
            stream.write(
                f"\n[{now()}] START {' '.join(task.command)}\n"
            )
            stream.flush()
            process = subprocess.Popen(
                task.command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            task_state.update({
                "status": "running",
                "started": now(),
                "pid": process.pid,
                "log": str(log_path),
                "command": task.command,
            })
            task_state.pop("ended", None)
            task_state.pop("returncode", None)
            active.append((task, process, stream))
        save_state(state)

        while active:
            remaining = []
            for task, process, stream in active:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((task, process, stream))
                    continue
                stream.write(f"[{now()}] EXIT {returncode}\n")
                stream.close()
                task_state = phase_state["tasks"][task.name]
                task_state.update({
                    "status": "completed" if returncode == 0 else "failed",
                    "ended": now(),
                    "returncode": returncode,
                })
                save_state(state)
                if returncode != 0:
                    phase_state.update({
                        "status": "failed",
                        "ended": now(),
                        "failed_task": task.name,
                    })
                    state["status"] = "failed"
                    save_state(state)
                    for _, other, other_stream in remaining:
                        if other.poll() is None:
                            other.terminate()
                        other_stream.close()
                    raise RuntimeError(
                        f"{phase}/{task.name} failed with code {returncode}"
                    )
            active = remaining
            if active:
                time.sleep(30)

    phase_state.update({"status": "completed", "ended": now()})
    state["events"].append({"time": now(), "phase_completed": phase})
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-phase")
    args = parser.parse_args()
    all_phases = phases()
    names = [name for name, _ in all_phases]
    if args.start_phase and args.start_phase not in names:
        raise SystemExit(f"unknown phase {args.start_phase}; choose from {names}")
    start_index = names.index(args.start_phase) if args.start_phase else 0
    state = load_state()
    state["status"] = "running"
    state["orchestrator_pid"] = os.getpid()
    state["ordered_phases"] = names
    save_state(state)
    try:
        for phase, tasks in all_phases[start_index:]:
            run_batch(phase, tasks, state)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        save_state(state)
        raise
    state["status"] = "completed"
    state["completed"] = now()
    save_state(state)


if __name__ == "__main__":
    main()
