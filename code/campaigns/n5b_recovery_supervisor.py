"""Restart-safe supervisor for the interrupted N5B production campaign.

The original NEXT24H orchestrator exited after a smoke/production checkpoint
collision and killed its remaining children.  Four corrected memory workers
for seeds 42--45 are already running outside this supervisor.  This process
waits for a slot, launches seed 46, then starts four disjoint pinned-control
chunks only after all memory branches have completed.
"""
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
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

from v5_paths import LOGS, RUNS


SEEDS_INITIAL = [42, 43, 44, 45]
PINNED_CHUNKS = [
    [42, 43, 44, 45, 46],
    [47, 48, 49, 50, 51],
    [52, 53, 54, 55, 56],
    [57, 58, 59, 60, 61],
]
STATE = RUNS / "N5B" / "recovery_state.json"
SCRIPT = Path(__file__).with_name("n5b_long_delay.py")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def memory_done(seed: int) -> bool:
    return (
        RUNS / "N5B" /
        f"memory_scan_N2000_s{seed}_production.json"
    ).exists()


def write_state(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE)


def command(*arguments: str) -> list[str]:
    return [sys.executable, str(SCRIPT), *arguments]


def launch(arguments: list[str], log_name: str) -> tuple[subprocess.Popen, object]:
    log_path = LOGS / "NEXT24H" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n[{now()}] RECOVERY START {' '.join(arguments)}\n")
    handle.flush()
    process = subprocess.Popen(
        arguments, stdout=handle, stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )
    return process, handle


def main() -> None:
    state = {
        "started": now(),
        "status": "waiting_for_memory_slot",
        "seed46": {},
        "pinned": {},
        "note": (
            "Seeds 42--45 run in external execution sessions; their production "
            "checkpoint files are the authoritative completion signal."
        ),
    }
    write_state(state)

    while not any(memory_done(seed) for seed in SEEDS_INITIAL):
        time.sleep(30)

    if not memory_done(46):
        process, handle = launch(
            command("--mode", "memory", "--seeds", "46"),
            "N5B_memory_seed46_recovery.log",
        )
        state["status"] = "seed46_running"
        state["seed46"] = {
            "pid": process.pid, "started": now(),
            "command": process.args,
        }
        write_state(state)
        returncode = process.wait()
        handle.write(f"[{now()}] RECOVERY EXIT {returncode}\n")
        handle.close()
        state["seed46"].update({
            "ended": now(), "returncode": returncode,
        })
        if returncode != 0:
            state["status"] = "failed_seed46"
            write_state(state)
            raise SystemExit(returncode)

    while not all(memory_done(seed) for seed in [42, 43, 44, 45, 46]):
        time.sleep(30)

    state["status"] = "pinned_running"
    pinned_processes = []
    for index, seeds in enumerate(PINNED_CHUNKS, start=1):
        arguments = command(
            "--mode", "pinned", "--pinned-seeds",
            *[str(seed) for seed in seeds],
        )
        process, handle = launch(
            arguments, f"N5B_pinned_recovery_chunk{index}.log"
        )
        key = f"chunk{index}"
        state["pinned"][key] = {
            "seeds": seeds, "pid": process.pid, "started": now(),
            "command": process.args,
        }
        pinned_processes.append((key, process, handle))
    write_state(state)

    failed = False
    for key, process, handle in pinned_processes:
        returncode = process.wait()
        handle.write(f"[{now()}] RECOVERY EXIT {returncode}\n")
        handle.close()
        state["pinned"][key].update({
            "ended": now(), "returncode": returncode,
        })
        failed = failed or returncode != 0
        write_state(state)
    state["status"] = "failed_pinned" if failed else "completed"
    state["ended"] = now()
    write_state(state)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
