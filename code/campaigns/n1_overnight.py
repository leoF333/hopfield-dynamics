"""Conservative restart-safe overnight orchestration for N1 dynamics.

The orchestrator never launches a seed whose final output exists or whose current
checkpoint/log has been updated recently by a pre-existing process.  It runs at
most four new seeds concurrently, writes a machine-readable state every minute,
and records the resource information available inside the Codex sandbox hourly.
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
from datetime import datetime, timezone
from pathlib import Path

from v5_paths import LOGS, ROOT, RUNS, environment_manifest, write_json


EXPECTED_DYNAMICS_SHA: str | None = None
# Outputs that completed before streaming relay accumulation are still valid:
# every successful ordered point stopped at five tours before the old 10,000-time
# pruning boundary. Only censored long trajectories were affected by that bug.
APPROVED_PRE_STREAMING_DYNAMICS_SHA = (
    "bfe196db07865c23a0870341833d6681dd38a8f7cdef0d328adf22bba7df6b2d"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def final_paths(seed: int) -> tuple[Path, Path]:
    base = RUNS / "N1" / f"dynamic_N2000_P100_seed{seed}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def checkpoint_path(seed: int) -> Path:
    return RUNS / "N1" / f"dynamic_N2000_P100_seed{seed}.checkpoint.npz"


def log_path(seed: int) -> Path:
    return LOGS / f"N1_production_seed{seed}.log"


def has_terminal_failure(seed: int) -> bool:
    """Recognize a finished wrapper error without matching older appended text."""
    path = log_path(seed)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        lines = [
            line.strip() for line in path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines() if line.strip()
        ]
    except OSError:
        return False
    return bool(lines) and lines[-1].startswith(
        "ERROR conda.cli.main_run:execute"
    )


def completed(seed: int) -> bool:
    npz, sidecar = final_paths(seed)
    if not (npz.exists() and sidecar.exists()):
        return False
    try:
        summary = json.loads(sidecar.read_text(encoding="utf-8"))
        source_hash = summary["environment"]["local_source_sha256"][
            "src/dynamics.py"
        ]
        config = summary["config"]
        return (
            summary.get("independent_history_control_passed") is True
            and "arrest_primary_motif" in summary
            and config.get("N") == 2000
            and config.get("P") == 100
            and config.get("seed") == seed
            and (
                EXPECTED_DYNAMICS_SHA is None
                or source_hash in {
                    EXPECTED_DYNAMICS_SHA,
                    APPROVED_PRE_STREAMING_DYNAMICS_SHA,
                }
            )
        )
    except (KeyError, OSError, ValueError, TypeError):
        return False


def latest_activity(seed: int) -> float | None:
    candidates = [
        path.stat().st_mtime
        for path in (checkpoint_path(seed), log_path(seed))
        if path.exists()
    ]
    return max(candidates) if candidates else None


def resource_snapshot() -> dict:
    snapshot = {
        "time_utc": utc_now(),
        "gpu": "not observable inside Codex sandbox; MLX/Metal disabled",
    }
    commands = {
        "cpu_load": ["uptime"],
        "memory_pressure": ["memory_pressure", "-Q"],
        "disk": ["df", "-h", str(ROOT)],
    }
    for key, command in commands.items():
        try:
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=15,
                check=False,
            )
            snapshot[key] = (result.stdout or result.stderr).strip()
        except Exception as exc:
            snapshot[key] = f"unavailable: {type(exc).__name__}: {exc}"
    return snapshot


def state_payload(
    *,
    phase: str,
    seeds: list[int],
    running: list[int],
    failures: dict[int, int],
    resources: list[dict],
    environment: dict,
    message: str = "",
) -> dict:
    return {
        "campaign": "N1 overnight dynamics",
        "updated_utc": utc_now(),
        "phase": phase,
        "message": message,
        "requested_seeds": seeds,
        "completed_seeds": [seed for seed in seeds if completed(seed)],
        "running_seeds": running,
        "checkpoint_activity_utc": {
            str(seed): (
                datetime.fromtimestamp(latest_activity(seed), timezone.utc).isoformat()
                if latest_activity(seed) is not None else None
            )
            for seed in seeds
        },
        "failures": {str(seed): code for seed, code in failures.items()},
        "resource_snapshots": resources,
        "environment": environment,
    }


def main() -> None:
    global EXPECTED_DYNAMICS_SHA

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(42, 62))
    )
    parser.add_argument(
        "--wait-active", type=int, nargs="*", default=[42, 43, 44, 45]
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stale-minutes", type=float, default=20.0)
    parser.add_argument("--retry-max-time", type=float, default=40000.0)
    parser.add_argument(
        "--retry-seeds", type=int, nargs="*", default=[],
        help=(
            "Explicitly route incomplete checkpointed seeds to the extended-time "
            "pass. This is intended for cleanly interrupted extended retries."
        ),
    )
    args = parser.parse_args()
    if args.max_parallel < 1 or args.max_parallel > 4:
        raise SystemExit("--max-parallel must be between 1 and 4")
    unknown_retry_seeds = sorted(set(args.retry_seeds) - set(args.seeds))
    if unknown_retry_seeds:
        raise SystemExit(
            f"--retry-seeds must be included in --seeds: {unknown_retry_seeds}"
        )

    status_path = RUNS / "N1" / "overnight_status.json"
    orchestrator_log = LOGS / "N1_overnight_orchestrator.log"
    environment = environment_manifest()
    EXPECTED_DYNAMICS_SHA = environment["local_source_sha256"]["src/dynamics.py"]
    resources = [resource_snapshot()]
    failures: dict[int, int] = {}
    last_resource = time.monotonic()

    def record(phase: str, running: list[int], message: str = "") -> None:
        write_json(
            status_path,
            state_payload(
                phase=phase, seeds=args.seeds, running=running,
                failures=failures, resources=resources[-24:],
                environment=environment,
                message=message,
            ),
        )
        with orchestrator_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {phase} {running} {message}\n")

    # Wait for the explicitly declared pre-existing batch.  Fresh checkpoints
    # prove liveness; stale activity is an error, never a license to duplicate.
    waiting = [seed for seed in args.wait_active if not completed(seed)]
    while waiting:
        terminal = [seed for seed in waiting if has_terminal_failure(seed)]
        if terminal:
            for seed in terminal:
                failures[seed] = 1
            waiting = [seed for seed in waiting if seed not in failures]
            record(
                "preexisting_batch_has_unresolved_seeds", waiting,
                f"terminal unresolved seeds {terminal}; no relaunch attempted",
            )
            if not waiting:
                break
        now = time.time()
        stale = [
            seed for seed in waiting
            if latest_activity(seed) is None
            or now - latest_activity(seed) > args.stale_minutes * 60.0
        ]
        if stale:
            record(
                "blocked_stale_preexisting_batch", waiting,
                f"stale seeds {stale}; no duplicate launch attempted",
            )
            raise SystemExit(3)
        if time.monotonic() - last_resource >= 3600.0:
            resources.append(resource_snapshot())
            last_resource = time.monotonic()
        record("waiting_for_preexisting_batch", waiting)
        time.sleep(min(args.poll_seconds, 60.0))
        waiting = [
            seed for seed in args.wait_active
            if not completed(seed) and seed not in failures
        ]

    # Recover terminal outcomes from earlier batches when this orchestrator is
    # itself restarted.  They are not rerun at the original time cap.
    for seed in args.seeds:
        if not completed(seed) and has_terminal_failure(seed):
            failures[seed] = 1
    for seed in args.retry_seeds:
        if not completed(seed):
            failures[seed] = failures.get(seed, 130)

    def run_batches(
        batch_seeds: list[int], *, max_time: float, phase: str, retry: bool
    ) -> None:
        nonlocal last_resource

        for start in range(0, len(batch_seeds), args.max_parallel):
            batch = batch_seeds[start:start + args.max_parallel]
            processes: dict[int, tuple[subprocess.Popen, object]] = {}
            for seed in batch:
                command = [
                    str(ROOT / "run.sh"), "N1-dynamic",
                    "--N", "2000", "--P", "100", "--seed", str(seed),
                    "--dt", "0.01", "--lambda-start", "0.40",
                    "--lambda-stop", "0.25", "--lambda-step", "0.002",
                    "--bracket-tol", "0.0005",
                    "--max-time", f"{max_time:g}",
                    "--required-tours", "5",
                ]
                process_environment = os.environ.copy()
                process_environment.update({
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                })
                log = log_path(seed).open("a", encoding="utf-8")
                process = subprocess.Popen(
                    command, cwd=ROOT, env=process_environment,
                    stdout=log, stderr=subprocess.STDOUT, text=True,
                )
                processes[seed] = (process, log)
            record(
                f"{phase}_running_batch", batch,
                f"launched with max_time={max_time:g}",
            )

            while processes:
                finished = []
                for seed, (process, log) in processes.items():
                    code = process.poll()
                    if code is not None:
                        log.flush()
                        log.close()
                        finished.append(seed)
                        if code == 0 and completed(seed):
                            if retry:
                                failures.pop(seed, None)
                        else:
                            failures[seed] = int(code)
                for seed in finished:
                    processes.pop(seed)
                if time.monotonic() - last_resource >= 3600.0:
                    resources.append(resource_snapshot())
                    last_resource = time.monotonic()
                record(f"{phase}_running_batch", sorted(processes), "poll")
                if processes:
                    time.sleep(min(args.poll_seconds, 60.0))

    pending = [
        seed for seed in args.seeds
        if not completed(seed) and seed not in failures
    ]
    run_batches(pending, max_time=20000.0, phase="primary", retry=False)

    # A timeout after roughly 4--5 strictly forward tours is an integration-time
    # limitation, not evidence for another attractor.  Resume exactly from the
    # saved preceding ordered history with a larger cap, while retaining the
    # five-tour criterion and all other classification gates.
    retry_seeds = sorted(failures)
    if retry_seeds:
        record(
            "extended_time_retry_planned", [],
            f"seeds {retry_seeds}; max_time={args.retry_max_time:g}; "
            "checkpointed continuation, five-tour criterion unchanged",
        )
        run_batches(
            retry_seeds, max_time=args.retry_max_time,
            phase="extended_retry", retry=True,
        )

    # The comparison explicitly excludes unresolved or stale-schema seeds and
    # applies the planned >=15 admissible-seed gate.
    if failures:
        record(
            "analysing_with_unresolved_seeds", [],
            f"unresolved seeds {sorted(failures)} excluded from comparison",
        )
    comparison = subprocess.run(
        [os.environ.get("PYTHON", "python"), str(ROOT / "src" / "n1_compare.py")],
        cwd=ROOT, check=False,
    )
    try:
        comparison_payload = json.loads(
            (RUNS / "N1" / "comparison.json").read_text(encoding="utf-8")
        )
        gate_passed = bool(comparison_payload["completion_test_passed"])
    except (KeyError, OSError, ValueError, TypeError):
        gate_passed = False
    if comparison.returncode or not gate_passed:
        record(
            "scientific_gate_failed", [],
            f"comparison={comparison.returncode}, gate_passed={gate_passed}",
        )
        raise SystemExit(4)

    figures = subprocess.run(
        [os.environ.get("PYTHON", "python"), str(ROOT / "src" / "n4_figures.py")],
        cwd=ROOT, check=False,
    )
    if figures.returncode:
        record(
            "postprocessing_failed", [],
            f"figures={figures.returncode}",
        )
        raise SystemExit(5)
    resources.append(resource_snapshot())
    record("complete", [], "all N1 dynamics, comparison and N4 rebuild completed")


if __name__ == "__main__":
    main()
