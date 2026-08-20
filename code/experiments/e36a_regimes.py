#!/usr/bin/env python3
"""
E36-A1 adaptive-sentinel campaign driver.

This file prepares (but does not autonomously authorize) the first regime scan.
Runs are strictly serial. ``--smoke`` is hard-capped; ``--production`` additionally
requires ``--confirm-heavy`` and is intended for a future user-approved launch.
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
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np

from layered_chain import LayeredChain, make_patterns_numpy


SENTINELS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SMOKE_CAPS = {
    "N": 128,
    "P": 8,
    "K": 4,
    "t_final": 0.25,
    "fine_steps": 100,
    "n_lambdas": 8,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert NumPy/non-finite values into strict JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                json_safe(payload), sort_keys=True, allow_nan=False
            ) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def directory_bytes(path: Path) -> int:
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    return total


class ObservableChunkWriter:
    """Atomic chunk writer for reduced observables, never full U trajectories."""

    def __init__(
        self,
        run_dir: Path,
        system: LayeredChain,
        config: dict,
        *,
        chunk_size: int,
        resume: bool,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >=1")
        self.run_dir = run_dir
        self.chunks_dir = run_dir / "chunks"
        self.manifest_path = run_dir / "manifest.json"
        self.system = system
        self.config = json_safe(config)
        self.chunk_size = chunk_size
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._times: list[float] = []
        self._physical_overlaps: list[np.ndarray] = []
        self._raw_overlaps: list[np.ndarray] = []
        self._saturation: list[np.ndarray] = []
        self._max_abs: list[np.ndarray] = []

        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"{self.manifest_path} exists; use --resume or a new output"
                )
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if self.manifest.get("config") != self.config:
                raise ValueError("Observable manifest configuration mismatch")
        else:
            self.manifest = {
                "schema": "E36A-observables-v2",
                "created_utc": utc_now(),
                "status": "running",
                "config": self.config,
                "chunks": [],
                "last_time": None,
            }
            atomic_json(self.manifest_path, self.manifest)
        if self.manifest.get("schema") != "E36A-observables-v2":
            raise ValueError(
                "Observable schema is not v2; raw/physical overlap ambiguity"
            )

    @property
    def last_time(self) -> float | None:
        value = self.manifest.get("last_time")
        return None if value is None else float(value)

    def __call__(self, time_value: float, U: np.ndarray) -> None:
        # A resumed integrator emits its checkpoint state once. Do not duplicate
        # the last already committed record.
        if (
            self.last_time is not None
            and time_value <= self.last_time + 1e-13
        ):
            return
        self._times.append(float(time_value))
        # Primary physical overlap follows the convention of the entire project:
        # m_mu(U)=xi_mu.tanh(beta*U)/N. The raw xi.U/N projection is retained only
        # to diagnose the O(lambda) amplitudes specific to E36.
        self._physical_overlaps.append(self.system.activation_overlaps(U))
        self._raw_overlaps.append(self.system.raw_overlaps(U))
        self._saturation.append(
            np.mean(np.abs(self.system.beta * U) > 10.0, axis=1)
        )
        self._max_abs.append(np.max(np.abs(U), axis=1))
        step = int(round(time_value/self.config["dt"]))
        at_checkpoint = (
            step > 0
            and step % self.config["checkpoint_every_steps"] == 0
        )
        # The record callback runs before LayeredChain commits its checkpoint.
        # Flushing here guarantees that a checkpoint is never newer than the
        # committed observable stream. If a crash occurs between both commits,
        # recomputation from the older checkpoint is safely de-duplicated by time.
        if len(self._times) >= self.chunk_size or at_checkpoint:
            self.flush()

    def flush(self) -> None:
        if not self._times:
            return
        index = len(self.manifest["chunks"])
        filename = f"chunk_{index:05d}.npz"
        target = self.chunks_dir / filename
        temporary = self.chunks_dir / (filename + ".tmp.npz")
        times = np.asarray(self._times, dtype=np.float64)
        physical_overlaps = np.asarray(
            self._physical_overlaps, dtype=np.float64
        )
        raw_overlaps = np.asarray(self._raw_overlaps, dtype=np.float64)
        saturation = np.asarray(self._saturation, dtype=np.float64)
        max_abs = np.asarray(self._max_abs, dtype=np.float64)
        np.savez(
            temporary,
            t=times,
            physical_overlaps=physical_overlaps,
            raw_overlaps=raw_overlaps,
            saturation=saturation,
            max_abs=max_abs,
        )
        os.replace(temporary, target)
        self.manifest["chunks"].append(
            {
                "file": f"chunks/{filename}",
                "n_records": len(times),
                "t_start": float(times[0]),
                "t_end": float(times[-1]),
            }
        )
        self.manifest["last_time"] = float(times[-1])
        self.manifest["updated_utc"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)
        self._times.clear()
        self._physical_overlaps.clear()
        self._raw_overlaps.clear()
        self._saturation.clear()
        self._max_abs.clear()

    def finalize(self) -> None:
        self.flush()
        self.manifest["status"] = "complete"
        self.manifest["completed_utc"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)

    def load(self) -> dict[str, np.ndarray]:
        self.flush()
        blocks = {
            "t": [],
            "physical_overlaps": [],
            "raw_overlaps": [],
            "saturation": [],
            "max_abs": [],
        }
        for chunk in self.manifest["chunks"]:
            with np.load(self.run_dir / chunk["file"], allow_pickle=False) as data:
                for key in blocks:
                    blocks[key].append(np.array(data[key], dtype=np.float64))
        if not blocks["t"]:
            return {
                "t": np.empty(0),
                "physical_overlaps": np.empty(
                    (0, self.system.K, self.system.P)
                ),
                "raw_overlaps": np.empty(
                    (0, self.system.K, self.system.P)
                ),
                "saturation": np.empty((0, self.system.K)),
                "max_abs": np.empty((0, self.system.K)),
            }
        return {
            key: np.concatenate(parts, axis=0) for key, parts in blocks.items()
        }


class HeartbeatLogger:
    """Append-only JSONL heartbeat callback for LayeredChain.integrate."""

    def __init__(
        self,
        path: Path,
        run_dir: Path,
        context: dict,
        min_wall_seconds: float,
    ):
        self.path = path
        self.run_dir = run_dir
        self.context = context
        self.min_wall_seconds = float(min_wall_seconds)
        self._last_write_monotonic: float | None = None

    def __call__(self, progress: dict, state: np.ndarray) -> None:
        now = time.monotonic()
        boundary = progress["step"] in (0, progress["total_steps"])
        if (
            not boundary
            and self._last_write_monotonic is not None
            and now-self._last_write_monotonic < self.min_wall_seconds
        ):
            return
        self._last_write_monotonic = now
        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_raw / (1024.0 * 1024.0) if sys.platform == "darwin" \
            else rss_raw / 1024.0
        append_jsonl(
            self.path,
            {
                "timestamp": utc_now(),
                "pid": os.getpid(),
                "cpu_time_seconds": time.process_time(),
                **self.context,
                **progress,
                "rss_mb": rss_mb,
                "max_abs_state": float(np.max(np.abs(state))),
                "finite": bool(np.all(np.isfinite(state))),
                "output_bytes": directory_bytes(self.run_dir),
            },
        )


def transition_events(
    times: np.ndarray, lead: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(lead[1:] != lead[:-1]) + 1
    return times[indices], lead[indices], lead[indices-1]


def analyze_observables(
    data: dict[str, np.ndarray],
    *,
    P: int,
    transient_fraction: float = 0.25,
) -> dict:
    """Classify order, transition clock, saturation, and inter-layer fronts."""
    times = np.asarray(data["t"], dtype=np.float64)
    physical_overlaps = np.asarray(
        data["physical_overlaps"], dtype=np.float64
    )
    raw_overlaps = np.asarray(data["raw_overlaps"], dtype=np.float64)
    saturation = np.asarray(data["saturation"], dtype=np.float64)
    max_abs = np.asarray(data["max_abs"], dtype=np.float64)
    if len(times) < 2:
        raise ValueError("Need at least two records for regime analysis")
    start = min(len(times)-2, int(math.floor(transient_fraction * len(times))))
    t = times[start:]
    # All lead-pattern identities, transitions, clocks, and front observables
    # are based exclusively on the physical m(U), never on xi.U/N.
    O = physical_overlaps[start:]
    raw_O = raw_overlaps[start:]
    sat = saturation[start:]
    amp = max_abs[start:]
    lead = np.argmax(O, axis=-1)  # (time, layer)
    raw_lead = np.argmax(raw_O, axis=-1)

    event_t, event_new, event_old = transition_events(t, lead[:, -1])
    increments = (event_new-event_old) % P
    order_fraction = (
        float(np.mean(increments == 1)) if len(increments) else None
    )
    dwell = np.diff(event_t)
    T1 = float(np.median(dwell)) if len(dwell) else None
    T1_cv = (
        float(np.std(dwell) / np.mean(dwell))
        if len(dwell) >= 2 and np.mean(dwell) > 0 else None
    )

    lags = []
    for k in range(lead.shape[1]-1):
        upstream_t, upstream_new, _ = transition_events(t, lead[:, k])
        downstream_t, downstream_new, _ = transition_events(t, lead[:, k+1])
        for td, target in zip(downstream_t, downstream_new):
            candidates = upstream_t[
                (upstream_new == target) & (upstream_t <= td + 1e-12)
            ]
            if len(candidates):
                lags.append(float(td-candidates[-1]))

    distinct_by_time = np.asarray([
        len(np.unique(row)) for row in lead
    ], dtype=np.float64)
    disagreement = np.mean(lead != lead[:, -1, None])
    overlap_drift = float(
        np.linalg.norm(O[-1]-O[max(0, len(O)//2)])
        / max(np.linalg.norm(O[-1]), 1e-14)
    )

    if len(event_t) >= 3:
        if (
            order_fraction is not None
            and order_fraction >= 0.9
            and (T1_cv is None or T1_cv <= 0.2)
        ):
            regime = "ordered_wave"
        else:
            regime = "irregular_switching"
    elif overlap_drift < 1e-3:
        regime = "stationary"
    else:
        regime = "unresolved_short_run"

    go_a1 = bool(
        len(event_t) >= 10
        and order_fraction is not None
        and order_fraction >= 0.95
        and T1_cv is not None
        and T1_cv < 0.10
    )

    return {
        "regime": regime,
        "go_A1_ordered_wave": go_a1,
        "n_records": len(t),
        "n_transitions_last_layer": len(event_t),
        "order_fraction": order_fraction,
        "T1_median": T1,
        "T1_cv": T1_cv,
        "mean_saturation_by_layer": np.mean(sat, axis=0),
        "max_abs_by_layer": np.max(amp, axis=0),
        "mean_distinct_patterns_across_layers": float(
            np.mean(distinct_by_time)
        ),
        "layer_disagreement_fraction": float(disagreement),
        "raw_physical_lead_disagreement_fraction": float(
            np.mean(raw_lead != lead)
        ),
        "interlayer_lag_median": (
            float(np.median(lags)) if lags else None
        ),
        "n_interlayer_lags": len(lags),
        "overlap_drift": overlap_drift,
        "analysis_t_start": float(t[0]),
        "analysis_t_end": float(t[-1]),
    }


def compare_dt(
    coarse: dict,
    fine: dict,
    coarse_final: np.ndarray,
    fine_final: np.ndarray,
) -> dict:
    state_error = float(
        np.linalg.norm(coarse_final-fine_final)
        / max(np.linalg.norm(fine_final), 1e-14)
    )
    coarse_t1 = coarse["T1_median"]
    fine_t1 = fine["T1_median"]
    period_error = None
    if coarse_t1 is not None and fine_t1 is not None and fine_t1 > 0:
        period_error = abs(coarse_t1-fine_t1)/fine_t1
    sat_error = float(np.max(np.abs(
        np.asarray(coarse["mean_saturation_by_layer"])
        - np.asarray(fine["mean_saturation_by_layer"])
    )))
    class_match = coarse["regime"] == fine["regime"]
    timing_ok = period_error < 0.02 if period_error is not None \
        else state_error < 0.02
    passed = class_match and timing_ok and sat_error < 0.02
    return {
        "status": "pass" if passed else "fail",
        "class_match": class_match,
        "relative_final_state_error": state_error,
        "relative_T1_error": period_error,
        "max_saturation_error": sat_error,
    }


def initial_condition(
    xi: np.ndarray,
    K: int,
    kind: str,
    *,
    seed: int,
    perturbation_sigma: float,
) -> np.ndarray:
    P, N = xi.shape
    U = np.empty((K, N), dtype=np.float64)
    U[1:] = 0.8 * xi[0][None, :]
    U[0] = 0.8 * xi[1 % P]
    if kind == "perturbed":
        rng = np.random.default_rng(seed + 10_003)
        U += perturbation_sigma * rng.standard_normal(U.shape)
    elif kind != "coherent":
        raise ValueError(f"Unknown initial condition {kind}")
    return U


def run_single(
    *,
    xi: np.ndarray,
    config: dict,
    U0: np.ndarray,
    run_dir: Path,
    heartbeat_path: Path,
    resume: bool,
) -> tuple[dict, np.ndarray]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"{summary_path} exists; use --resume")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(run_dir / "final_state.npz", allow_pickle=False) as data:
            return summary, np.array(data["state"], dtype=np.float64)

    system = LayeredChain(
        xi,
        beta=config["beta"],
        lam=config["lambda"],
        K=config["K"],
        t0=config["t0"],
    )
    writer = ObservableChunkWriter(
        run_dir,
        system,
        config,
        chunk_size=config["chunk_size"],
        resume=resume,
    )
    checkpoint = run_dir / "checkpoint.npz"
    can_resume = resume and checkpoint.exists()
    heartbeat = HeartbeatLogger(
        heartbeat_path,
        run_dir,
        {
            "experiment": "E36A1",
            "run_id": config["run_id"],
            "lambda": config["lambda"],
            "initial_condition": config["initial_condition"],
            "dt": config["dt"],
            "checkpoint": str(checkpoint),
            "parameter_index": config["parameter_index"],
            "parameter_total": config["parameter_total"],
        },
        min_wall_seconds=config["heartbeat_min_wall_seconds"],
    )
    result = system.integrate(
        None if can_resume else U0,
        t_final=config["t_final"],
        dt=config["dt"],
        record_every=config["record_every"],
        checkpoint_path=checkpoint,
        checkpoint_every_steps=config["checkpoint_every_steps"],
        resume=can_resume,
        record_callback=writer,
        heartbeat_callback=heartbeat,
        heartbeat_every_steps=config["heartbeat_every_steps"],
        store_records=False,
    )
    writer.finalize()
    metrics = analyze_observables(
        writer.load(),
        P=config["P"],
        transient_fraction=config["transient_fraction"],
    )
    summary = {
        "schema": "E36A-run-summary-v1",
        "completed_utc": utc_now(),
        "config": config,
        "metrics": metrics,
    }
    temporary = run_dir / "final_state.tmp.npz"
    np.savez(temporary, state=result["final_state"])
    os.replace(temporary, run_dir / "final_state.npz")
    # The summary is the completion marker and is committed last.
    atomic_json(summary_path, summary)
    return summary, result["final_state"]


def lambda_tag(value: float) -> str:
    return f"{value:.6g}".replace(".", "p")


def parse_lambdas(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("lambda list cannot be empty")
    if any(not math.isfinite(value) or value < 0 or value > 8 for value in values):
        raise argparse.ArgumentTypeError("all lambdas must be in [0,8]")
    return values


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--production", action="store_true")
    parser.add_argument("--confirm-heavy", action="store_true")
    parser.add_argument("--N", type=int)
    parser.add_argument("--P", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-final", type=float)
    parser.add_argument("--record-every", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int)
    parser.add_argument("--heartbeat-every-steps", type=int)
    parser.add_argument("--heartbeat-min-wall-seconds", type=float)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--transient-fraction", type=float, default=0.25)
    parser.add_argument("--perturbation-sigma", type=float, default=0.3)
    parser.add_argument(
        "--lambdas",
        type=parse_lambdas,
        default=SENTINELS,
        help="comma-separated sentinels in [0,8]",
    )
    parser.add_argument(
        "--dt-check",
        choices=("all", "high", "none"),
        default=None,
        help="dt/2 validation for all, lambda>=2, or no sentinels",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    if args.production and not args.confirm_heavy:
        raise SystemExit(
            "REFUSED: --production requires --confirm-heavy after user approval"
        )
    smoke_like = args.smoke or args.dry_run
    N = args.N if args.N is not None else (64 if smoke_like else 500)
    P = args.P if args.P is not None else (4 if smoke_like else 25)
    K = args.K if args.K is not None else (3 if smoke_like else 6)
    dt = args.dt if args.dt is not None else (0.01 if smoke_like else 0.01)
    t_final = (
        args.t_final if args.t_final is not None else (0.1 if smoke_like else 200.0)
    )
    dt_check = args.dt_check or ("all" if smoke_like else "high")
    heartbeat_min_wall_seconds = (
        args.heartbeat_min_wall_seconds
        if args.heartbeat_min_wall_seconds is not None
        else (0.0 if smoke_like else 300.0)
    )
    record_every = (
        args.record_every if args.record_every is not None
        else (1 if smoke_like else 5)
    )
    checkpoint_every_steps = (
        args.checkpoint_every_steps
        if args.checkpoint_every_steps is not None
        else (10 if smoke_like else 5_000)
    )
    heartbeat_every_steps = (
        args.heartbeat_every_steps
        if args.heartbeat_every_steps is not None
        else (10 if smoke_like else 100)
    )
    chunk_size = (
        args.chunk_size if args.chunk_size is not None
        else (64 if smoke_like else 256)
    )
    if not (1 <= P <= N) or K < 2 or dt <= 0 or t_final <= 0:
        raise SystemExit("REFUSED: require 1<=P<=N, K>=2, dt>0, t_final>0")
    if (
        args.beta <= 0 or args.t0 <= 0 or record_every < 1
        or checkpoint_every_steps < 1 or heartbeat_every_steps < 1
        or chunk_size < 1 or heartbeat_min_wall_seconds < 0
        or args.perturbation_sigma < 0
    ):
        raise SystemExit("REFUSED: invalid positive cadence/model parameter")
    if checkpoint_every_steps % record_every != 0:
        raise SystemExit(
            "REFUSED: checkpoint cadence must be a multiple of record cadence"
        )
    coarse_steps = round(t_final/dt)
    fine_steps_exact = round(t_final/(dt/2))
    if (
        abs(coarse_steps*dt-t_final) > 1e-12*max(1.0, t_final)
        or abs(fine_steps_exact*(dt/2)-t_final) > 1e-12*max(1.0, t_final)
    ):
        raise SystemExit("REFUSED: t_final must be divisible by dt and dt/2")
    if not (0 <= args.transient_fraction < 1):
        raise SystemExit("REFUSED: transient fraction must be in [0,1)")
    config = {
        "mode": (
            "production" if args.production else
            "smoke" if args.smoke else "dry-run"
        ),
        "N": N,
        "P": P,
        "K": K,
        "beta": args.beta,
        "t0": args.t0,
        "seed": args.seed,
        "dt": dt,
        "t_final": t_final,
        "record_every": record_every,
        "checkpoint_every_steps": checkpoint_every_steps,
        "heartbeat_every_steps": heartbeat_every_steps,
        "heartbeat_min_wall_seconds": heartbeat_min_wall_seconds,
        "chunk_size": chunk_size,
        "transient_fraction": args.transient_fraction,
        "perturbation_sigma": args.perturbation_sigma,
        "lambdas": list(args.lambdas),
        "initial_conditions": ["coherent", "perturbed"],
        "dt_check": dt_check,
    }
    if args.smoke:
        fine_steps = fine_steps_exact
        checks = {
            "N": N <= SMOKE_CAPS["N"],
            "P": P <= SMOKE_CAPS["P"],
            "K": K <= SMOKE_CAPS["K"],
            "t_final": t_final <= SMOKE_CAPS["t_final"],
            "fine_steps": fine_steps <= SMOKE_CAPS["fine_steps"],
            "n_lambdas": len(args.lambdas) <= SMOKE_CAPS["n_lambdas"],
        }
        failed = [key for key, okay in checks.items() if not okay]
        if failed:
            raise SystemExit(
                "REFUSED: smoke caps exceeded for " + ", ".join(failed)
            )
    n_pairs = len(args.lambdas) * 2
    n_dt_checks = (
        n_pairs if dt_check == "all" else
        sum(value >= 2 for value in args.lambdas) * 2
        if dt_check == "high" else 0
    )
    config["planned_parameter_pairs"] = n_pairs
    config["planned_integrations"] = n_pairs + n_dt_checks
    config["planned_coarse_steps"] = coarse_steps
    config["planned_fine_steps_per_check"] = fine_steps_exact
    return config


def should_dt_check(value: float, mode: str) -> bool:
    return mode == "all" or (mode == "high" and value >= 2.0)


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    config = resolve_config(args)
    if args.dry_run:
        print(json.dumps(json_safe(config), indent=2, sort_keys=True))
        return 0

    if args.output_dir is None:
        root = Path(__file__).resolve().parents[1]
        if args.smoke:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = root / "results/7_tau_implicite/smoke" / f"E36A_{tag}"
        else:
            output_dir = (
                root / "results/7_tau_implicite/data"
                / f"E36A_regimes_seed{config['seed']}"
            )
    else:
        output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"REFUSED: non-empty output {output_dir}; use --resume or a new path"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output_dir / "heartbeat.jsonl"
    manifest_path = output_dir / "campaign_manifest.json"

    campaign = {
        "schema": "E36A-campaign-v1",
        "created_utc": utc_now(),
        "status": "running",
        "config": config,
        "output_dir": str(output_dir),
        "runs": [],
    }
    if args.resume and manifest_path.exists():
        campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
        if campaign.get("config") != json_safe(config):
            raise SystemExit("REFUSED: campaign configuration mismatch")
        campaign["status"] = "running"
        campaign["resumed_utc"] = utc_now()
    atomic_json(manifest_path, campaign)

    xi, _ = make_patterns_numpy(config["N"], config["P"], config["seed"])
    completed_by_id = {
        entry["run_id"]: entry for entry in campaign.get("runs", [])
        if entry.get("status") == "complete"
    }
    campaign_results = []

    parameter_pairs = [
        (lam, ic_kind)
        for lam in config["lambdas"]
        for ic_kind in config["initial_conditions"]
    ]
    for parameter_index, (lam, ic_kind) in enumerate(parameter_pairs, start=1):
        pair_id = f"lam{lambda_tag(lam)}_{ic_kind}"
        if pair_id in completed_by_id:
            campaign_results.append(completed_by_id[pair_id])
            continue
        U0 = initial_condition(
            xi, config["K"], ic_kind,
            seed=config["seed"],
            perturbation_sigma=config["perturbation_sigma"],
        )
        base_run = {
            "N": config["N"],
            "P": config["P"],
            "K": config["K"],
            "beta": config["beta"],
            "t0": config["t0"],
            "seed": config["seed"],
            "lambda": lam,
            "initial_condition": ic_kind,
            "t_final": config["t_final"],
            "record_every": config["record_every"],
            "checkpoint_every_steps": config["checkpoint_every_steps"],
            "heartbeat_every_steps": config["heartbeat_every_steps"],
            "heartbeat_min_wall_seconds": (
                config["heartbeat_min_wall_seconds"]
            ),
            "chunk_size": config["chunk_size"],
            "transient_fraction": config["transient_fraction"],
            "parameter_index": parameter_index,
            "parameter_total": len(parameter_pairs),
        }
        coarse_config = {
            **base_run,
            "run_id": pair_id + "_coarse",
            "dt": config["dt"],
        }
        coarse_summary, coarse_final = run_single(
            xi=xi,
            config=coarse_config,
            U0=U0,
            run_dir=output_dir / "runs" / pair_id / "coarse",
            heartbeat_path=heartbeat_path,
            resume=args.resume,
        )
        pair_result = {
            "run_id": pair_id,
            "status": "complete",
            "lambda": lam,
            "initial_condition": ic_kind,
            "coarse": coarse_summary["metrics"],
            "dt_check": None,
        }
        if should_dt_check(lam, config["dt_check"]):
            fine_config = {
                **base_run,
                "run_id": pair_id + "_fine",
                "dt": config["dt"]/2,
                "record_every": config["record_every"]*2,
                "checkpoint_every_steps": (
                    config["checkpoint_every_steps"]*2
                ),
                "heartbeat_every_steps": (
                    config["heartbeat_every_steps"]*2
                ),
            }
            fine_summary, fine_final = run_single(
                xi=xi,
                config=fine_config,
                U0=U0,
                run_dir=output_dir / "runs" / pair_id / "fine",
                heartbeat_path=heartbeat_path,
                resume=args.resume,
            )
            pair_result["fine"] = fine_summary["metrics"]
            pair_result["dt_check"] = compare_dt(
                coarse_summary["metrics"],
                fine_summary["metrics"],
                coarse_final,
                fine_final,
            )
        campaign_results.append(pair_result)
        campaign["runs"] = campaign_results
        campaign["updated_utc"] = utc_now()
        atomic_json(manifest_path, campaign)

    campaign["runs"] = campaign_results
    campaign["status"] = "complete"
    campaign["completed_utc"] = utc_now()
    campaign["output_bytes"] = directory_bytes(output_dir)
    atomic_json(manifest_path, campaign)
    print(json.dumps({
        "status": campaign["status"],
        "output_dir": str(output_dir),
        "completed_pairs": len(campaign_results),
        "heartbeat": str(heartbeat_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
