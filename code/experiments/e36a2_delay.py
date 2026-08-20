#!/usr/bin/env python3
"""
Guarded E36-A2 delay-scaling campaign.

Production target: lambda={1,2,4}, K={4,6,8,11}, one coherent initial condition.
The completed K=6 E36-A1 trajectories are reused rather than recomputed. New runs
are strictly serial and store compressed physical overlaps only. Production always
requires ``--confirm-heavy``; this module never launches it autonomously.
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
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from e36a_regimes import (
    HeartbeatLogger,
    atomic_json,
    directory_bytes,
    initial_condition,
    json_safe,
    lambda_tag,
    utc_now,
)
from layered_chain import LayeredChain, make_patterns_numpy


TARGET_K = (4, 6, 8, 11)
TARGET_LAMBDAS = (1.0, 2.0, 4.0)
SMOKE_CAPS = {
    "N": 128,
    "P": 8,
    "K_max": 4,
    "steps": 100,
    "n_runs": 6,
}


def parse_numbers(text: str, cast=float) -> tuple:
    try:
        values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def load_physical_trajectory(run_dir: Path) -> dict[str, np.ndarray]:
    """Load and validate a v2 A1 or A2 chunk stream."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete trajectory manifest: {manifest_path}")
    chunks = manifest.get("chunks", [])
    if not chunks:
        raise ValueError(f"No chunks in {manifest_path}")
    blocks = {
        "t": [],
        "physical_overlaps": [],
        "saturation": [],
        "max_abs": [],
    }
    for chunk in chunks:
        path = run_dir / chunk["file"]
        with np.load(path, allow_pickle=False) as data:
            if "physical_overlaps" not in data.files:
                raise ValueError(f"Physical overlap missing from {path}")
            for key in blocks:
                blocks[key].append(np.asarray(data[key], dtype=np.float64))
    result = {
        key: np.concatenate(parts, axis=0) for key, parts in blocks.items()
    }
    if not np.all(np.diff(result["t"]) > 0):
        raise ValueError(f"Non-increasing record times in {run_dir}")
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"Non-finite trajectory observable in {run_dir}")
    return result


class DelayChunkWriter:
    """Compressed, atomic A2 stream: no full state and no raw overlaps."""

    def __init__(
        self,
        run_dir: Path,
        system: LayeredChain,
        config: dict,
        *,
        chunk_size: int,
        resume: bool,
    ):
        self.run_dir = run_dir
        self.chunks_dir = run_dir / "chunks"
        self.manifest_path = run_dir / "manifest.json"
        self.system = system
        self.config = json_safe(config)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >=1")
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._t: list[float] = []
        self._m: list[np.ndarray] = []
        self._sat: list[np.ndarray] = []
        self._amp: list[np.ndarray] = []
        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"{self.manifest_path} exists; use --resume"
                )
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if self.manifest.get("config") != self.config:
                raise ValueError("A2 stream configuration mismatch")
        else:
            self.manifest = {
                "schema": "E36A2-observables-v1",
                "created_utc": utc_now(),
                "status": "running",
                "config": self.config,
                "chunks": [],
                "last_time": None,
            }
            atomic_json(self.manifest_path, self.manifest)
        if self.manifest.get("schema") != "E36A2-observables-v1":
            raise ValueError("Unsupported A2 stream schema")

    @property
    def last_time(self) -> float | None:
        value = self.manifest.get("last_time")
        return None if value is None else float(value)

    def __call__(self, time_value: float, U: np.ndarray) -> None:
        if self.last_time is not None and time_value <= self.last_time + 1e-13:
            return
        self._t.append(float(time_value))
        self._m.append(self.system.activation_overlaps(U))
        self._sat.append(
            np.mean(np.abs(self.system.beta*U) > 10.0, axis=1)
        )
        self._amp.append(np.max(np.abs(U), axis=1))
        step = int(round(time_value/self.config["dt"]))
        at_checkpoint = (
            step > 0
            and step % self.config["checkpoint_every_steps"] == 0
        )
        if len(self._t) >= self.chunk_size or at_checkpoint:
            self.flush()

    def flush(self) -> None:
        if not self._t:
            return
        index = len(self.manifest["chunks"])
        filename = f"chunk_{index:05d}.npz"
        target = self.chunks_dir / filename
        temporary = self.chunks_dir / (filename + ".tmp.npz")
        t = np.asarray(self._t, dtype=np.float64)
        np.savez_compressed(
            temporary,
            t=t,
            physical_overlaps=np.asarray(self._m, dtype=np.float64),
            saturation=np.asarray(self._sat, dtype=np.float64),
            max_abs=np.asarray(self._amp, dtype=np.float64),
        )
        os.replace(temporary, target)
        self.manifest["chunks"].append({
            "file": f"chunks/{filename}",
            "n_records": len(t),
            "t_start": float(t[0]),
            "t_end": float(t[-1]),
        })
        self.manifest["last_time"] = float(t[-1])
        self.manifest["updated_utc"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)
        self._t.clear()
        self._m.clear()
        self._sat.clear()
        self._amp.clear()

    def finalize(self) -> None:
        self.flush()
        self.manifest["status"] = "complete"
        self.manifest["completed_utc"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values)-1:
        return float(index)
    left, center, right = values[index-1:index+2]
    denominator = left - 2.0*center + right
    if abs(denominator) < 1e-15:
        return float(index)
    offset = 0.5*(left-right)/denominator
    return float(index + np.clip(offset, -1.0, 1.0))


def regularize_record_grid(
    times: np.ndarray,
    overlaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Accept a regular grid or trim one shorter terminal callback interval.

    LayeredChain always records the final state. If t_final is not a multiple of
    ``record_every*dt``, that creates one final short interval. It contains no
    information needed for lag estimation and is removed explicitly. Any interior
    irregularity, multiple irregularities, non-positive interval, or longer final
    interval remains a hard error.
    """
    times = np.asarray(times, dtype=np.float64)
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if times.ndim != 1 or overlaps.shape[0] != len(times) or len(times) < 3:
        raise ValueError("Need >=3 aligned time/overlap records")
    differences = np.diff(times)
    if np.any(differences <= 0):
        raise ValueError("Record times must be strictly increasing")
    # The interior intervals define the intended recording cadence. A terminal
    # callback is the only permitted exception.
    nominal = float(np.median(differences[:-1]))
    tolerance = max(1e-12, 1e-8*nominal)
    interior_bad = np.flatnonzero(
        np.abs(differences[:-1]-nominal) > tolerance
    )
    if len(interior_bad):
        raise ValueError(
            "Record grid has interior irregularities at intervals "
            + ",".join(str(int(index)) for index in interior_bad)
        )
    terminal = float(differences[-1])
    trimmed = False
    if abs(terminal-nominal) > tolerance:
        if 0 < terminal < nominal-tolerance:
            times = times[:-1]
            overlaps = overlaps[:-1]
            trimmed = True
        else:
            raise ValueError(
                "Record grid has a non-short terminal irregularity"
            )
    return times, overlaps, {
        "nominal_record_dt": nominal,
        "terminal_interval": terminal,
        "trimmed_final_sample": trimmed,
        "input_records": len(differences)+1,
        "used_records": len(times),
        "action": "trim_short_terminal_interval" if trimmed else "none",
    }


def xcorr_lag(
    upstream: np.ndarray,
    downstream: np.ndarray,
    *,
    dt: float,
    max_lag: float,
    min_peak_correlation: float = 0.5,
    min_peak_margin: float = 0.02,
) -> dict:
    """
    Estimate positive delay from vector cross-correlation.

    C(s)=corr(upstream(t), downstream(t+s)); a delayed downstream signal peaks
    at positive s. Peak uniqueness excludes a neighborhood around the best peak.
    """
    upstream = np.asarray(upstream, dtype=np.float64)
    downstream = np.asarray(downstream, dtype=np.float64)
    if upstream.shape != downstream.shape or upstream.ndim != 2:
        raise ValueError("xcorr inputs must have the same shape (time,P)")
    n = len(upstream)
    max_samples = min(int(math.ceil(max_lag/dt)), max(1, n//3))
    if n < max(12, 3*max_samples):
        return {
            "lag": None, "peak": None, "margin": None,
            "unique": False, "reason": "trace_too_short",
        }
    x = upstream - np.mean(upstream, axis=0, keepdims=True)
    y = downstream - np.mean(downstream, axis=0, keepdims=True)
    correlations = np.empty(max_samples+1, dtype=np.float64)
    for lag in range(max_samples+1):
        xs = x if lag == 0 else x[:-lag]
        ys = y if lag == 0 else y[lag:]
        denominator = np.linalg.norm(xs)*np.linalg.norm(ys)
        correlations[lag] = (
            np.sum(xs*ys)/denominator if denominator > 0 else -np.inf
        )
    best = int(np.argmax(correlations))
    refined = _parabolic_peak(correlations, best)
    exclusion = max(2, int(round(0.10*max_samples)))
    mask = np.ones(len(correlations), dtype=bool)
    mask[max(0, best-exclusion):min(len(mask), best+exclusion+1)] = False
    second = float(np.max(correlations[mask])) if np.any(mask) else -1.0
    peak = float(correlations[best])
    margin = peak-second
    interior = 0 < best < max_samples
    unique = bool(
        interior
        and peak >= min_peak_correlation
        and margin >= min_peak_margin
    )
    return {
        "lag": refined*dt if unique else None,
        "candidate_lag": refined*dt,
        "peak": peak,
        "margin": margin,
        "unique": unique,
        "best_index": best,
        "max_lag": max_samples*dt,
    }


def crossing_events(
    times: np.ndarray, overlaps: np.ndarray
) -> list[list[dict]]:
    """Linearly interpolate old/new physical-overlap crossings per layer."""
    times = np.asarray(times, dtype=np.float64)
    overlaps = np.asarray(overlaps, dtype=np.float64)
    lead = np.argmax(overlaps, axis=-1)
    all_events: list[list[dict]] = []
    for k in range(overlaps.shape[1]):
        events = []
        indices = np.flatnonzero(lead[1:, k] != lead[:-1, k]) + 1
        for index in indices:
            old = int(lead[index-1, k])
            new = int(lead[index, k])
            d0 = overlaps[index-1, k, new] - overlaps[index-1, k, old]
            d1 = overlaps[index, k, new] - overlaps[index, k, old]
            if d1 == d0:
                fraction = 0.5
            else:
                fraction = float(np.clip(-d0/(d1-d0), 0.0, 1.0))
            crossing_time = (
                times[index-1] + fraction*(times[index]-times[index-1])
            )
            events.append({
                "time": float(crossing_time),
                "old": old,
                "new": new,
            })
        all_events.append(events)
    return all_events


def relay_lags(
    times: np.ndarray,
    overlaps: np.ndarray,
    *,
    P: int,
    max_pair_lag: float,
) -> dict:
    """Estimate layer-to-layer lags by matching the same relay crossing."""
    events = crossing_events(times, overlaps)
    per_layer = []
    counts = []
    ordered_counts = []
    for k in range(len(events)-1):
        values = []
        ordered = 0
        for downstream in events[k+1]:
            if (downstream["new"]-downstream["old"]) % P == 1:
                ordered += 1
            candidates = [
                upstream["time"] for upstream in events[k]
                if upstream["new"] == downstream["new"]
                and 0 <= downstream["time"]-upstream["time"] <= max_pair_lag
            ]
            if candidates:
                values.append(downstream["time"]-max(candidates))
        per_layer.append(float(np.median(values)) if values else None)
        counts.append(len(values))
        ordered_counts.append(ordered)
    valid = [value for value in per_layer if value is not None]
    last_events = events[-1]
    last_ordered = [
        event for event in last_events
        if (event["new"]-event["old"]) % P == 1
    ]
    last_times = np.asarray([event["time"] for event in last_ordered])
    dwell = np.diff(last_times)
    return {
        "delta_by_layer": per_layer,
        "count_by_layer": counts,
        "ordered_event_count_by_layer": ordered_counts,
        "tau_eff": float(np.sum(valid)) if len(valid) == len(per_layer) else None,
        "T1_median": float(np.median(dwell)) if len(dwell) else None,
        "T1_cv": (
            float(np.std(dwell)/np.mean(dwell))
            if len(dwell) >= 2 and np.mean(dwell) > 0 else None
        ),
        "n_last_layer_relays": len(last_ordered),
    }


def estimate_delay(
    times: np.ndarray,
    overlaps: np.ndarray,
    *,
    P: int,
    T1_hint: float | None = None,
) -> dict:
    """Apply both lag estimators to a physical-overlap trajectory."""
    times, overlaps, grid_diagnostic = regularize_record_grid(times, overlaps)
    dt_values = np.diff(times)
    dt = float(np.median(dt_values))
    if np.max(np.abs(dt_values-dt)) > 1e-8*max(1.0, dt):
        raise ValueError("Record grid must be uniform")
    provisional = relay_lags(
        times, overlaps, P=P,
        max_pair_lag=max(2.0, 2.0*(T1_hint or 2.0)),
    )
    T1 = provisional["T1_median"] or T1_hint
    if T1 is None:
        return {
            "valid": False,
            "reason": "no_relay_clock",
            "relay": provisional,
            "xcorr": None,
            "record_grid": grid_diagnostic,
        }
    max_lag = min(1.5*T1, 0.30*(times[-1]-times[0]))
    xcorr_layers = [
        xcorr_lag(
            overlaps[:, k, :],
            overlaps[:, k+1, :],
            dt=dt,
            max_lag=max_lag,
        )
        for k in range(overlaps.shape[1]-1)
    ]
    xcorr_values = [
        item["lag"] for item in xcorr_layers if item["lag"] is not None
    ]
    tau_xcorr = (
        float(np.sum(xcorr_values))
        if len(xcorr_values) == overlaps.shape[1]-1 else None
    )
    relay = relay_lags(
        times, overlaps, P=P, max_pair_lag=max_lag
    )
    tau_relay = relay["tau_eff"]
    agreement = None
    if tau_xcorr is not None and tau_relay is not None:
        agreement = abs(tau_xcorr-tau_relay) / max(
            0.5*(abs(tau_xcorr)+abs(tau_relay)), 1e-14
        )
    return {
        "valid": bool(
            tau_xcorr is not None and tau_relay is not None
        ),
        "record_dt": dt,
        "record_grid": grid_diagnostic,
        "T1_median": relay["T1_median"],
        "T1_cv": relay["T1_cv"],
        "n_last_layer_relays": relay["n_last_layer_relays"],
        "xcorr": {
            "by_layer": xcorr_layers,
            "tau_eff": tau_xcorr,
            "unique_fraction": float(np.mean([
                item["unique"] for item in xcorr_layers
            ])),
        },
        "relay": relay,
        "relative_estimator_disagreement": agreement,
    }


def block_uncertainty(
    times: np.ndarray,
    overlaps: np.ndarray,
    *,
    P: int,
    T1: float,
    max_blocks: int = 6,
) -> dict:
    """Contiguous-block uncertainty, preserving within-block correlations."""
    duration = float(times[-1]-times[0])
    n_blocks = min(max_blocks, int(duration//max(4.0*T1, 1e-12)))
    if n_blocks < 3:
        return {
            "n_blocks": n_blocks,
            "tau_xcorr_samples": [],
            "tau_relay_samples": [],
            "tau_xcorr_ci95": None,
            "tau_relay_ci95": None,
        }
    boundaries = np.linspace(0, len(times), n_blocks+1, dtype=int)
    samples_x, samples_r = [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop-start < 12:
            continue
        estimate = estimate_delay(
            times[start:stop],
            overlaps[start:stop],
            P=P,
            T1_hint=T1,
        )
        if estimate["xcorr"] is not None \
                and estimate["xcorr"]["tau_eff"] is not None:
            samples_x.append(estimate["xcorr"]["tau_eff"])
        if estimate["relay"]["tau_eff"] is not None:
            samples_r.append(estimate["relay"]["tau_eff"])

    def interval(values: list[float]) -> list[float] | None:
        if len(values) < 3:
            return None
        array = np.asarray(values)
        half = 1.96*np.std(array, ddof=1)/math.sqrt(len(array))
        center = float(np.mean(array))
        return [center-half, center+half]

    return {
        "n_blocks": n_blocks,
        "tau_xcorr_samples": samples_x,
        "tau_relay_samples": samples_r,
        "tau_xcorr_ci95": interval(samples_x),
        "tau_relay_ci95": interval(samples_r),
        "method": "contiguous-block normal CI (descriptive)",
    }


def analyze_delay_trajectory(
    data: dict[str, np.ndarray],
    *,
    P: int,
    transient_fraction: float,
    T1_hint: float | None = None,
) -> dict:
    n = len(data["t"])
    for key in ("physical_overlaps", "saturation", "max_abs"):
        if len(data[key]) != n:
            raise ValueError(f"Observable {key} is not aligned with time")
    start = min(n-2, int(math.floor(transient_fraction*n)))
    recorded_times = data["t"][start:]
    recorded_overlaps = data["physical_overlaps"][start:]
    times, overlaps, grid_diagnostic = regularize_record_grid(
        recorded_times, recorded_overlaps
    )
    estimate = estimate_delay(
        times, overlaps, P=P, T1_hint=T1_hint
    )
    # Preserve the diagnostic from the full analysis interval: estimate_delay
    # sees the already-cleaned grid and would otherwise report action='none'.
    estimate["record_grid"] = grid_diagnostic
    if estimate.get("T1_median") is not None:
        estimate["block_uncertainty"] = block_uncertainty(
            times,
            overlaps,
            P=P,
            T1=estimate["T1_median"],
        )
    else:
        estimate["block_uncertainty"] = None
    estimate["analysis_t_start"] = float(times[0])
    estimate["analysis_t_end_recorded"] = float(recorded_times[-1])
    estimate["analysis_t_end_used"] = float(times[-1])
    stop = start+len(times)
    estimate["mean_saturation_by_layer"] = np.mean(
        data["saturation"][start:stop], axis=0
    )
    estimate["max_abs_by_layer"] = np.max(
        data["max_abs"][start:stop], axis=0
    )
    return estimate


def fit_k_scaling(entries: list[dict]) -> dict:
    """Fit T1=a+b(K-1) and assign strong/limited/no-go."""
    usable = [
        entry for entry in entries
        if entry["analysis"].get("valid")
        and entry["analysis"].get("T1_median") is not None
    ]
    if len(usable) < 3:
        return {
            "verdict": "NO-GO",
            "reason": "fewer_than_three_valid_K",
            "n_valid_K": len(usable),
        }
    K = np.asarray([entry["K"] for entry in usable], dtype=np.float64)
    T1 = np.asarray([
        entry["analysis"]["T1_median"] for entry in usable
    ], dtype=np.float64)
    x = K-1.0
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, T1, rcond=None)[0]
    predicted = intercept+slope*x
    ss_res = float(np.sum((T1-predicted)**2))
    ss_tot = float(np.sum((T1-np.mean(T1))**2))
    r2 = 1.0-ss_res/ss_tot if ss_tot > 0 else 1.0
    disagreements = [
        entry["analysis"]["relative_estimator_disagreement"]
        for entry in usable
        if entry["analysis"]["relative_estimator_disagreement"] is not None
    ]
    unique_fractions = [
        entry["analysis"]["xcorr"]["unique_fraction"] for entry in usable
    ]
    lag_ok = (
        len(disagreements) == len(usable)
        and max(disagreements) <= 0.05
        and min(unique_fractions) == 1.0
    )
    linear_ok = r2 > 0.98 and slope > 0
    if linear_ok and lag_ok and 0.8 <= slope <= 1.2:
        verdict = "GO-strong"
        reason = "linear_K_unique_lags_unit_quantum"
    elif linear_ok and lag_ok:
        verdict = "GO-limited"
        reason = "linear_distributed_delay_nonunit_quantum"
    else:
        verdict = "NO-GO"
        reason = "nonlinear_K_or_ambiguous_lags"
    return {
        "verdict": verdict,
        "reason": reason,
        "n_valid_K": len(usable),
        "K": K,
        "T1": T1,
        "intercept": float(intercept),
        "slope": float(slope),
        "R2": r2,
        "residuals": T1-predicted,
        "max_relative_estimator_disagreement": (
            max(disagreements) if disagreements else None
        ),
        "min_xcorr_unique_fraction": min(unique_fractions),
    }


def planned_duration(
    K: int,
    T1_K6: float | None,
    *,
    dt: float,
    record_every: int,
    transient_fraction: float,
    target_relays: int,
    fixed_t_final: float | None,
) -> float:
    if fixed_t_final is not None:
        raw = fixed_t_final
    elif T1_K6 is not None:
        expected = T1_K6*(K-1)/5.0
        raw = max(
            120.0,
            expected*(target_relays+2)/(1.0-transient_fraction),
        )
    else:
        raw = 200.0
    record_dt = dt*record_every
    return math.ceil(raw/record_dt)*record_dt


def preserve_resume_run_config(
    run_dir: Path,
    proposed_config: dict,
    *,
    resume: bool,
) -> dict:
    """
    Preserve the exact configuration recorded by an interrupted/partial run.

    Duration planning may become stricter between code revisions (for example,
    rounding ``t_final`` to the record cadence). A pre-existing trajectory must
    nevertheless be resumed with its original ``t_final`` so its manifest and
    checkpoint remain an immutable, coherent unit. All other proposed fields
    must still match; only this legacy duration alignment difference is allowed.
    """
    manifest_path = run_dir / "manifest.json"
    if not resume or not manifest_path.exists():
        return proposed_config
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = manifest.get("config")
    if not isinstance(existing, dict):
        raise ValueError(f"Missing run configuration in {manifest_path}")
    all_keys = set(existing) | set(proposed_config)
    incompatible = [
        key for key in sorted(all_keys)
        if key != "t_final"
        and json_safe(existing.get(key)) != json_safe(proposed_config.get(key))
    ]
    if incompatible:
        raise ValueError(
            "A2 resume run configuration mismatch outside t_final: "
            + ", ".join(incompatible)
        )
    return existing


def run_new_trajectory(
    *,
    xi: np.ndarray,
    config: dict,
    run_dir: Path,
    heartbeat_path: Path,
    resume: bool,
) -> tuple[dict, Path]:
    summary_path = run_dir / "delay_summary.json"
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"{summary_path} exists; use --resume")
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            run_dir,
        )
    system = LayeredChain(
        xi,
        beta=config["beta"],
        lam=config["lambda"],
        K=config["K"],
        t0=config["t0"],
    )
    writer = DelayChunkWriter(
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
            "experiment": "E36A2",
            "run_id": config["run_id"],
            "lambda": config["lambda"],
            "K": config["K"],
            "parameter_index": config["parameter_index"],
            "parameter_total": config["parameter_total"],
            "checkpoint": str(checkpoint),
        },
        min_wall_seconds=config["heartbeat_min_wall_seconds"],
    )
    U0 = initial_condition(
        xi,
        config["K"],
        "coherent",
        seed=config["seed"],
        perturbation_sigma=0.0,
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
    data = load_physical_trajectory(run_dir)
    analysis = analyze_delay_trajectory(
        data,
        P=config["P"],
        transient_fraction=config["transient_fraction"],
        T1_hint=config.get("T1_hint"),
    )
    final_tmp = run_dir / "final_state.tmp.npz"
    np.savez(final_tmp, state=result["final_state"])
    os.replace(final_tmp, run_dir / "final_state.npz")
    summary = {
        "schema": "E36A2-run-summary-v1",
        "completed_utc": utc_now(),
        "config": config,
        "analysis": analysis,
    }
    atomic_json(summary_path, summary)
    return summary, run_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--production", action="store_true")
    parser.add_argument("--confirm-heavy", action="store_true")
    parser.add_argument("--N", type=int)
    parser.add_argument("--P", type=int)
    parser.add_argument("--K-list", type=lambda text: parse_numbers(text, int))
    parser.add_argument("--lambdas", type=lambda text: parse_numbers(text, float))
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-final", type=float)
    parser.add_argument("--target-relays", type=int, default=10)
    parser.add_argument("--transient-fraction", type=float, default=0.25)
    parser.add_argument("--record-every", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int)
    parser.add_argument("--heartbeat-every-steps", type=int)
    parser.add_argument("--heartbeat-min-wall-seconds", type=float)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--reuse-a1-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    if args.production and not args.confirm_heavy:
        raise SystemExit(
            "REFUSED: --production requires --confirm-heavy after user approval"
        )
    # Dry-run describes the actual production plan but performs no filesystem
    # writes or integrations. Only --smoke switches to tiny dimensions.
    small = args.smoke
    N = args.N if args.N is not None else (32 if small else 500)
    P = args.P if args.P is not None else (4 if small else 25)
    K_list = args.K_list or ((2, 3, 4) if small else TARGET_K)
    lambdas = args.lambdas or ((1.0,) if small else TARGET_LAMBDAS)
    dt = args.dt if args.dt is not None else (0.01 if small else 0.01)
    record_every = args.record_every or (1 if small else 5)
    checkpoint_every = args.checkpoint_every_steps or (
        10 if small else 5_000
    )
    heartbeat_every = args.heartbeat_every_steps or (10 if small else 100)
    heartbeat_min = (
        args.heartbeat_min_wall_seconds
        if args.heartbeat_min_wall_seconds is not None
        else (0.0 if small else 300.0)
    )
    chunk_size = args.chunk_size or (32 if small else 256)
    if (
        not (1 <= P <= N) or min(K_list) < 2 or dt <= 0
        or args.beta <= 0 or args.t0 <= 0
        or any(value < 0 or value > 8 for value in lambdas)
        or record_every < 1 or checkpoint_every < 1
        or heartbeat_every < 1 or heartbeat_min < 0 or chunk_size < 1
        or args.target_relays < 3
        or not (0 <= args.transient_fraction < 1)
    ):
        raise SystemExit("REFUSED: invalid A2 parameter")
    if len(set(K_list)) != len(K_list) or len(set(lambdas)) != len(lambdas):
        raise SystemExit("REFUSED: duplicate K or lambda values")
    if checkpoint_every % record_every != 0:
        raise SystemExit(
            "REFUSED: checkpoint cadence must be a multiple of record cadence"
        )
    if args.t_final is not None:
        steps = round(args.t_final/dt)
        if abs(steps*dt-args.t_final) > 1e-12*max(1.0, args.t_final):
            raise SystemExit("REFUSED: t_final must be divisible by dt")
    config = {
        "mode": (
            "production" if args.production else
            "smoke" if args.smoke else "dry-run"
        ),
        "N": N,
        "P": P,
        "K_list": list(K_list),
        "lambdas": list(lambdas),
        "beta": args.beta,
        "t0": args.t0,
        "seed": args.seed,
        "dt": dt,
        "fixed_t_final": args.t_final,
        "target_relays": args.target_relays,
        "transient_fraction": args.transient_fraction,
        "record_every": record_every,
        "checkpoint_every_steps": checkpoint_every,
        "heartbeat_every_steps": heartbeat_every,
        "heartbeat_min_wall_seconds": heartbeat_min,
        "chunk_size": chunk_size,
        "strict_serial": True,
    }
    if args.smoke:
        t_final = args.t_final if args.t_final is not None else 0.04
        steps = round(t_final/dt)
        checks = {
            "N": N <= SMOKE_CAPS["N"],
            "P": P <= SMOKE_CAPS["P"],
            "K": max(K_list) <= SMOKE_CAPS["K_max"],
            "steps": steps <= SMOKE_CAPS["steps"],
            "runs": len(K_list)*len(lambdas) <= SMOKE_CAPS["n_runs"],
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise SystemExit(
                "REFUSED: A2 smoke caps exceeded for " + ", ".join(failed)
            )
        config["fixed_t_final"] = t_final
    total_entries = len(K_list)*len(lambdas)
    reused = len(lambdas) if (not args.smoke and 6 in K_list) else 0
    config["planned_entries"] = total_entries
    config["planned_reused_K6"] = reused
    config["planned_new_runs"] = total_entries-reused
    return config


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    config = resolve_config(args)
    root = Path(__file__).resolve().parents[1]
    reuse_root = (
        args.reuse_a1_root.resolve() if args.reuse_a1_root is not None
        else root / "results/7_tau_implicite/data/E36A_regimes_seed42"
    )
    if args.dry_run:
        plan = {
            **config,
            "production_target_K": TARGET_K,
            "production_target_lambdas": TARGET_LAMBDAS,
            "reuse_K6": str(reuse_root),
            "note": "No integration or output is performed in dry-run.",
        }
        print(json.dumps(json_safe(plan), indent=2, sort_keys=True))
        return 0

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.smoke:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            root / "results/7_tau_implicite/smoke" / f"E36A2_{tag}"
        )
    else:
        output_dir = (
            root / "results/7_tau_implicite/data"
            / f"E36A2_delay_seed{config['seed']}"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"REFUSED: non-empty output {output_dir}; use --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output_dir / "heartbeat.jsonl"
    manifest_path = output_dir / "campaign_manifest.json"

    campaign = {
        "schema": "E36A2-campaign-v1",
        "created_utc": utc_now(),
        "status": "running",
        "config": config,
        "reuse_a1_root": str(reuse_root),
        "entries": [],
        "fits": {},
    }
    if args.resume and manifest_path.exists():
        campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
        if campaign.get("config") != json_safe(config):
            raise SystemExit("REFUSED: A2 campaign configuration mismatch")
        campaign["status"] = "running"
        campaign["resumed_utc"] = utc_now()
    atomic_json(manifest_path, campaign)

    xi, _ = make_patterns_numpy(config["N"], config["P"], config["seed"])
    completed = {
        (entry["lambda"], entry["K"]): entry
        for entry in campaign.get("entries", [])
        if entry.get("status") == "complete"
    }
    entries = []
    parameter_total = len(config["lambdas"])*len(config["K_list"])
    parameter_index = 0

    # K=6 A1 hints determine only run duration, never fitted values at other K.
    K6_hints: dict[float, float] = {}
    if config["mode"] == "production":
        reuse_campaign_path = reuse_root / "campaign_manifest.json"
        reuse_compatible = False
        if reuse_campaign_path.exists():
            reuse_campaign = json.loads(
                reuse_campaign_path.read_text(encoding="utf-8")
            )
            reuse_config = reuse_campaign.get("config", {})
            reuse_compatible = all([
                reuse_campaign.get("status") == "complete",
                reuse_config.get("N") == config["N"],
                reuse_config.get("P") == config["P"],
                reuse_config.get("K") == 6,
                reuse_config.get("beta") == config["beta"],
                reuse_config.get("seed") == config["seed"],
            ])
        if 6 in config["K_list"] and not reuse_compatible:
            raise SystemExit(
                "REFUSED: reusable K6 campaign is missing or incompatible"
            )
        if reuse_compatible:
            for lam in config["lambdas"]:
                summary_path = (
                    reuse_root / "runs"
                    / f"lam{lambda_tag(lam)}_coherent/coarse/summary.json"
                )
                if not summary_path.exists():
                    if 6 in config["K_list"]:
                        raise SystemExit(
                            f"REFUSED: missing reusable K6 {summary_path}"
                        )
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                K6_hints[lam] = summary["metrics"]["T1_median"]

    for lam in config["lambdas"]:
        for K in config["K_list"]:
            parameter_index += 1
            key = (lam, K)
            if key in completed:
                entries.append(completed[key])
                continue
            if config["mode"] == "production" and K == 6:
                run_dir = (
                    reuse_root / "runs"
                    / f"lam{lambda_tag(lam)}_coherent/coarse"
                )
                data = load_physical_trajectory(run_dir)
                analysis = analyze_delay_trajectory(
                    data,
                    P=config["P"],
                    transient_fraction=config["transient_fraction"],
                    T1_hint=K6_hints.get(lam),
                )
                entry = {
                    "lambda": lam,
                    "K": K,
                    "status": "complete",
                    "source": "reused_E36A1",
                    "source_path": str(run_dir),
                    "analysis": analysis,
                }
            else:
                duration = planned_duration(
                    K,
                    K6_hints.get(lam),
                    dt=config["dt"],
                    record_every=config["record_every"],
                    transient_fraction=config["transient_fraction"],
                    target_relays=config["target_relays"],
                    fixed_t_final=config["fixed_t_final"],
                )
                run_config = {
                    "N": config["N"],
                    "P": config["P"],
                    "K": K,
                    "beta": config["beta"],
                    "t0": config["t0"],
                    "seed": config["seed"],
                    "lambda": lam,
                    "dt": config["dt"],
                    "t_final": duration,
                    "record_every": config["record_every"],
                    "checkpoint_every_steps": (
                        config["checkpoint_every_steps"]
                    ),
                    "heartbeat_every_steps": (
                        config["heartbeat_every_steps"]
                    ),
                    "heartbeat_min_wall_seconds": (
                        config["heartbeat_min_wall_seconds"]
                    ),
                    "chunk_size": config["chunk_size"],
                    "transient_fraction": config["transient_fraction"],
                    "T1_hint": K6_hints.get(lam),
                    "run_id": f"lam{lambda_tag(lam)}_K{K}",
                    "parameter_index": parameter_index,
                    "parameter_total": parameter_total,
                }
                run_dir = (
                    output_dir / "runs"
                    / f"lam{lambda_tag(lam)}_K{K}"
                )
                run_config = preserve_resume_run_config(
                    run_dir,
                    run_config,
                    resume=args.resume,
                )
                summary, run_dir = run_new_trajectory(
                    xi=xi,
                    config=run_config,
                    run_dir=run_dir,
                    heartbeat_path=heartbeat_path,
                    resume=args.resume,
                )
                entry = {
                    "lambda": lam,
                    "K": K,
                    "status": "complete",
                    "source": "E36A2_new",
                    "source_path": str(run_dir),
                    "analysis": summary["analysis"],
                }
            entries.append(entry)
            campaign["entries"] = entries
            campaign["updated_utc"] = utc_now()
            atomic_json(manifest_path, campaign)

    fits = {}
    for lam in config["lambdas"]:
        fits[str(lam)] = fit_k_scaling([
            entry for entry in entries if entry["lambda"] == lam
        ])
    campaign["entries"] = entries
    campaign["fits"] = fits
    campaign["status"] = "complete"
    campaign["completed_utc"] = utc_now()
    campaign["output_bytes"] = directory_bytes(output_dir)
    atomic_json(manifest_path, campaign)
    print(json.dumps({
        "status": "complete",
        "output_dir": str(output_dir),
        "n_entries": len(entries),
        "fits": json_safe(fits),
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
