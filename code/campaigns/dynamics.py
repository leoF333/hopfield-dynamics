"""Exact reduced DDE dynamics, event classification, and warm-start descent."""
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from couplings import make_patterns
from v5_paths import RUNS, activate_reference_modules, environment_manifest, write_json

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402


@dataclass
class DynamicConfig:
    N: int = 2000
    P: int = 100
    seed: int = 42
    beta: float = 20.0
    tau: float = 10.0
    t0: float = 1.0
    dt: float = 0.01
    record_dt: float = 0.05
    chunk_time: float = 200.0
    max_time: float = 20000.0
    no_tour_arrest_time: float = 5000.0
    required_tours: int = 5
    forward_fraction: float = 0.99


def signed_steps(winner: np.ndarray, P: int) -> np.ndarray:
    raw = np.diff(winner)
    return ((raw + P // 2) % P) - P // 2


def relay_statistics(times: np.ndarray, winner: np.ndarray, P: int) -> dict:
    changes = np.flatnonzero(np.diff(winner) != 0) + 1
    if not changes.size:
        return {
            "changes": 0, "advance": 0.0, "tours": 0.0,
            "forward_fraction": 0.0, "dwell": np.full(P, np.nan),
            "slowest_bond": -1, "reversal_fraction": 0.0,
            "relay_times": np.empty(0),
            "relay_before": np.empty(0, dtype=int),
            "relay_after": np.empty(0, dtype=int),
        }
    before = winner[changes - 1]
    after = winner[changes]
    steps = signed_steps(np.r_[before[0], after], P)
    single = np.abs(steps) == 1
    forward = steps == 1
    advance = float(steps.sum())
    ffrac = float(forward.sum() / max(single.sum(), 1))
    dwell_lists = [[] for _ in range(P)]
    for left, right in zip(changes[:-1], changes[1:]):
        mu = int(winner[left])
        dwell_lists[mu].append(float(times[right] - times[left]))
    dwell = np.array(
        [np.median(x) if x else np.nan for x in dwell_lists], dtype=float
    )
    return {
        "changes": int(changes.size),
        "advance": advance,
        "tours": max(0.0, advance / P),
        "forward_fraction": ffrac,
        "dwell": dwell,
        "slowest_bond": int(np.nanargmax(dwell)) if np.isfinite(dwell).any() else -1,
        "reversal_fraction": float((steps < 0).sum() / max(len(steps), 1)),
        "relay_times": times[changes],
        "relay_before": before,
        "relay_after": after,
    }


def relay_statistics_from_events(
    event_times: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    P: int,
) -> dict:
    """Exact relay statistics from a streaming, untruncated event record."""
    event_times = np.asarray(event_times, dtype=float)
    before = np.asarray(before, dtype=int)
    after = np.asarray(after, dtype=int)
    if not event_times.size:
        return {
            "changes": 0, "advance": 0.0, "tours": 0.0,
            "forward_fraction": 0.0, "dwell": np.full(P, np.nan),
            "slowest_bond": -1, "reversal_fraction": 0.0,
            "relay_times": np.empty(0),
            "relay_before": np.empty(0, dtype=int),
            "relay_after": np.empty(0, dtype=int),
        }
    if not (len(event_times) == len(before) == len(after)):
        raise ValueError("relay event arrays must have equal lengths")
    steps = signed_steps(np.r_[before[0], after], P)
    single = np.abs(steps) == 1
    forward = steps == 1
    dwell_lists = [[] for _ in range(P)]
    for index, duration in enumerate(np.diff(event_times)):
        dwell_lists[int(after[index])].append(float(duration))
    dwell = np.array(
        [np.median(x) if x else np.nan for x in dwell_lists], dtype=float
    )
    return {
        "changes": int(len(event_times)),
        "advance": float(steps.sum()),
        "tours": max(0.0, float(steps.sum()) / P),
        "forward_fraction": float(forward.sum() / max(single.sum(), 1)),
        "dwell": dwell,
        "slowest_bond": (
            int(np.nanargmax(dwell)) if np.isfinite(dwell).any() else -1
        ),
        "reversal_fraction": float((steps < 0).sum() / max(len(steps), 1)),
        "relay_times": event_times,
        "relay_before": before,
        "relay_after": after,
    }


def reduced_fixed_residual(system: ReducedDDE, history: np.ndarray) -> float:
    a = history[-1]
    return float(np.linalg.norm(system.rhs(a, a)) / np.sqrt(system.P))


def periodic_closure(times: np.ndarray, states: np.ndarray,
                     tau: float) -> tuple[float, float]:
    """Best tail recurrence and candidate period, excluding near-zero lags."""
    if len(times) < 100:
        return np.inf, np.nan
    tail_start = max(0, len(times) - 40_000)
    tail = states[tail_start:]
    t_tail = times[tail_start:]
    scale = max(float(np.linalg.norm(tail[-1])), 1e-12)
    eligible = np.flatnonzero((t_tail[-1] - t_tail) >= max(tau, 1.0))
    if not eligible.size:
        return np.inf, np.nan
    # Coarse search, then exact search around the best coarse candidate.
    coarse = eligible[::max(1, len(eligible) // 2000)]
    error = np.linalg.norm(tail[coarse] - tail[-1], axis=1) / scale
    best_coarse = int(coarse[np.argmin(error)])
    radius = max(2, len(eligible) // 2000)
    local = np.arange(
        max(eligible[0], best_coarse - radius),
        min(eligible[-1] + 1, best_coarse + radius + 1),
    )
    local_error = np.linalg.norm(tail[local] - tail[-1], axis=1) / scale
    best = int(local[np.argmin(local_error)])
    return float(local_error.min()), float(t_tail[-1] - t_tail[best])


def integrate_and_classify(
    system: ReducedDDE,
    history: np.ndarray,
    derivative_history: np.ndarray | None,
    config: DynamicConfig,
    *,
    keep_trajectory: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray]:
    rec_every = max(1, round(config.record_dt / config.dt))
    t_offset = 0.0
    all_t: list[np.ndarray] = []
    all_a: list[np.ndarray] = []
    event_times_parts: list[np.ndarray] = []
    event_before_parts: list[np.ndarray] = []
    event_after_parts: list[np.ndarray] = []
    previous_time: float | None = None
    previous_winner: int | None = None
    final_hist = np.asarray(history, dtype=float)
    final_dhist = derivative_history
    started = time.perf_counter()
    label = "timeout_other"
    while t_offset < config.max_time:
        duration = min(config.chunk_time, config.max_time - t_offset)
        sol = system.integrate(
            final_hist, duration, config.dt, record_every=rec_every,
            da_hist0=final_dhist,
        )
        final_hist, final_dhist = sol["hist"], sol["dhist"]
        chunk_times = sol["t"][1:] + t_offset
        chunk_states = sol["a"][1:]
        chunk_winner = np.argmax(chunk_states, axis=1)
        if previous_winner is None:
            event_times_input = chunk_times
            event_winner_input = chunk_winner
        else:
            event_times_input = np.r_[previous_time, chunk_times]
            event_winner_input = np.r_[previous_winner, chunk_winner]
        changes = np.flatnonzero(np.diff(event_winner_input) != 0) + 1
        if changes.size:
            event_times_parts.append(event_times_input[changes])
            event_before_parts.append(event_winner_input[changes - 1])
            event_after_parts.append(event_winner_input[changes])
        if len(chunk_winner):
            previous_time = float(chunk_times[-1])
            previous_winner = int(chunk_winner[-1])
        all_t.append(chunk_times)
        all_a.append(chunk_states)
        t_offset += duration
        times = np.concatenate(all_t)
        states = np.concatenate(all_a)
        relay = relay_statistics_from_events(
            (
                np.concatenate(event_times_parts)
                if event_times_parts else np.empty(0)
            ),
            (
                np.concatenate(event_before_parts)
                if event_before_parts else np.empty(0, dtype=int)
            ),
            (
                np.concatenate(event_after_parts)
                if event_after_parts else np.empty(0, dtype=int)
            ),
            system.P,
        )
        residual = reduced_fixed_residual(system, final_hist)
        closure, candidate_period = periodic_closure(
            times, states, config.tau
        )
        if (
            relay["tours"] >= config.required_tours
            and relay["forward_fraction"] > config.forward_fraction
        ):
            label = "ordered_cycle"
            break
        if residual < 1e-10:
            label = "stationary_arrest"
            break
        if (
            t_offset >= config.no_tour_arrest_time
            and relay["tours"] < 1.0
            and closure < 1e-6
        ):
            label = "noncycling_periodic"
            break
        # Retain at most the latest six-tour-scale record to bound RAM. The full
        # final history is never pruned.
        if len(times) > 200_000:
            all_t = [times[-100_000:]]
            all_a = [states[-100_000:]]

    times = np.concatenate(all_t) if all_t else np.array([])
    states = np.concatenate(all_a) if all_a else np.empty((0, system.P))
    winner = np.argmax(states, axis=1) if len(states) else np.array([], dtype=int)
    relay = relay_statistics_from_events(
        np.concatenate(event_times_parts) if event_times_parts else np.empty(0),
        (
            np.concatenate(event_before_parts)
            if event_before_parts else np.empty(0, dtype=int)
        ),
        (
            np.concatenate(event_after_parts)
            if event_after_parts else np.empty(0, dtype=int)
        ),
        system.P,
    )
    residual = reduced_fixed_residual(system, final_hist)
    closure, candidate_period = periodic_closure(times, states, config.tau)
    result = {
        "label": label,
        "time": float(t_offset),
        "wall_seconds": time.perf_counter() - started,
        "fixed_residual": residual,
        "periodic_closure": closure,
        "candidate_period": candidate_period,
        **relay,
    }
    if keep_trajectory:
        result.update(record_t=times, record_a=states, winner=winner)
    return result, final_hist, final_dhist


def run_descent(
    config: DynamicConfig,
    output: Path,
    lambda_start: float = 0.40,
    lambda_stop: float = 0.25,
    lambda_step: float = 0.002,
    bracket_tol: float = 5e-4,
    xi_override: np.ndarray | None = None,
) -> dict:
    if (
        output.exists()
        and output.with_suffix(".json").exists()
        and os.environ.get("V5_FORCE", "0") != "1"
    ):
        return json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    xi = (
        make_patterns(config.N, config.P, config.seed)[0]
        if xi_override is None else np.asarray(xi_override, dtype=float)
    )
    system = ReducedDDE(xi, config.beta, lambda_start, config.tau, config.t0)
    checkpoint = output.with_suffix(".checkpoint.npz")
    initial = np.zeros(config.P)
    initial[0] = 0.99
    initial[1 % config.P] = 0.05
    history: np.ndarray = initial
    dhistory = None
    rows: list[dict] = []
    cycle_history = None
    cycle_dhistory = None
    arrest_history = None
    arrest_dhistory = None

    scalar_keys = [
        "label", "time", "wall_seconds", "fixed_residual",
        "periodic_closure", "candidate_period", "changes", "advance",
        "tours", "forward_fraction", "slowest_bond", "reversal_fraction",
        "lambda",
    ]

    def save_checkpoint(next_lambda: float) -> None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save = {
            key: np.array([row.get(key, np.nan) for row in rows])
            for key in scalar_keys
        }
        save.update(
            next_lambda=np.array(next_lambda),
            history=history,
            dhistory=(np.empty((0, config.P)) if dhistory is None else dhistory),
            cycle_history=(
                np.empty((0, config.P)) if cycle_history is None else cycle_history
            ),
            cycle_dhistory=(
                np.empty((0, config.P)) if cycle_dhistory is None else cycle_dhistory
            ),
            dwell=np.stack([
                np.asarray(row.get("dwell", np.full(config.P, np.nan)))
                for row in rows
            ]),
        )
        event_counts = np.array([
            len(row.get("relay_times", [])) for row in rows
        ], dtype=int)
        save["event_offsets"] = np.r_[0, np.cumsum(event_counts)]
        for key in ("relay_times", "relay_before", "relay_after"):
            save[key] = np.concatenate([
                np.asarray(row.get(key, [])) for row in rows
            ]) if event_counts.sum() else np.empty(0)
        np.savez_compressed(checkpoint, **save)

    lam = lambda_start
    if checkpoint.exists():
        saved = np.load(checkpoint)
        row_count = len(saved["lambda"])
        rows = [
            {
                key: (
                    saved[key][i].item()
                    if isinstance(saved[key][i], np.generic) else saved[key][i]
                )
                for key in scalar_keys
            }
            for i in range(row_count)
        ]
        if "dwell" in saved.files:
            offsets = saved["event_offsets"]
            for i, row in enumerate(rows):
                row["dwell"] = saved["dwell"][i]
                left, right = int(offsets[i]), int(offsets[i + 1])
                row["relay_times"] = saved["relay_times"][left:right]
                row["relay_before"] = saved["relay_before"][left:right].astype(int)
                row["relay_after"] = saved["relay_after"][left:right].astype(int)
        history = saved["history"]
        dhistory = saved["dhistory"] if len(saved["dhistory"]) else None
        cycle_history = (
            saved["cycle_history"] if len(saved["cycle_history"]) else None
        )
        cycle_dhistory = (
            saved["cycle_dhistory"] if len(saved["cycle_dhistory"]) else None
        )
        lam = float(saved["next_lambda"])
        print(
            f"[dynamic seed {config.seed}] resumed at lambda={lam:.6f} "
            f"after {row_count} completed points",
            flush=True,
        )

    while lam >= lambda_stop - 1e-12:
        system.lam = lam
        result, history, dhistory = integrate_and_classify(
            system, history, dhistory, config,
        )
        result["lambda"] = float(lam)
        rows.append(result)
        print(
            f"[dynamic seed {config.seed}] lambda={lam:.6f} "
            f"label={result['label']} tours={result.get('tours', 0):.2f} "
            f"forward={result.get('forward_fraction', 0):.4f} "
            f"sim_t={result['time']:.0f} wall={result['wall_seconds']:.1f}s",
            flush=True,
        )
        if result["label"] == "ordered_cycle":
            cycle_history, cycle_dhistory = history.copy(), dhistory.copy()
            lam -= lambda_step
            save_checkpoint(lam)
            continue
        if cycle_history is not None and result["label"] in {
            "stationary_arrest", "noncycling_periodic"
        }:
            arrest_history = history.copy()
            arrest_dhistory = dhistory.copy()
            save_checkpoint(lam)
            break
        if cycle_history is not None:
            raise RuntimeError(
                f"encountered {result['label']} at lambda={lam}; "
                "the dynamic boundary is intentionally left unresolved until "
                "Lyapunov/attractor classification is run"
            )
        raise RuntimeError(
            f"no coherent cycle constructed at lambda={lambda_start}: {result['label']}"
        )

    if cycle_history is None or arrest_history is None:
        raise RuntimeError("dynamic boundary not bracketed by the descent")
    low = rows[-1]["lambda"]
    high = next(r["lambda"] for r in reversed(rows[:-1])
                if r["label"] == "ordered_cycle")
    # high cycles; low does not. Each bisection trial starts from the independently
    # saved cycling history and never consults a static threshold.
    while high - low > bracket_tol:
        mid = 0.5 * (low + high)
        system.lam = mid
        result, h_mid, dh_mid = integrate_and_classify(
            system, cycle_history.copy(), cycle_dhistory.copy(), config,
        )
        result["lambda"] = float(mid)
        rows.append(result)
        print(
            f"[dynamic seed {config.seed}] bisection lambda={mid:.6f} "
            f"label={result['label']} tours={result.get('tours', 0):.2f} "
            f"wall={result['wall_seconds']:.1f}s",
            flush=True,
        )
        if result["label"] == "ordered_cycle":
            high = mid
            cycle_history, cycle_dhistory = h_mid, dh_mid
        elif result["label"] in {"stationary_arrest", "noncycling_periodic"}:
            low = mid
            arrest_history = h_mid
            arrest_dhistory = dh_mid
        else:
            raise RuntimeError(
                f"bisection encountered {result['label']} at lambda={mid}; "
                "no forced cycle/arrest assignment was made"
            )

    # Required history sensitivity: construct a second coherent cycle from a
    # macroscopically different initial memory, then bring it independently to the
    # final cycling bracket. This is not a tiny perturbation of the first history.
    second_initial = np.zeros(config.P)
    second_mu = config.P // 2
    second_initial[second_mu] = 0.99
    second_initial[(second_mu + 1) % config.P] = 0.05
    system.lam = lambda_start
    second_start, second_hist, second_dhist = integrate_and_classify(
        system, second_initial, None, config,
    )
    if second_start["label"] != "ordered_cycle":
        raise RuntimeError(
            "second independent cycle construction failed at lambda_start: "
            f"{second_start['label']}"
        )
    system.lam = high
    second_high, second_hist, second_dhist = integrate_and_classify(
        system, second_hist, second_dhist, config,
    )
    if second_high["label"] != "ordered_cycle":
        raise RuntimeError(
            "second independent cycle history did not survive at the final "
            f"cycling bracket: {second_high['label']}"
        )

    # Required history sensitivity at the last cycling and first arrested points.
    repeats = []
    independent_histories = [
        ("primary", cycle_history, cycle_dhistory),
        ("independent_memory", second_hist, second_dhist),
    ]
    for lam_test in (high, low):
        for name, stored_history, stored_derivative in independent_histories:
            system.lam = lam_test
            result, _, _ = integrate_and_classify(
                system, stored_history.copy(), stored_derivative.copy(), config,
            )
            repeats.append({
                "lambda": float(lam_test),
                "history": name,
                **{
                    key: (value.item() if isinstance(value, np.generic) else value)
                    for key, value in result.items()
                    if np.isscalar(value) or isinstance(value, str)
                },
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = sorted(
        set.intersection(*[
            {k for k, v in row.items() if np.isscalar(v) or isinstance(v, str)}
            for row in rows
        ])
    )
    save = {
        key: np.array([row[key] for row in rows])
        for key in scalar_keys
    }
    save.update(
        dynamic_low=np.array(low), dynamic_high=np.array(high),
        last_cycle_history=cycle_history,
        last_cycle_dhistory=cycle_dhistory,
        first_noncycle_history=arrest_history,
        first_noncycle_dhistory=arrest_dhistory,
        N=np.array(config.N), P=np.array(config.P), seed=np.array(config.seed),
        dwell=np.stack([
            np.asarray(row.get("dwell", np.full(config.P, np.nan)))
            for row in rows
        ]),
    )
    event_counts = np.array([
        len(row.get("relay_times", [])) for row in rows
    ], dtype=int)
    save["event_offsets"] = np.r_[0, np.cumsum(event_counts)]
    for key in ("relay_times", "relay_before", "relay_after"):
        save[key] = np.concatenate([
            np.asarray(row.get(key, [])) for row in rows
        ]) if event_counts.sum() else np.empty(0)
    np.savez_compressed(output, **save)
    summary = {
        "config": asdict(config),
        "output": str(output),
        "dynamic_noncycle_low": float(low),
        "dynamic_cycle_high": float(high),
        "last_cycle_slowest_bond": int(
            next(r["slowest_bond"] for r in reversed(rows)
                 if r["label"] == "ordered_cycle")
        ),
        "arrest_primary_motif": int(np.argmax(arrest_history[-1])),
        "arrest_top_two_motifs": np.argsort(
            arrest_history[-1]
        )[-2:][::-1].astype(int).tolist(),
        "outcomes": {label: sum(r["label"] == label for r in rows)
                     for label in sorted({r["label"] for r in rows})},
        "history_repeats": repeats,
        "independent_history_control_passed": (
            all(
                row["label"] == "ordered_cycle"
                for row in repeats if row["lambda"] == high
            )
            and all(
                row["label"] in {"stationary_arrest", "noncycling_periodic"}
                for row in repeats if row["lambda"] == low
            )
        ),
        "environment": environment_manifest(),
    }
    write_json(output.with_suffix(".json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--lambda-start", type=float, default=0.40)
    parser.add_argument("--lambda-stop", type=float, default=0.25)
    parser.add_argument("--lambda-step", type=float, default=0.002)
    parser.add_argument("--bracket-tol", type=float, default=5e-4)
    parser.add_argument("--max-time", type=float, default=20000)
    parser.add_argument("--required-tours", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = DynamicConfig(
        N=args.N, P=args.P, seed=args.seed, beta=args.beta, tau=args.tau,
        dt=args.dt, max_time=args.max_time, required_tours=args.required_tours,
    )
    output = args.out or (
        RUNS / "N1" / f"dynamic_N{args.N}_P{args.P}_seed{args.seed}.npz"
    )
    print(run_descent(
        cfg, output, args.lambda_start, args.lambda_stop,
        args.lambda_step, args.bracket_tol,
    ))


if __name__ == "__main__":
    main()
