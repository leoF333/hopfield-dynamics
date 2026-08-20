"""N1 branch/fold audit and restart-safe dynamic convergence controls."""
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
from dataclasses import asdict
from pathlib import Path

import numpy as np

from couplings import make_patterns
from dynamics import DynamicConfig, integrate_and_classify
from fold_refinement import (
    FoldResult,
    refine_fold_from_coefficients,
    refine_fold_from_full_state,
)
from n2_local_laws import resample_history
from v5_paths import RUNS, activate_reference_modules, environment_manifest, write_json

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402


ROOT = RUNS / "N1_AUDIT"


def fold_payload(result: FoldResult) -> dict:
    payload = asdict(result)
    payload.pop("coefficients")
    payload.pop("null_vector")
    return payload


def audit_branches(seed: int, N: int, P: int) -> dict:
    output = ROOT / f"branch_audit_N{N}_P{P}_s{seed}.json"
    arrays = output.with_suffix(".npz")
    if output.exists() and arrays.exists() and os.environ.get("V5_FORCE") != "1":
        return json.loads(output.read_text(encoding="utf-8"))

    n1 = RUNS / "N1"
    static_path = n1 / f"static_N{N}_P{P}_seed{seed}.npz"
    dynamic_path = n1 / f"dynamic_N{N}_P{P}_seed{seed}.npz"
    dynamic_json = dynamic_path.with_suffix(".json")
    if not (static_path.exists() and dynamic_path.exists() and dynamic_json.exists()):
        raise FileNotFoundError(f"complete N1 static/dynamic files required for seed {seed}")

    static = np.load(static_path)
    dynamic = np.load(dynamic_path)
    summary = json.loads(dynamic_json.read_text(encoding="utf-8"))
    xi, _ = make_patterns(N, P, seed)
    valid = np.flatnonzero(static["resolved"])
    order = valid[np.argsort(static["lambda_low"][valid])[::-1]]
    selected = int(order[0])
    runner_up = int(order[1])

    memory_guess = 0.5 * (
        float(static["lambda_low"][selected])
        + float(static["lambda_high"][selected])
    )
    memory_fold = refine_fold_from_full_state(
        xi, static["state"][selected], memory_guess
    )
    runner_guess = 0.5 * (
        float(static["lambda_low"][runner_up])
        + float(static["lambda_high"][runner_up])
    )
    runner_fold = refine_fold_from_full_state(
        xi, static["state"][runner_up], runner_guess
    )

    arrest_coefficients = np.asarray(dynamic["first_noncycle_history"][-1], float)
    arrest_guess = float(dynamic["dynamic_low"])
    arrest_fold = refine_fold_from_coefficients(
        xi, arrest_coefficients, arrest_guess
    )
    fold_state_distance = float(
        np.linalg.norm(memory_fold.coefficients - arrest_fold.coefficients)
        / np.sqrt(P)
    )
    arrest_to_memory_distance = float(
        np.linalg.norm(arrest_coefficients - memory_fold.coefficients)
        / np.sqrt(P)
    )
    dyn_low = float(dynamic["dynamic_low"])
    dyn_high = float(dynamic["dynamic_high"])
    dyn_mid = 0.5 * (dyn_low + dyn_high)
    payload = {
        "seed": seed,
        "N": N,
        "P": P,
        "static_selected_motif": selected,
        "static_runner_up_motif": runner_up,
        "dynamic_slowest_motif": int(summary["last_cycle_slowest_bond"]),
        "dynamic_arrest_motif": int(summary["arrest_primary_motif"]),
        "original_static_bracket": [
            float(static["lambda_low"][selected]),
            float(static["lambda_high"][selected]),
        ],
        "original_dynamic_bracket": [dyn_low, dyn_high],
        "memory_fold": fold_payload(memory_fold),
        "runner_up_fold": fold_payload(runner_fold),
        "arrest_branch_fold": fold_payload(arrest_fold),
        "dynamic_mid_minus_memory_fold": dyn_mid - memory_fold.lambda_fold,
        "dynamic_mid_minus_arrest_fold": dyn_mid - arrest_fold.lambda_fold,
        "fold_state_distance": fold_state_distance,
        "arrest_to_memory_fold_distance": arrest_to_memory_distance,
        "same_fold_numerically": bool(fold_state_distance < 1e-8),
        "environment": environment_manifest(),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays,
        memory_coefficients=memory_fold.coefficients,
        memory_null_vector=memory_fold.null_vector,
        runner_up_coefficients=runner_fold.coefficients,
        runner_up_null_vector=runner_fold.null_vector,
        arrest_coefficients=arrest_fold.coefficients,
        arrest_null_vector=arrest_fold.null_vector,
    )
    write_json(output, payload)
    return payload


def _point_tag(lam: float, dt: float, tours: int, direction: str) -> str:
    return (
        f"{direction}_lam{lam:.8f}_dt{dt:g}_tours{tours}"
        .replace(".", "p")
    )


def classify_saved_point(
    *,
    seed: int,
    N: int,
    P: int,
    xi: np.ndarray,
    lam: float,
    dt: float,
    tours: int,
    history: np.ndarray,
    derivative: np.ndarray,
    direction: str,
) -> dict:
    tag = _point_tag(lam, dt, tours, direction)
    output = ROOT / "points" / f"N{N}_P{P}_s{seed}_{tag}.npz"
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists() and os.environ.get("V5_FORCE") != "1":
        return json.loads(sidecar.read_text(encoding="utf-8"))

    system = ReducedDDE(xi, 20.0, lam, 10.0, 1.0)
    config = DynamicConfig(
        N=N,
        P=P,
        seed=seed,
        dt=dt,
        required_tours=tours,
        forward_fraction=0.99,
        chunk_time=200.0,
        max_time=max(40_000.0, 5_000.0 * tours),
        no_tour_arrest_time=5_000.0,
        record_dt=max(0.05, dt),
    )
    started = time.perf_counter()
    result, final_history, final_derivative = integrate_and_classify(
        system, history.copy(), derivative.copy(), config
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        final_history=final_history,
        final_derivative=final_derivative,
        relay_times=result["relay_times"],
        relay_before=result["relay_before"],
        relay_after=result["relay_after"],
        dwell=result["dwell"],
    )
    payload = {
        "seed": seed,
        "N": N,
        "P": P,
        "lambda": lam,
        "dt": dt,
        "required_tours": tours,
        "direction": direction,
        "label": result["label"],
        "simulated_time": result["time"],
        "wall_seconds": time.perf_counter() - started,
        "tours": result["tours"],
        "forward_fraction": result["forward_fraction"],
        "fixed_residual": result["fixed_residual"],
        "slowest_bond": result["slowest_bond"],
        "output": str(output),
        "environment": environment_manifest(),
    }
    write_json(sidecar, payload)
    return payload


def refine_dynamic_boundary(
    seed: int,
    N: int,
    P: int,
    *,
    dt: float,
    tours: int,
    bracket_tol: float,
    direction: str = "descending",
) -> dict:
    tolerance_tag = f"{bracket_tol:g}".replace(".", "p")
    output = ROOT / (
        f"dynamic_audit_N{N}_P{P}_s{seed}_dt{dt:g}_tours{tours}_"
        f"tol{tolerance_tag}_{direction}.json"
    )
    if output.exists() and os.environ.get("V5_FORCE") != "1":
        return json.loads(output.read_text(encoding="utf-8"))

    n1_path = RUNS / "N1" / f"dynamic_N{N}_P{P}_seed{seed}.npz"
    dynamic = np.load(n1_path)
    branch = audit_branches(seed, N, P)
    xi, _ = make_patterns(N, P, seed)
    if direction == "descending":
        source_history = dynamic["last_cycle_history"]
        source_derivative = dynamic["last_cycle_dhistory"]
    elif direction == "ascending":
        source_history = dynamic["first_noncycle_history"]
        source_derivative = dynamic["first_noncycle_dhistory"]
    else:
        raise ValueError("direction must be descending or ascending")
    history, derivative = resample_history(
        source_history, source_derivative, 10.0, 0.01, dt
    )

    old_low = float(dynamic["dynamic_low"])
    old_high = float(dynamic["dynamic_high"])
    fold_values = [
        branch["memory_fold"]["lambda_fold"],
        branch["arrest_branch_fold"]["lambda_fold"],
    ]
    low = min(old_low, *fold_values) - 0.001
    high = max(old_high, *fold_values) + 0.001
    rows: list[dict] = []

    def evaluate(lam: float) -> dict:
        row = classify_saved_point(
            seed=seed,
            N=N,
            P=P,
            xi=xi,
            lam=lam,
            dt=dt,
            tours=tours,
            history=history,
            derivative=derivative,
            direction=direction,
        )
        rows.append(row)
        return row

    low_row = evaluate(low)
    high_row = evaluate(high)
    cycle_labels = {"ordered_cycle"}
    arrest_labels = {"stationary_arrest", "noncycling_periodic"}
    if direction == "descending":
        low_is_arrest = low_row["label"] in arrest_labels
        high_is_cycle = high_row["label"] in cycle_labels
    else:
        low_is_arrest = low_row["label"] in arrest_labels
        high_is_cycle = high_row["label"] in cycle_labels
    if not (low_is_arrest and high_is_cycle):
        payload = {
            "seed": seed,
            "dt": dt,
            "required_tours": tours,
            "direction": direction,
            "resolved": False,
            "reason": "initial bracket did not classify as arrest/cycle",
            "rows": rows,
            "environment": environment_manifest(),
        }
        write_json(output, payload)
        return payload

    while high - low > bracket_tol:
        mid = 0.5 * (low + high)
        row = evaluate(mid)
        if row["label"] in cycle_labels:
            high = mid
        elif row["label"] in arrest_labels:
            low = mid
        else:
            payload = {
                "seed": seed,
                "dt": dt,
                "required_tours": tours,
                "direction": direction,
                "resolved": False,
                "reason": f"unresolved label {row['label']} at lambda={mid}",
                "rows": rows,
                "environment": environment_manifest(),
            }
            write_json(output, payload)
            return payload

    payload = {
        "seed": seed,
        "N": N,
        "P": P,
        "dt": dt,
        "required_tours": tours,
        "direction": direction,
        "resolved": True,
        "dynamic_low": low,
        "dynamic_high": high,
        "rows": rows,
        "environment": environment_manifest(),
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 44, 48, 61])
    parser.add_argument(
        "--mode", choices=["branches", "dynamic", "all"], default="branches"
    )
    parser.add_argument("--dts", type=float, nargs="+", default=[0.005])
    parser.add_argument("--tours", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--bracket-tol", type=float, default=2.5e-5)
    parser.add_argument("--hysteresis", action="store_true")
    args = parser.parse_args()
    branch_rows = [audit_branches(seed, args.N, args.P) for seed in args.seeds]
    dynamic_rows = []
    if args.mode in {"dynamic", "all"}:
        for seed in args.seeds:
            for dt in args.dts:
                for tours in args.tours:
                    dynamic_rows.append(refine_dynamic_boundary(
                        seed,
                        args.N,
                        args.P,
                        dt=dt,
                        tours=tours,
                        bracket_tol=args.bracket_tol,
                        direction="descending",
                    ))
                    if args.hysteresis:
                        dynamic_rows.append(refine_dynamic_boundary(
                            seed,
                            args.N,
                            args.P,
                            dt=dt,
                            tours=tours,
                            bracket_tol=args.bracket_tol,
                            direction="ascending",
                        ))
    payload = {
        "campaign": "N1 convergence and branch audit",
        "N": args.N,
        "P": args.P,
        "seeds": args.seeds,
        "branch_rows": branch_rows,
        "dynamic_rows": dynamic_rows,
        "environment": environment_manifest(),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    seed_tag = "_".join(map(str, args.seeds))
    mode_tag = args.mode
    dt_tag = "_".join(f"{value:g}" for value in args.dts).replace(".", "p")
    tours_tag = "_".join(map(str, args.tours))
    tol_tag = f"{args.bracket_tol:g}".replace(".", "p")
    summary_path = ROOT / (
        f"summary_{mode_tag}_seeds_{seed_tag}_dt{dt_tag}_"
        f"tours{tours_tag}_tol{tol_tag}_hyst{int(args.hysteresis)}.json"
    )
    write_json(summary_path, payload)
    print(
        f"N1 audit completed {len(branch_rows)} branch audits and "
        f"{len(dynamic_rows)} dynamic refinements"
    )


if __name__ == "__main__":
    main()
