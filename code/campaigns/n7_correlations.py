"""N7 multi-seed Markov-flip and smooth periodic correlation controls."""
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
from pathlib import Path

import numpy as np

from observables import save_observables, simulate_observables
from patterns import (
    adjacent_correlations, gp_arcsine_prediction, gp_circle, markov_flip,
)
from static_thresholds import StaticConfig, run as run_static
from v5_paths import RUNS, environment_manifest, write_json

CS = [0.0, 0.2, 0.4, 0.6, 0.8]
KAPPAS = [0.8, 0.4, 0.2, 0.1, 0.05]


def _is_moving(classification: str) -> bool:
    return classification in {
        "periodic_sequence_candidate", "moving_requires_lyapunov"
    }


def dynamic_descent(
    xi: np.ndarray, seed: int, family: str, parameter: float,
    N: int, P: int, lambda_min: float, smoke: bool,
) -> list[dict]:
    lambdas = (
        [0.9, 0.5, lambda_min] if smoke
        else np.arange(0.9, lambda_min - 1e-12, -0.02).tolist()
    )
    history = None
    dhistory = None
    rows = []
    for index, lam in enumerate(lambdas):
        tag = str(parameter).replace(".", "p")
        path = (
            RUNS / "N7" /
            f"{family}_{tag}_N{N}_P{P}_s{seed}_lam{lam:.3f}.npz"
        )
        if path.exists():
            saved = np.load(path)
            history = saved["final_history"]
            dhistory = saved["final_derivative_history"]
            scalar = {
                key: saved[key].item()
                for key in saved.files
                if saved[key].ndim == 0
            }
            rows.append({
                "family": family, "parameter": parameter, "seed": seed,
                "lambda": float(lam), "file": str(path),
                "_history": history.copy(), "_dhistory": dhistory.copy(),
                **scalar,
            })
            continue
        result, history, dhistory = simulate_observables(
            xi, beta=20.0, lam=float(lam), tau=10.0, t0=1.0, dt=0.01,
            transient=(20.0 if smoke else (300.0 if index == 0 else 50.0)),
            observation=(40.0 if smoke else 1000.0),
            initial_history=history,
            initial_derivative_history=dhistory,
            record_dt=0.1,
        )
        save_observables(path, result, history, dhistory, {
            "N": N, "P": P, "seed": seed, "family": family,
            "ensemble_parameter": parameter, "lambda": lam,
        })
        rows.append({
            "family": family, "parameter": parameter, "seed": seed,
            "lambda": float(lam), "file": str(path),
            "_history": history.copy(), "_dhistory": dhistory.copy(),
            **{key: value for key, value in result.items()
               if np.isscalar(value) or isinstance(value, str)},
        })
    if not smoke:
        moving = np.array([_is_moving(r["classification"]) for r in rows])
        transitions = np.flatnonzero(moving[:-1] != moving[1:])
        for index in transitions:
            high = max(rows[index]["lambda"], rows[index + 1]["lambda"])
            low = min(rows[index]["lambda"], rows[index + 1]["lambda"])
            refined_history = rows[index]["_history"].copy()
            refined_dhistory = rows[index]["_dhistory"].copy()
            for lam in np.arange(high - 0.005, low, -0.005):
                refined_path = (
                    RUNS / "N7" /
                    f"{family}_{str(parameter).replace('.', 'p')}_N{N}_P{P}"
                    f"_s{seed}_lam{lam:.3f}_refined.npz"
                )
                if refined_path.exists():
                    saved = np.load(refined_path)
                    refined_history = saved["final_history"]
                    refined_dhistory = saved["final_derivative_history"]
                    scalar = {
                        key: saved[key].item()
                        for key in saved.files
                        if saved[key].ndim == 0
                    }
                    result = scalar
                else:
                    result, refined_history, refined_dhistory = simulate_observables(
                        xi, beta=20.0, lam=float(lam), tau=10.0, t0=1.0, dt=0.01,
                        transient=100.0, observation=1500.0,
                        initial_history=refined_history,
                        initial_derivative_history=refined_dhistory,
                        record_dt=0.1,
                    )
                    save_observables(
                        refined_path, result, refined_history, refined_dhistory,
                        {
                            "N": N, "P": P, "seed": seed, "family": family,
                            "ensemble_parameter": parameter, "lambda": lam,
                            "refined": True,
                        },
                    )
                rows.append({
                    "family": family, "parameter": parameter, "seed": seed,
                    "lambda": float(lam), "file": str(refined_path),
                    "_history": refined_history.copy(),
                    "_dhistory": refined_dhistory.copy(),
                    **{key: value for key, value in result.items()
                       if np.isscalar(value) or isinstance(value, str)},
                })
    for row in rows:
        row.pop("_history", None)
        row.pop("_dhistory", None)
    return rows


def run_one(family: str, parameter: float, seed: int,
            N: int, P: int, smoke: bool, nproc: int) -> dict:
    if family == "markov":
        xi = markov_flip(N, P, parameter, seed)
        predicted = parameter
        lambda_min = 0.1
    else:
        xi = gp_circle(N, P, parameter, seed)
        predicted = gp_arcsine_prediction(P, parameter)
        lambda_min = 0.05
    correlations = adjacent_correlations(xi)
    tag = str(parameter).replace(".", "p")
    static_path = (
        RUNS / "N7" / f"static_{family}_{tag}_N{N}_P{P}_s{seed}.npz"
    )
    if static_path.exists() and static_path.with_suffix(".json").exists():
        static_summary = json.loads(
            static_path.with_suffix(".json").read_text(encoding="utf-8")
        )
    else:
        static_summary = run_static(
            StaticConfig(
                N=N, P=P, seed=seed, relaxed_identity=True,
                bracket_tol=(2e-3 if smoke else 2e-4),
            ),
            static_path,
            nproc=nproc,
            xi_override=xi,
            motifs=(list(range(min(P, 6))) if smoke else None),
        )
    static = np.load(static_path)
    thresholds = static["lambda_low"]
    resolved = static["resolved"]
    seam_rank = None
    if family == "markov" and len(thresholds) == P and resolved[-1]:
        seam_rank = int(np.sum(thresholds[resolved] <= thresholds[-1]))
    dynamics = dynamic_descent(
        xi, seed, family, parameter, N, P, lambda_min, smoke
    )
    moving_lambdas = [
        row["lambda"] for row in dynamics if _is_moving(row["classification"])
    ]
    return {
        "family": family, "parameter": parameter, "seed": seed,
        "measured_q_mean": float(np.mean(correlations[:-1])
                                 if family == "markov" else np.mean(correlations)),
        "measured_q_std": float(np.std(correlations)),
        "seam_q": float(correlations[-1]),
        "analytic_q": float(predicted),
        "static": static_summary,
        "seam_rank": seam_rank,
        "lowest_tested_moving_lambda": (
            float(min(moving_lambdas)) if moving_lambdas else None
        ),
        "dynamics": dynamics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["markov", "gp", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    parameters = []
    if args.family in {"markov", "both"}:
        parameters.extend(("markov", c) for c in (CS[:2] if args.smoke else CS))
    if args.family in {"gp", "both"}:
        parameters.extend(("gp", k) for k in (KAPPAS[-2:] if args.smoke else KAPPAS))
    rows = [
        run_one(family, parameter, seed, args.N, args.P, args.smoke, args.nproc)
        for seed in args.seeds for family, parameter in parameters
    ]
    summary_name = (
        "smoke.json" if args.smoke else
        ("summary.json" if args.seeds == [42, 43, 44, 45, 46]
         else "summary_seeds_" + "_".join(map(str, args.seeds)) + ".json")
    )
    write_json(
        RUNS / "N7" / summary_name,
        {
            "campaign": "N7", "rows": rows, "smoke": args.smoke,
            "environment": environment_manifest(),
        },
    )
    print(f"N7 completed {len(rows)} pattern ensembles")


if __name__ == "__main__":
    main()
