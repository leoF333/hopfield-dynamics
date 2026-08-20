"""N8 certification of high-load moving states by float64 Lyapunov analysis."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
from pathlib import Path

import numpy as np

from couplings import make_patterns
from lyapunov_stream import LyapunovConfig, run as run_lyapunov
from observables import save_observables, simulate_observables
from v5_paths import RUNS, environment_manifest, write_json

ALPHAS = [0.05, 0.09, 0.13, 0.15]
LAMBDAS = [0.35, 0.40, 0.50, 0.70, 0.90]


def moving_observed(
    observables: dict, minimum_transitions: int,
) -> bool:
    return (
        int(observables.get("relay_count", 0)) >= minimum_transitions
        and observables["forward_fraction"] > 0.99
    )


def final_label(
    observables: dict, maximal: float | None,
    minimum_transitions: int, tolerance: float = 2e-3,
) -> str:
    if observables["fixed_residual"] < 1e-10:
        return "fixed_point"
    moving = moving_observed(observables, minimum_transitions)
    if maximal is None:
        return "moving_unresolved" if moving else "other_unresolved"
    if moving and maximal > tolerance:
        return "chaotic_sequence"
    if moving and maximal <= tolerance and observables["periodic_closure"] < 1e-4:
        return "periodic_sequence"
    if not moving and maximal > tolerance:
        return "nonmoving_chaos"
    return "other"


def run_cell(N: int, alpha: float, lam: float, seed: int,
             smoke: bool, skip_lyapunov: bool) -> dict:
    P = round(alpha * N)
    xi, _ = make_patterns(N, P, seed)
    tag = f"a{alpha:.2f}_lam{lam:.2f}_s{seed}".replace(".", "p")
    trajectory_path = RUNS / "N8" / f"trajectory_N{N}_{tag}.npz"
    if trajectory_path.exists():
        saved = np.load(trajectory_path)
        history = saved["final_history"]
        dhistory = saved["final_derivative_history"]
        observables = {
            key: saved[key].item()
            for key in saved.files
            if saved[key].ndim == 0
        }
        observables["relay_count"] = int(
            saved["relay_count"].item()
            if "relay_count" in saved.files
            else len(saved["relay_times"])
        )
    else:
        observables, history, dhistory = simulate_observables(
            xi, beta=20.0, lam=lam, tau=10.0, t0=1.0,
            dt=(0.02 if smoke else 0.01),
            transient=(20.0 if smoke else 400.0),
            observation=(40.0 if smoke else 1600.0),
            record_dt=(0.1 if not smoke else 0.2),
        )
        save_observables(trajectory_path, observables, history, dhistory, {
            "N": N, "P": P, "alpha": alpha, "lambda": lam, "seed": seed,
        })
    maximal = None
    spectrum = None
    lyap_file = None
    minimum_transitions = 5 if smoke else 50
    moving_candidate = moving_observed(observables, minimum_transitions)
    if moving_candidate and not skip_lyapunov:
        representative = seed == 42 and abs(lam - 0.50) < 1e-12
        k = 4 if representative else 1
        lyap_path = RUNS / "N8" / f"lyapunov_N{N}_{tag}_k{k}.npz"
        lyap = run_lyapunov(
            LyapunovConfig(
                N=N, P=P, seed=seed, lam=lam, dt=(0.02 if smoke else 0.01),
                transient=(10.0 if smoke else 400.0),
                accumulation=(10.0 if smoke else 1000.0),
                k=k, qr_time=(0.4 if smoke else 2.0),
            ),
            lyap_path, initial_history=history,
        )
        spectrum = lyap["lyapunov"]
        maximal = spectrum[0]
        lyap_file = str(lyap_path)
    return {
        "N": N, "P": P, "alpha": alpha, "lambda": lam, "seed": seed,
        "trajectory_file": str(trajectory_path),
        "lyapunov_file": lyap_file,
        "lyapunov": spectrum,
        "classification": final_label(
            observables, maximal, minimum_transitions
        ),
        **{key: value for key, value in observables.items()
           if np.isscalar(value) or isinstance(value, str)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-lyapunov", action="store_true")
    args = parser.parse_args()
    alphas = [0.05, 0.13] if args.smoke else ALPHAS
    lambdas = [0.4, 0.9] if args.smoke else LAMBDAS
    rows = [
        run_cell(
            args.N, alpha, lam, seed, args.smoke, args.skip_lyapunov
        )
        for alpha in alphas for lam in lambdas for seed in args.seeds
    ]
    summary_name = (
        "smoke.json" if args.smoke else
        ("summary.json" if args.seeds == [42, 43, 44, 45, 46]
         else "summary_seeds_" + "_".join(map(str, args.seeds)) + ".json")
    )
    write_json(
        RUNS / "N8" / summary_name,
        {
            "campaign": "N8", "rows": rows, "smoke": args.smoke,
            "environment": environment_manifest(),
        },
    )
    print(f"N8 completed {len(rows)} cells")


if __name__ == "__main__":
    main()
