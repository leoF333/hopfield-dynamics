"""N5A short-delay coherence and seed-resolved cycle/chaos boundary."""
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

TAUS_COHERENCE = [
    0.25, 0.5, 0.75, 1.0, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
    1.40, 1.45, 1.50, 1.60, 1.80, 2.0, 3.0, 5.0, 10.0,
]
TAUS_BOUNDARY = [0.5, 1.0, 1.2, 1.3, 1.4, 2.0, 3.0, 5.0, 10.0]
LAMBDA_BOUNDARY = np.arange(0.36, 0.30 - 1e-12, -0.002)


def _dt(tau: float) -> float:
    return min(0.01, tau / 25.0)


def coherence_seed(seed: int, N: int, P: int, smoke: bool = False) -> list[dict]:
    xi, _ = make_patterns(N, P, seed)
    taus = [0.5, 1.3, 3.0] if smoke else TAUS_COHERENCE
    rows = []
    for tau in taus:
        dts = [_dt(tau)]
        if tau in (1.2, 1.3, 1.4):
            dts.append(_dt(tau) / 2.0)
        for dt in dts:
            path = (
                RUNS / "N5A" /
                f"coherence_N{N}_P{P}_s{seed}_tau{tau:g}_dt{dt:g}.npz"
            )
            if path.exists():
                saved = np.load(path)
                scalar = {
                    key: saved[key].item()
                    for key in saved.files
                    if saved[key].ndim == 0
                }
                rows.append({
                    "seed": seed, "tau": tau, "dt": dt, "file": str(path),
                    **scalar,
                })
                continue
            transient = 20.0 if smoke else max(200.0, 10.0 * (tau + 1.0))
            observation = 40.0 if smoke else max(600.0, 60.0 * (tau + 1.0))
            result, history, dhistory = simulate_observables(
                xi, beta=20.0, lam=0.9, tau=tau, t0=1.0, dt=dt,
                transient=transient, observation=observation,
                record_dt=max(dt, 0.05),
            )
            save_observables(path, result, history, dhistory, {
                "N": N, "P": P, "seed": seed, "tau": tau,
                "lambda": 0.9, "dt": dt,
            })
            rows.append({
                "seed": seed, "tau": tau, "dt": dt, "file": str(path),
                **{key: value for key, value in result.items()
                   if np.isscalar(value) or isinstance(value, str)},
            })
    return rows


def boundary_seed(seed: int, N: int, P: int, smoke: bool = False,
                  with_lyapunov: bool = True) -> list[dict]:
    xi, _ = make_patterns(N, P, seed)
    taus = [1.3] if smoke else TAUS_BOUNDARY
    lambdas = [0.36, 0.33, 0.30] if smoke else LAMBDA_BOUNDARY
    rows = []
    for tau in taus:
        history = None
        dhistory = None
        tau_rows = []
        for index, lam in enumerate(lambdas):
            dt = _dt(tau)
            path = (
                RUNS / "N5A" /
                f"boundary_N{N}_P{P}_s{seed}_tau{tau:g}_lam{lam:.3f}.npz"
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
                tau_rows.append({
                    "seed": seed, "tau": tau, "lambda": float(lam),
                    "file": str(path), "history": history.copy(), **scalar,
                })
                continue
            result, history, dhistory = simulate_observables(
                xi, beta=20.0, lam=float(lam), tau=tau, t0=1.0, dt=dt,
                transient=(20.0 if smoke else (200.0 if index == 0 else 50.0)),
                observation=(40.0 if smoke else max(800.0, 80.0 * (tau + 1.0))),
                initial_history=history,
                initial_derivative_history=dhistory,
                record_dt=max(dt, 0.05),
            )
            save_observables(path, result, history, dhistory, {
                "N": N, "P": P, "seed": seed, "tau": tau,
                "lambda": lam, "dt": dt,
            })
            row = {
                "seed": seed, "tau": tau, "lambda": float(lam),
                "file": str(path), "history": history.copy(),
                **{key: value for key, value in result.items()
                   if np.isscalar(value) or isinstance(value, str)},
            }
            tau_rows.append(row)
        moving = [
            i for i, row in enumerate(tau_rows)
            if row["classification"] in {
                "periodic_sequence_candidate", "moving_requires_lyapunov"
            }
        ]
        if with_lyapunov and not smoke:
            # Lambda is scanned downwards.  The boundary is therefore the last
            # moving point, not the first moving point at the top of the scan.
            # Evaluate exactly the last two moving cells and the first two
            # subsequent cells, as required by N5A.
            if moving:
                boundary_index = max(moving)
                selected = sorted(set(
                    moving[-2:]
                    + list(range(
                        boundary_index + 1,
                        min(len(tau_rows), boundary_index + 3),
                    ))
                ))
            else:
                # If the whole bracket is post-boundary, retain the first two
                # cells as a diagnostic that the initial bracket must be moved.
                selected = list(range(min(2, len(tau_rows))))
            for i in selected:
                row = tau_rows[i]
                lyap_path = Path(row["file"]).with_name(
                    Path(row["file"]).stem + "_lyap.npz"
                )
                lyap = run_lyapunov(
                    LyapunovConfig(
                        N=N, P=P, seed=seed, tau=tau, lam=row["lambda"],
                        dt=_dt(tau), transient=200.0, accumulation=1000.0, k=1,
                    ),
                    lyap_path, initial_history=row["history"],
                )
                row["maximal_lyapunov"] = lyap["lyapunov"][0]
                row["lyapunov_file"] = str(lyap_path)
        for row in tau_rows:
            row.pop("history", None)
        rows.extend(tau_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["coherence", "boundary", "all"],
                        default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-lyapunov", action="store_true")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        if args.mode in {"coherence", "all"}:
            rows.extend(coherence_seed(seed, args.N, args.P, args.smoke))
        if args.mode in {"boundary", "all"}:
            rows.extend(boundary_seed(
                seed, args.N, args.P, args.smoke, not args.skip_lyapunov
            ))
    payload = {
        "campaign": "N5A", "smoke": args.smoke,
        "rows": rows, "environment": environment_manifest(),
    }
    if args.smoke:
        summary_name = "smoke.json"
    elif args.seeds == [42, 43, 44, 45, 46]:
        summary_name = "summary.json"
    else:
        summary_name = "summary_seeds_" + "_".join(map(str, args.seeds)) + ".json"
    write_json(RUNS / "N5A" / summary_name, payload)
    print(f"N5A completed {len(rows)} cells")


if __name__ == "__main__":
    main()
