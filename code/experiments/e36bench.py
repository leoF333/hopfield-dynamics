#!/usr/bin/env python3
"""
Guarded microbenchmark for E36.

Only ``--dry-run`` and tiny ``--pilot`` modes exist. The hard caps prevent this
script from starting a reference-size or production simulation.
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
import time

import numpy as np

from layered_chain import LayeredChain, make_patterns_numpy


CAPS = {"N": 128, "P": 8, "K": 4, "steps": 100}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--P", type=int, default=8)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--lambda", type=float, default=8.0, dest="lam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--steps", type=int, default=40)
    return p


def validate(args: argparse.Namespace) -> None:
    for key in ("N", "P", "K", "steps"):
        value = getattr(args, key)
        if value > CAPS[key]:
            raise SystemExit(
                f"REFUSED: {key}={value} exceeds pilot cap {CAPS[key]}"
            )
    if not (1 <= args.P <= args.N):
        raise SystemExit("REFUSED: require 1 <= P <= N")
    if args.K < 2 or args.steps < 1:
        raise SystemExit("REFUSED: require K>=2 and steps>=1")
    if not (0.0 <= args.lam <= 8.0):
        raise SystemExit("REFUSED: guarded E36 pilot requires 0<=lambda<=8")
    if args.dt <= 0:
        raise SystemExit("REFUSED: dt must be positive")


def main() -> None:
    args = parser().parse_args()
    validate(args)
    config = {
        "mode": "dry-run" if args.dry_run else "pilot",
        "N": args.N,
        "P": args.P,
        "K": args.K,
        "beta": args.beta,
        "lambda": args.lam,
        "seed": args.seed,
        "dt": args.dt,
        "steps": args.steps,
        "caps": CAPS,
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    xi, _ = make_patterns_numpy(args.N, args.P, args.seed)
    system = LayeredChain(
        xi, beta=args.beta, lam=args.lam, K=args.K
    )
    rng = np.random.default_rng(args.seed + 1)
    U0 = 0.1 * rng.standard_normal((args.K, args.N))

    # Warm BLAS and tanh dispatch once, then time RHS and a very short RK4 run.
    system.rhs_full(U0)
    start = time.perf_counter()
    for _ in range(args.steps):
        system.rhs_full(U0)
    rhs_seconds = time.perf_counter() - start

    t_final = args.steps * args.dt
    start = time.perf_counter()
    result = system.integrate(
        U0, t_final=t_final, dt=args.dt, record_every=args.steps
    )
    rk_seconds = time.perf_counter() - start
    final = result["final_state"]
    report = {
        **config,
        "rhs_calls_per_second": args.steps / rhs_seconds,
        "rk4_steps_per_second": args.steps / rk_seconds,
        "rhs_elapsed_seconds": rhs_seconds,
        "rk4_elapsed_seconds": rk_seconds,
        "max_abs_final": float(np.max(np.abs(final))),
        "fraction_beta_u_gt_10": float(
            np.mean(np.abs(args.beta * final) > 10.0)
        ),
        "finite": bool(np.all(np.isfinite(final))),
        "gram_condition": system.gram_condition,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
