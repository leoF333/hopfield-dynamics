"""Refine resolution-stable N5B pinned-front complex-root crossings.

For every (seed, tau) slice whose three stored lambda offsets bracket a change
of sign of the leading complex margin, bisect in lambda, track the same complex
root family, and verify the refined root at M=48, 64 and 80.  Nonlinear probes
are deliberately left to a separate gated step.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from couplings import Couplings, make_patterns
from n5b_long_delay import (
    conjugacy_aware_difference,
    roots_at,
    tracked_complex,
)
from robust_branch import woodbury_newton
from v5_paths import RUNS, environment_manifest, write_json


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def combined_summary() -> Path:
    matches = sorted((RUNS / "N5B").glob("summary_all_memory_*_pinned_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one combined N5B summary, found {matches}")
    return matches[0]


def crossing_brackets() -> list[dict]:
    payload = json.loads(combined_summary().read_text(encoding="utf-8"))
    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in payload["pinned_controls"]:
        grouped.setdefault((int(row["seed"]), float(row["tau"])), []).append(row)
    brackets = []
    for (seed, tau), rows in sorted(grouped.items()):
        # lambda increases as relative_below_fold decreases.
        ordered = sorted(rows, key=lambda row: -float(row["relative_below_fold"]))
        for left, right in zip(ordered[:-1], ordered[1:]):
            left_margin = float(left["complex_margin"])
            right_margin = float(right["complex_margin"])
            if left_margin <= 0.0 < right_margin:
                brackets.append({
                    "seed": seed,
                    "tau": tau,
                    "left": left,
                    "right": right,
                })
                break
    return brackets


def refine(bracket: dict, N: int, tolerance: float) -> dict:
    seed = int(bracket["seed"])
    tau = float(bracket["tau"])
    left = bracket["left"]
    right = bracket["right"]
    checkpoint = (
        RUNS / "N5B"
        / f"pinned_crossing_N{N}_s{seed}_tau{tag(tau)}.json"
    )
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))

    data = np.load(RUNS / "N1" / f"static_N{N}_P100_seed{seed}.npz")
    valid = np.flatnonzero(data["resolved"])
    motif = int(valid[np.argmax(data["lambda_low"][valid])])
    fold = 0.5 * (
        float(data["lambda_low"][motif]) + float(data["lambda_high"][motif])
    )
    base_state = data["state"][motif]
    xi, xis = make_patterns(N, 100, seed)
    coup = Couplings(xi, xis)

    lo = fold - float(left["relative_below_fold"])
    hi = fold - float(right["relative_below_fold"])
    root_lo = complex(
        float(left["complex_margin"]), float(left["complex_frequency"])
    )
    root_hi = complex(
        float(right["complex_margin"]), float(right["complex_frequency"])
    )
    state_lo, ok_lo = woodbury_newton(coup, base_state, lo, 20.0, tol=1e-12)
    state_hi, ok_hi = woodbury_newton(coup, base_state, hi, 20.0, tol=1e-12)
    if not (ok_lo and ok_hi and root_lo.real <= 0.0 < root_hi.real):
        raise RuntimeError(
            f"invalid crossing bracket seed={seed}, tau={tau}: "
            f"ok=({ok_lo},{ok_hi}), roots=({root_lo},{root_hi})"
        )

    iterations = 0
    while hi - lo > tolerance and iterations < 40:
        mid = 0.5 * (lo + hi)
        state_mid, ok_mid = woodbury_newton(
            coup, 0.5 * (state_lo + state_hi), mid, 20.0, tol=1e-12
        )
        if not ok_mid:
            raise RuntimeError(
                f"midpoint solve failed seed={seed}, tau={tau}, lambda={mid}"
            )
        reference = 0.5 * (root_lo + root_hi)
        root_mid, residual_mid = tracked_complex(
            roots_at(coup, state_mid, mid, tau, 64), reference
        )
        if not np.isfinite(root_mid.real) or residual_mid >= 1e-10:
            raise RuntimeError(
                f"root failed seed={seed}, tau={tau}, lambda={mid}, "
                f"root={root_mid}, residual={residual_mid}"
            )
        if root_mid.real <= 0.0:
            lo, state_lo, root_lo = mid, state_mid, root_mid
        else:
            hi, state_hi, root_hi = mid, state_mid, root_mid
        iterations += 1

    lam = 0.5 * (lo + hi)
    state_cross, ok = woodbury_newton(
        coup, 0.5 * (state_lo + state_hi), lam, 20.0, tol=1e-12
    )
    if not ok:
        raise RuntimeError(
            f"crossing solve failed seed={seed}, tau={tau}, lambda={lam}"
        )
    reference = 0.5 * (root_lo + root_hi)
    spectra = {m: roots_at(coup, state_cross, lam, tau, m) for m in (48, 64, 80)}
    roots = {}
    residuals = {}
    for resolution, spectrum in spectra.items():
        root, residual = tracked_complex(spectrum, reference)
        roots[resolution] = root
        residuals[resolution] = residual

    differences = [
        conjugacy_aware_difference(roots[48], roots[64]),
        conjugacy_aware_difference(roots[48], roots[80]),
        conjugacy_aware_difference(roots[64], roots[80]),
    ]
    fixed_residual = float(
        np.linalg.norm(coup.field_F(state_cross, lam, 20.0)) / np.sqrt(N)
    )
    leading_real = complex(spectra[64]["leading_real"])
    result = {
        "seed": seed,
        "motif": motif,
        "tau": tau,
        "fold": fold,
        "lambda_bracket": [lo, hi],
        "lambda_crossing": lam,
        "lambda_width": hi - lo,
        "relative_crossing": fold - lam,
        "root_M48": [float(roots[48].real), float(roots[48].imag)],
        "root_M64": [float(roots[64].real), float(roots[64].imag)],
        "root_M80": [float(roots[80].real), float(roots[80].imag)],
        "leading_real_M64": [float(leading_real.real), float(leading_real.imag)],
        "selected_residuals": {
            str(key): float(value) for key, value in residuals.items()
        },
        "all_spectral_residual_gates_pass": bool(
            all(spectrum["all_residuals_pass"] for spectrum in spectra.values())
        ),
        "maximum_resolution_difference": float(max(differences)),
        "fixed_residual": fixed_residual,
        "transversality_dreal_dlambda": float(
            (root_hi.real - root_lo.real) / (hi - lo)
        ),
        "iterations": iterations,
    }
    write_json(checkpoint, result)
    return result


def worker(arguments: tuple[dict, int, float]) -> dict:
    bracket, N, tolerance = arguments
    return refine(bracket, N, tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--indices", type=int, nargs="+")
    args = parser.parse_args()
    brackets = crossing_brackets()
    if args.indices is not None:
        selected = [brackets[index] for index in args.indices]
    else:
        selected = brackets[: args.limit] if args.limit is not None else brackets
    jobs = [(bracket, args.N, args.tolerance) for bracket in selected]
    if args.workers == 1:
        rows = [worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as pool:
            rows = list(pool.map(worker, jobs))
    summary = {
        "campaign": "N5B_pinned_crossing_refinement",
        "candidate_crossing_count": len(brackets),
        "refined_count": len(rows),
        "all_resolution_gates_pass": bool(
            rows and all(
                row["all_spectral_residual_gates_pass"]
                and row["maximum_resolution_difference"] < 1e-6
                for row in rows
            )
        ),
        "all_transverse": bool(
            rows and all(
                abs(row["transversality_dreal_dlambda"]) > 1e-6
                for row in rows
            )
        ),
        "all_real_modes_stable": bool(
            rows and all(row["leading_real_M64"][0] < 0.0 for row in rows)
        ),
        "rows": rows,
        "environment": environment_manifest(),
    }
    write_json(RUNS / "N5B" / "pinned_crossing_refinement.json", summary)
    print(
        f"N5B refined {len(rows)}/{len(brackets)} pinned-front crossings"
    )


if __name__ == "__main__":
    main()
