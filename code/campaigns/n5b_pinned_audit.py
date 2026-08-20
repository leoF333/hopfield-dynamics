"""Resolution audit for positive N5B pinned-front complex margins.

The production pinned scan used M=64.  This script recomputes only cells whose
stored leading complex root has non-negative real part at M=48 and M=80.  It
does not classify a Hopf bifurcation; it only tests whether the candidate root
is stable under spectral-resolution changes before any refinement or nonlinear
probe is considered.
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
from pathlib import Path
from time import perf_counter

import numpy as np

from couplings import Couplings, make_patterns
from n5b_long_delay import conjugacy_aware_difference, roots_at
from robust_branch import woodbury_newton
from v5_paths import RUNS, environment_manifest, write_json


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def combined_summary() -> Path:
    matches = sorted((RUNS / "N5B").glob("summary_all_memory_*_pinned_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one combined N5B summary, found {matches}")
    return matches[0]


def candidate_rows() -> list[dict]:
    payload = json.loads(combined_summary().read_text(encoding="utf-8"))
    return [
        row for row in payload["pinned_controls"]
        if float(row["complex_margin"]) >= 0.0
    ]


def audit(row: dict, N: int, resolutions: tuple[int, ...]) -> dict:
    seed = int(row["seed"])
    tau = float(row["tau"])
    relative = float(row["relative_below_fold"])
    checkpoint = (
        RUNS / "N5B"
        / f"pinned_audit_N{N}_s{seed}_tau{tag(tau)}_rel{tag(relative)}.json"
    )
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))

    static_path = RUNS / "N1" / f"static_N{N}_P100_seed{seed}.npz"
    data = np.load(static_path)
    valid = np.flatnonzero(data["resolved"])
    motif = int(valid[np.argmax(data["lambda_low"][valid])])
    fold = 0.5 * (
        float(data["lambda_low"][motif]) + float(data["lambda_high"][motif])
    )
    lam = fold - relative
    state = data["state"][motif]
    xi, xis = make_patterns(N, 100, seed)
    coup = Couplings(xi, xis)
    solved, ok = woodbury_newton(coup, state, lam, 20.0, tol=1e-12)
    fixed_residual = float(
        np.linalg.norm(coup.field_F(solved, lam, 20.0)) / np.sqrt(N)
    )
    if not ok or fixed_residual >= 1e-11:
        raise RuntimeError(
            f"fixed-point solve failed for seed={seed}, tau={tau}, "
            f"relative={relative}: ok={ok}, residual={fixed_residual}"
        )

    stored = complex(
        float(row["complex_margin"]), float(row["complex_frequency"])
    )
    roots = {"64_stored": [stored.real, stored.imag]}
    residual_gates = {"64_stored": bool(row["residual_gate"])}
    wall_times = {}
    computed = {}
    for resolution in resolutions:
        start = perf_counter()
        spectrum = roots_at(coup, solved, lam, tau, resolution)
        wall_times[str(resolution)] = perf_counter() - start
        root = complex(spectrum["leading_complex"])
        computed[resolution] = root
        roots[str(resolution)] = [float(root.real), float(root.imag)]
        residual_gates[str(resolution)] = bool(
            spectrum["all_residuals_pass"]
        )

    differences = {
        f"{resolution}_vs_64": conjugacy_aware_difference(
            computed[resolution], stored
        )
        for resolution in resolutions
    }
    if len(resolutions) > 1:
        for left, right in zip(resolutions[:-1], resolutions[1:]):
            differences[f"{left}_vs_{right}"] = (
                conjugacy_aware_difference(computed[left], computed[right])
            )
    signs = [stored.real] + [computed[m].real for m in resolutions]
    result = {
        "seed": seed,
        "motif": motif,
        "tau": tau,
        "relative_below_fold": relative,
        "fold": fold,
        "lambda": lam,
        "fixed_residual": fixed_residual,
        "roots": roots,
        "residual_gates": residual_gates,
        "conjugacy_aware_differences": differences,
        "positive_at_all_resolutions": bool(all(value > 0 for value in signs)),
        "maximum_resolution_difference": float(max(differences.values())),
        "wall_times_seconds": wall_times,
    }
    write_json(checkpoint, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[48, 80])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    candidates = candidate_rows()
    if args.limit is not None:
        candidates = candidates[: args.limit]
    rows = [
        audit(row, args.N, tuple(args.resolutions))
        for row in candidates
    ]
    summary = {
        "campaign": "N5B_pinned_resolution_audit",
        "candidate_count": len(candidate_rows()),
        "audited_count": len(rows),
        "all_positive_at_all_resolutions": bool(
            rows and all(row["positive_at_all_resolutions"] for row in rows)
        ),
        "all_residual_gates_pass": bool(
            rows and all(
                all(row["residual_gates"].values()) for row in rows
            )
        ),
        "maximum_resolution_difference": (
            float(max(row["maximum_resolution_difference"] for row in rows))
            if rows else None
        ),
        "rows": rows,
        "environment": environment_manifest(),
    }
    write_json(RUNS / "N5B" / "pinned_resolution_audit.json", summary)
    print(
        f"N5B pinned audit completed {len(rows)}/{len(candidate_rows())} "
        "positive-margin cells"
    )


if __name__ == "__main__":
    main()
