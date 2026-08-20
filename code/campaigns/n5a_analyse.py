"""Derived midpoints, jump brackets and boundary classifications for N5A."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import re
import argparse
from collections import defaultdict

import numpy as np

from v5_paths import FIGURES, RUNS, environment_manifest, write_json

COHERENCE_NAME = re.compile(
    r"coherence_N(?P<N>\d+)_P(?P<P>\d+)_s(?P<seed>\d+)_"
    r"tau(?P<tau>[\d.]+)_dt(?P<dt>[\d.]+)\.npz"
)
BOUNDARY_NAME = re.compile(
    r"boundary_N(?P<N>\d+)_P(?P<P>\d+)_s(?P<seed>\d+)_"
    r"tau(?P<tau>[\d.]+)_lam(?P<lam>[\d.]+)\.npz"
)


def base_dt(tau: float) -> float:
    return min(0.01, tau / 25.0)


def interpolate_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float:
    for i in range(len(x) - 1):
        if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (target-y[i])/(y[i+1]-y[i])*(x[i+1]-x[i]))
    return np.nan


def coherence_analysis(N_target: int, P_target: int) -> list[dict]:
    grouped = defaultdict(list)
    for path in (RUNS / "N5A").glob("coherence_*.npz"):
        match = COHERENCE_NAME.fullmatch(path.name)
        if not match:
            continue
        if int(match.group("N")) != N_target or int(match.group("P")) != P_target:
            continue
        values = {key: float(match.group(key)) for key in ("tau", "dt")}
        seed = int(match.group("seed"))
        if abs(values["dt"] - base_dt(values["tau"])) > 1e-10:
            continue
        data = np.load(path)
        grouped[seed].append({
            "tau": values["tau"],
            "coherence": float(data["coherence_mean"]),
            "participation": float(data["participation_mean"]),
            "front_width": float(data["front_width_mean"]),
            "binary": float(data["binary_retrieval_mean"]),
            "file": str(path),
        })
    results = []
    for seed, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["tau"])
        tau = np.array([row["tau"] for row in rows])
        coherence = np.array([row["coherence"] for row in rows])
        inverse_part = 1.0 / np.array([row["participation"] for row in rows])
        score = 0.5 * (coherence + inverse_part)
        short = tau <= 0.75
        long = tau >= 2.0
        target = 0.5 * (score[short].mean() + score[long].mean())
        midpoint = interpolate_crossing(tau, score, target)
        oriented = np.c_[
            coherence, inverse_part,
            -np.array([row["front_width"] for row in rows]),
            np.array([row["binary"] for row in rows]),
        ]
        standardized = (oriented - oriented.mean(axis=0)) / np.maximum(
            oriented.std(axis=0), 1e-12
        )
        jump = np.mean(np.abs(np.diff(standardized, axis=0)), axis=1)
        index = int(np.argmax(jump))
        results.append({
            "seed": seed, "midpoint": midpoint, "midpoint_target": float(target),
            "direct_jump_bracket": [float(tau[index]), float(tau[index+1])],
            "direct_jump_score": float(jump[index]),
            "rows": rows,
        })
    return results


def boundary_analysis(N_target: int, P_target: int) -> list[dict]:
    grouped = defaultdict(list)
    for path in (RUNS / "N5A").glob("boundary_*.npz"):
        match = BOUNDARY_NAME.fullmatch(path.name)
        if not match:
            continue
        if int(match.group("N")) != N_target or int(match.group("P")) != P_target:
            continue
        data = np.load(path)
        seed = int(match.group("seed"))
        tau = float(match.group("tau"))
        lam = float(match.group("lam"))
        lyap_path = path.with_name(path.stem + "_lyap.npz")
        maximal = None
        if lyap_path.exists():
            maximal = float(np.load(lyap_path)["lyapunov"][0])
        classification = str(data["classification"])
        if maximal is not None:
            if maximal > 2e-3 and "moving" in classification:
                classification = "chaotic_moving"
            elif maximal > 2e-3:
                classification = "chaotic_nonmoving"
            elif "sequence" in classification:
                classification = "periodic_sequence"
        grouped[(seed, tau)].append({
            "lambda": lam, "classification": classification,
            "maximal_lyapunov": maximal, "file": str(path),
        })
    output = []
    for (seed, tau), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["lambda"])
        moving = [
            row for row in rows
            if row["classification"] in {
                "periodic_sequence", "chaotic_moving",
                "periodic_sequence_candidate", "moving_requires_lyapunov",
            }
        ]
        nonmoving = [row for row in rows if row not in moving]
        output.append({
            "seed": seed, "tau": tau,
            "moving_min_lambda": (
                min(row["lambda"] for row in moving) if moving else None
            ),
            "nonmoving_max_lambda": (
                max(row["lambda"] for row in nonmoving) if nonmoving else None
            ),
            "lyapunov_resolved_points": sum(
                row["maximal_lyapunov"] is not None for row in rows
            ),
            "rows": rows,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    args = parser.parse_args()
    coherence = coherence_analysis(args.N, args.P)
    boundary = boundary_analysis(args.N, args.P)
    seeds = {row["seed"] for row in coherence}
    completion = (
        seeds == {42, 43, 44, 45, 46}
        and all(np.isfinite(row["midpoint"]) for row in coherence)
        and boundary
        and all(row["lyapunov_resolved_points"] >= 4 for row in boundary)
    )
    payload = {
        "campaign": "N5A", "coherence": coherence, "boundary": boundary,
        "completion_test_passed": bool(completion),
        "environment": environment_manifest(),
    }
    write_json(RUNS / "N5A" / "analysis.json", payload)
    if coherence:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
        seed_colors = {
            seed: color for seed, color in zip(
                sorted({row["seed"] for row in coherence}),
                plt.cm.viridis(np.linspace(0.12, 0.88, len(coherence))),
            )
        }
        for row in coherence:
            tau = np.array([item["tau"] for item in row["rows"]])
            score = 0.5 * (
                np.array([item["coherence"] for item in row["rows"]])
                + 1.0/np.array([item["participation"] for item in row["rows"]])
            )
            axes[0].plot(
                tau, score, "o-", ms=3, color=seed_colors[row["seed"]],
                label=f"seed {row['seed']}",
            )
            axes[0].axvspan(*row["direct_jump_bracket"], alpha=.05)
        axes[0].set(xlabel=r"$\tau/t_0$", ylabel="coherence score",
                    title="Seed-resolved short-delay crossover")
        boundary_by_seed = defaultdict(list)
        for row in boundary:
            boundary_by_seed[row["seed"]].append(row)
        static_references = {}
        for seed, seed_rows in sorted(boundary_by_seed.items()):
            valid_rows = [
                row for row in seed_rows
                if row["moving_min_lambda"] is not None
            ]
            if valid_rows:
                axes[1].plot(
                    [row["tau"] for row in valid_rows],
                    [row["moving_min_lambda"] for row in valid_rows],
                    "o-", ms=3, color=seed_colors.get(seed),
                    label=f"seed {seed}",
                )
            static_path = (
                RUNS / "N1" / f"static_N{args.N}_P{args.P}_seed{seed}.npz"
            )
            if static_path.exists():
                static = np.load(static_path)
                valid = np.flatnonzero(static["resolved"])
                if len(valid):
                    motif = int(valid[np.argmax(static["lambda_low"][valid])])
                    reference = 0.5 * (
                        float(static["lambda_low"][motif])
                        + float(static["lambda_high"][motif])
                    )
                    static_references[str(seed)] = reference
                    axes[1].axhline(
                        reference, color=seed_colors.get(seed), ls="--",
                        lw=.8, alpha=.55,
                    )
        axes[1].set(xlabel=r"$\tau/t_0$", ylabel="lowest moving $\\lambda$",
                    title="Dynamic boundary; dashed static extremes")
        for ax in axes:
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        if boundary_by_seed:
            axes[1].legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(FIGURES / "N5A_delay_boundary.png", dpi=600)
        fig.savefig(FIGURES / "N5A_delay_boundary.pdf")
    print(f"N5A analysis: {len(coherence)} coherence seeds, {len(boundary)} boundaries")


if __name__ == "__main__":
    main()
