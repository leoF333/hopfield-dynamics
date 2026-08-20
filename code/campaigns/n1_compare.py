"""N1 static-extreme versus independently measured dynamic onset."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import csv
import math
from pathlib import Path

import numpy as np

from v5_paths import FIGURES, REPORTS, RUNS, environment_manifest, write_json


APPROVED_PRE_STREAMING_DYNAMICS_SHA = (
    "bfe196db07865c23a0870341833d6681dd38a8f7cdef0d328adf22bba7df6b2d"
)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [np.nan, np.nan]
    p = successes / total
    denominator = 1.0 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center - half, center + half]


def load_seed(
    seed: int, N: int, P: int, expected_dynamics_sha: str
) -> dict | None:
    static_path = RUNS / "N1" / f"static_N{N}_P{P}_seed{seed}.npz"
    dynamic_path = RUNS / "N1" / f"dynamic_N{N}_P{P}_seed{seed}.npz"
    dynamic_json = dynamic_path.with_suffix(".json")
    if not static_path.exists() or not dynamic_path.exists() or not dynamic_json.exists():
        return None
    import json
    static = np.load(static_path)
    dynamic = np.load(dynamic_path)
    summary = json.loads(dynamic_json.read_text(encoding="utf-8"))
    if (
        summary.get("independent_history_control_passed") is not True
        or "arrest_primary_motif" not in summary
        or summary.get("environment", {}).get("local_source_sha256", {}).get(
            "src/dynamics.py"
        ) not in {
            expected_dynamics_sha,
            APPROVED_PRE_STREAMING_DYNAMICS_SHA,
        }
    ):
        return None
    valid = np.flatnonzero(static["resolved"])
    if len(valid) < 98:
        return None
    order = valid[np.argsort(static["lambda_low"][valid])[::-1]]
    first, second = int(order[0]), int(order[1])
    stat_low = float(static["lambda_low"][first])
    stat_high = float(static["lambda_high"][first])
    dyn_low = float(dynamic["dynamic_low"])
    dyn_high = float(dynamic["dynamic_high"])
    dynamic_motif = int(summary["last_cycle_slowest_bond"])
    arrest_top_two = [
        int(value) for value in summary.get("arrest_top_two_motifs", [])
    ]
    overlap = max(stat_low, dyn_low) <= min(stat_high, dyn_high)
    combined_resolution = (stat_high-stat_low) + (dyn_high-dyn_low)
    second_low = float(static["lambda_low"][second])
    second_high = float(static["lambda_high"][second])
    gap = 0.5 * (stat_low + stat_high - second_low - second_high)
    combined_resolution += second_high - second_low
    return {
        "seed": seed, "static_motif": first, "dynamic_motif": dynamic_motif,
        "static_low": stat_low, "static_high": stat_high,
        "static_mid": 0.5*(stat_low+stat_high),
        "dynamic_low": dyn_low, "dynamic_high": dyn_high,
        "dynamic_mid": 0.5*(dyn_low+dyn_high),
        "difference": 0.5*(dyn_low+dyn_high-stat_low-stat_high),
        "identity_match": first == dynamic_motif,
        "arrest_contains_static_extreme": first in arrest_top_two,
        "arrest_top_two_motifs": arrest_top_two,
        "bracket_overlap": overlap,
        "first_second_gap": gap,
        "gap_below_resolution": gap <= combined_resolution,
        "static_extreme_brackets_overlap": stat_low <= second_high,
        "second_low": second_low, "second_high": second_high,
        "second_motif": second,
        "history_control_passed": bool(
            summary.get("independent_history_control_passed", False)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    args = parser.parse_args()
    environment = environment_manifest()
    expected_dynamics_sha = environment["local_source_sha256"]["src/dynamics.py"]
    rows = [
        row for seed in args.seeds
        if (
            row := load_seed(
                seed, args.N, args.P, expected_dynamics_sha
            )
        ) is not None
    ]
    matches = sum(row["identity_match"] for row in rows)
    overlaps = sum(row["bracket_overlap"] for row in rows)
    discrepancies = np.array([abs(row["difference"]) for row in rows])
    completion = (
        len(rows) >= 15 and overlaps / max(len(rows), 1) >= 0.9
        and all(
            (
                row["identity_match"]
                and row["arrest_contains_static_extreme"]
            )
            or row["gap_below_resolution"]
            for row in rows
        )
        and all(row["history_control_passed"] for row in rows)
    )
    payload = {
        "campaign": "N1", "admissible_seeds": len(rows),
        "identity_matches": matches,
        "identity_wilson_95": wilson(matches, len(rows)),
        "bracket_overlaps": overlaps,
        "bracket_overlap_fraction": overlaps / len(rows) if rows else None,
        "mean_absolute_discrepancy": (
            float(np.mean(discrepancies)) if len(discrepancies) else None
        ),
        "maximum_absolute_discrepancy": (
            float(np.max(discrepancies)) if len(discrepancies) else None
        ),
        "completion_test_passed": completion,
        "rows": rows, "environment": environment,
    }
    write_json(RUNS / "N1" / "comparison.json", payload)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "N1_static_dynamic.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))
        x = np.array([row["static_mid"] for row in rows])
        y = np.array([row["dynamic_mid"] for row in rows])
        xerr = np.array([
            [row["static_mid"]-row["static_low"] for row in rows],
            [row["static_high"]-row["static_mid"] for row in rows],
        ])
        yerr = np.array([
            [row["dynamic_mid"]-row["dynamic_low"] for row in rows],
            [row["dynamic_high"]-row["dynamic_mid"] for row in rows],
        ])
        axes[0].errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=4, capsize=2)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        axes[0].plot(limits, limits, "k--")
        axes[0].set(xlabel=r"$\max_\mu\lambda_c^{\rm stat}$",
                    ylabel=r"$\lambda_{\rm dyn}$", title="Threshold equality")
        axes[1].axhline(0, color="k", lw=1)
        axes[1].scatter(
            [r["seed"] for r in rows], [r["difference"] for r in rows],
            c=["#2463a6" if r["bracket_overlap"] else "#bd3f32" for r in rows],
        )
        axes[1].set(xlabel="disorder seed",
                    ylabel=r"$\lambda_{\rm dyn}-\lambda_{\max}^{\rm stat}$",
                    title="Seed-resolved discrepancy")
        axes[2].scatter(
            [r["first_second_gap"] for r in rows],
            [abs(r["difference"]) for r in rows],
            c=["#2f855a" if r["identity_match"] else "#bd3f32" for r in rows],
        )
        axes[2].set(xlabel="first–second static gap",
                    ylabel="absolute discrepancy", title="Extreme-gap sensitivity")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "N1_static_dynamic.png", dpi=600)
        fig.savefig(FIGURES / "N1_static_dynamic.pdf")
    print(payload)


if __name__ == "__main__":
    main()
