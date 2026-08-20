"""N4 publication-style Figures 1--4 from archived and newly derived tables."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v5_paths import FIGURES, MANIFESTS, REFERENCE_ROOT, REPORTS, RUNS, write_json

DATA = REFERENCE_ROOT / "results"
UNIFICATION = DATA / "3_unification_seuils_scaling" / "data"
CYCLE = DATA / "2_cycle_rappel_snic" / "data"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "figure.dpi": 130,
    })
    return plt


def label(ax, letter: str) -> None:
    ax.text(
        0.012, 0.985, letter, transform=ax.transAxes,
        fontweight="bold", fontsize=11, va="top", ha="left", zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
              "pad": 0.8},
    )


def figure1(plt) -> dict:
    cycle_path = CYCLE / "cycle_lam0.9_tau10.0_N2000.npz"
    cycle = np.load(cycle_path)
    threshold_path = RUNS / "N1" / "static_N2000_P100_seed42.npz"
    threshold = np.load(threshold_path)
    lam = threshold["lambda_low"][threshold["resolved"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1))

    ax = axes[0, 0]
    angles = 2*np.pi*np.arange(8)/8
    x, y = np.cos(angles), np.sin(angles)
    ax.scatter(x, y, s=190, c=np.arange(8), cmap="viridis", edgecolor="k")
    for i in range(8):
        j = (i + 1) % 8
        ax.annotate("", xy=(x[j], y[j]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color="#bb3e3e", lw=1.5))
    ax.text(0, 0.18, r"$J$: reciprocal memory", ha="center")
    ax.text(0, -0.10, r"$K$: delayed sequence", ha="center", color="#a02c2c")
    ax.text(0, -0.38, r"exact state: $u=Xa$", ha="center")
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal",
           title="Non-reciprocal time-delayed Hopfield network")
    ax.axis("off")
    label(ax, "a")

    ax = axes[0, 1]
    A = cycle["a_grid"]
    dt = float(cycle["dt"])
    L = int(cycle["L"])
    start = L
    stop = min(len(A), start + int(8*float(cycle["T1_mean"])/dt))
    image = A[start:stop].T
    ax.imshow(image, origin="lower", aspect="auto", interpolation="nearest",
              extent=[0, (stop-start)*dt, 0, A.shape[1]], cmap="viridis")
    ax.set(xlabel="time", ylabel="pattern index", title="Sequential-recall wave")
    label(ax, "b")

    ax = axes[1, 0]
    ax.hist(lam, bins=16, density=True, color="#476f9f", alpha=0.8)
    ax.axvline(np.min(lam), color="#2f855a", ls="--", label="first fold")
    ax.axvline(np.max(lam), color="#bd3f32", ls="--", label="last fold")
    ax.set(xlabel=r"sample threshold $\lambda_c(\mu)$", ylabel="density",
           title="Measured ensemble of quenched bifurcations")
    ax.legend()
    label(ax, "c")

    ax = axes[1, 1]
    lo, hi = float(np.min(lam)), float(np.max(lam))
    lower_axis = max(0.0, lo - 0.02)
    ax.axvspan(lower_axis, lo, color="#b9d8be")
    ax.axvspan(lo, hi, color="#edd9a3")
    ax.axvspan(hi, 0.40, color="#b8cee8")
    ax.text((lower_axis+lo)/2, 0.5, "all 100\nbranches",
            ha="center", va="center", fontsize=7.5)
    ax.text((lo+hi)/2, 0.5, "progressive\nfolds",
            ha="center", va="center", fontsize=7.5)
    ax.text((hi+0.40)/2, 0.5, "ordered\ncycle",
            ha="center", va="center", fontsize=7.5)
    ax.set(xlim=(lower_axis, 0.40), ylim=(0, 1), yticks=[],
           xlabel=r"$\lambda$", title="Reference realization: demonstrated regimes")
    label(ax, "d")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"N4_Figure1.{suffix}", dpi=600)
    derived = FIGURES / "N4_Figure1_data.npz"
    np.savez_compressed(derived, threshold=lam, kymograph=image, dt=dt)
    return {
        "output": str(FIGURES / "N4_Figure1.pdf"),
        "sources": [str(cycle_path), str(threshold_path)],
        "derived": str(derived),
        "panels": {
            "a": {"kind": "schematic", "measured": False},
            "b": {"file": str(cycle_path), "keys": ["a_grid", "dt", "L", "T1_mean"],
                  "parameters": {"N": 2000, "P": 100, "seed": 42, "lambda": 0.9}},
            "c": {"file": str(threshold_path),
                  "keys": ["lambda_low", "resolved"],
                  "parameters": {"N": 2000, "P": 100, "seed": 42}},
            "d": {"file": str(threshold_path),
                  "keys": ["lambda_low", "lambda_high", "resolved"],
                  "note": "labels state only regimes directly resolved for seed 42"},
        },
    }


def figure2(plt) -> dict:
    curves_path = UNIFICATION / "E24_curves_N2000_s42.npz"
    death_path = UNIFICATION / "E25_deathshape.npz"
    data = np.load(curves_path)
    death = np.load(death_path)
    lam = data["lam_c"]
    order = np.argsort(lam)
    selected = [int(order[0]), int(order[len(order)//2]), int(order[-1])]
    colors = ["#2f855a", "#476f9f", "#bd3f32"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for mu, color, name in zip(selected, colors, ("minimum", "median", "maximum")):
        curve = data[f"curve_{mu}"]
        axes[0, 0].plot(curve[:, 0], curve[:, 1], "o-", ms=2.5,
                        color=color, label=f"{name}: $\\mu={mu}$")
        axes[0, 1].plot(curve[:, 0], curve[:, 3], "o-", ms=2.5, color=color)
    for ax in axes[0]:
        ax.axvspan(0.3262, 0.3280, color="#bd3f32", alpha=0.10,
                   label="reference twin/threshold resolution" if ax is axes[0, 0] else None)
    axes[0, 0].set(xlabel=r"$\lambda$", ylabel=r"$a_\mu$",
                   title="Memory-connected terminal branches")
    axes[0, 0].legend()
    axes[0, 1].axhline(0, color="k", lw=1)
    axes[0, 1].set(xlabel=r"$\lambda$", ylabel="leading real eigenvalue",
                   title="Real-mode softening at each fold")
    # Compare the *shape* of all terminal branches on their individual
    # distance-to-fold scale.  Both axes are normalized from the first saved
    # point (0) to the last resolved point (1), so this panel does not imply a
    # universal amplitude or threshold.  The raw curves remain visible behind
    # the pointwise median and interquartile band.
    collapse_grid = np.linspace(0.0, 1.0, 101)
    collapse = []
    for mu in range(len(lam)):
        curve = data[f"curve_{mu}"]
        x = (curve[:, 0] - curve[0, 0]) / (curve[-1, 0] - curve[0, 0])
        denominator = curve[0, 1] - curve[-1, 1]
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            continue
        y = (curve[0, 1] - curve[:, 1]) / denominator
        axes[1, 0].plot(x, y, color="#7f9bbb", alpha=.10, lw=.65)
        collapse.append(np.interp(collapse_grid, x, y))
    collapse = np.asarray(collapse)
    q25, median, q75 = np.nanpercentile(collapse, [25, 50, 75], axis=0)
    axes[1, 0].fill_between(
        collapse_grid, q25, q75, color="#476f9f", alpha=.25,
        label="interquartile range",
    )
    axes[1, 0].plot(collapse_grid, median, color="#233e63", lw=1.8,
                    label="median")
    axes[1, 0].set(
        xlabel=r"normalized distance along branch",
        ylabel=r"normalized loss of $a_\mu$",
        title="All-branch shape collapse",
    )
    axes[1, 0].legend(fontsize=6.5)
    axes[1, 1].scatter(
        death["lam_c"], death["r_c"], c=death["q"], cmap="coolwarm",
        s=20, edgecolor="none",
    )
    axes[1, 1].set(xlabel=r"$\lambda_c(\mu)$",
                   ylabel=r"$|a_{\mu+1}/a_\mu|$ at fold",
                   title="Terminal-shape variation")
    for index, ax in enumerate(axes.flat):
        label(ax, "abcd"[index])
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"N4_Figure2.{suffix}", dpi=600)
    derived = FIGURES / "N4_Figure2_data.npz"
    np.savez_compressed(
        derived, thresholds=lam, selected=np.array(selected),
        collapse_grid=collapse_grid, collapse=collapse,
        collapse_q25=q25, collapse_median=median, collapse_q75=q75,
        death_lambda=death["lam_c"], death_ratio=death["r_c"], death_q=death["q"],
    )
    return {
        "output": str(FIGURES / "N4_Figure2.pdf"),
        "sources": [str(curves_path), str(death_path)],
        "derived": str(derived),
        "panels": {
            "a": {"file": str(curves_path),
                  "keys": [f"curve_{mu}" for mu in selected]},
            "b": {"file": str(curves_path),
                  "keys": [f"curve_{mu}" for mu in selected],
                  "columns": ["lambda", "a_mu", "a_successor", "leading_real"]},
            "c": {
                "file": str(curves_path),
                "keys": [f"curve_{mu}" for mu in range(len(lam))],
                "columns": ["lambda", "a_mu", "a_successor", "leading_real"],
                "normalization": (
                    "each branch maps its first/last saved lambda and a_mu "
                    "values to 0/1; lines plus pointwise median and IQR"
                ),
            },
            "d": {"file": str(death_path), "keys": ["lam_c", "r_c", "q"]},
        },
    }


def figure3(plt) -> dict:
    table_path = REPORTS / "N3_E30_seed_statistics.csv"
    analysis_path = RUNS / "N3" / "E30_seed_aware_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    with table_path.open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    fixed = [
        row for row in table if abs(float(row["alpha"]) - 0.05) < 5e-4
    ]
    groups = defaultdict(list)
    for row in fixed:
        groups[int(row["N"])].append(row)
    sizes = np.array(sorted(groups))
    mean_std = np.array([
        np.mean([float(r["std"]) for r in groups[N]]) for N in sizes
    ])
    sem_std = np.array([
        np.std([float(r["std"]) for r in groups[N]], ddof=1)
        / np.sqrt(len(groups[N])) if len(groups[N]) > 1 else np.nan
        for N in sizes
    ])
    mean_gap = np.array([
        np.mean([float(r["gap"]) for r in groups[N]]) for N in sizes
    ])
    sem_gap = np.array([
        np.std([float(r["gap"]) for r in groups[N]], ddof=1)
        / np.sqrt(len(groups[N])) if len(groups[N]) > 1 else np.nan
        for N in sizes
    ])
    observed = np.array([float(r["max"]) for r in fixed])
    predicted = []
    for row in fixed:
        P = int(row["n_resolved"])
        from scipy.stats import norm
        q = norm.ppf((P - 0.375)/(P + 0.25))
        predicted.append(float(row["mean"]) + float(row["std"])*q)
    predicted = np.asarray(predicted)
    fig = plt.figure(figsize=(7.2, 5.0))
    grid_spec = fig.add_gridspec(2, 3)
    axes = [
        fig.add_subplot(grid_spec[0, 0]),
        fig.add_subplot(grid_spec[0, 1]),
        fig.add_subplot(grid_spec[0, 2]),
        fig.add_subplot(grid_spec[1, 0]),
        fig.add_subplot(grid_spec[1, 1:]),
    ]
    axes[0].errorbar(sizes, mean_std, yerr=sem_std, fmt="o",
                     color="#476f9f", capsize=2, label="seed mean ± s.e.m.")
    std_power = next(
        row for row in analysis["fixed_alpha_0p05"]["std"]["models"]
        if row["model"] == "power"
    )
    grid = np.geomspace(sizes.min(), sizes.max(), 200)
    axes[0].plot(
        grid, std_power["parameters"][0] * grid**(-std_power["parameters"][1]),
        "-", color="#233e63", label="power fit",
    )
    std_ci = analysis["fixed_alpha_0p05"]["std"]["bootstrap"]["exponent_ci95"]
    axes[0].text(
        .04, .05,
        rf"$b={std_power['parameters'][1]:.3f}$"
        + "\n" + rf"95% CI [{std_ci[0]:.3f},{std_ci[1]:.3f}]",
        transform=axes[0].transAxes, fontsize=6.5,
    )
    axes[0].set(xlabel="$N$", ylabel=r"$\sigma_{\lambda_c}$",
                title="Narrowing")
    axes[1].errorbar(sizes, mean_gap, yerr=sem_gap, fmt="o",
                     color="#bd3f32", capsize=2, label="seed mean ± s.e.m.")
    gap_power = next(
        row for row in analysis["fixed_alpha_0p05"]["gap"]["models"]
        if row["model"] == "power"
    )
    axes[1].plot(
        grid, gap_power["parameters"][0] * grid**(-gap_power["parameters"][1]),
        "-", color="#76291f", label="power fit",
    )
    gap_ci = analysis["fixed_alpha_0p05"]["gap"]["bootstrap"]["exponent_ci95"]
    axes[1].text(
        .04, .05,
        rf"$d={gap_power['parameters'][1]:.3f}$"
        + "\n" + rf"95% CI [{gap_ci[0]:.3f},{gap_ci[1]:.3f}]",
        transform=axes[1].transAxes, fontsize=6.5,
    )
    axes[1].set(xlabel="$N$", ylabel="max–min", title="Extreme interval")
    pooled = []
    for row in groups[max(groups)]:
        d = np.load(row["path"])
        x = d["lam_c"]
        pooled.extend((x-np.mean(x))/np.std(x, ddof=1))
    from scipy import stats
    osm, osr = stats.probplot(np.asarray(pooled), dist="norm", fit=False)
    axes[2].plot(osm, osr, ".", ms=2)
    axes[2].plot([-3, 3], [-3, 3], "k--", lw=1)
    axes[2].set(xlabel="normal quantile", ylabel="standardized threshold",
                title="QQ diagnostic")
    axes[3].scatter(observed, predicted, s=9, alpha=.55)
    lo, hi = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
    axes[3].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[3].set(xlabel="observed max", ylabel="Gaussian prediction",
                title="Finite-$P$ extreme")
    comparison_path = (
        REPORTS / "N1_static_extreme_vs_dynamic_onset_summary.json"
    )
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        rows = [
            row for row in comparison["rows"]
            if row["admissible_for_equality_test"]
        ]
        tolerance = float(comparison["tolerance"])
        lo = min(row["static_extreme_lambda"] for row in rows)
        hi = max(row["dynamic_onset_midpoint"] for row in rows)
        line = np.linspace(lo, hi, 200)
        axes[4].fill_between(
            line, line - tolerance, line + tolerance,
            color="0.90", linewidth=0,
            label=rf"$|\Delta\lambda|<10^{{-3}}$",
        )
        axes[4].errorbar(
            [r["static_extreme_lambda"] for r in rows],
            [r["dynamic_onset_midpoint"] for r in rows],
            yerr=[
                0.5 * (
                    r["dynamic_onset_high"] - r["dynamic_onset_low"]
                )
                for r in rows
            ],
            fmt="o", ms=3, capsize=1.2, color="#0072B2",
        )
        if rows:
            axes[4].plot([lo, hi], [lo, hi], "k--", lw=1)
            axes[4].text(
                .04, .96,
                f"{comparison['retained_agreement_count']}/"
                f"{comparison['retained_count']} at $10^{{-3}}$",
                transform=axes[4].transAxes, ha="left", va="top",
                fontsize=6.5,
            )
        axes[4].set(xlabel="static extreme", ylabel="dynamic onset",
                    title="Sample-wise extreme selection")
    else:
        axes[4].text(.5, .5, "N1 dynamic campaign\nin progress",
                     ha="center", va="center", transform=axes[4].transAxes)
        axes[4].set(title="Independent pairing", xticks=[], yticks=[])
    for index, ax in enumerate(axes):
        label(ax, "abcde"[index])
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=5.8)
    axes[1].legend(fontsize=5.8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"N4_Figure3.{suffix}", dpi=600)
    return {
        "output": str(FIGURES / "N4_Figure3.pdf"),
        "sources": [str(table_path), str(analysis_path), str(comparison_path)],
        "panels": {
            "a": {"file": str(table_path), "columns": ["N", "alpha", "seed", "std"]},
            "b": {"file": str(table_path), "columns": ["N", "alpha", "seed", "gap"]},
            "c": {"files": [row["path"] for row in groups[max(groups)]],
                  "key": "lam_c", "standardization": "within seed"},
            "d": {"file": str(table_path),
                  "columns": ["mean", "std", "max", "n_resolved"],
                  "prediction": "finite-P Gaussian order statistic"},
            "e": {"file": str(comparison_path), "status": (
                "filled" if comparison_path.exists() else "blank until N1 completion"
            )},
        },
    }


def figure4(plt) -> dict:
    e9_path = CYCLE / "E9_bond_times.npz"
    e29_path = CYCLE / "E29_cycle_ghosts.npz"
    floquet_path = CYCLE / "floquet_scan_N2000_tau10.0.npz"
    e9, e29, floquet = np.load(e9_path), np.load(e29_path), np.load(floquet_path)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    axes[0, 0].scatter(e9["bonds"], e9["dts"], s=12, alpha=.65)
    axes[0, 0].set(xlabel="bond index", ylabel="relay time",
                   title=f"Localized slowing at $\\lambda={float(e9['lam']):.3f}$")
    delta = e29["lams"] - float(e29["lam_star"])
    axes[0, 1].loglog(delta[delta > 0], e29["Ts"][delta > 0], "o-")
    axes[0, 1].set(xlabel=r"$\lambda-\lambda_*$", ylabel="tour period",
                   title="Cycle-side critical slowing")
    tag = "0p330"
    axes[1, 0].plot(e29["fold_lc"], e29[f"dwell_{tag}"], "o", ms=3)
    critical = int(np.nanargmax(e29["fold_lc"]))
    axes[1, 0].plot(
        e29["fold_lc"][critical], e29[f"dwell_{tag}"][critical],
        "*", ms=12, color="#bd3f32",
    )
    axes[1, 0].set(xlabel=r"static $\lambda_c(\mu)$",
                   ylabel="cycle dwell time", title="Static barrier selects bottleneck")
    # Archived multipliers are at the numerical floor. Plot only honest lower
    # bounds on total contraction, never the floor values as measurements.
    lower_bound = -np.log(np.full(len(floquet["lam"]), 10*np.finfo(float).eps))
    axes[1, 1].plot(floquet["lam"], lower_bound, "_", ms=12, color="#476f9f")
    for x, y in zip(floquet["lam"], lower_bound):
        axes[1, 1].annotate("", xy=(x, y+2), xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color="#476f9f", lw=.8))
    axes[1, 1].set_ylim(lower_bound[0] - 2.0, lower_bound[0] + 3.0)
    axes[1, 1].set(xlabel=r"$\lambda$", ylabel=r"$A=-\log|\mu_1|$",
                   title="Floquet contraction: lower bounds at floor")
    for index, ax in enumerate(axes.flat):
        label(ax, "abcd"[index])
        ax.grid(alpha=.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"N4_Figure4.{suffix}", dpi=600)
    return {
        "output": str(FIGURES / "N4_Figure4.pdf"),
        "sources": [str(e9_path), str(e29_path), str(floquet_path)],
        "note": "Floquet panel uses lower-bound arrows for floor-limited multipliers.",
        "panels": {
            "a": {"file": str(e9_path), "keys": ["bonds", "dts", "lam"]},
            "b": {"file": str(e29_path), "keys": ["lam_star", "lams", "Ts"]},
            "c": {"file": str(e29_path), "keys": ["fold_lc", "dwell_0p330"]},
            "d": {"file": str(floquet_path), "keys": ["lam", "mu_star"],
                  "rendering": "machine-floor values converted to lower-bound arrows"},
        },
    }


def main() -> None:
    plt = style()
    provenance = {
        "N4_Figure1": figure1(plt),
        "N4_Figure2": figure2(plt),
        "N4_Figure3": figure3(plt),
        "N4_Figure4": figure4(plt),
    }
    write_json(MANIFESTS / "N4_figure_provenance.json", provenance)
    print("N4 figures built:", ", ".join(provenance))


if __name__ == "__main__":
    main()
