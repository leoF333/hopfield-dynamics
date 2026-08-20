"""Publication plots for the multi-seed N1 identity results."""
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
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from v5_paths import FIGURES, REPORTS, RUNS


TOLERANCE = 1e-3
AUDIT = RUNS / "N1_AUDIT"


def load_rows() -> list[dict]:
    comparison = json.loads(
        (RUNS / "N1" / "comparison.json").read_text(encoding="utf-8")
    )
    comparison_by_seed = {
        int(row["seed"]): row for row in comparison["rows"]
    }
    rows = []
    for seed in range(42, 62):
        branch = json.loads(
            (
                AUDIT / f"branch_audit_N2000_P100_s{seed}.json"
            ).read_text(encoding="utf-8")
        )
        base = comparison_by_seed[seed]
        dynamic_low = float(base["dynamic_low"])
        dynamic_high = float(base["dynamic_high"])
        dynamic_mid = 0.5 * (dynamic_low + dynamic_high)
        memory_fold = float(branch["memory_fold"]["lambda_fold"])
        arrest_converged = bool(branch["arrest_branch_fold"]["converged"])
        if arrest_converged:
            relevant_fold = float(
                branch["arrest_branch_fold"]["lambda_fold"]
            )
            relevant_source = (
                "same memory/arrest branch"
                if branch["same_fold_numerically"]
                else "distinct arrest branch"
            )
        else:
            relevant_fold = memory_fold
            relevant_source = "memory fold; arrest solve unresolved"
        rows.append({
            "seed": seed,
            "static_motif_code_index": int(base["static_motif"]),
            "dynamic_death_motif_code_index": int(base["dynamic_motif"]),
            "static_motif_paper_index": int(base["static_motif"]) + 1,
            "dynamic_death_motif_paper_index": int(base["dynamic_motif"]) + 1,
            "memory_fold": memory_fold,
            "relevant_stationary_fold": relevant_fold,
            "relevant_fold_source": relevant_source,
            "arrest_fold_converged": arrest_converged,
            "same_fold_numerically": bool(branch["same_fold_numerically"]),
            "dynamic_low": dynamic_low,
            "dynamic_high": dynamic_high,
            "dynamic_mid": dynamic_mid,
            "delta_memory": dynamic_mid - memory_fold,
            "delta_relevant": dynamic_mid - relevant_fold,
            "motif_identity": (
                int(base["static_motif"]) == int(base["dynamic_motif"])
            ),
            "memory_agreement_at_1e-3": (
                abs(dynamic_mid - memory_fold) < TOLERANCE
            ),
            "relevant_agreement_at_1e-3": (
                abs(dynamic_mid - relevant_fold) < TOLERANCE
            ),
        })
    return rows


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13, 1.04, label, transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="bottom", ha="left",
    )


def plot_motif(ax: plt.Axes, rows: list[dict]) -> None:
    seeds = np.array([row["seed"] for row in rows])
    static = np.array([row["static_motif_paper_index"] for row in rows])
    dynamic = np.array([
        row["dynamic_death_motif_paper_index"] for row in rows
    ])
    blue = "#0072B2"
    orange = "#D55E00"
    ax.scatter(
        seeds, static, s=30, marker="o", facecolors="none",
        edgecolors=blue, linewidths=1.2,
        label=r"last surviving static motif $\mu_{\rm stat}$",
        zorder=3,
    )
    ax.scatter(
        seeds, dynamic, s=22, marker="x", color=orange, linewidths=1.1,
        label=r"cycle-death motif $\mu_{\rm dyn}$",
        zorder=4,
    )
    ax.set_xlim(41.2, 61.8)
    ax.set_ylim(0, 101)
    ax.set_xticks(np.arange(42, 62, 2))
    ax.set_xlabel("disorder seed")
    ax.set_ylabel(r"pattern index $\mu$")
    ax.grid(axis="y", color="0.88", linewidth=0.45)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4)
    ax.text(
        0.98, 0.97, "20/20 exact matches",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white",
              "ec": "0.75", "lw": 0.6},
    )
    panel_label(ax, "(a)")


def plot_lambda(ax: plt.Axes, residual_ax: plt.Axes,
                rows: list[dict]) -> None:
    seeds = np.array([row["seed"] for row in rows])
    dynamic = np.array([row["dynamic_mid"] for row in rows])
    dynamic_err = 0.5 * np.array([
        row["dynamic_high"] - row["dynamic_low"] for row in rows
    ])
    relevant = np.array([
        row["relevant_stationary_fold"] for row in rows
    ])
    memory = np.array([row["memory_fold"] for row in rows])
    distinct = np.array([
        row["relevant_fold_source"] == "distinct arrest branch"
        for row in rows
    ])
    unresolved = np.array([
        not row["arrest_fold_converged"] for row in rows
    ])
    same = ~(distinct | unresolved)

    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    gray = "0.45"

    ax.errorbar(
        seeds + 0.07, dynamic, yerr=dynamic_err,
        fmt="x", ms=4.3, mew=1.0, color=orange,
        ecolor=orange, elinewidth=0.65, capsize=1.5,
        label=r"dynamic midpoint $\lambda_{\rm dyn}$",
        zorder=4,
    )
    ax.scatter(
        seeds[same] - 0.07, relevant[same],
        s=24, marker="o", facecolors="none", edgecolors=blue,
        linewidths=1.0, label="stationary fold: same branch", zorder=3,
    )
    ax.scatter(
        seeds[distinct] - 0.07, relevant[distinct],
        s=26, marker="s", color=green, edgecolors="none",
        label="stationary fold: distinct arrest branch", zorder=3,
    )
    ax.scatter(
        seeds[unresolved] - 0.07, relevant[unresolved],
        s=28, marker="^", facecolors="none", edgecolors=gray,
        linewidths=1.0, label="arrest solve unresolved", zorder=3,
    )
    for seed, left, right in zip(seeds, relevant, dynamic):
        ax.plot(
            [seed - 0.04, seed + 0.04], [left, right],
            color="0.78", lw=0.5, zorder=1,
        )
    ax.set_xlim(41.2, 61.8)
    ax.set_ylim(0.310, 0.342)
    ax.set_xticks([])
    ax.set_ylabel(r"critical coupling $\lambda_c$")
    ax.grid(axis="y", color="0.88", linewidth=0.45)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.015),
        frameon=False, ncol=2, columnspacing=0.9,
        handletextpad=0.4, borderaxespad=0.0,
    )
    ax.text(
        0.98, 0.97,
        r"20/20: $|\lambda_{\rm dyn}-\lambda_{\rm fold}|<10^{-3}$",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white",
              "ec": "0.75", "lw": 0.6},
    )
    panel_label(ax, "(b)")

    residual_ax.axhspan(
        -1.0, 1.0, color="0.92", zorder=0,
        label=r"$10^{-3}$ resolution band",
    )
    residual_ax.axhline(0.0, color="0.35", lw=0.65, zorder=1)
    residual_ax.scatter(
        seeds - 0.08, 1e3 * (dynamic - memory),
        s=22, marker="o", facecolors="none", edgecolors=gray,
        linewidths=0.9, label="relative to memory fold", zorder=2,
    )
    residual_ax.scatter(
        seeds[same] + 0.08, 1e3 * (dynamic[same] - relevant[same]),
        s=20, marker="o", color=blue, edgecolors="none", zorder=3,
    )
    residual_ax.scatter(
        seeds[distinct] + 0.08,
        1e3 * (dynamic[distinct] - relevant[distinct]),
        s=22, marker="s", color=green, edgecolors="none",
        label="relative to arrest-reached fold", zorder=3,
    )
    residual_ax.scatter(
        seeds[unresolved] + 0.08,
        1e3 * (dynamic[unresolved] - relevant[unresolved]),
        s=24, marker="^", facecolors="none", edgecolors=gray,
        linewidths=0.9, zorder=3,
    )
    residual_ax.annotate(
        "seed 48",
        xy=(48 - 0.08, 1e3 * (dynamic[6] - memory[6])),
        xytext=(49.0, 3.35),
        arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "0.35"},
        fontsize=7.2,
    )
    residual_ax.set_xlim(41.2, 61.8)
    residual_ax.set_ylim(-1.15, 3.85)
    residual_ax.set_xticks(np.arange(42, 62, 2))
    residual_ax.set_yticks([-1, 0, 1, 2, 3])
    residual_ax.set_xlabel("disorder seed")
    residual_ax.set_ylabel(
        r"$10^3(\lambda_{\rm dyn}-\lambda_{\rm fold})$"
    )
    residual_ax.grid(axis="y", color="0.88", linewidth=0.45)
    residual_ax.legend(
        loc="upper right", frameon=False, ncol=1,
        handletextpad=0.4, borderaxespad=0.2,
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.svg")
    fig.savefig(FIGURES / f"{stem}.png", dpi=400)


def write_report(rows: list[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "N1_multiseed_identity_1e-3.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    motif_matches = sum(row["motif_identity"] for row in rows)
    memory_matches = sum(row["memory_agreement_at_1e-3"] for row in rows)
    relevant_matches = sum(row["relevant_agreement_at_1e-3"] for row in rows)
    caption = (
        r"\textbf{Multi-realization identity of the dynamically selected "
        r"bottleneck.} (a) For all 20 disorder realizations, the motif whose "
        r"memory-connected stationary branch survives up to the largest "
        r"coupling coincides with the slowest motif at cycle death "
        r"($20/20$). Code indices have been shifted to the paper convention "
        r"$\mu=1,\ldots,P$. (b) Dynamic critical couplings (bracket midpoints; "
        r"error bars show bracket half-widths) compared with the exact fold of "
        r"the stationary branch reached after arrest. Filled squares denote "
        r"realizations in which this branch differs from the memory-connected "
        r"branch; for seed 43 the arrest-branch augmented solve is unresolved "
        r"and the memory fold is shown as an open triangle. The lower panel "
        r"shows residuals relative to both the memory fold and the "
        r"arrest-reached fold. At resolution $10^{-3}$, "
        rf"{relevant_matches}/20 branch-aware critical values agree "
        rf"({memory_matches}/20 when only the memory fold is used)."
    )
    (REPORTS / "N1_multiseed_identity_caption.tex").write_text(
        caption + "\n", encoding="utf-8"
    )
    summary = {
        "motif_matches": motif_matches,
        "memory_lambda_matches_at_1e-3": memory_matches,
        "branch_aware_lambda_matches_at_1e-3": relevant_matches,
        "tolerance": TOLERANCE,
        "rows": rows,
    }
    (REPORTS / "N1_multiseed_identity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    style()
    rows = load_rows()
    write_report(rows)

    fig = plt.figure(figsize=(8.15, 4.55), constrained_layout=True)
    grid = fig.add_gridspec(
        2, 2, width_ratios=[0.94, 1.25], height_ratios=[1.55, 1.0]
    )
    motif_ax = fig.add_subplot(grid[:, 0])
    lambda_ax = fig.add_subplot(grid[0, 1])
    residual_ax = fig.add_subplot(grid[1, 1], sharex=lambda_ax)
    plot_motif(motif_ax, rows)
    plot_lambda(lambda_ax, residual_ax, rows)
    save_all(fig, "N1_multiseed_identity")
    plt.close(fig)

    fig_motif, ax_motif = plt.subplots(
        figsize=(4.15, 3.45), constrained_layout=True
    )
    plot_motif(ax_motif, rows)
    save_all(fig_motif, "N1_multiseed_motif_identity")
    plt.close(fig_motif)

    fig_lambda = plt.figure(figsize=(4.45, 4.65), constrained_layout=True)
    lambda_grid = fig_lambda.add_gridspec(2, 1, height_ratios=[1.55, 1.0])
    lambda_ax = fig_lambda.add_subplot(lambda_grid[0, 0])
    residual_ax = fig_lambda.add_subplot(
        lambda_grid[1, 0], sharex=lambda_ax
    )
    plot_lambda(lambda_ax, residual_ax, rows)
    save_all(fig_lambda, "N1_multiseed_lambda_identity_1e-3")
    plt.close(fig_lambda)

    print(
        "N1 plots complete:",
        f"motifs={sum(row['motif_identity'] for row in rows)}/20,",
        "memory lambda=",
        f"{sum(row['memory_agreement_at_1e-3'] for row in rows)}/20,",
        "branch-aware lambda=",
        f"{sum(row['relevant_agreement_at_1e-3'] for row in rows)}/20",
    )


if __name__ == "__main__":
    main()
