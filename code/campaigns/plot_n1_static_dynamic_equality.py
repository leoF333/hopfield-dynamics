"""Direct multi-seed equality plot for static extremes and dynamic onsets."""
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from v5_paths import FIGURES, REPORTS, RUNS


TOLERANCE = 1e-3


def load_rows() -> list[dict]:
    comparison = json.loads(
        (RUNS / "N1" / "comparison.json").read_text(encoding="utf-8")
    )
    rows = []
    for base in comparison["rows"]:
        seed = int(base["seed"])
        branch = json.loads(
            (
                RUNS / "N1_AUDIT"
                / f"branch_audit_N2000_P100_s{seed}.json"
            ).read_text(encoding="utf-8")
        )
        static_fold = float(branch["memory_fold"]["lambda_fold"])
        dynamic_low = float(base["dynamic_low"])
        dynamic_high = float(base["dynamic_high"])
        dynamic_onset = 0.5 * (dynamic_low + dynamic_high)
        unique_static_extreme = not bool(
            base["static_extreme_brackets_overlap"]
        )
        static_fold_converged = bool(branch["memory_fold"]["converged"])
        history_control_passed = bool(base["history_control_passed"])
        admissible = (
            unique_static_extreme
            and static_fold_converged
            and history_control_passed
        )
        exclusion_reason = ""
        if not unique_static_extreme:
            exclusion_reason = "overlapping first/second static-extreme brackets"
        elif not static_fold_converged:
            exclusion_reason = "augmented static-fold solve did not converge"
        elif not history_control_passed:
            exclusion_reason = "independent dynamic-history control failed"
        rows.append({
            "seed": seed,
            "static_extreme_motif_code_index": int(base["static_motif"]),
            "dynamic_death_motif_code_index": int(base["dynamic_motif"]),
            "static_extreme_lambda": static_fold,
            "dynamic_onset_low": dynamic_low,
            "dynamic_onset_high": dynamic_high,
            "dynamic_onset_midpoint": dynamic_onset,
            "delta_lambda": dynamic_onset - static_fold,
            "absolute_delta_lambda": abs(dynamic_onset - static_fold),
            "unique_static_extreme": unique_static_extreme,
            "static_fold_converged": static_fold_converged,
            "history_control_passed": history_control_passed,
            "admissible_for_equality_test": admissible,
            "exclusion_reason": exclusion_reason,
            "agreement_at_1e-3": (
                abs(dynamic_onset - static_fold) < TOLERANCE
            ),
        })
    return rows


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def save_figure(fig: plt.Figure) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "N1_static_extreme_vs_dynamic_onset"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)


def write_outputs(rows: list[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (
        REPORTS / "N1_static_extreme_vs_dynamic_onset.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    retained = [row for row in rows if row["admissible_for_equality_test"]]
    excluded = [row for row in rows if not row["admissible_for_equality_test"]]
    agreement = sum(row["agreement_at_1e-3"] for row in retained)
    summary = {
        "reliability_rule": {
            "unique_static_extreme": (
                "the top two static-extreme brackets do not overlap"
            ),
            "static_fold_converged": True,
            "independent_dynamic_history_control_passed": True,
        },
        "tolerance": TOLERANCE,
        "retained_seeds": [row["seed"] for row in retained],
        "excluded_seeds": [row["seed"] for row in excluded],
        "retained_agreement_count": agreement,
        "retained_count": len(retained),
        "maximum_retained_absolute_delta": max(
            row["absolute_delta_lambda"] for row in retained
        ),
        "rows": rows,
    }
    (
        REPORTS / "N1_static_extreme_vs_dynamic_onset_summary.json"
    ).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    caption = (
        r"\textbf{Sample-wise correspondence between the static extreme and "
        r"the dynamic onset.} The abscissa is the exact augmented-solve fold "
        r"of the uniquely selected extreme memory-connected branch; the "
        r"ordinate is the midpoint of the independently measured dynamic "
        r"cycle-death bracket, with error bars showing its half-width. The "
        r"shaded region denotes $|\lambda_{\rm dyn}-\lambda_{\rm stat}|<10^{-3}$. "
        rf"All {agreement}/{len(retained)} admissible disorder realizations lie "
        r"inside this resolution band. Admissibility was defined without using "
        r"the static--dynamic discrepancy: convergence of the static fold, a "
        r"successful independent-history dynamic control, and non-overlap of "
        r"the first and second static-extreme brackets. Seed 48 is excluded "
        r"from the plotted set because its two leading static brackets overlap; "
        r"the subsequent branch audit also identifies a distinct arrest-reached "
        r"stationary branch."
    )
    (
        REPORTS / "N1_static_extreme_vs_dynamic_onset_caption.tex"
    ).write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    rows = load_rows()
    write_outputs(rows)

    seeds = np.array([row["seed"] for row in rows])
    static = np.array([row["static_extreme_lambda"] for row in rows])
    dynamic = np.array([row["dynamic_onset_midpoint"] for row in rows])
    dynamic_err = 0.5 * np.array([
        row["dynamic_onset_high"] - row["dynamic_onset_low"]
        for row in rows
    ])
    admissible = np.array([
        row["admissible_for_equality_test"] for row in rows
    ])
    excluded = ~admissible

    blue = "#0072B2"
    figure = plt.figure(figsize=(4.55, 5.65), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[1.7, 1.0])
    identity_ax = figure.add_subplot(grid[0, 0])
    residual_ax = figure.add_subplot(grid[1, 0])

    lo = 0.310
    hi = 0.342
    line = np.linspace(lo, hi, 300)
    identity_ax.fill_between(
        line, line - TOLERANCE, line + TOLERANCE,
        color="0.92", linewidth=0, zorder=0,
        label=r"$|\Delta\lambda|<10^{-3}$",
    )
    identity_ax.plot(
        line, line, color="0.25", linewidth=0.8, zorder=1
    )
    identity_ax.errorbar(
        static[admissible], dynamic[admissible],
        yerr=dynamic_err[admissible],
        fmt="o", ms=5.2, mfc=blue, mec=blue, mew=0.8,
        ecolor=blue, elinewidth=0.7, capsize=1.7,
        label="admissible disorder realizations", zorder=3,
    )
    identity_ax.set_xlim(lo, hi)
    identity_ax.set_ylim(lo, hi)
    identity_ax.set_aspect("equal", adjustable="box")
    identity_ax.set_xlabel(
        r"static extreme $\lambda_{\rm stat}$"
    )
    identity_ax.set_ylabel(
        r"dynamic onset $\lambda_{\rm dyn}$"
    )
    identity_ax.grid(color="0.88", linewidth=0.45)
    retained_count = int(np.sum(admissible))
    retained_agreement = int(np.sum(
        np.abs(dynamic[admissible] - static[admissible]) < TOLERANCE
    ))
    identity_ax.text(
        0.03, 0.97,
        rf"{retained_agreement}/{retained_count} admissible seeds "
        rf"agree at $10^{{-3}}$",
        transform=identity_ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white",
              "ec": "0.75", "lw": 0.6},
    )
    identity_ax.text(
        -0.13, 1.04, "(a)", transform=identity_ax.transAxes,
        fontsize=11, fontweight="bold", ha="left", va="bottom",
    )

    residual = 1e3 * (dynamic - static)
    residual_ax.axhspan(-1, 1, color="0.92", zorder=0)
    residual_ax.axhline(0, color="0.25", linewidth=0.7, zorder=1)
    residual_ax.errorbar(
        seeds[admissible], residual[admissible],
        yerr=1e3 * dynamic_err[admissible],
        fmt="o", ms=4.7, mfc=blue, mec=blue, mew=0.8,
        ecolor=blue, elinewidth=0.65, capsize=1.5, zorder=3,
    )
    residual_ax.set_xlim(41.3, 61.7)
    residual_ax.set_ylim(-1.15, 1.15)
    residual_ax.set_xticks(np.arange(42, 62, 2))
    residual_ax.set_yticks([-1, 0, 1])
    residual_ax.set_xlabel("disorder seed")
    residual_ax.set_ylabel(
        r"$10^3(\lambda_{\rm dyn}-\lambda_{\rm stat})$"
    )
    residual_ax.grid(axis="y", color="0.88", linewidth=0.45)
    residual_ax.text(
        -0.13, 1.04, "(b)", transform=residual_ax.transAxes,
        fontsize=11, fontweight="bold", ha="left", va="bottom",
    )

    save_figure(figure)
    plt.close(figure)
    print(
        f"retained equality: {retained_agreement}/{retained_count}; "
        f"excluded seeds: {seeds[excluded].tolist()}"
    )


if __name__ == "__main__":
    main()
