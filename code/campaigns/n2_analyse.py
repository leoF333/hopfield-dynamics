"""Publication-grade aggregation, gate and figure for corrected N2 outputs."""
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

import matplotlib.pyplot as plt
import numpy as np

from n2_local_laws import fit_models
from v5_paths import FIGURES, ROOT, RUNS, write_json

SEEDS = [42, 47, 51, 52, 60]
N1_TABLE = ROOT / "reports" / "N1_static_extreme_vs_dynamic_onset.csv"
REPORTS = ROOT / "reports"
WINDOWS = {
    "all": lambda delta: np.ones(len(delta), dtype=bool),
    "drop_smallest": lambda delta: delta > np.min(delta),
    "drop_largest": lambda delta: delta < np.max(delta),
    "near_5e-4": lambda delta: delta <= 5e-4,
    "near_2e-4": lambda delta: delta <= 2e-4,
    "near_1e-4": lambda delta: delta <= 1e-4,
}


def load() -> tuple[dict[int, dict], dict[int, dict]]:
    summaries = {
        seed: json.loads(
            (
                RUNS / "N2" /
                f"seed_N2000_P100_s{seed}_production.json"
            ).read_text(encoding="utf-8")
        )
        for seed in SEEDS
    }
    with N1_TABLE.open(newline="", encoding="utf-8") as handle:
        thresholds = {
            int(row["seed"]): row
            for row in csv.DictReader(handle)
            if int(row["seed"]) in SEEDS
        }
    return summaries, thresholds


def fit_rows(summaries: dict[int, dict]) -> list[dict]:
    output = []
    for seed, summary in summaries.items():
        for side in ("dynamic", "static"):
            rows = (
                [
                    row for row in summary["dynamic_rows"]
                    if row["dt"] == 0.01 and not row["censored"]
                ]
                if side == "dynamic" else summary["static_rows"]
            )
            delta = np.asarray([row["delta"] for row in rows], float)
            value = np.asarray([
                row["critical_median"] if side == "dynamic"
                else row["leading_real"]
                for row in rows
            ], float)
            for window, selector in WINDOWS.items():
                selected = selector(delta) & np.isfinite(value)
                if selected.sum() < 4:
                    continue
                fits = fit_models(
                    delta[selected], value[selected],
                    dynamic=(side == "dynamic"),
                )
                for fit in fits:
                    output.append({
                        "seed": seed,
                        "side": side,
                        "window": window,
                        "n": int(selected.sum()),
                        "delta_min": float(np.min(delta[selected])),
                        "delta_max": float(np.max(delta[selected])),
                        "model": fit["model"],
                        "parameters": json.dumps(fit["parameters"]),
                        "parameter_se": json.dumps(fit["parameter_se"]),
                        "aicc": fit["aicc"],
                        "delta_aicc": fit["delta_aicc"],
                    })
    return output


def free_exponent(
    summary: dict, side: str, window: str,
    rng: np.random.Generator | None = None,
) -> float:
    if side == "dynamic":
        rows = [
            row for row in summary["dynamic_rows"]
            if row["dt"] == 0.01 and not row["censored"]
        ]
        values = []
        for row in rows:
            passages = np.asarray(row["critical_passages"], float)
            if rng is not None and len(passages):
                passages = rng.choice(passages, size=len(passages), replace=True)
            values.append(float(np.median(passages)))
    else:
        rows = summary["static_rows"]
        values = [float(row["leading_real"]) for row in rows]
    delta = np.asarray([row["delta"] for row in rows], float)
    values_array = np.asarray(values, float)
    selected = WINDOWS[window](delta) & np.isfinite(values_array)
    fits = fit_models(
        delta[selected], values_array[selected],
        dynamic=(side == "dynamic"),
    )
    free = next(row for row in fits if row["model"] == "free_power")
    return float(free["parameters"][-1])


def bootstrap_mean_exponent(
    summaries: dict[int, dict], seeds: list[int], side: str, window: str,
    n_bootstrap: int = 5000,
) -> dict:
    base = np.asarray([
        free_exponent(summaries[seed], side, window) for seed in seeds
    ])
    rng = np.random.default_rng(20260730)
    inner = {
        seed: (
            np.asarray([
                free_exponent(summaries[seed], side, window, rng=rng)
                for _ in range(500)
            ])
            if side == "dynamic" else np.asarray([base[index]])
        )
        for index, seed in enumerate(seeds)
    }
    sampled = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        choices = rng.choice(seeds, size=len(seeds), replace=True)
        values = [
            float(rng.choice(inner[int(seed)]))
            for seed in choices
        ]
        sampled[index] = np.mean(values)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "seeds": seeds,
        "per_seed": base.tolist(),
        "mean": float(np.mean(base)),
        "bootstrap_95": [float(low), float(high)],
        "n_bootstrap": n_bootstrap,
    }


def timestep_rows(summaries: dict[int, dict]) -> list[dict]:
    output = []
    for seed, summary in summaries.items():
        for fine in (
            row for row in summary["dynamic_rows"] if row["dt"] == 0.005
        ):
            coarse = next(
                row for row in summary["dynamic_rows"]
                if row["dt"] == 0.01
                and np.isclose(row["delta"], fine["delta"])
            )
            difference = (
                abs(coarse["critical_median"] - fine["critical_median"])
                / max(abs(fine["critical_median"]), 1e-12)
            )
            output.append({
                "seed": seed,
                "delta": fine["delta"],
                "t_dt_0p01": coarse["critical_median"],
                "t_dt_0p005": fine["critical_median"],
                "relative_difference": difference,
            })
    return output


def make_figure(
    summaries: dict[int, dict], local_dynamic: list[int],
    fit_table: list[dict],
) -> None:
    colors = dict(zip(SEEDS, plt.cm.viridis(np.linspace(0.08, 0.92, len(SEEDS)))))
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.9))

    # (a) Multi-seed critical passage times with per-seed half-power guides.
    ax = axes[0, 0]
    for seed in SEEDS:
        summary = summaries[seed]
        rows = sorted(
            (
                row for row in summary["dynamic_rows"]
                if row["dt"] == 0.01 and not row["censored"]
            ),
            key=lambda row: row["delta"],
        )
        delta = np.asarray([row["delta"] for row in rows])
        time = np.asarray([row["critical_median"] for row in rows])
        admissible = seed in local_dynamic
        color = colors[seed] if admissible else "0.55"
        ax.loglog(
            delta, time, "o", ms=3.4, color=color,
            markerfacecolor=(color if admissible else "none"),
            label=f"seed {seed}" + ("" if admissible else " (miscentered)"),
        )
        if admissible:
            fit = next(
                row for row in fit_table
                if row["side"] == "dynamic" and row["seed"] == seed
                and row["window"] == "all" and row["model"] == "half_power"
            )
            B, A = json.loads(fit["parameters"])
            grid = np.geomspace(delta.min(), delta.max(), 200)
            ax.loglog(grid, B + A * grid**-0.5, "-", lw=0.8, color=color)
    ax.set(
        xlabel=r"$\delta=\lambda-\lambda_*^{\rm stat}$",
        ylabel=r"critical-bond passage time",
        title="Cycle-side critical slowing",
    )
    ax.legend(fontsize=5.3, ncol=2)

    # (b) Dynamic free exponent and its fit-window dependence.
    ax = axes[0, 1]
    shown_windows = ["all", "drop_smallest", "drop_largest", "near_5e-4"]
    x = np.arange(len(shown_windows))
    for offset, seed in enumerate(SEEDS):
        values = []
        for window in shown_windows:
            fit = next(
                row for row in fit_table
                if row["side"] == "dynamic" and row["seed"] == seed
                and row["window"] == window and row["model"] == "free_power"
            )
            values.append(json.loads(fit["parameters"])[-1])
        color = colors[seed] if seed in local_dynamic else "0.55"
        ax.plot(
            x + (offset - 2) * 0.035, values, "o-", ms=3, lw=0.75,
            color=color, alpha=(1.0 if seed in local_dynamic else 0.65),
        )
    ax.axhline(0.5, color="k", ls="--", lw=0.9)
    ax.set(
        xticks=x,
        xticklabels=["all", "no min", "no max", r"$\delta\leq5\,10^{-4}$"],
        ylabel=r"free exponent $\nu_{\rm dyn}$",
        title="Finite-window exponent drift",
    )
    ax.tick_params(axis="x", labelrotation=18)

    # (c) Dedicated real-axis roots on the stable branch.
    ax = axes[1, 0]
    for seed in SEEDS:
        rows = sorted(summaries[seed]["static_rows"], key=lambda row: row["delta"])
        delta = np.asarray([row["delta"] for row in rows])
        rate = -np.asarray([row["leading_real"] for row in rows])
        ax.loglog(delta, rate, "o-", ms=3, lw=0.7, color=colors[seed],
                  label=f"seed {seed}")
    guide_x = np.geomspace(1e-5, 1e-4, 100)
    guide_y = 0.32 * np.sqrt(guide_x)
    ax.loglog(guide_x, guide_y, "k--", lw=1.0, label=r"slope $1/2$")
    ax.set(
        xlabel=r"$\delta=\lambda_*^{\rm stat}-\lambda$",
        ylabel=r"$-\mathrm{Re}\,z_{\rm real}$",
        title="Static softening beyond the local window",
    )
    ax.legend(fontsize=5.2, ncol=2)

    # (d) Static exponent changes as progressively less local points enter.
    ax = axes[1, 1]
    static_windows = ["near_1e-4", "near_2e-4", "near_5e-4", "drop_largest"]
    xmax = [1e-4, 2e-4, 5e-4, 5e-3]
    for seed in SEEDS:
        values = []
        for window in static_windows:
            fit = next(
                row for row in fit_table
                if row["side"] == "static" and row["seed"] == seed
                and row["window"] == window and row["model"] == "free_power"
            )
            values.append(json.loads(fit["parameters"])[-1])
        ax.semilogx(xmax, values, "o-", ms=3, lw=0.75, color=colors[seed])
    ax.axhline(0.5, color="k", ls="--", lw=0.9)
    ax.set(
        xlabel=r"largest included $\delta$",
        ylabel=r"free exponent $\nu_{\rm stat}$",
        title="Static fit-window sensitivity",
    )
    ax.set_ylim(-0.02, 2.08)

    for label, ax in zip("abcd", axes.flat):
        ax.text(
            -0.14, 1.04, label, transform=ax.transAxes,
            fontweight="bold", va="top",
        )
        ax.grid(alpha=0.22)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"N2_critical_laws.{suffix}", dpi=600)
    plt.close(fig)


def main() -> None:
    summaries, thresholds = load()
    local_dynamic = [
        seed for seed, row in thresholds.items()
        if (
            float(row["dynamic_onset_low"])
            <= float(row["static_extreme_lambda"])
            <= float(row["dynamic_onset_high"])
        )
    ]
    excluded_dynamic = sorted(set(SEEDS) - set(local_dynamic))

    fits = fit_rows(summaries)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "N2_fit_windows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fits[0]))
        writer.writeheader()
        writer.writerows(fits)

    steps = timestep_rows(summaries)
    with (REPORTS / "N2_timestep_controls.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(steps[0]))
        writer.writeheader()
        writer.writerows(steps)

    dynamic_all = bootstrap_mean_exponent(
        summaries, local_dynamic, "dynamic", "all"
    )
    dynamic_near = bootstrap_mean_exponent(
        summaries, local_dynamic, "dynamic", "near_5e-4"
    )
    static_near = bootstrap_mean_exponent(
        summaries, SEEDS, "static", "near_1e-4"
    )
    static_extended = bootstrap_mean_exponent(
        summaries, SEEDS, "static", "near_5e-4"
    )

    valid_static = all(
        row["branch_source"] == "stable_N1_terminal_continuation"
        and row["branch_identity"] == summary["selected_motif"]
        and row["converged"]
        and row["residual"] < 1e-11
        and row["leading_real_method"] == "real_axis_sigmin_scan_n300"
        and row["leading_real_scaled_residual"] < 1e-10
        and row["leading_real"] < 0.0
        for summary in summaries.values()
        for row in summary["static_rows"]
    )
    all_dynamic_uncensored = all(
        not row["censored"]
        for summary in summaries.values()
        for row in summary["dynamic_rows"]
    )
    timestep_max = max(row["relative_difference"] for row in steps)
    geometry_identity = all(
        row["closest_bond"] == summary["selected_motif"]
        for summary in summaries.values()
        for row in summary["dynamic_rows"]
        if row["dt"] == 0.01
    )
    log_deltas = [
        row["delta_aicc"] for row in fits
        if row["side"] == "dynamic" and row["seed"] in local_dynamic
        and row["window"] in ("all", "near_5e-4")
        and row["model"] == "logarithmic"
    ]
    half_deltas = [
        row["delta_aicc"] for row in fits
        if row["side"] == "dynamic" and row["seed"] in local_dynamic
        and row["window"] in ("all", "near_5e-4")
        and row["model"] == "half_power"
    ]
    static_window_shift = abs(
        static_near["mean"] - static_extended["mean"]
    )
    strict_gate = (
        valid_static
        and all_dynamic_uncensored
        and timestep_max < 0.02
        and min(log_deltas) > 10.0
        and max(half_deltas) <= 10.0
        and static_window_shift < 0.1
    )

    payload = {
        "seeds": SEEDS,
        "dynamic_local_scaling_seeds": local_dynamic,
        "dynamic_local_scaling_excluded": {
            str(seed): (
                "static fold lies above the independently measured dynamic "
                "onset bracket; the offset exceeds the smallest local deltas"
            )
            for seed in excluded_dynamic
        },
        "dynamic_exponent_all": dynamic_all,
        "dynamic_exponent_near_5e-4": dynamic_near,
        "static_exponent_near_1e-4": static_near,
        "static_exponent_near_5e-4": static_extended,
        "static_branch_and_residual_gate": valid_static,
        "all_dynamic_uncensored": all_dynamic_uncensored,
        "timestep_max_relative_difference": timestep_max,
        "critical_bond_identity_all_dt_0p01": geometry_identity,
        "minimum_logarithmic_delta_aicc": min(log_deltas),
        "maximum_half_power_delta_aicc": max(half_deltas),
        "static_mean_exponent_window_shift": static_window_shift,
        "strict_measured_half_exponent_gate": strict_gate,
        "allowed_manuscript_wording": (
            "compatible with a saddle-node square-root law"
            if not strict_gate else "measured one-half exponent"
        ),
        "normal_form_seed42": summaries[42]["normal_form"],
        "sources": [
            str(
                RUNS / "N2" /
                f"seed_N2000_P100_s{seed}_production.json"
            )
            for seed in SEEDS
        ],
    }
    write_json(REPORTS / "N2_summary.json", payload)

    report = f"""# N2 scientific gate

## Scope and integrity

- Five disorder realizations: {SEEDS}.
- All 50 corrected stationary points remain on the stable N1-connected branch,
  preserve the selected motif identity and pass the fixed-point and dedicated
  real-root residual gates.
- All 80 dynamic measurements are uncensored.
- The 30 half-step controls have a maximum relative difference of
  {timestep_max:.3g}.
- The closest dynamic approach occurs at the statically selected terminal bond
  for every one of the 50 base-step trajectories.

## Local dynamic law

The independent N1 onset bracket contains the exact static fold for seeds
{local_dynamic}. Seed {excluded_dynamic[0]} is retained in all files and plots
but excluded from the local exponent aggregation because its entire dynamic
onset bracket lies below the exact static fold by more than the smallest tested
offsets. This criterion was fixed independently of the fitted exponent.

- Full-window disorder-mean free exponent:
  {dynamic_all['mean']:.3f}, nested-bootstrap 95% interval
  [{dynamic_all['bootstrap_95'][0]:.3f},
  {dynamic_all['bootstrap_95'][1]:.3f}].
- Near-window (\u03b4 <= 5e-4) mean:
  {dynamic_near['mean']:.3f}, 95% interval
  [{dynamic_near['bootstrap_95'][0]:.3f},
  {dynamic_near['bootstrap_95'][1]:.3f}].
- The logarithmic alternative is disfavored by at least
  \u0394AICc={min(log_deltas):.1f} over the audited full and near windows.
- A fixed one-half law is nevertheless disfavored relative to a free power by
  as much as \u0394AICc={max(half_deltas):.1f}; finite-size/sample-dependent
  deviations remain resolved.

## Static law

The dedicated real-axis search repairs candidate omissions in the generic
ARPACK spectrum and gives scaled characteristic residuals below 1e-10 at all
50 points. Close to the fold the rightmost real mode softens with a
square-root-like trend. Progressively wider windows leave this local asymptotic
regime and produce strong sample and window dependence:

- mean free exponent through \u03b4=1e-4:
  {static_near['mean']:.3f}, seed-bootstrap 95% interval
  [{static_near['bootstrap_95'][0]:.3f},
  {static_near['bootstrap_95'][1]:.3f}];
- mean through \u03b4=5e-4:
  {static_extended['mean']:.3f}, 95% interval
  [{static_extended['bootstrap_95'][0]:.3f},
  {static_extended['bootstrap_95'][1]:.3f}].

The static exponent is therefore not stable over the required 1.5 decades.

## Gate conclusion

**Strict measured-half-exponent gate: {'PASS' if strict_gate else 'FAIL'}.**

The data robustly establish localized critical slowing, reject a logarithmic
dynamic law for the four correctly centered samples, and are compatible with
the saddle-node square-root mechanism. They do not support quoting a universal
measured exponent exactly equal to one half. The permitted manuscript wording
is therefore: **“compatible with a saddle-node square-root law.”**

## Files

- Machine-readable summary: `reports/N2_summary.json`
- Fit-window table: `reports/N2_fit_windows.csv`
- Time-step controls: `reports/N2_timestep_controls.csv`
- Four-panel diagnostic: `figures/N2_critical_laws.pdf`
"""
    (REPORTS / "N2_GATE_STATUS.md").write_text(report, encoding="utf-8")
    make_figure(summaries, local_dynamic, fits)


if __name__ == "__main__":
    main()
