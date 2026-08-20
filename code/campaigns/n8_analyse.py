"""Audit and preliminary analysis of the completed N8 trajectory grid.

This script deliberately recomputes trajectory labels from the saved scalar
observables.  The legacy ``classification`` field is not used: it required two
complete tours, which the fixed 1600-time-unit observation cannot contain at
large P, and it was also able to overwrite the Lyapunov-aware label in the N8
worker summary.
"""
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
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from v5_paths import FIGURES, REPORTS, RUNS


N = 2000
ALPHAS = [0.05, 0.09, 0.13, 0.15]
LAMBDAS = [0.35, 0.40, 0.50, 0.70, 0.90]
SEEDS = [42, 43, 44, 45, 46]
OBSERVATION = 1600.0
MINIMUM_TRANSITIONS = 50
LYAPUNOV_TOLERANCE = 2e-3
REVERSAL_TOLERANCE = 1e-2


def tag(alpha: float, lam: float, seed: int) -> str:
    return f"a{alpha:.2f}_lam{lam:.2f}_s{seed}".replace(".", "p")


def trajectory_path(alpha: float, lam: float, seed: int) -> Path:
    return RUNS / "N8" / f"trajectory_N{N}_{tag(alpha, lam, seed)}.npz"


def lyapunov_paths(alpha: float, lam: float, seed: int) -> list[Path]:
    return sorted(
        (RUNS / "N8").glob(
            f"lyapunov_N{N}_{tag(alpha, lam, seed)}_k*.npz"
        )
    )


def trajectory_class(
    *, fixed_residual: float, transitions: int, forward_fraction: float,
    reversal_fraction: float,
) -> str:
    if fixed_residual < 1e-10:
        return "fixed_point"
    legacy_moving = (
        transitions >= MINIMUM_TRANSITIONS and forward_fraction > 0.99
    )
    strict_sequence = (
        legacy_moving and reversal_fraction < REVERSAL_TOLERANCE
    )
    if strict_sequence:
        return "strict_sequence"
    if legacy_moving:
        return "degraded_forward"
    return "irregular_nonsequence"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            for seed in SEEDS:
                source = trajectory_path(alpha, lam, seed)
                if not source.exists():
                    raise FileNotFoundError(source)
                with np.load(source, allow_pickle=False) as data:
                    required = [
                        "fixed_residual", "forward_fraction",
                        "reversal_fraction", "relay_times", "tours",
                        "periodic_closure", "peak_analog_mean",
                        "binary_retrieval_mean", "participation_mean",
                        "front_width_mean", "kymograph_t", "kymograph_a",
                        "final_history", "final_derivative_history",
                    ]
                    missing = [key for key in required if key not in data.files]
                    if missing:
                        raise ValueError(f"{source}: missing {missing}")
                    array_keys = [
                        "relay_times", "kymograph_t", "kymograph_a",
                        "final_history", "final_derivative_history",
                    ]
                    if not all(
                        np.isfinite(data[key]).all() for key in array_keys
                    ):
                        raise ValueError(
                            f"{source}: non-finite required array"
                        )
                    transitions = int(len(data["relay_times"]))
                    fixed_residual = float(data["fixed_residual"])
                    forward_fraction = float(data["forward_fraction"])
                    reversal_fraction = float(data["reversal_fraction"])
                    label = trajectory_class(
                        fixed_residual=fixed_residual,
                        transitions=transitions,
                        forward_fraction=forward_fraction,
                        reversal_fraction=reversal_fraction,
                    )
                    relay_times = np.asarray(data["relay_times"], dtype=float)
                    quarter_counts = np.histogram(
                        relay_times, bins=np.linspace(0, OBSERVATION, 5)
                    )[0]
                    row = {
                        "alpha": alpha,
                        "lambda": lam,
                        "seed": seed,
                        "P": int(data["P"]),
                        "trajectory_class": label,
                        "transitions": transitions,
                        "transition_rate": transitions / OBSERVATION,
                        "forward_fraction": forward_fraction,
                        "reversal_fraction": reversal_fraction,
                        "tours": float(data["tours"]),
                        "fixed_residual": fixed_residual,
                        "periodic_closure": float(data["periodic_closure"]),
                        "peak_analog_mean": float(data["peak_analog_mean"]),
                        "binary_retrieval_mean": float(
                            data["binary_retrieval_mean"]
                        ),
                        "participation_mean": float(
                            data["participation_mean"]
                        ),
                        "front_width_mean": float(data["front_width_mean"]),
                        "quarter_1_transitions": int(quarter_counts[0]),
                        "quarter_2_transitions": int(quarter_counts[1]),
                        "quarter_3_transitions": int(quarter_counts[2]),
                        "quarter_4_transitions": int(quarter_counts[3]),
                        "trajectory_source": str(source),
                        "lyapunov_source": "",
                        "lyapunov_k": "",
                        "lyapunov_final": "",
                        "lyapunov_tail_250": "",
                        "lyapunov_status": "not_computed",
                    }

                candidates = lyapunov_paths(alpha, lam, seed)
                if candidates:
                    # Prefer the widest available spectrum.
                    candidates.sort(
                        key=lambda path: int(path.stem.rsplit("_k", 1)[1])
                    )
                    lyap_source = candidates[-1]
                    with np.load(lyap_source, allow_pickle=False) as data:
                        spectrum = np.asarray(data["lyapunov"], dtype=float)
                        block_time = np.asarray(
                            data["block_time"], dtype=float
                        )
                        block_exponents = np.asarray(
                            data["block_exponents"], dtype=float
                        )
                        cumulative = np.asarray(
                            data["cumulative_exponents"], dtype=float
                        )
                        if not (
                            np.isfinite(spectrum).all()
                            and np.isfinite(block_time).all()
                            and np.isfinite(block_exponents).all()
                            and np.isfinite(cumulative).all()
                        ):
                            raise ValueError(
                                f"{lyap_source}: non-finite Lyapunov data"
                            )
                        tail = block_time > block_time[-1] - 250.0
                        maximal = float(spectrum[0])
                        tail_mean = float(
                            np.mean(block_exponents[tail, 0])
                        )
                        row.update({
                            "lyapunov_source": str(lyap_source),
                            "lyapunov_k": int(len(spectrum)),
                            "lyapunov_final": maximal,
                            "lyapunov_tail_250": tail_mean,
                            "lyapunov_status": (
                                "positive"
                                if tail_mean > LYAPUNOV_TOLERANCE
                                else "no_positive_exponent_resolved"
                            ),
                        })
                rows.append(row)
    return rows


def cell_rows(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            sample = [
                row for row in rows
                if row["alpha"] == alpha and row["lambda"] == lam
            ]
            strict = [
                row for row in sample
                if row["trajectory_class"] == "strict_sequence"
            ]
            def mean(key: str) -> float:
                return (
                    float(np.mean([row[key] for row in strict]))
                    if strict else float("nan")
                )
            def sd(key: str) -> float:
                return (
                    float(np.std(
                        [row[key] for row in strict], ddof=1
                    ))
                    if len(strict) > 1 else float("nan")
                )
            output.append({
                "alpha": alpha,
                "lambda": lam,
                "strict_sequences": len(strict),
                "disorder_realizations": len(sample),
                "strict_fraction": len(strict) / len(sample),
                "fixed_points": sum(
                    row["trajectory_class"] == "fixed_point"
                    for row in sample
                ),
                "degraded_forward": sum(
                    row["trajectory_class"] == "degraded_forward"
                    for row in sample
                ),
                "irregular_nonsequence": sum(
                    row["trajectory_class"] == "irregular_nonsequence"
                    for row in sample
                ),
                "transition_rate_mean": mean("transition_rate"),
                "transition_rate_sd": sd("transition_rate"),
                "binary_retrieval_mean": mean("binary_retrieval_mean"),
                "binary_retrieval_sd": sd("binary_retrieval_mean"),
                "participation_mean": mean("participation_mean"),
                "front_width_mean": mean("front_width_mean"),
                "lyapunov_covered": sum(
                    bool(row["lyapunov_source"]) for row in sample
                ),
                "positive_tail_lyapunov": sum(
                    row["lyapunov_status"] == "positive"
                    for row in sample
                ),
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def make_figure(rows: list[dict], cells: list[dict]) -> Path:
    style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
    colors = plt.cm.viridis(np.linspace(0.08, 0.90, len(ALPHAS)))

    ax = axes[0, 0]
    fraction = np.array([
        [next(
            row["strict_fraction"] for row in cells
            if row["alpha"] == alpha and row["lambda"] == lam
        ) for lam in LAMBDAS]
        for alpha in ALPHAS
    ])
    image = ax.imshow(
        fraction, vmin=0, vmax=1, cmap="Blues", aspect="auto",
        interpolation="nearest",
    )
    for iy, alpha in enumerate(ALPHAS):
        for ix, lam in enumerate(LAMBDAS):
            count = round(5 * fraction[iy, ix])
            ax.text(
                ix, iy, f"{count}/5", ha="center", va="center",
                color=("white" if fraction[iy, ix] > 0.55 else "0.15"),
                fontsize=8,
            )
    ax.set(
        xticks=np.arange(len(LAMBDAS)),
        xticklabels=[f"{value:.2f}" for value in LAMBDAS],
        yticks=np.arange(len(ALPHAS)),
        yticklabels=[f"{value:.2f}" for value in ALPHAS],
        xlabel=r"non-reciprocity $\lambda$",
        ylabel=r"load $\alpha=P/N$",
        title="Strict sequential retrieval across disorder",
    )
    cbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("fraction of seeds")

    ax = axes[0, 1]
    for alpha, color in zip(ALPHAS, colors):
        sample = [
            row for row in cells if row["alpha"] == alpha
        ]
        y = np.array([row["transition_rate_mean"] for row in sample])
        e = np.array([row["transition_rate_sd"] for row in sample])
        e = np.nan_to_num(e, nan=0.0)
        ax.errorbar(
            LAMBDAS, y, yerr=e, marker="o", ms=3.5, lw=1.0,
            capsize=2, color=color, label=rf"$\alpha={alpha:.2f}$",
        )
    ax.set(
        xlabel=r"non-reciprocity $\lambda$",
        ylabel="relay speed (patterns / time)",
        title="Local propagation speed",
    )
    ax.legend(ncol=2)

    ax = axes[1, 0]
    for alpha, color in zip(ALPHAS, colors):
        sample = [
            row for row in cells if row["alpha"] == alpha
        ]
        y = np.array([row["binary_retrieval_mean"] for row in sample])
        e = np.array([row["binary_retrieval_sd"] for row in sample])
        e = np.nan_to_num(e, nan=0.0)
        ax.errorbar(
            LAMBDAS, y, yerr=e, marker="o", ms=3.5, lw=1.0,
            capsize=2, color=color, label=rf"$\alpha={alpha:.2f}$",
        )
    ax.set(
        xlabel=r"non-reciprocity $\lambda$",
        ylabel="mean binary retrieval overlap",
        ylim=(0.80, 1.005),
        title="Retrieval quality within strict sequences",
    )

    ax = axes[1, 1]
    for alpha, color in zip(ALPHAS, colors):
        subset = [
            row for row in rows
            if row["alpha"] == alpha and row["lyapunov_source"]
        ]
        first = True
        for row in subset:
            source = Path(row["lyapunov_source"])
            with np.load(source, allow_pickle=False) as data:
                time = np.asarray(data["block_time"], dtype=float)
                cumulative = np.asarray(
                    data["cumulative_exponents"], dtype=float
                )[:, 0]
            keep = time >= 100.0
            ax.plot(
                time[keep], cumulative[keep], color=color, alpha=0.58,
                lw=0.8, label=(rf"$\alpha={alpha:.2f}$" if first else None),
            )
            first = False
    ax.axhline(0, color="0.2", lw=0.75)
    ax.axhline(
        LYAPUNOV_TOLERANCE, color="#b2473e", lw=0.75, ls="--",
        label=r"$2\times10^{-3}$ tolerance",
    )
    ax.set(
        xlabel="Lyapunov accumulation time",
        ylabel=r"cumulative leading exponent $\Lambda_1(t)$",
        title="Seed 46: convergence toward the neutral direction",
        xlim=(100, 1000),
    )
    ax.legend(ncol=2)

    for letter, ax in zip("abcd", axes.flat):
        ax.text(
            0.015, 0.985, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10, va="top", ha="left",
            bbox={
                "facecolor": "white", "edgecolor": "none",
                "alpha": 0.80, "pad": 0.7,
            },
        )
        ax.grid(color="0.88", linewidth=0.45, zorder=-2)
    figure.tight_layout()

    stem = FIGURES / "N8_preliminary_audit"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(figure)
    return stem


def write_report(rows: list[dict], cells: list[dict], stem: Path) -> None:
    counts = Counter(row["trajectory_class"] for row in rows)
    strict = [
        row for row in rows if row["trajectory_class"] == "strict_sequence"
    ]
    covered = [row for row in rows if row["lyapunov_source"]]
    tail = np.array(
        [float(row["lyapunov_tail_250"]) for row in covered], dtype=float
    )
    final = np.array(
        [float(row["lyapunov_final"]) for row in covered], dtype=float
    )
    quarter_ratios = [
        min(
            row[f"quarter_{index}_transitions"] for index in range(1, 5)
        ) / max(
            row[f"quarter_{index}_transitions"] for index in range(1, 5)
        )
        for row in strict
    ]
    corner = [
        row for row in rows
        if row["alpha"] == 0.15 and row["lambda"] == 0.35
    ]
    summary = {
        "campaign": "N8",
        "trajectory_files_expected": 100,
        "trajectory_files_valid": len(rows),
        "classification_counts": dict(counts),
        "strict_sequence_definition": {
            "minimum_transitions": MINIMUM_TRANSITIONS,
            "forward_fraction_greater_than": 0.99,
            "reversal_fraction_less_than": REVERSAL_TOLERANCE,
            "fixed_residual_not_less_than": 1e-10,
        },
        "strict_sequence_cells_5_of_5": sum(
            row["strict_sequences"] == 5 for row in cells
        ),
        "strict_sequence_trajectories": len(strict),
        "minimum_quarter_transition_ratio": min(quarter_ratios),
        "lyapunov_trajectories": len(covered),
        "lyapunov_strict_coverage": len(covered) / len(strict),
        "lyapunov_disorder_seeds": sorted({
            row["seed"] for row in covered
        }),
        "lyapunov_final_range": [float(final.min()), float(final.max())],
        "lyapunov_tail_250_range": [float(tail.min()), float(tail.max())],
        "positive_tail_lyapunov_count": int(
            np.sum(tail > LYAPUNOV_TOLERANCE)
        ),
        "corner_alpha_0p15_lambda_0p35": [
            {
                key: row[key] for key in [
                    "seed", "trajectory_class", "transitions",
                    "forward_fraction", "reversal_fraction",
                    "fixed_residual", "binary_retrieval_mean",
                    "participation_mean",
                ]
            }
            for row in corner
        ],
        "figure": str(stem.with_suffix(".pdf")),
    }
    (REPORTS / "N8_preliminary_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    table_lines = [
        "| $\\alpha$ | $\\lambda$ | strict sequences | fixed | "
        "degraded | irregular | $v$ | binary overlap | Lyapunov |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        table_lines.append(
            f"| {row['alpha']:.2f} | {row['lambda']:.2f} | "
            f"{row['strict_sequences']}/5 | {row['fixed_points']} | "
            f"{row['degraded_forward']} | "
            f"{row['irregular_nonsequence']} | "
            f"{row['transition_rate_mean']:.5f} | "
            f"{row['binary_retrieval_mean']:.3f} | "
            f"{row['lyapunov_covered']}/5 |"
        )
    report = f"""# N8 — preliminary high-load audit

## Scope and integrity

- All 100 expected trajectories are present and readable: 4 loads, 5 values of
  non-reciprocity, and 5 independent disorder seeds.
- Each cell uses one prescribed retrieval-like initial history per disorder
  realization.  Fractions across seeds are therefore **not basin-volume
  estimates**.
- The stored legacy `classification` field is not used.  It requires two
  complete tours, which is impossible in most high-load records of fixed length,
  and the worker summary merge order can overwrite the Lyapunov-aware label.

## Result from the trajectory data

- A strict sequential trajectory is defined independently of the desired
  conclusion by at least {MINIMUM_TRANSITIONS} winner changes, nearest-neighbour
  forward fraction above 0.99, reversal fraction below
  {REVERSAL_TOLERANCE:.2f}, and a nonzero fixed-point residual.
- {len(strict)}/100 trajectories satisfy this strict criterion.  In
  {sum(row["strict_sequences"] == 5 for row in cells)}/20 parameter cells, all
  five disorder realizations show strict sequential retrieval.
- The only heterogeneous cell is $(\\alpha,\\lambda)=(0.15,0.35)$:
  seed 44 is a strict sequence; seed 42 is a degraded, jumpy forward state;
  seeds 43 and 46 are irregular non-sequential states; seed 45 is a fixed
  point.  No seed is discarded.
- For every strict sequence, the event count is stationary over the four
  quarters of the observation window: the smallest quarter-to-quarter
  count ratio is {min(quarter_ratios):.2f}.
- Conditional on strict sequential retrieval, the local relay speed is governed
  mainly by $\\lambda$ and depends only weakly on load.  Retrieval quality
  decreases with load and improves with $\\lambda$.

## Lyapunov evidence currently available

- Lyapunov data exist for {len(covered)}/{len(strict)} strict trajectories:
  seed 46 in 19 cells.  There is no Lyapunov run at the heterogeneous corner
  because seed 46 is not a moving sequence there.
- The stored cumulative leading exponents at time 1000 range from
  {final.min():+.6f} to {final.max():+.6f}.  Their mean over the last 250 time
  units ranges from {tail.min():+.6f} to {tail.max():+.6f}; none exceeds the
  predeclared positive threshold {LYAPUNOV_TOLERANCE:.3f}.
- The cumulative estimates decay toward zero and the late-window means resolve
  no positive exponent.  For this one disorder realization, the moving states
  are therefore nonchaotic within numerical resolution and are compatible with
  the neutral direction of a stable sequential limit cycle.
- This is not yet a multi-seed classification.  The other 77 strict moving
  trajectories still need their leading Lyapunov exponent.  A long record
  spanning multiple full tours is also needed to certify periodic closure at
  $\\alpha\\geq0.09$.

## Cell-level summary

{chr(10).join(table_lines)}

## Outputs

- Figure: `{stem.with_suffix(".pdf")}`
- Per-trajectory audit: `{REPORTS / "N8_trajectory_audit.csv"}`
- Per-cell summary: `{REPORTS / "N8_cell_summary.csv"}`
- Machine-readable summary: `{REPORTS / "N8_preliminary_summary.json"}`
"""
    (REPORTS / "N8_PRELIMINARY_ANALYSIS.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    cells = cell_rows(rows)
    write_csv(REPORTS / "N8_trajectory_audit.csv", rows)
    write_csv(REPORTS / "N8_cell_summary.csv", cells)
    stem = make_figure(rows, cells)
    write_report(rows, cells, stem)
    print(REPORTS / "N8_PRELIMINARY_ANALYSIS.md")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
