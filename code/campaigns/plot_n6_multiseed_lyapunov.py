"""Publication plot for the validated N6 multi-seed Lyapunov result."""
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


SEEDS = [42, 43, 44, 45, 46]
LAMBDA = 0.31
DTS = [0.01, 0.005]


def path(seed: int, dt: float):
    lam_tag = f"{LAMBDA:.3f}".replace(".", "p")
    dt_tag = f"{dt:.3f}".rstrip("0").replace(".", "p")
    return (
        RUNS / "N6" /
        f"lyapunov_N2000_P100_s{seed}_lam{lam_tag}_dt{dt_tag}.npz"
    )


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


def main() -> None:
    style()
    spectra: dict[tuple[int, float], np.ndarray] = {}
    rows = []
    for seed in SEEDS:
        for dt in DTS:
            source = path(seed, dt)
            data = np.load(source)
            spectrum = np.asarray(data["lyapunov"], dtype=float)
            spectra[(seed, dt)] = spectrum
            for index, value in enumerate(spectrum, start=1):
                rows.append({
                    "seed": seed,
                    "dt": dt,
                    "index": index,
                    "lyapunov_exponent": value,
                    "source": str(source),
                })

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(SEEDS)))

    ax = axes[0]
    indices = np.arange(1, 9)
    for seed, color in zip(SEEDS, colors):
        ax.plot(
            indices, spectra[(seed, 0.01)], "o-", ms=3.0, lw=0.9,
            color=color, label=f"seed {seed}",
        )
    ax.axhline(0, color="0.25", lw=0.7)
    ax.axvspan(0.5, 3.5, color="#d9ecdf", alpha=0.7, zorder=-1)
    ax.set(
        xlabel="Lyapunov index $i$",
        ylabel=r"$\Lambda_i$",
        title=r"Five disorder realizations at $\lambda=0.31$",
        xticks=indices,
    )
    ax.legend(ncol=2, handlelength=1.4, columnspacing=0.7)

    ax = axes[1]
    marker_colors = ["#0072B2", "#D55E00", "#009E73"]
    lo, hi = 0.008, 0.058
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=0.8)
    for index, color in zip(range(3), marker_colors):
        x = np.array([spectra[(seed, 0.01)][index] for seed in SEEDS])
        y = np.array([spectra[(seed, 0.005)][index] for seed in SEEDS])
        ax.scatter(x, y, s=24, color=color, edgecolor="white", linewidth=0.4,
                   label=rf"$\Lambda_{index + 1}$")
    ax.set(
        xlim=(lo, hi), ylim=(lo, hi),
        xlabel=r"$dt=0.01$",
        ylabel=r"$dt=0.005$",
        title="Half-step control of expanding directions",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    ax = axes[2]
    width = 0.36
    x = np.arange(len(SEEDS))
    third_dt1 = np.array([spectra[(seed, 0.01)][2] for seed in SEEDS])
    third_dt2 = np.array([spectra[(seed, 0.005)][2] for seed in SEEDS])
    ax.bar(x - width/2, third_dt1, width, color="#476f9f", label="$dt=0.01$")
    ax.bar(x + width/2, third_dt2, width, color="#bd6a3c", label="$dt=0.005$")
    ax.axhline(0, color="0.25", lw=0.7)
    ax.set(
        xticks=x, xticklabels=SEEDS,
        xlabel="disorder seed",
        ylabel=r"third exponent $\Lambda_3$",
        title="At least three positive exponents",
    )
    ax.legend()

    for letter, ax in zip("abc", axes):
        ax.text(
            0.015, 0.985, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10, va="top", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.78, "pad": 0.7},
        )
        ax.grid(color="0.88", linewidth=0.45, zorder=-2)
    figure.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "N6_multiseed_lyapunov"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=400)

    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "N6_multiseed_lyapunov.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seeds": SEEDS,
        "lambda": LAMBDA,
        "time_steps": DTS,
        "minimum_third_exponent": min(
            spectra[(seed, dt)][2] for seed in SEEDS for dt in DTS
        ),
        "statement": (
            "All five tested disorder realizations retain at least three "
            "positive Lyapunov exponents under a factor-two time-step control."
        ),
        "sources": [str(path(seed, dt)) for seed in SEEDS for dt in DTS],
    }
    (REPORTS / "N6_multiseed_lyapunov_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS / "N6_multiseed_lyapunov_caption.tex").write_text(
        r"\textbf{Robust multi-seed hyperchaos at $\lambda=0.31$.} "
        r"(a) The leading eight Lyapunov exponents for five independent "
        r"disorder realizations at $dt=0.01$.  The shaded region marks the "
        r"first three indices.  (b) The three expanding directions remain "
        r"positive under a factor-two reduction of the integration step. "
        r"(c) In every seed and at both time steps, the third exponent remains "
        r"strictly positive.  Thus the tested attractor has at least three "
        r"robustly expanding directions; the calculation does not resolve the "
        r"full Kaplan--Yorke dimension."
        "\n",
        encoding="utf-8",
    )
    print(stem)


if __name__ == "__main__":
    main()
