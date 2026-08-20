#!/usr/bin/env python3
"""
Build the merged correlation panel for the manuscript.

Replaces the former Figure10 (Markov, E27) and Figure11 (circular GP, E28), whose
lower rows (death-shape family, threshold vs bond overlap) are dropped. What
survives is the comparison that carries the argument: correlating the memories
either concentrates pinning onto one architectural defect (the seam of a finite
Markov chain closed into a ring) or removes exceptional links altogether (an
exactly periodic, seam-free Gaussian process on the circle).

Inputs   3_numerics/results/6_motifs_correles/data/E27_markov_N10000.npz
         3_numerics/results/6_motifs_correles/data/E28_gpcircle_N10000.npz
Output   Figure10_correlations.png   (N = 10^4, P = 500, beta = 20, seed 42)

Colour: the level index is an ordered quantity in (a) and (c), so both use a
single perceptually-uniform sequential ramp (plasma / viridis).  In (b) and (d)
the series are named statistics, so each carries a distinct marker as well as a
distinct hue: identity is never colour-alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "3_numerics/results/6_motifs_correles/data")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Figure10_correlations.png")

P = 500
CS = [0.0, 0.2, 0.4, 0.6, 0.8]          # Markov frame correlations
KS = [400, 200, 100, 50, 25]            # circular-GP latent Fourier modes

INK = "#1b1b1b"
MUTED = "#6b6b6b"
GRID = dict(alpha=0.25, lw=0.7)

plt.rcParams.update({
    "font.size": 11.5,
    "axes.labelsize": 12.5,
    "axes.titlesize": 12.5,
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#d8d8d8",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def panel_label(ax, letter: str) -> None:
    ax.text(-0.135, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left", color=INK)


def main() -> None:
    mk = np.load(DATA / "E27_markov_N10000.npz")
    gp = np.load(DATA / "E28_gpcircle_N10000.npz")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    fig.subplots_adjust(left=0.072, right=0.985, top=0.835, bottom=0.075,
                        hspace=0.46, wspace=0.225)

    # ---------------------------------------------------------------- (a) ---
    # Markov chain: bulk threshold distributions, seam branch as a dashed rule.
    ax = axes[0, 0]
    cols = plt.cm.plasma(np.linspace(0.05, 0.72, len(CS)))
    shown, empty = [], []
    for c, col in zip(CS, cols):
        lc = mk[f"lam_c_{c}"]
        bulk = lc[:P - 1][np.isfinite(lc[:P - 1])]
        if bulk.size < 2:
            empty.append(c)
            continue
        shown.append(c)
        ax.hist(bulk, bins=22, histtype="step", lw=2.0, color=col, density=True,
                label=rf"$c={c}$  (bulk, $n={bulk.size}$)")
        seam = lc[P - 1]
        if np.isfinite(seam):
            ax.axvline(seam, color=col, ls="--", lw=1.8, ymin=0.0, ymax=0.70)
            ax.annotate("seam", xy=(seam, 0.705), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=10.5, color=col, va="bottom", ha="center")
    ax.set_xlim(0.0, 0.32)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
    ax.set_xlabel(r"$\lambda_c(\mu)$")
    ax.set_ylabel("density")
    ax.set_title("Correlation collapses the bulk;\nthe uncorrelated seam does not move",
                 pad=9)
    ax.grid(**GRID)
    ax.legend(fontsize=9.5, loc="upper left")
    if empty:
        ax.text(0.985, 0.975,
                "no localized branch survives\n"
                rf"for $c \geq {min(empty)}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=MUTED)
    panel_label(ax, "a")

    # ---------------------------------------------------------------- (b) ---
    # Markov chain: bulk order statistics and the seam, versus correlation.
    ax = axes[0, 1]
    cs_ok = [c for c in CS if np.isfinite(mk[f"lam_c_{c}"][:P - 1]).sum() >= 2]
    series = [
        (np.nanmax, "^", "#2f9e44", "bulk max"),
        (np.nanmedian, "o", "#2b6cb0", "bulk median"),
        (np.nanmin, "v", "#b8860b", "bulk min"),
    ]
    for fn, marker, colour, label in series:
        ax.plot(cs_ok, [fn(mk[f"lam_c_{c}"][:P - 1]) for c in cs_ok],
                marker=marker, ls="-", lw=2.0, ms=9, color=colour, label=label)
    ax.plot(cs_ok, [mk[f"lam_c_{c}"][P - 1] for c in cs_ok], "s--", lw=2.4, ms=10,
            color="#c2255c", label=r"seam branch ($q \approx 0$)")
    seam02 = mk["lam_c_0.2"][P - 1]
    med02 = np.nanmedian(mk["lam_c_0.2"][:P - 1])
    ax.annotate("", xy=(0.2, seam02), xytext=(0.2, med02),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4))
    ax.annotate("the last\nbarrier", xy=(0.2, 0.5 * (seam02 + med02)),
                xytext=(11, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10, color=MUTED)
    ax.set_xlabel(r"frame correlation $c$")
    ax.set_ylabel(r"$\lambda_c$")
    ax.set_xlim(-0.012, 0.262)
    ax.set_title("Extreme selection survives, but the extreme\n"
                 "is now an architectural defect", pad=9)
    ax.grid(**GRID)
    ax.set_ylim(0.0, 0.315)
    ax.legend(fontsize=9.5, loc="lower left")
    panel_label(ax, "b")

    # ---------------------------------------------------------------- (c) ---
    # Circular GP: threshold distributions versus smoothness, no seam anywhere.
    ax = axes[1, 0]
    colsK = plt.cm.viridis(np.linspace(0.08, 0.80, len(KS)))
    goneK = []
    for K, col in zip(KS, colsK):
        lc = gp[f"lam_c_{K}"]
        fin = lc[np.isfinite(lc)]
        q1 = gp[f"q_{K}"].mean()
        if fin.size < 2:
            goneK.append((K, q1, fin.size))
            continue
        ax.hist(fin, bins=22, histtype="step", lw=2.0, color=col, density=True,
                label=rf"$K={K}$  ($q_1={q1:+.2f}$, $n={fin.size}$)")
    ax.set_xlim(0.0, 0.32)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.34)
    ax.set_xlabel(r"$\lambda_c(\mu)$")
    ax.set_ylabel("density")
    ax.set_title("Seam-free periodic correlations melt the barriers\n"
                 "instead of concentrating them", pad=9)
    ax.grid(**GRID)
    ax.legend(fontsize=9.5, loc="upper left")
    if goneK:
        txt = "\n".join(rf"$K={K}$ ($q_1={q1:+.2f}$): {n}/500 branches"
                        for K, q1, n in goneK)
        ax.text(0.985, 0.975, "localized branches almost gone:\n" + txt,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=MUTED)
    panel_label(ax, "c")

    # ---------------------------------------------------------------- (d) ---
    # Circular GP: the whole ensemble falls together, with no outlying link.
    ax = axes[1, 1]
    ok = [K for K in KS if np.isfinite(gp[f"lam_c_{K}"]).sum() >= 2]
    q1s = [gp[f"q_{K}"].mean() for K in ok]
    ax.plot(q1s, [np.nanmax(gp[f"lam_c_{K}"]) for K in ok], "^-", lw=2.0, ms=9,
            color="#2f9e44", label=r"max $=\lambda^{*}$")
    ax.errorbar(q1s, [np.nanmedian(gp[f"lam_c_{K}"]) for K in ok],
                yerr=[np.nanstd(gp[f"lam_c_{K}"]) for K in ok],
                fmt="o-", lw=2.0, ms=9, capsize=4, color="#2b6cb0",
                label=r"median $\pm\ \sigma$")
    ax.plot(q1s, [np.nanmin(gp[f"lam_c_{K}"]) for K in ok], "v-", lw=2.0, ms=9,
            color="#b8860b", label="min (first death)")
    for K, q1 in zip(ok, q1s):
        ax.annotate(rf"$K={K}$", xy=(q1, np.nanmax(gp[f"lam_c_{K}"])),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10, color=MUTED)
    ax.axhline(0.20, color="#c2255c", ls=":", lw=1.8)
    ax.annotate("forward and unpinned\n" r"down to $\lambda=0.20$",
                xy=(0.030, 0.20), xycoords=("axes fraction", "data"),
                xytext=(0, -7), textcoords="offset points",
                ha="left", va="top", fontsize=9.5, color="#c2255c")
    ax.set_xlabel(r"nearest-neighbour correlation $q_1$")
    ax.set_ylabel(r"$\lambda_c$")
    ax.set_xlim(-0.20, 0.63)
    ax.set_ylim(0.0, 0.335)
    ax.set_title("The whole ensemble falls together;\n"
                 "no link is left to pin on", pad=9)
    ax.grid(**GRID)
    ax.legend(fontsize=9.5, loc="lower left")
    panel_label(ax, "d")

    # ------------------------------------------------------------- framing --
    fig.text(0.5, 0.985,
             "Correlated memories: one construction concentrates pinning, "
             "the other removes it",
             ha="center", va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.5, 0.947,
             r"(a,b) Markov flip chain closed into a ring: one uncorrelated seam.   "
             r"(c,d) exactly periodic circular Gaussian process: no seam.   "
             r"$N=10^4$, $P=500$, $\beta=20$, seed 42.",
             ha="center", va="top", fontsize=11, color=MUTED)

    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"wrote {OUT}  ({', '.join(str(v) for v in plt.imread(OUT).shape[:2][::-1])} px)")


if __name__ == "__main__":
    main()
