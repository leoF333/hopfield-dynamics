#!/usr/bin/env python3
"""Create the publication schematic summarizing the E34 bifurcation architecture."""

from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_e34")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

import os as _os
from pathlib import Path as _Path

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
CHICAGO_S = str(CHICAGO)
# ---------------------------------------------------------------------------

# --- PATCHED 2026-08-19 by folder reorganisation -----------------------------
# All absolute paths below now resolve from a single root, overridable with the
# CHICAGO_ROOT environment variable.  See ../../REORGANISATION_LOG.md
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
# -----------------------------------------------------------------------------



FIGURE_DIR = CHICAGO / "obsolete/02_superseded_manuscripts/paper_v5/figures_ready"
MANIFEST_DIR = CHICAGO / "4_campaigns/numerical_workplan/manifests"
SOURCE_REPORT = CHICAGO / "3_numerics/results/2_cycle_rappel_snic/E34_INVARIANT_CIRCLE_RESULTS.md"
SOURCE_FIGURES = [
    Path(
        CHICAGO_S + "/3_numerics/results/2_cycle_rappel_snic/"
        "figures/E34/figE34_invariant_circle.png"
    ),
    Path(
        CHICAGO_S + "/3_numerics/results/2_cycle_rappel_snic/"
        "figures/E34/figE34_bifurcation.png"
    ),
]

OUT_STEM = FIGURE_DIR / "Figure_E34_bifurcation_architecture"


# Colorblind-safe, print-friendly palette.
INK = "#17212B"
MUTED = "#627181"
PANEL = "#F6F8FA"
GRID = "#D8E0E7"
NODE = "#2474A6"
NODE_LIGHT = "#DCECF5"
SADDLE = "#CF4B55"
TWIN = "#E5A23B"
CYCLE = "#7257B5"
GHOST = "#9AA7B2"
OPEN = "#8A97A3"
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def add_panel(ax, xy, width, height):
    panel = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=PANEL,
        edgecolor=GRID,
        linewidth=0.9,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(panel)


def state(ax, x, y, kind, radius=0.012, label=None, label_dx=0.0, label_dy=-0.035):
    if kind == "node":
        patch = Circle(
            (x, y),
            radius,
            transform=ax.transAxes,
            facecolor=NODE,
            edgecolor=WHITE,
            linewidth=1.6,
            zorder=8,
        )
    elif kind == "twin":
        patch = Circle(
            (x, y),
            radius,
            transform=ax.transAxes,
            facecolor=TWIN,
            edgecolor=WHITE,
            linewidth=1.6,
            zorder=8,
        )
    elif kind == "saddle":
        vertices = [
            (x, y + radius * 1.15),
            (x + radius * 1.15, y),
            (x, y - radius * 1.15),
            (x - radius * 1.15, y),
        ]
        patch = Polygon(
            vertices,
            closed=True,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=SADDLE,
            linewidth=2.0,
            zorder=8,
        )
    elif kind == "ghost":
        patch = Circle(
            (x, y),
            radius,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=GHOST,
            linewidth=1.4,
            linestyle=(0, (2.0, 2.0)),
            alpha=0.95,
            zorder=6,
        )
    else:
        raise ValueError(kind)
    ax.add_patch(patch)
    if label:
        ax.text(
            x + label_dx,
            y + label_dy,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.6,
            color=INK if kind != "ghost" else MUTED,
            zorder=10,
        )
    return patch


def ellipse_connection(
    ax,
    cx,
    cy,
    width,
    height,
    *,
    color=NODE,
    linestyle="-",
    linewidth=2.0,
    alpha=1.0,
    zorder=2,
):
    ellipse = Ellipse(
        (cx, cy),
        width,
        height,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(ellipse)
    # Direction marker tangent to the upper-right part of the ellipse.
    arrow = FancyArrowPatch(
        (cx + 0.19 * width, cy + 0.48 * height),
        (cx + 0.32 * width, cy + 0.39 * height),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        color=color,
        transform=ax.transAxes,
        alpha=alpha,
        zorder=zorder + 1,
    )
    ax.add_patch(arrow)


def arrow(ax, start, end, *, color=MUTED, lw=1.5, style="-|>", rad=0.0, ls="-", z=4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def bracket(ax, x0, x1, y, text, color=INK):
    ax.plot([x0, x0, x1, x1], [y - 0.012, y, y, y - 0.012], color=color, lw=1.2, transform=ax.transAxes)
    ax.text(
        (x0 + x1) / 2,
        y + 0.008,
        text,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8.8,
        color=color,
    )


def make_figure():
    fig = plt.figure(figsize=(15.8, 9.1), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.045,
        0.955,
        "Successive saddle-node beads on a continuing invariant necklace culminate in the terminal SNIC",
        ha="left",
        va="top",
        fontsize=19.5,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.045,
        0.918,
        r"Schematic synthesis for the reference realization ($N=2000$, $P=100$, seed 42, $\beta=20$, $\tau=10$)",
        ha="left",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )

    # ------------------------------------------------------------------
    # Panel (a): global bifurcation sequence.
    # ------------------------------------------------------------------
    add_panel(ax, (0.035, 0.405), 0.93, 0.475)
    ax.text(
        0.052,
        0.845,
        "(a) Global sequence as non-reciprocity increases",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="semibold",
        color=INK,
        va="center",
    )
    ax.text(
        0.949,
        0.845,
        "geometry of the state-space connection",
        transform=ax.transAxes,
        fontsize=9.1,
        color=MUTED,
        ha="right",
        va="center",
    )

    global_y = 0.635

    # Many-bead, twin-mediated necklace: dashed because full closure is not yet certified.
    c1 = 0.145
    ellipse_connection(
        ax,
        c1,
        global_y,
        0.165,
        0.175,
        color=OPEN,
        linestyle=(0, (4.0, 3.0)),
        linewidth=2.0,
    )
    positions = [
        (c1 - 0.072, global_y + 0.025, "node"),
        (c1 - 0.045, global_y + 0.072, "saddle"),
        (c1 + 0.010, global_y + 0.085, "node"),
        (c1 + 0.058, global_y + 0.050, "saddle"),
        (c1 + 0.076, global_y - 0.005, "node"),
        (c1 + 0.045, global_y - 0.070, "saddle"),
        (c1 - 0.015, global_y - 0.086, "twin"),
        (c1 - 0.065, global_y - 0.046, "saddle"),
    ]
    for x, y, kind in positions:
        state(ax, x, y, kind, radius=0.0098)
    ax.text(
        c1,
        0.745,
        "many beads",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.8,
        fontweight="semibold",
        color=INK,
    )
    ax.text(
        c1,
        0.505,
        "canonical nodes + twins\nlocal links certified;\nfull closure still open",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.35,
    )

    # Sequential fold arrow and ghost annotation.
    arrow(ax, (0.235, global_y), (0.285, global_y), color=SADDLE, lw=1.6)
    ax.text(
        0.260,
        global_y + 0.052,
        "successive\nlocal folds",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.7,
        color=SADDLE,
        fontweight="semibold",
    )
    ax.text(
        0.260,
        global_y - 0.056,
        "beads disappear;\nghosts remain",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )

    # Two-bead circle: directly closed.
    c2 = 0.365
    ellipse_connection(ax, c2, global_y, 0.135, 0.175, color=NODE, linewidth=2.3)
    state(ax, c2 - 0.055, global_y + 0.032, "node", radius=0.0105, label="node 43")
    state(ax, c2 + 0.053, global_y + 0.035, "node", radius=0.0105, label="node 91")
    state(ax, c2, global_y + 0.083, "saddle", radius=0.010)
    state(ax, c2, global_y - 0.084, "saddle", radius=0.010)
    ax.text(
        c2,
        0.745,
        r"$\lambda=0.307629$: two beads",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.7,
        fontweight="semibold",
        color=INK,
    )
    ax.text(
        c2,
        0.505,
        r"circle closed: $43\rightarrow91\rightarrow43$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
        color=NODE,
    )

    arrow(ax, (0.445, global_y), (0.500, global_y), color=SADDLE, lw=1.6)
    state(ax, 0.472, global_y + 0.038, "ghost", radius=0.009)
    state(ax, 0.478, global_y - 0.040, "ghost", radius=0.009)

    # One-bead circle: directly closed.
    c3 = 0.585
    ellipse_connection(ax, c3, global_y, 0.135, 0.175, color=NODE, linewidth=2.3)
    state(ax, c3 - 0.020, global_y + 0.081, "node", radius=0.011, label="node 91", label_dx=-0.010)
    state(ax, c3 + 0.032, global_y + 0.079, "saddle", radius=0.0105, label="saddle 91", label_dx=0.015)
    ax.text(
        c3,
        0.745,
        r"$\lambda=0.322629$: last bead",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.7,
        fontweight="semibold",
        color=INK,
    )
    ax.text(
        c3,
        0.505,
        r"one full tour returns to node 91" "\n" r"(distance $<4\times10^{-14}$)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=NODE,
        linespacing=1.35,
    )

    # Terminal collision / SNIC.
    snic_x = 0.735
    ax.plot(
        [snic_x, snic_x],
        [0.465, 0.805],
        transform=ax.transAxes,
        color=CYCLE,
        lw=1.6,
        alpha=0.75,
        zorder=1,
    )
    arrow(ax, (0.657, global_y), (snic_x - 0.018, global_y), color=SADDLE, lw=1.8)
    Circle(
        (snic_x, global_y),
        0.014,
        transform=ax.transAxes,
        facecolor=NODE,
        edgecolor=SADDLE,
        linewidth=2.5,
        zorder=9,
    )
    ax.add_patch(
        Circle(
            (snic_x, global_y),
            0.014,
            transform=ax.transAxes,
            facecolor=NODE,
            edgecolor=SADDLE,
            linewidth=2.5,
            zorder=9,
        )
    )
    ax.text(
        snic_x,
        0.758,
        r"terminal fold at $\lambda^\ast=0.32763$",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.1,
        fontweight="semibold",
        color=CYCLE,
    )
    ax.text(
        snic_x,
        0.568,
        "last equilibrium pair disappears",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK,
    )
    ax.text(
        snic_x,
        0.480,
        "SNIC",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=CYCLE,
    )
    ax.text(
        snic_x,
        0.528,
        r"$T_{\rm tour}^{-}\;\longrightarrow\;T_{\rm cycle}^{+}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        color=CYCLE,
    )

    # Above threshold: stable travelling cycle.
    c4 = 0.885
    ellipse_connection(ax, c4, global_y, 0.145, 0.175, color=CYCLE, linewidth=2.6)
    ax.add_patch(
        Circle(
            (c4 + 0.048, global_y + 0.067),
            0.012,
            transform=ax.transAxes,
            facecolor=CYCLE,
            edgecolor=WHITE,
            linewidth=1.6,
            zorder=9,
        )
    )
    # Faint remembered positions visited by the wave.
    for xx, yy in [
        (c4 - 0.054, global_y + 0.038),
        (c4 - 0.028, global_y - 0.074),
        (c4 + 0.055, global_y - 0.037),
    ]:
        ax.add_patch(
            Circle(
                (xx, yy),
                0.006,
                transform=ax.transAxes,
                facecolor=CYCLE,
                edgecolor="none",
                alpha=0.22,
                zorder=3,
            )
        )
    ax.text(
        c4,
        0.745,
        r"$\lambda>\lambda^\ast$: travelling recall cycle",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.7,
        fontweight="semibold",
        color=INK,
    )
    ax.text(
        c4,
        0.505,
        "stable sequential wave observed;\nglobal uniqueness remains to be certified",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.4,
        color=MUTED,
        linespacing=1.35,
    )
    arrow(ax, (snic_x + 0.018, global_y), (0.805, global_y), color=CYCLE, lw=1.8)

    # Lambda axis.
    arrow(ax, (0.073, 0.445), (0.943, 0.445), color=INK, lw=1.1)
    ax.text(
        0.508,
        0.422,
        r"increasing non-reciprocity $\lambda$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        color=INK,
    )

    # ------------------------------------------------------------------
    # Panel (b): local bead anatomy and terminology.
    # ------------------------------------------------------------------
    add_panel(ax, (0.035, 0.075), 0.93, 0.295)
    ax.text(
        0.052,
        0.337,
        "(b) Local anatomy of an ordinary motif - the subtlety behind “all folds are SNICs”",
        transform=ax.transAxes,
        fontsize=12.2,
        fontweight="semibold",
        color=INK,
        va="center",
    )

    chain_y = 0.225
    x_node, x_s1, x_twin, x_s2, x_next = 0.115, 0.220, 0.335, 0.445, 0.565
    # Underlying necklace segment.
    ax.plot(
        [0.075, 0.615],
        [chain_y, chain_y],
        transform=ax.transAxes,
        color=GRID,
        lw=3.0,
        solid_capstyle="round",
        zorder=1,
    )
    state(ax, x_node, chain_y, "node", radius=0.0125, label=r"canonical node $\mu$", label_dy=-0.030)
    state(ax, x_s1, chain_y, "saddle", radius=0.012, label=r"saddle$_1$", label_dy=-0.030)
    state(ax, x_twin, chain_y, "twin", radius=0.0125, label=r"twin of $\mu$", label_dy=-0.030)
    state(ax, x_s2, chain_y, "saddle", radius=0.012, label=r"saddle$_2$", label_dy=-0.030)
    state(ax, x_next, chain_y, "node", radius=0.0125, label=r"next node $\nu$", label_dy=-0.030)

    # Certified downstream unstable branches.
    arrow(ax, (x_s1 + 0.015, chain_y), (x_twin - 0.016, chain_y), color=SADDLE, lw=1.5)
    arrow(ax, (x_s2 + 0.015, chain_y), (x_next - 0.016, chain_y), color=SADDLE, lw=1.5)
    bracket(ax, x_node - 0.025, x_s1 + 0.025, 0.285, r"fold A: node$_\mu$ + saddle$_1$")
    bracket(ax, x_twin - 0.025, x_s2 + 0.025, 0.285, r"fold B: twin$_\mu$ + saddle$_2$")

    # Optional/incomplete cascade.
    arrow(
        ax,
        (x_twin, chain_y - 0.016),
        (x_twin + 0.055, 0.148),
        color=OPEN,
        lw=1.2,
        ls=(0, (3.0, 2.0)),
        rad=0.15,
    )
    state(
        ax,
        x_twin + 0.068,
        0.145,
        "ghost",
        radius=0.009,
        label=r"possible twin$^{(2)}$",
        label_dy=-0.018,
    )
    ax.text(
        x_twin + 0.145,
        0.141,
        r"$\mu=96$: cascade observed; not yet closed",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.6,
        color=MUTED,
    )

    # Terminal-bead inset.
    inset = FancyBboxPatch(
        (0.665, 0.115),
        0.275,
        0.175,
        boxstyle="round,pad=0.012,rounding_size=0.01",
        transform=ax.transAxes,
        facecolor=WHITE,
        edgecolor=GRID,
        linewidth=0.9,
        zorder=0,
    )
    ax.add_patch(inset)
    ax.text(
        0.682,
        0.265,
        "Terminal bead 91: no twin",
        transform=ax.transAxes,
        fontsize=9.7,
        fontweight="semibold",
        color=INK,
    )
    state(ax, 0.718, 0.205, "node", radius=0.012, label="node 91", label_dy=-0.030)
    state(ax, 0.790, 0.205, "saddle", radius=0.0115, label="saddle 91", label_dy=-0.030)
    arrow(ax, (0.805, 0.205), (0.858, 0.205), color=SADDLE, lw=1.5)
    ax.add_patch(
        Circle(
            (0.876, 0.205),
            0.013,
            transform=ax.transAxes,
            facecolor=NODE,
            edgecolor=SADDLE,
            linewidth=2.2,
            zorder=9,
        )
    )
    ax.text(
        0.876,
        0.165,
        r"$\lambda^\ast$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.0,
        color=CYCLE,
        fontweight="semibold",
    )
    ax.text(
        0.802,
        0.105,
        "Only this final annihilation releases a periodic orbit:\n"
        "the SNIC in the strict dynamical-systems sense.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color=CYCLE,
        fontweight="semibold",
        linespacing=1.35,
    )

    # Evidence legend.
    ax.plot([0.055, 0.083], [0.095, 0.095], transform=ax.transAxes, color=NODE, lw=2.2)
    ax.text(
        0.088,
        0.095,
        "solid: directly closed or certified in E34",
        transform=ax.transAxes,
        va="center",
        fontsize=7.8,
        color=MUTED,
    )
    ax.plot(
        [0.255, 0.283],
        [0.095, 0.095],
        transform=ax.transAxes,
        color=OPEN,
        lw=1.7,
        linestyle=(0, (4.0, 3.0)),
    )
    ax.text(
        0.288,
        0.095,
        "dashed: supported continuation, not globally closed",
        transform=ax.transAxes,
        va="center",
        fontsize=7.8,
        color=MUTED,
    )

    # Footer.
    fig.text(
        0.046,
        0.035,
        "Interpretation: all observed folds are consistent with saddle-nodes embedded in the same continuing necklace; "
        "the terminal fold is the directly demonstrated SNIC.",
        ha="left",
        va="center",
        fontsize=8.4,
        color=MUTED,
    )
    fig.text(
        0.955,
        0.035,
        "E34 - reference realization",
        ha="right",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )

    return fig


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    manifest = {
        "figure": str(OUT_STEM),
        "created_from": {
            "report": str(SOURCE_REPORT),
            "figures": [str(p) for p in SOURCE_FIGURES],
        },
        "reference_parameters": {
            "N": 2000,
            "P": 100,
            "seed": 42,
            "beta": 20,
            "tau": 10,
            "lambda_star": 0.32763,
        },
        "evidence_encoding": {
            "solid": "direct closure or local connection certified in E34",
            "dashed": "continuation supported but full global closure not certified",
        },
        "scientific_scope": [
            "Direct terminal SNIC for the reference realization.",
            "Direct closed invariant circle with one and two surviving canonical beads.",
            "Certified local twin-mediated links for motifs 9 and 22.",
            "The full twin-inclusive necklace with many beads remains incompletely closed.",
            "Global uniqueness of the post-threshold limit cycle remains pending.",
        ],
    }
    (MANIFEST_DIR / "Figure_E34_bifurcation_architecture.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
