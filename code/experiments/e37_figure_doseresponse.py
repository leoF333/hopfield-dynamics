"""E37 figure 1 -- what modernising K buys, as a dose-response in correlation.

Encoding choice: the finding is about the SEQUENCE field K, so K carries the
colour (two hues, categorical slots 1-2) and the storage field J carries the line
style. Four arbitrary hues would have hidden the 2x2 structure that is the whole
point. Marker fill is the frozen E35 cycle-validity verdict, so validity never
rests on colour alone.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/6_motifs_correles/data"
FIGS = ROOT / "results/6_motifs_correles/figures"

K_COLOR = {"KH": "#2a78d6", "KP": "#eb6834"}          # categorical slots 1, 2
J_STYLE = {"JH": "-", "JP": "--"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

LABEL = {"JH_KH": "J Hebb · K Hebb   (système usuel)",
         "JP_KH": "J moderne · K Hebb",
         "JH_KP": "J Hebb · K moderne",
         "JP_KP": "J moderne · K moderne"}
ORDER = ("JH_KH", "JP_KH", "JH_KP", "JP_KP")


def load(tag: str, seed: int) -> dict:
    rows = {}
    for f in sorted(glob.glob(str(DATA / f"E37_modernK_c*_seed{seed}_{tag}_*.json"))):
        d = json.loads(Path(f).read_text())
        for r in d["rows"]:
            rows.setdefault(r["arm"], []).append(r)
    for arm in rows:
        rows[arm].sort(key=lambda r: r["c"])
    return rows


def _style(ax, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    t = ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=14 if subtitle else 6)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.015), xycoords="axes fraction",
                    color=MUTED, fontsize=9, va="bottom")
    return t


def _plot(ax, rows, key, scale=1.0):
    for arm in ORDER:
        rs = rows.get(arm)
        if not rs:
            continue
        j, k = arm.split("_")
        x = [r["c"] for r in rs]
        y = [scale * r[key] for r in rs]
        ax.plot(x, y, J_STYLE[j], color=K_COLOR[k], lw=2.0, zorder=3,
                solid_capstyle="round", dash_capstyle="round")
        # marker fill = the frozen cycle-validity verdict (secondary encoding)
        for xi, yi, r in zip(x, y, rs):
            valid = bool(r["valid_cycle"])
            ax.plot([xi], [yi], "o", ms=8, zorder=4,
                    color=K_COLOR[k] if valid else SURFACE,
                    mec=K_COLOR[k], mew=1.8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=FIGS / "figE37_modernK_doseresponse.png")
    args = p.parse_args(argv)

    rows = load(args.tag, args.seed)
    if not rows:
        raise SystemExit("no E37 json found")
    lam = rows[ORDER[0]][0]["lam"]
    n, p_ = rows[ORDER[0]][0]["n"], rows[ORDER[0]][0]["p"]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4), facecolor=SURFACE)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    _plot(ax_a, rows, "forward_fraction")
    ax_a.axhline(0.95, color=MUTED, lw=1.0, ls=":", zorder=2)
    ax_a.annotate("seuil de cycle valide (0.95)", xy=(0.02, 0.95), xytext=(0, -14),
                  textcoords="offset points", color=MUTED, fontsize=8.5)
    _style(ax_a, "corrélation entre trames consécutives  $c$",
           "fraction des transitions vers l'avant",
           "(a) L'ordre de la séquence",
           "1.0 = chaque changement de meneur va vers la trame suivante")
    # the K-Hebb arms reach exactly 0 at c>=0.6 (the state freezes): the axis has
    # to show that, not clip it away.
    ax_a.set_ylim(-0.04, 1.05)
    ax_a.annotate("le meneur ne change plus :\nl'état se fige sur une trame",
                  xy=(0.6, 0.0), xytext=(0.44, 0.22), fontsize=8.5, color=MUTED,
                  arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))

    _plot(ax_b, rows, "coverage_fraction")
    _style(ax_b, "corrélation entre trames consécutives  $c$",
           "fraction des trames effectivement visitées",
           "(b) La couverture de la séquence",
           "1.0 = les 100 trames défilent")
    ax_b.set_ylim(-0.03, 1.05)

    _plot(ax_c, rows, "selectivity_median")
    _style(ax_c, "corrélation entre trames consécutives  $c$",
           "marge $m_{\\rm meneur} - \\max_{\\nu\\neq}\\, m_\\nu$  au pic",
           "(c) La netteté du rappel",
           "de combien la trame lue domine sa suivante immédiate")
    ax_c.set_ylim(bottom=0)

    _plot(ax_d, rows, "packet_width_median")
    _style(ax_d, "corrélation entre trames consécutives  $c$",
           "nombre de trames à plus de la moitié du pic",
           "(d) L'étalement du paquet",
           "1 = une seule trame est lue à la fois")
    ax_d.set_yscale("log")
    ax_d.set_yticks([1, 2, 5, 10, 20, 50])
    ax_d.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # direct labels, at the right edge of panel (a); legend carries the rest
    handles = [plt.Line2D([], [], color=K_COLOR[a.split("_")[1]],
                          ls=J_STYLE[a.split("_")[0]], lw=2.0, marker="o", ms=7,
                          label=LABEL[a]) for a in ORDER]
    handles.append(plt.Line2D([], [], color=MUTED, ls="none", marker="o", ms=7,
                              mfc=SURFACE, mec=MUTED, mew=1.8,
                              label="marqueur creux = cycle non valide (critère E35)"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=9.5, bbox_to_anchor=(0.5, -0.005),
                     handlelength=3.6)   # long enough for dashed vs solid to read
    for txt in leg.get_texts():
        txt.set_color(INK2)

    fig.suptitle("E37 — ce que la composante Hopfield moderne apporte sur une "
                 "séquence corrélée", fontsize=13.5, color=INK, x=0.5, y=0.985)
    fig.text(0.5, 0.947, f"vidéo de Markov (E27) : $\\xi^{{\\mu+1}}_i=\\xi^\\mu_i s_i$, "
             f"$P(s_i=-1)=(1-c)/2$   ·   $N$={n}, $P$={p_}, $\\beta$=20, $\\tau$=10, "
             f"$\\lambda$={lam}   ·   lecture identique pour les quatre bras : "
             f"$m=\\xi\\tanh(\\beta u)/N$",
             ha="center", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.935))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
