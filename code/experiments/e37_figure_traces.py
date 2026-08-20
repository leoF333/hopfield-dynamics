"""E37 figure 2 -- the same contrast, read directly on the trajectories.

Panels (a),(b) show the recall wave itself: the overlap m_nu(t) as a field over
(time, frame index). A clean sequential recall is a straight diagonal band one
frame wide; a failed one is a smeared, wandering, or stalled band. Colour is a
DIVERGING ramp with a neutral midpoint at zero because overlaps carry a sign and
an anti-correlated frame is not the same thing as an absent one.

Panel (c) is the instantaneous profile at one peak: the shape of what the network
is actually holding. Panel (d) is the leader index over time for both arms, which
is what the forward-fraction statistic counts.
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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/6_motifs_correles/data"
FIGS = ROOT / "results/6_motifs_correles/figures"

K_COLOR = {"JH_KH": "#2a78d6", "JP_KP": "#eb6834"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
TITLE = {"JH_KH": "J Hebb · K Hebb — le système usuel",
         "JP_KP": "J moderne · K moderne"}

# diverging blue<->red, neutral gray midpoint (palette reference instance)
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "e37_div", ["#104281", "#256abf", "#86b6ef", "#f0efec",
                "#f0a3a3", "#d03b3b", "#8f1f1f"])


def _find(c: float, tag: str):
    npz = sorted(glob.glob(str(DATA / f"E37_modernK_c{c:.2f}_*_{tag}_*.npz")))
    js = sorted(glob.glob(str(DATA / f"E37_modernK_c{c:.2f}_*_{tag}_*.json")))
    if not npz:
        raise SystemExit(f"no trace npz for c={c} tag={tag}")
    return np.load(npz[-1]), json.loads(Path(js[-1]).read_text())


def _style(ax, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11.5, loc="left",
                 pad=14 if subtitle else 6)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.015), xycoords="axes fraction",
                    color=MUTED, fontsize=9, va="bottom")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--c", type=float, default=0.6)
    p.add_argument("--tag", default="trace")
    p.add_argument("--window", type=float, default=700.0)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    d, meta = _find(args.c, args.tag)
    out = args.output or FIGS / f"figE37_traces_c{args.c:.2f}.png"
    rows = {r["arm"]: r for r in meta["rows"]}

    fig = plt.figure(figsize=(13.0, 9.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42,
                          wspace=0.24)
    ax = {"JH_KH": fig.add_subplot(gs[0, 0]), "JP_KP": fig.add_subplot(gs[0, 1])}
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    vmax = 0.0
    seg = {}
    for arm in ("JH_KH", "JP_KP"):
        t, m = d[f"t_{arm}"], d[f"m_{arm}"]
        w = t >= t[-1] - args.window
        seg[arm] = (t[w], m[w])
        vmax = max(vmax, float(np.abs(m[w]).max()))

    for arm in ("JH_KH", "JP_KP"):
        t, m = seg[arm]
        a = ax[arm]
        im = a.imshow(m.T, aspect="auto", origin="lower", cmap=DIVERGING,
                      vmin=-vmax, vmax=vmax,
                      extent=(t[0], t[-1], -0.5, m.shape[1] - 0.5),
                      interpolation="nearest")
        r = rows[arm]
        _style(a, "temps  $t$", "indice de trame  $\\nu$",
               f"({'a' if arm == 'JH_KH' else 'b'}) {TITLE[arm]}",
               f"couverture {r['coverage_fraction']:.2f} · "
               f"avant {r['forward_fraction']:.2f} · "
               f"largeur {r['packet_width_median']:.0f} trames")
        cb = fig.colorbar(im, ax=a, pad=0.02, fraction=0.046)
        cb.set_label("recouvrement  $m_\\nu$", color=INK2, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(AXIS)

    # ---- (c) the instantaneous profile, at each arm's own strongest peak -----
    for arm in ("JH_KH", "JP_KP"):
        t, m = seg[arm]
        lead = np.argmax(m, axis=1)
        i = int(np.argmax(m[np.arange(len(m)), lead]))
        nu0 = int(lead[i])
        off = np.arange(m.shape[1]) - nu0
        off = (off + m.shape[1] // 2) % m.shape[1] - m.shape[1] // 2
        o = np.argsort(off)
        a_ = ax_c
        a_.plot(off[o], m[i][o], "-", color=K_COLOR[arm], lw=2.0,
                label=TITLE[arm], solid_capstyle="round")
        a_.plot([0], [m[i][nu0]], "o", ms=8, color=K_COLOR[arm], zorder=4)
    ax_c.axhline(0.0, color=AXIS, lw=1.0)
    ax_c.set_xlim(-14, 14)
    ax_c.set_ylim(top=1.28)      # headroom so the legend never sits on the peak
    ax_c.grid(True, color=GRID, lw=0.8)
    ax_c.set_axisbelow(True)
    _style(ax_c, "décalage par rapport à la trame lue  $\\nu-\\nu_{\\rm meneur}$",
           "recouvrement  $m_\\nu$", "(c) Ce que le réseau tient à un instant",
           "profil instantané au pic le plus haut de chaque bras")
    leg = ax_c.legend(frameon=False, fontsize=9.5, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK2)

    # ---- (d) the leader index over time -------------------------------------
    for arm in ("JH_KH", "JP_KP"):
        t, m = seg[arm]
        lead = np.argmax(m, axis=1).astype(float)
        # unwrap the ring so a clean tour reads as a straight ramp
        u = np.copy(lead)
        step = np.diff(lead)
        u[1:] = lead[0] + np.cumsum(np.where(step < -m.shape[1] / 2,
                                             step + m.shape[1],
                                             np.where(step > m.shape[1] / 2,
                                                      step - m.shape[1], step)))
        ax_d.plot(t, u, "-", color=K_COLOR[arm], lw=2.0, label=TITLE[arm],
                  solid_capstyle="round")
    ax_d.grid(True, color=GRID, lw=0.8)
    ax_d.set_axisbelow(True)
    _style(ax_d, "temps  $t$", "trame lue, déroulée (cumulée)",
           "(d) La progression le long de la séquence",
           "une droite montante = un défilement régulier vers l'avant")
    leg = ax_d.legend(frameon=False, fontsize=9.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(INK2)

    fig.suptitle(f"E37 — vidéo de Markov à corrélation $c$={args.c:g} : "
                 f"la même dynamique, lue de deux façons", fontsize=13.5,
                 color=INK, y=0.98)
    fig.text(0.5, 0.943, f"$N$={meta['n']}, $P$={meta['p']}, $\\beta$=20, "
             f"$\\tau$=10, $\\lambda$={meta['lam']} · fenêtre de "
             f"{args.window:g} unités de temps en fin de simulation",
             ha="center", color=MUTED, fontsize=9.5)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.97)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
