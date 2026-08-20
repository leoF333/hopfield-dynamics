"""Figure: is the SNIC carried by ONE invariant circle for all the saddle-nodes?

Panels (dpi 170, English labels), built only from artefacts on disk:

  (a) mu=91, the last surviving bead: the forward W^u branch tours the whole ring
      and returns to node_91 -- the invariant circle CLOSES (delta=0.005, 0.010).
      ET1'' had stopped at t=1200, ~800 t.u. short of the return, which is why
      the cell then read "non-stationary, far from every node".
  (b) the same shooting at delta=0.020, where a second bead (43) is alive: the
      branch stops on node_43 instead -- the circle now carries more than one
      saddle-node pair, which is exactly what test B has to chain up.
  (c) measured survival threshold of every bead (warm-start ladder) against the
      E24 table, coloured by reading: table consistent / table early / branch
      fragmented into several stable windows.
  (d) test B: the forward map mu -> landing at a fixed lambda, drawn on the ring
      over the corrected alive set, with the closure verdict.
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
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/2_cycle_rappel_snic/data/E34"
FIGS = ROOT / "results/2_cycle_rappel_snic/figures/E34"

COL = {"table_consistent": "#999999", "table_early": "#ff7f0e",
       "S_shaped_branch": "#d62728", "table_late": "#1f77b4"}
LBL = {"table_consistent": "table consistent", "table_early": "table early",
       "S_shaped_branch": "branch fragmented (>1 stable window)",
       "table_late": "table late"}


def _latest(pat):
    hits = sorted(DATA.glob(pat))
    return hits[-1] if hits else None


def _closure(delta):
    p = _latest(f"E34_snic_closure_mu91_d{delta}_*.npz")
    j = _latest(f"E34_snic_closure_mu91_d{delta}_*.json")
    if p is None or j is None:
        return None
    d = np.load(p, allow_pickle=True)
    return dict(t=d["t"], lead=d["lead"], d_node=d["d_node"],
                rec=json.loads(j.read_text()))


def _panel_shoot(ax, cl, title, ring_color="#1f77b4"):
    t, lead, dn = cl["t"], cl["lead"], cl["d_node"]
    ax.plot(t, lead, ".", ms=1.6, color=ring_color, rasterized=True)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"leading bead $\arg\max_\nu |m_\nu|$", color=ring_color)
    ax.tick_params(axis="y", labelcolor=ring_color)
    ax.set_ylim(-4, 103)
    ax2 = ax.twinx()
    ax2.semilogy(t, np.maximum(dn, 1e-16), lw=1.1, color="#d62728", alpha=0.8)
    ax2.set_ylabel(r"$d_G$ to node$_{91}$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    rec = cl["rec"]
    if rec.get("tour_times"):
        for tt in rec["tour_times"]:
            ax.axvline(tt, color="#2ca02c", ls="--", lw=1.0)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)
    return ax2


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chain", type=Path, default=None,
                   help="chain-closure JSON for panel (d)")
    p.add_argument("--output", type=Path,
                   default=FIGS / "figE34_snic_closure.png")
    args = p.parse_args(argv)

    c005, c010, c020 = _closure("0.005"), _closure("0.01"), _closure("0.02")
    lad = _latest("E34_alive_ladder_*.json")
    ladder = json.loads(lad.read_text()) if lad else None
    chain_path = args.chain or _latest("E34_chain_lam*.json")
    chain = json.loads(chain_path.read_text()) if chain_path else None

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # --- (a) the circle closes -------------------------------------------
    if c005 is not None:
        rec = c005["rec"]
        tour = rec["tour_times"][0] if rec.get("tour_times") else float("nan")
        _panel_shoot(ax_a, c005,
                     "(a) $\\mu$=91, $\\delta$=0.005: forward $W^u$ tours the ring "
                     f"and RETURNS\ntour {tour:.0f} t.u., "
                     f"$d_G$(node$_{{91}}$) final {rec['d_node_final']:.1e} "
                     "— invariant circle CLOSED")
        if c010 is not None:
            ax_a.plot(c010["t"], c010["lead"], ".", ms=1.0, color="#7f7f7f",
                      alpha=0.55, rasterized=True)
            ax_a.text(0.02, 0.06,
                      f"grey: $\\delta$=0.010 (tour "
                      f"{c010['rec']['tour_times'][0]:.0f} t.u., also closes)",
                      transform=ax_a.transAxes, fontsize=8.5, color="#555555")
    else:
        ax_a.text(0.5, 0.5, "closure run missing", ha="center")

    # --- (b) delta = 0.020: lands on another bead -------------------------
    if c020 is not None:
        rec = c020["rec"]
        _panel_shoot(ax_b, c020,
                     "(b) $\\mu$=91, $\\delta$=0.020: a second bead is alive\n"
                     f"the branch STOPS on node$_{{{rec.get('settled_on')}}}$ "
                     f"($d_G$={rec.get('settled_dist', float('nan')):.1e}) — the "
                     "circle carries several pairs", "#9467bd")
    else:
        ax_b.text(0.5, 0.5, "delta=0.020 closure run missing", ha="center")

    # --- (c) measured survival vs table ----------------------------------
    if ladder is not None:
        rows = [r for r in ladder["rows"] if r.get("status") == "ok"]
        for kind in ("table_consistent", "table_early", "S_shaped_branch",
                     "table_late"):
            sel = [r for r in rows if r.get("reading") == kind]
            if not sel:
                continue
            ax_c.scatter([r["lam_c_table"] for r in sel],
                         [r["lam_survive"] for r in sel],
                         s=[18 + 12 * (r.get("n_intervals", 1) - 1) for r in sel],
                         color=COL[kind], label=f"{LBL[kind]} ({len(sel)})",
                         alpha=0.85, zorder=3)
        lo = min(r["lam_c_table"] for r in rows)
        hi = max(r["lam_survive"] for r in rows)
        ax_c.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5, zorder=1)
        ax_c.set_xlabel(r"tabulated $\lambda_c(\mu)$  (E24)")
        ax_c.set_ylabel(r"measured survival $\lambda$  (warm-start ladder)")
        ax_c.set_title("(c) the memory branches are FRAGMENTED\n"
                       f"marker size = number of stable windows; "
                       f"{sum(1 for r in rows if r.get('n_intervals', 1) > 1)}"
                       f"/{len(rows)} beads have more than one", fontsize=11)
        ax_c.legend(fontsize=8.5, loc="upper left")
        ax_c.grid(alpha=0.25)
    else:
        ax_c.text(0.5, 0.5, "ladder missing", ha="center")

    # --- (d) the forward map on the ring at fixed lambda -----------------
    if chain is not None:
        alive = chain["alive"]
        fmap = {int(r["mu"]): r.get("forward_bead") for r in chain["rows"]
                if r.get("status") == "brick_positive"}
        th = {m: 2 * np.pi * m / 100.0 for m in range(100)}
        ax_d.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#dddddd",
                                  lw=1.2))
        for m in range(100):
            x, y = np.cos(th[m]), np.sin(th[m])
            ax_d.plot([x], [y], ".", ms=2.0, color="#dddddd", zorder=1)
        for m in alive:
            x, y = np.cos(th[m]), np.sin(th[m])
            ax_d.plot([x], [y], "o", ms=7, color="#2ca02c", zorder=3)
            ax_d.annotate(str(m), (1.09 * x, 1.09 * y), ha="center",
                          va="center", fontsize=8.5)
        for m, nxt in fmap.items():
            if nxt is None:
                continue
            x0, y0 = 0.93 * np.cos(th[m]), 0.93 * np.sin(th[m])
            x1, y1 = 0.93 * np.cos(th[nxt]), 0.93 * np.sin(th[nxt])
            ax_d.annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                          lw=1.6, connectionstyle="arc3,rad=0.22"),
                          zorder=2)
        ax_d.set_xlim(-1.25, 1.25); ax_d.set_ylim(-1.25, 1.25)
        ax_d.set_aspect("equal"); ax_d.axis("off")
        verdict = chain["verdict"].get("closure", "")
        ax_d.set_title(f"(d) test B at $\\lambda$={chain['lam']:.6f}: forward map "
                       f"over the {len(alive)} alive beads\n{verdict}",
                       fontsize=11)
    else:
        ax_d.text(0.5, 0.5, "chain-closure run missing", ha="center")
        ax_d.axis("off")

    fig.suptitle("E34 — one invariant circle for all the saddle-nodes? "
                 "$N$=2000, $P$=100, seed 42, "
                 r"$\beta$=20, $\tau$=10", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.952))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
