"""The invariant circle of the last beads, drawn from the recorded trajectories.

(a) lambda = 0.307629, the two last surviving beads (43 and 91): the closed
    invariant circle in the (m_43, m_91) plane -- two stable nodes, two index-1
    fold partners, four W^u branches, and the loop closes.
(b) the same circle unrolled: |m_nu(t)| along the long connection 43 -> 91,
    showing the travelling wave sweeping the whole ring through the ghosts of
    the dead beads.
(c) closure certificate: Gram distance to each node along both connections.
(d) lambda = 0.322629, the LAST bead alone: the forward branch tours the entire
    ring and returns to node_91 (d_G falls back to 1e-14) -- one saddle-node
    pair on one circle, the textbook SNIC.
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/2_cycle_rappel_snic/data/E34"
FIGS = ROOT / "results/2_cycle_rappel_snic/figures/E34"

C43, C91 = "#d62728", "#1f77b4"


def _latest(pat):
    hits = sorted(DATA.glob(pat))
    if not hits:
        raise SystemExit(f"missing {pat}")
    return hits[-1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=FIGS / "figE34_invariant_circle.png")
    args = p.parse_args(argv)

    d2 = np.load(_latest("E34_circle_traj_circle2_*.npz"))
    d1 = np.load(_latest("E34_circle_traj_circle1_*.npz"))
    k43, k91 = "lam0.307629_mu43", "lam0.307629_mu91"
    k1 = "lam0.322629_mu91"

    fig = plt.figure(figsize=(13.6, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- (a) the circle in the (m_43, m_91) plane ------------------------
    def seg(key, branch):
        return d2[f"{key}_{branch}_m"]

    pieces = [(k43, "plus", C43, "$W^u_+(\\mathrm{saddle}_{43})\\to$ node$_{91}$"),
              (k91, "plus", C91, "$W^u_+(\\mathrm{saddle}_{91})\\to$ node$_{43}$"),
              (k43, "minus", C43, "$W^u_-(\\mathrm{saddle}_{43})\\to$ node$_{43}$"),
              (k91, "minus", C91, "$W^u_-(\\mathrm{saddle}_{91})\\to$ node$_{91}$")]
    for key, br, col, lab in pieces:
        M = seg(key, br)
        style = "-" if br == "plus" else "--"
        ax_a.plot(M[:, 43], M[:, 91], style, color=col, lw=2.0 if br == "plus"
                  else 1.3, alpha=0.95, label=lab)
        j = len(M) // 2
        ax_a.annotate("", xy=(M[j + 1, 43], M[j + 1, 91]),
                      xytext=(M[j, 43], M[j, 91]),
                      arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
    for key, col, name in ((k43, C43, "43"), (k91, C91, "91")):
        mn, ms = d2[f"{key}_m_node"], d2[f"{key}_m_saddle"]
        ax_a.plot(mn[43], mn[91], "o", ms=12, color=col, mec="k", mew=1.2,
                  zorder=5)
        ax_a.plot(ms[43], ms[91], "X", ms=11, color=col, mec="k", mew=1.0,
                  zorder=5)
        ax_a.annotate(f"node$_{{{name}}}$", (mn[43], mn[91]),
                      textcoords="offset points", xytext=(10, -14), fontsize=10)
        ax_a.annotate(f"saddle$_{{{name}}}$", (ms[43], ms[91]),
                      textcoords="offset points", xytext=(10, 8), fontsize=9,
                      color=col)
    ax_a.annotate("passage through the ghosts of the\ndead beads:"
                  " both $m_{43}, m_{91}\\approx 0$\n"
                  "(the wave is elsewhere on the ring — panel b)",
                  xy=(0.03, 0.02), xytext=(0.30, 0.30), fontsize=8.2,
                  color="#444444",
                  arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0))
    ax_a.set_xlabel(r"$m_{43}$")
    ax_a.set_ylabel(r"$m_{91}$")
    ax_a.set_title("(a) $\\lambda$=0.307629, the two last beads:\n"
                   "the invariant circle CLOSES  (43 $\\to$ 91 $\\to$ 43)",
                   fontsize=11.5)
    ax_a.legend(fontsize=7.8, loc="upper right", framealpha=0.92)
    ax_a.grid(alpha=0.25)

    # ---- (b) kymograph of the long connection ---------------------------
    M = d2[f"{k43}_plus_m"]
    t = d2[f"{k43}_plus_t"]
    im = ax_b.imshow(np.abs(M).T, aspect="auto", origin="lower", cmap="magma",
                     extent=[t[0], t[-1], -0.5, M.shape[1] - 0.5], vmin=0,
                     vmax=1.0)
    ax_b.axhline(43, color=C43, lw=1.0, ls="--", alpha=0.8)
    ax_b.axhline(91, color=C91, lw=1.0, ls="--", alpha=0.8)
    ax_b.set_xlabel("time $t$")
    ax_b.set_ylabel(r"bead index $\nu$")
    ax_b.set_title("(b) the same connection unrolled: $|m_\\nu(t)|$\n"
                   "a wave sweeping the ring through the dead beads' ghosts",
                   fontsize=11.5)
    fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02, label=r"$|m_\nu|$")

    # ---- (c) closure certificate ----------------------------------------
    def dist_to(Mtraj, m_ref):
        return np.linalg.norm(Mtraj - m_ref[None, :], axis=1)

    m43n, m91n = d2[f"{k43}_m_node"], d2[f"{k91}_m_node"]
    for key, br, col, lab in ((k43, "plus", C43, "from saddle$_{43}$, forward"),
                              (k91, "plus", C91, "from saddle$_{91}$, forward")):
        M = seg(key, br)
        tt = d2[f"{key}_{br}_t"]
        tgt = m91n if key == k43 else m43n
        ax_c.semilogy(tt, np.maximum(dist_to(M, tgt), 1e-16), color=col, lw=1.8,
                      label=lab + r" $\to$ target node")
    ax_c.set_xlabel("time $t$")
    ax_c.set_ylabel(r"$\|m(t) - m_{\rm target}\|$")
    ax_c.set_title("(c) closure certificate: each forward branch reaches\n"
                   "the other node to machine precision", fontsize=11.5)
    ax_c.legend(fontsize=9, loc="upper right")
    ax_c.grid(alpha=0.25)

    # ---- (d) the last bead alone: one pair, one circle -------------------
    Mp = d1[f"{k1}_plus_m"]
    tp = d1[f"{k1}_plus_t"]
    lead = np.argmax(np.abs(Mp), axis=1)
    m91n1 = d1[f"{k1}_m_node"]
    ax_d.plot(tp, lead, ".", ms=1.8, color="#2ca02c", rasterized=True)
    ax_d.set_xlabel("time $t$")
    ax_d.set_ylabel(r"leading bead $\arg\max_\nu|m_\nu|$", color="#2ca02c")
    ax_d.tick_params(axis="y", labelcolor="#2ca02c")
    ax_d.set_ylim(-4, 103)
    ax_d2 = ax_d.twinx()
    ax_d2.semilogy(tp, np.maximum(np.linalg.norm(Mp - m91n1[None, :], axis=1),
                                  1e-16), color="#d62728", lw=1.2, alpha=0.85)
    ax_d2.set_ylabel(r"$\|m(t)-m_{\rm node_{91}}\|$", color="#d62728")
    ax_d2.tick_params(axis="y", labelcolor="#d62728")
    ax_d.set_title("(d) $\\lambda$=0.322629, bead 91 ALONE: the forward branch\n"
                   "tours the whole ring and returns to node$_{91}$",
                   fontsize=11.5)
    ax_d.grid(alpha=0.2)

    fig.suptitle("E34 — the invariant circle carrying the last saddle-node pairs "
                 "($N$=2000, $P$=100, seed 42, $\\beta$=20, $\\tau$=10)",
                 fontsize=12.8)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
