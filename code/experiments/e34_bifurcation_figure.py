"""Bifurcation diagram of the delayed mixed Hopfield ring, from the measured data.

(a) the LAST bead (91): node and its index-1 partner annihilate at lambda* --
    a single fold, no twin, and the closure test showed the pair sits ON a
    closed invariant circle. This is the SNIC that gives birth to the cycle.
(b) an ordinary bead (9): FOUR branches and TWO folds -- node/saddle_1 and
    twin/saddle_2 -- with a window where two stable states carrying the same
    pattern coexist. The ring connection is handed on through the twin.
(c) the ring as a whole: how many beads survive at each lambda (measured by
    warm-start continuation), with the lambdas where the chain was tested.
(d) the cycle branch: traversal time of the invariant circle below lambda* and
    period of the limit cycle above it.
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

LAM_STAR = 0.32763
STYLE = {"node": ("#1f77b4", "-", 2.2, "node (stable)"),
         "saddle1": ("#d62728", "--", 1.8, "saddle 1 (index 1)"),
         "twin": ("#2ca02c", "-", 2.2, "twin (stable)"),
         "saddle2": ("#ff7f0e", "--", 1.8, "saddle 2 (index 1)")}


def _load(pat, many=False):
    hits = sorted(DATA.glob(pat))
    if not hits:
        return [] if many else None
    if many:
        return [json.loads(h.read_text()) for h in hits]
    return json.loads(hits[-1].read_text())


def _branch_panel(ax, d, title, annotate_star=False, xlim=None):
    for name in ("node", "saddle1", "twin", "saddle2"):
        b = d["branches"].get(name)
        if not b:
            continue
        col, ls, lw, lab = STYLE[name]
        ax.plot(b["lam"], b["m_mu"], ls, color=col, lw=lw, label=lab)
        # mark the end of each branch: that is where the fold is
        ax.plot([b["lam"][-1]], [b["m_mu"][-1]], "o", ms=5, color=col,
                mec="k", mew=0.8, zorder=5)
    # fold = where a stable and an unstable branch terminate together
    ends = {n: (d["branches"][n]["lam"][-1], d["branches"][n]["m_mu"][-1])
            for n in d["branches"]}
    pairs = [("node", "saddle1"), ("twin", "saddle2")]
    for a, b in pairs:
        if a in ends and b in ends and abs(ends[a][0] - ends[b][0]) < 5e-4:
            lam_f = 0.5 * (ends[a][0] + ends[b][0])
            ax.axvline(lam_f, color="#888888", lw=1.0, ls=":", zorder=1)
            ax.annotate(f"fold\n{lam_f:.5f}",
                        (lam_f, 0.5 * (ends[a][1] + ends[b][1])),
                        textcoords="offset points", xytext=(-46, 0),
                        fontsize=8.2, color="#444444", ha="center")
    if annotate_star:
        ax.axvline(LAM_STAR, color="#9467bd", lw=1.4, alpha=0.8)
        ax.annotate(r"$\lambda^*$", (LAM_STAR, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(4, 12), fontsize=11,
                    color="#9467bd")
    # shade the window where two stable branches coexist
    st = [d["branches"][n] for n in ("node", "twin") if n in d["branches"]]
    if len(st) == 2:
        lo = max(min(st[0]["lam"]), min(st[1]["lam"]))
        hi = min(max(st[0]["lam"]), max(st[1]["lam"]))
        if hi > lo:
            ax.axvspan(lo, hi, color="#2ca02c", alpha=0.10, zorder=0)
            ax.annotate("two stable states\nwith the same pattern",
                        (0.5 * (lo + hi), ax.get_ylim()[0]),
                        textcoords="offset points", xytext=(0, 16),
                        fontsize=8, color="#2ca02c", ha="center")
    if xlim:
        ax.set_xlim(*xlim)
        b = d["branches"].get("node")
        if b and min(b["lam"]) < xlim[0]:
            ax.annotate(f"the node branch continues down to "
                        f"$\\lambda$={min(b['lam']):.3f}",
                        (0.02, 0.97), xycoords="axes fraction", fontsize=8,
                        color="#1f77b4", va="top")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$m_\mu$  (overlap with its own pattern)")
    ax.set_title(title, fontsize=11.5)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=FIGS / "figE34_bifurcation.png")
    args = p.parse_args(argv)

    d91 = _load("E34_branch_diagram_mu91_*.json")
    d9 = _load("E34_branch_diagram_mu9_*.json")
    ladder = _load("E34_alive_ladder_*.json")
    closures = _load("E34_snic_closure_mu91_*.json", many=True)
    periods = _load("E34_cycle_uniqueness_lam*periodscan*.json", many=True)
    merged = _load("E34_cycle_uniqueness_MERGED_lam*.json", many=True)
    chains = _load("E34_chain_lam0.3*.json", many=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    if d91:
        _branch_panel(ax_a, d91,
                      "(a) bead 91, the LAST to die: ONE fold, no twin\n"
                      "the pair annihilates at $\\lambda^*$, on a closed "
                      "invariant circle $\\Rightarrow$ SNIC", annotate_star=True)
    if d9:
        _branch_panel(ax_b, d9,
                      "(b) bead 9, an ordinary bead: TWO folds\n"
                      "node/saddle$_1$ and twin/saddle$_2$ — the ring "
                      "connection passes through the twin",
                      xlim=(0.2995, 0.3075))

    # ---- (c) how many beads survive --------------------------------------
    if ladder:
        iv = {int(r["mu"]): [tuple(x) for x in r["intervals"]]
              for r in ladder["rows"] if r.get("status") == "ok"}
        lam_g = np.linspace(0.19, 0.335, 600)
        n_alive = [sum(1 for v in iv.values()
                       if any(a <= L <= b for a, b in v)) for L in lam_g]
        ax_c.plot(lam_g, n_alive, "-", color="#1f77b4", lw=2.0)
        ax_c.fill_between(lam_g, 0, n_alive, color="#1f77b4", alpha=0.15)
        ax_c.axvline(LAM_STAR, color="#9467bd", lw=1.4)
        ax_c.annotate(r"$\lambda^*=%.5f$" % LAM_STAR, (LAM_STAR, 60),
                      textcoords="offset points", xytext=(-96, 0), fontsize=9,
                      color="#9467bd")
        tested = {}
        for c in chains:
            lam = float(c["lam"])
            ok = sum(1 for r in c["rows"] if r.get("status") == "brick_positive")
            tested.setdefault(lam, [0, 0])
            tested[lam][0] += ok
            tested[lam][1] += len(c["rows"])
        for i, (lam, (ok, tot)) in enumerate(sorted(tested.items())):
            n = sum(1 for v in iv.values() if any(a <= lam <= b for a, b in v))
            ax_c.plot([lam], [n], "v", ms=9, color="#d62728", zorder=5)
            ax_c.annotate(f"$\\lambda$={lam:.4f}\n{ok}/{tot} links",
                          (lam, n), textcoords="offset points",
                          xytext=(-4, 16 + 26 * (i % 3)), fontsize=7.6,
                          color="#d62728", ha="right",
                          arrowprops=dict(arrowstyle="-", color="#d62728",
                                          lw=0.7))
        ax_c.set_xlabel(r"$\lambda$")
        ax_c.set_ylabel("number of beads carrying a stable memory")
        ax_c.set_title("(c) the ring empties one bead at a time\n"
                       "red markers: $\\lambda$ where the chain was tested "
                       "(certified links / tested)", fontsize=11.5)
        ax_c.set_xlim(0.19, 0.335)
        ax_c.grid(alpha=0.25)

    # ---- (d) circle traversal time and cycle period ----------------------
    lam_b, T_b = [], []
    for c in closures:
        if c.get("tour_times"):
            lam_b.append(float(c["lam"]))
            T_b.append(float(c["tour_times"][0]))
    lam_a, T_a = [], []
    for j in periods:
        for r in j["rows"]:
            if np.isfinite(r.get("T_refined", np.nan)):
                lam_a.append(float(r["lam"]))
                T_a.append(float(r["T_refined"]))
    for j in merged:
        if j.get("period_mean"):
            lam_a.append(float(j["lam"]))
            T_a.append(float(j["period_mean"]))
    if lam_b:
        o = np.argsort(lam_b)
        ax_d.plot(np.array(lam_b)[o], np.array(T_b)[o], "s--", color="#2ca02c",
                  lw=1.6, ms=8, label="below $\\lambda^*$: one tour of the\n"
                                      "invariant circle (transient)")
    if lam_a:
        o = np.argsort(lam_a)
        la, Ta = np.array(lam_a)[o], np.array(T_a)[o]
        ax_d.plot(la, Ta, "o-", color="#1f77b4", lw=1.8, ms=7,
                  label="above $\\lambda^*$: period of the limit cycle")
        m = la > LAM_STAR + 1e-6
        if m.sum() >= 3:
            x = 1.0 / np.sqrt(la[m] - LAM_STAR)
            A = np.polyfit(x, Ta[m], 1)
            xx = np.linspace(la[m].min(), la[m].max(), 200)
            ax_d.plot(xx, A[0] / np.sqrt(xx - LAM_STAR) + A[1], ":",
                      color="#666666", lw=1.6,
                      label=r"SNIC law $a/\sqrt{\lambda-\lambda^*}+b$"
                            f"\n(a={A[0]:.2f}, b={A[1]:.0f})")
    ax_d.axvline(LAM_STAR, color="#9467bd", lw=1.4)
    ax_d.set_xlabel(r"$\lambda$")
    ax_d.set_ylabel("time to go once around the ring")
    ax_d.set_title("(d) the circle survives the bifurcation\n"
                   "its traversal time becomes the cycle period at "
                   "$\\lambda^*$", fontsize=11.5)
    ax_d.legend(fontsize=8.5, loc="upper right")
    ax_d.grid(alpha=0.25)

    fig.suptitle("E34 — bifurcation structure of the delayed mixed Hopfield ring "
                 "($N$=2000, $P$=100, seed 42, $\\beta$=20, $\\tau$=10)",
                 fontsize=12.8)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
