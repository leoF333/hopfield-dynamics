"""
E20 figures: merge the main basin-fraction sweep (lam<=0.325) with the high-lam
completion run and render
  figN_basin_competition.png -- stacked basin fractions vs lam (memory/front/cycle/chaos/other)
  figO_metastability.png     -- front metastability: front basin fraction (log) and
                                the front:(front+memory) retrieval share vs lam.
Pure rendering from npz. English labels, dpi 175.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")
FIG = os.path.join(BASE, "figures")
LABELS = ["memory", "front", "cycle", "chaos", "other"]

def load(name):
    d = np.load(os.path.join(BASE, name), allow_pickle=True)
    return np.asarray(d["lam_grid"], float), np.asarray(d["fractions"], float)

lam1, fr1 = load("E20a_basin_fractions.npz")
try:
    lam2, fr2 = load("E20a_basin_fractions_hi.npz")
    lam = np.concatenate([lam1, lam2]); fr = np.vstack([fr1, fr2])
except FileNotFoundError:
    lam, fr = lam1, fr1
order = np.argsort(lam); lam = lam[order]; fr = fr[order]
F = {L: fr[:, i] for i, L in enumerate(LABELS)}

LAM_C, LAM_STAR = 0.2822, 0.327
colors = dict(memory="#2b6cb0", front="#e07b00", cycle="#2f9e44",
              chaos="#b03a5b", other="#9aa0a6")

# -------- figN: stacked basin fractions --------
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.stackplot(lam, [F[L] for L in LABELS],
             labels=[L for L in LABELS],
             colors=[colors[L] for L in LABELS], alpha=0.9)
ax.axvline(LAM_C, color="0.1", ls="--", lw=1.4)
ax.axvline(LAM_STAR, color="black", lw=1.7)
ax.text(LAM_C - 0.004, 0.55, r"$\lambda_c=0.282$", ha="right", va="center",
        rotation=90, fontsize=9.5, color="0.1")
ax.text(LAM_STAR + 0.004, 0.30, r"$\lambda^*\approx0.327$", ha="left", va="center",
        rotation=90, fontsize=9.5, color="black")
ax.set_xlim(lam.min(), lam.max()); ax.set_ylim(0, 1)
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("basin fraction (random ICs)")
ax.set_title("E20a -- basin competition of the delayed mixed Hopfield network\n"
             r"($\tau=10$, $N=2000$, $\alpha=0.05$, seed 42; K=250 ICs/$\lambda$)")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
fig.tight_layout()
fp = os.path.join(FIG, "figN_basin_competition.png")
fig.savefig(fp, dpi=175, bbox_inches="tight"); plt.close(fig); print("wrote", fp)

# -------- figO: front metastability --------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.0))
# left: front basin fraction (linear + log inset feel) vs lam, with memory & cycle for scale
a1.plot(lam, F["memory"], "o-", color=colors["memory"], label="memory basin")
a1.plot(lam, F["cycle"], "s-", color=colors["cycle"], label="cycle basin")
a1.plot(lam, F["front"], "D-", color=colors["front"], lw=2.4, ms=7,
        label="front basin (metastable)")
a1.axvline(LAM_C, color="0.15", ls="--", lw=1.1); a1.axvline(LAM_STAR, color="crimson", lw=1.3)
a1.set_xlabel(r"$\lambda$"); a1.set_ylabel("basin fraction")
a1.set_title("The pinned-front basin is a thin sliver at all $\\lambda$\n"
             "(front $\\leq$ 0.05 while memory/cycle are O(0.5))")
a1.legend(fontsize=9); a1.grid(alpha=.3)
# right: front basin RELATIVE to the dominant competing attractor -- stays <<1
winner = np.maximum.reduce([F["memory"], F["cycle"], F["chaos"]])
ratio = F["front"] / np.maximum(winner, 1e-9)
a2.semilogy(lam, np.maximum(ratio, 1e-3), "D-", color=colors["front"], lw=2.2, ms=7)
a2.axhline(1.0, color="0.3", ls="-", lw=1.0)
a2.text(0.5, 1.15, "equal footing (front would win above this)", fontsize=8.5, color="0.3")
a2.axvline(LAM_C, color="0.15", ls="--", lw=1.1)
a2.axvline(LAM_STAR, color="crimson", lw=1.3)
a2.text(LAM_C, 0.5, r"$\lambda_c$", ha="center", fontsize=9)
a2.text(LAM_STAR, 0.5, r"$\lambda^*$", ha="center", fontsize=9, color="crimson")
a2.set_ylim(1e-3, 3)
a2.set_xlabel(r"$\lambda$")
a2.set_ylabel(r"front basin / dominant basin (log)")
a2.set_title("Metastability index: front basin vs the winning basin\n"
             r"stays $\lesssim 0.1$ everywhere -- the weak-bond state never wins")
a2.grid(alpha=.3, which="both")
fig.suptitle("E20 -- pinned fronts are METASTABLE relative to memory patterns "
             r"($\tau=10$, seed 42)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fp = os.path.join(FIG, "figO_metastability.png")
fig.savefig(fp, dpi=175, bbox_inches="tight"); plt.close(fig); print("wrote", fp)

# -------- print merged table --------
print("\n  lam  " + "".join(f"{L:>9}" for L in LABELS))
for i, lm in enumerate(lam):
    print(f"{lm:6.3f}" + "".join(f"{F[L][i]:9.3f}" for L in LABELS))
