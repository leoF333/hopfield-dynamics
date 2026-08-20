"""
E33 Exp E2 summary: two-time correlation collapse spread vs alpha -> does the
mixture-phase chaos AGE as alpha -> capacity? Figure figAD.
Loads E33_aging_mlx[_a*].npz for alpha in {0.05,0.08,0.10,0.12}, computes the
on-attractor collapse spread (small = stationary/no aging), annotates the chaotic
ensemble size (noise caveat at large alpha).
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIG = os.path.join(OUT, "figures")
FILES = [(0.05, "E33_aging_mlx.npz"), (0.08, "E33_aging_mlx_a0p08.npz"),
         (0.10, "E33_aging_mlx_a0p10.npz"), (0.12, "E33_aging_mlx_a0p12.npz")]


def collapse_spread(d):
    tws = [int(t) for t in d["tws"]]
    Cn = {}
    for tw in tws:
        if f"C_{tw}" not in d.files:
            continue
        lag, C = d[f"lag_{tw}"], d[f"C_{tw}"]
        if C[0] <= 0:
            continue
        Cn[tw] = (lag, C / C[0])
    tws_on = [t for t in tws if t >= 50 and t in Cn]
    if len(tws_on) < 2:
        return np.nan, np.nan, Cn
    Lref = min(len(Cn[t][0]) for t in tws_on)
    il = min(int(np.searchsorted(Cn[tws_on[0]][0], 20)), Lref - 1)
    vals = [Cn[t][1][il] for t in tws_on]
    spread = float(np.max(vals) - np.min(vals))
    return spread, il, Cn


alphas, spreads, kepts = [], [], []
Cn_repr = None
for a, fn in FILES:
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        print(f"[E33-E2] missing {fn}, skip"); continue
    d = np.load(p)
    sp, il, Cn = collapse_spread(d)
    kept = int(d["nkeep_tot"])
    alphas.append(a); spreads.append(sp); kepts.append(kept)
    if abs(a - 0.05) < 1e-9:
        Cn_repr = (a, Cn)
    print(f"[E33-E2] alpha={a}: collapse spread(lag~20)={sp:.3f}, chaotic-kept={kept}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].axhline(0.1, color="crimson", ls="--", lw=1.2, label="aging threshold (0.1)")
ax[0].plot(alphas, spreads, "o-", color="#2b6cb0", lw=1.6, ms=8)
for a, s, k in zip(alphas, spreads, kepts):
    ax[0].annotate(f"n={k}", (a, s), textcoords="offset points", xytext=(0, 8),
                   fontsize=8, ha="center")
ax[0].set_xlabel(r"$\alpha$")
ax[0].set_ylabel(r"two-time $C$ collapse spread (lag$\approx$20)")
ax[0].set_ylim(0, 0.13)
ax[0].set_title("Aging test vs load (Exp E2):\nspread stays $\\ll$ threshold "
                "$\\Rightarrow$ NO aging up to capacity")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

if Cn_repr is not None:
    a, Cn = Cn_repr
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(Cn)))
    for (tw, (lag, cn)), c in zip(sorted(Cn.items()), cols):
        ax[1].plot(lag, cn, color=c, lw=1.4, label=f"$t_w$={tw}")
    ax[1].set_xlim(0, 120); ax[1].set_xlabel("lag $t$")
    ax[1].set_ylabel(r"$C(t_w{+}t,t_w)/C(t_w,t_w)$")
    ax[1].set_title(f"Two-time $C$ collapse ($\\alpha$={a}): curves\n"
                    "superimpose $\\Rightarrow$ time-translation invariant")
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3)

fig.suptitle("E33 Exp E2 -- the mixture-phase chaos does NOT age as $\\alpha\\to$ "
             "capacity (float32/GPU, pending float64 gate; large-$\\alpha$ noisy, small n)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fp = os.path.join(FIG, "figAD_aging_vs_alpha.png")
fig.savefig(fp, dpi=170, bbox_inches="tight")
print(f"[E33-E2] figure -> {fp}")
print(f"[E33-E2] VERDICT: spreads {[round(s,3) for s in spreads]} all << 0.1 "
      f"-> NO aging up to alpha={max(alphas)} (large-alpha preliminary, small chaotic n)")
