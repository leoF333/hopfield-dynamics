"""
E32 analysis -- (alpha, lambda) basin phase diagram from E32_basins_*.npz.
Heatmaps of static/cycle/chaos fractions + the cycle-fraction pinch-off
alpha_cycle(lambda) = where f_cycle drops below 0.05 as alpha grows (the
memory->chaos route: the cycle region closing). Figure figAB.
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
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "E32_basins_N2000_mlx.npz")
d = np.load(fn, allow_pickle=True)
grid = d["grid"]                      # (nA, nL, nS, 4) static,cycle,chaos,other
alphas, lams = d["alphas"], d["lams"]
order = [str(x) for x in d["order"]]
fmean = grid.mean(axis=2)             # seed-avg (nA, nL, 4)
fstd = grid.std(axis=2)
iC = order.index("cycle"); iH = order.index("chaos"); iS = order.index("static")

print(f"[E32an] {fn}: alphas={list(alphas)}, lams={list(lams)}, "
      f"seeds={list(d['seeds'])}, K={int(d['K'])}")
print("[E32an] cycle fraction (seed-avg):")
print("   a\\lam " + " ".join(f"{l:5.2f}" for l in lams))
for ia, a in enumerate(alphas):
    print(f"   {a:.3f} " + " ".join(f"{fmean[ia,il,iC]:5.2f}" for il in range(len(lams))))

# pinch-off: for each lam, the alpha where cycle fraction first drops below 0.05
print("\n[E32an] cycle pinch-off alpha_cycle(lam) (f_cycle<0.05):")
for il, l in enumerate(lams):
    below = np.where(fmean[:, il, iC] < 0.05)[0]
    ac = alphas[below[0]] if len(below) and fmean[0, il, iC] >= 0.05 else (
        alphas[0] if fmean[0, il, iC] < 0.05 else np.nan)
    if l >= 0.35:
        print(f"   lam={l:.2f}: alpha_cycle={ac if np.isfinite(ac) else '>max'}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for k, (idx, name, cmap) in enumerate([(iS, "static (memory+front)", "Greens"),
                                       (iC, "cycle (recall)", "Blues"),
                                       (iH, "chaos", "Reds")]):
    im = ax[k].imshow(fmean[:, :, idx], origin="lower", aspect="auto",
                      vmin=0, vmax=1, cmap=cmap,
                      extent=[lams[0], lams[-1], alphas[0], alphas[-1]])
    ax[k].set_xlabel(r"$\lambda$"); ax[k].set_ylabel(r"$\alpha$")
    ax[k].set_title(f"{name} fraction")
    fig.colorbar(im, ax=ax[k], shrink=0.85)
    # overlay the cycle=0.05 contour on all panels for reference
    try:
        ax[k].contour(lams, alphas, fmean[:, :, iC], levels=[0.05, 0.5],
                      colors="k", linewidths=0.8, linestyles=[":", "-"])
    except Exception:
        pass
fig.suptitle(r"E32 -- $(\alpha,\lambda)$ basin phase diagram (MLX float32 GPU, "
             f"K={int(d['K'])}, {len(d['seeds'])} seeds): the cycle region closes as "
             r"$\alpha\to$ capacity (memory$\to$chaos)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fp = os.path.join(FIG, "figAB_alpha_lambda_basins.png")
fig.savefig(fp, dpi=170, bbox_inches="tight")
print(f"[E32an] figure -> {fp}")
