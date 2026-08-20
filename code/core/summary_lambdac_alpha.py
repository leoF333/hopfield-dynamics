"""Summary figure: lambda_c(alpha) from the multi-seed N=10000 fold study."""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
CHICAGO_S = str(CHICAGO)
# ---------------------------------------------------------------------------


D = CHICAGO_S + "/3_numerics/results/N=10000"
alphas = [0.01, 0.03, 0.05, 0.10]
mean, std = [], []
for a in alphas:
    d = np.load(os.path.join(D, f"fold_N10000_a{a}.npz"))
    mean.append(float(d["lam_c_mean"])); std.append(float(d["lam_c_std"]))
mean, std = np.array(mean), np.array(std)

la_a, la = [0.01, 0.05, 0.10], [0.403, 0.277, 0.246]   # legacy analytic, tau=10

fig, ax = plt.subplots(figsize=(7.5, 5))
clean = np.array([True, True, True, False])
ax.errorbar(np.array(alphas)[clean], mean[clean], yerr=std[clean], fmt="o-",
            ms=8, capsize=4, color="navy",
            label="numeric $N=10^4$ (clean, 5 seeds)")
ax.errorbar([0.10], [mean[3]], yerr=[std[3]], fmt="s", ms=9, capsize=4,
            color="crimson",
            label=r"$\alpha=0.10$ (near-capacity, 8 seeds, unreliable)")
ax.plot(la_a, la, "x--", ms=10, color="gray",
        label=r"legacy analytic $\lambda_c^{(1)}$")
ax.axvline(0.138, color="green", ls=":", lw=1.2,
           label=r"Hopfield capacity $\alpha_c\approx0.138$")
for a, m in zip(alphas, mean):
    ax.annotate(f"{m:.3f}", (a, m), textcoords="offset points",
                xytext=(8, 6), fontsize=8)
ax.set_xlabel(r"$\alpha$ (memory load)")
ax.set_ylabel(r"$\lambda_c$ (static-memory fold)")
ax.set_title(r"Critical mixing $\lambda_c(\alpha)$ for static-memory loss "
             r"($N=10^4,\ \tau=10,\ \beta=20$)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8.5); fig.tight_layout()
p = os.path.join(D, "lambda_c_vs_alpha.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print("saved", p)
print("lambda_c(alpha):",
      {a: f"{m:.3f}+/-{s:.3f}" for a, m, s in zip(alphas, mean, std)})
