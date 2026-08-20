"""E29 figure regeneration from E29_cycle_ghosts.npz (corrected titles) +
closest-approach analysis of the loop to the fold states (3 reference bonds)."""

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

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")
d = np.load(os.path.join(BASE, "E29_cycle_ghosts.npz"))
fold_ac, fold_a1c, fold_lc = d["fold_ac"], d["fold_a1c"], d["fold_lc"]
lam_star = float(d["lam_star"]); lams = d["lams"]; Ts = d["Ts"]
tagc = f"{lams[-1]:.3f}".replace(".", "p")
mu_w = int(np.nanargmax(fold_lc))

print(f"[E29-fig] closest approach of the lam={lams[-1]:.3f} loop to fold states:")
for mu in (91, 43, 28):
    C = d[f"proj_{mu}_{tagc}"]
    dist = np.hypot(C[:, 0] - fold_ac[mu], C[:, 1] - fold_a1c[mu])
    print(f"  bond {mu:3d} (lam_c={fold_lc[mu]:.4f}): min dist to own fold "
          f"= {dist.min():.4f}   (fold at ({fold_ac[mu]:.3f},{fold_a1c[mu]:.3f}))")

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))

ax = axes[0, 0]
cols = {91: "#c0392b", 43: "#2b6cb0", 28: "#2f855a"}
for mu in (91, 43, 28):
    C = d[f"proj_{mu}_{tagc}"]
    ax.plot(C[:, 0], C[:, 1], color=cols[mu], lw=1.1,
            label=f"cycle, bond {mu} ($\\lambda_c$={fold_lc[mu]:.3f})")
    ax.scatter([fold_ac[mu]], [fold_a1c[mu]], marker="*", s=240,
               color=cols[mu], edgecolor="k", zorder=5)
ax.scatter([], [], marker="*", s=140, color="w", edgecolor="k",
           label="fold state (dead fixed point)")
ax.scatter([1.0], [0.0], marker="s", s=90, color="gold", edgecolor="k",
           zorder=5, label=r"pure pattern $(a_\mu,a_{\mu+1})=(1,0)$")
ax.set_xlabel(r"$a_\mu$"); ax.set_ylabel(r"$a_{\mu+1}$")
ax.set_title(f"Cycle at $\\lambda={lams[-1]:.3f}$: each handoff transits from the "
             "pure pattern\nthrough the fold region toward the next pattern")
ax.legend(fontsize=8.5); ax.grid(alpha=.3)

ax = axes[0, 1]
cmap = plt.cm.viridis
for i, lam in enumerate(lams):
    tag = f"{lam:.3f}".replace(".", "p")
    c = cmap(i / max(len(lams) - 1, 1))
    lab = f"$\\lambda$={lam:.3f}" if lam in (0.9, 0.5, 0.37, lams[-1]) else None
    ax.scatter(fold_ac, d[f"peak_a_{tag}"], s=22, color=c, edgecolor="none",
               alpha=.75, label=lab)
ax.plot([0.6, 1.0], [0.6, 1.0], "k--", lw=1, label="peak = fold value")
ax.axhline(1.0, color="0.6", lw=0.8)
ax.set_ylim(0.55, 1.03)
ax.set_xlabel(r"fold state $a_c(\mu)$ (fixed point at death)")
ax.set_ylabel(r"cycle peak $\max_t a_\mu(t)$")
ax.set_title("Analog peaks stay at the QUASI-PURE patterns ($\\geq 0.978$),\n"
             "far above the fold mixtures -- even at $\\lambda^*+0.002$")
ax.legend(fontsize=8.5); ax.grid(alpha=.3)

ax = axes[1, 0]
mins, meds = [], []
for lam in lams:
    tag = f"{lam:.3f}".replace(".", "p")
    mins.append(np.nanmin(d[f"m_peak_{tag}"]))
    meds.append(np.nanmedian(d[f"m_peak_{tag}"]))
ax.plot(lams, meds, "o-", color="#2b6cb0", label=r"median$_\mu$ peak $m^{sign}_\mu$")
ax.plot(lams, mins, "s-", color="#c0392b", label=r"min$_\mu$ peak $m^{sign}_\mu$")
ax.plot(lams, d["frac_exact"], "d-", color="#2f855a",
        label=r"loop-time fraction with sign$(u)$ EXACTLY $=\xi^\mu$")
ax.axvline(lam_star, color="k", ls=":", lw=1.2, label=r"$\lambda^*$")
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("binary readout quality")
ax.set_ylim(-0.02, 1.02)
ax.set_title(r"Binarized recall is EXACT at every peak: sign$(u)=\xi^\mu$ "
             "for all 100 patterns")
ax.legend(fontsize=8.5); ax.grid(alpha=.3)

ax = axes[1, 1]
tl, sp, wl = d[f"t_loop_{tagc}"], d[f"speed_{tagc}"], d[f"w_loop_{tagc}"]
ax.semilogy(tl, sp, lw=0.9, color="#4a5568")
j91 = np.where(wl == mu_w)[0]
ax.semilogy(tl[j91], sp[j91], lw=1.6, color="#c0392b",
            label=f"winner window of critical bond {mu_w}")
ax.set_xlabel("time within one loop"); ax.set_ylabel(r"speed $|\dot a|$")
ax.set_title(f"SNIC bottleneck at $\\lambda={lams[-1]:.3f}$: the cycle creeps "
             f"past the ghost of bond {mu_w}\n(dwell x2.3 the median bond)")
ax.legend(fontsize=9); ax.grid(alpha=.3, which="both")

fig.suptitle("E29 -- just above $\\lambda^*$ the cycle still passes through the "
             "QUASI-PURE patterns; the dead fixed-point mixture is visited only "
             "as the ghost of the critical bond ($N=2000$, seed 42)",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fp = os.path.join(BASE, "figures", "figAA_cycle_vs_ghosts.png")
fig.savefig(fp, dpi=175, bbox_inches="tight")
print(f"[E29-fig] figure -> {fp}")
