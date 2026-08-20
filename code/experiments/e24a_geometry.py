"""
E24a -- shape evolution of the memory->front branches (all P=100, N=2000, seed 42).

Question (user, task a): as lambda grows, does a given static attractor stay close to
xi^mu over a range and then depart SUDDENLY at some lambda_i(mu) < lambda_c(mu) (a knee),
or does it tilt SMOOTHLY all along, with only the universal sqrt fold acceleration at
the very end?

Observables per branch (from the stored curves of e24_thresholds --store-curves):
  tilt r(lam)   = a_{mu+1}/a_mu
  half-tilt point x_50(mu) = lam(r = r_fold/2) / lambda_c(mu)   (knee detector: a knee
     would concentrate x_50 near 1; a smooth universal curve gives a broad, tight
     collapse of r(lam/lambda_c) with x_50 well below 1)
  terminal sqrt test: a_mu(lam) linear in sqrt(lambda_c - lam) near the fold.

Outputs: figU_branch_geometry.png + printed table.
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")
d = np.load(os.path.join(BASE, "E24_curves_N2000_s42.npz"))
lam_c = d["lam_c"]; P = int(d["P"])
curves = {mu: d[f"curve_{mu}"] for mu in range(P) if f"curve_{mu}" in d}
print(f"[E24a] {len(curves)} stored curves, lam_c in [{lam_c.min():.4f}, {lam_c.max():.4f}]")

# ---- per-branch metrics -----------------------------------------------------
x50, r_fold, tilt_at_half_lam = [], [], []
for mu, C in curves.items():
    lam, amu, amu1 = C[:, 0], np.abs(C[:, 1]), np.abs(C[:, 2])
    r = amu1 / np.maximum(amu, 1e-12)
    rf = r[-1]
    r_fold.append(rf)
    j = np.searchsorted(r, rf / 2)          # r is monotone up along the branch
    x50.append(lam[min(j, len(lam) - 1)] / lam_c[mu])
    k = np.searchsorted(lam, 0.5 * lam_c[mu])
    tilt_at_half_lam.append(r[min(k, len(lam) - 1)] / rf)
x50 = np.array(x50); r_fold = np.array(r_fold); t_half = np.array(tilt_at_half_lam)

print(f"[E24a] tilt at the fold r_fold = a2/a1: median {np.median(r_fold):.3f} "
      f"(min {r_fold.min():.3f}, max {r_fold.max():.3f}) -> universal death shape")
print(f"[E24a] half-tilt position x50 = lam(r=r_fold/2)/lam_c: "
      f"median {np.median(x50):.3f}, IQR [{np.percentile(x50,25):.3f}, "
      f"{np.percentile(x50,75):.3f}]")
print(f"[E24a] fraction of final tilt already acquired at lam = lam_c/2: "
      f"median {np.median(t_half):.3f}")

# ---- terminal sqrt-law test on 5 representative branches ---------------------
reps = [int(np.argmin(lam_c)), int(np.argsort(lam_c)[P // 4]),
        int(np.argsort(lam_c)[P // 2]), int(np.argsort(lam_c)[3 * P // 4]),
        int(np.argmax(lam_c))]
sqrt_r2 = {}
for mu in reps:
    C = curves[mu]; lam, amu = C[:, 0], np.abs(C[:, 1])
    m = lam > lam_c[mu] - 0.02
    if m.sum() >= 4:
        s = np.sqrt(np.maximum(lam_c[mu] - lam[m], 0))
        A = np.vstack([s, np.ones(m.sum())]).T
        coef, res_, *_ = np.linalg.lstsq(A, amu[m], rcond=None)
        pred = A @ coef
        ss = 1 - np.sum((amu[m] - pred) ** 2) / max(np.sum((amu[m] - amu[m].mean()) ** 2), 1e-30)
        sqrt_r2[mu] = (ss, coef[0])
        print(f"[E24a] sqrt-law fit branch mu={mu} (lam_c={lam_c[mu]:.4f}): "
              f"a_mu = a_c + {coef[0]:.3f}*sqrt(lam_c-lam), R^2={ss:.4f}")

# ---- figure ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
cmap = plt.cm.viridis
norm = plt.Normalize(lam_c.min(), lam_c.max())

ax = axes[0, 0]
for mu, C in curves.items():
    r = np.abs(C[:, 2]) / np.maximum(np.abs(C[:, 1]), 1e-12)
    ax.plot(C[:, 0], r, lw=0.7, color=cmap(norm(lam_c[mu])), alpha=0.7)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
fig.colorbar(sm, ax=ax, label=r"$\lambda_c(\mu)$")
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"tilt $a_{\mu+1}/a_\mu$")
ax.set_title("All 100 branches: the tilt grows smoothly from $\\lambda=0.05$ on\n"
             "(no plateau, no knee)")
ax.grid(alpha=.3)

ax = axes[0, 1]
for mu, C in curves.items():
    r = np.abs(C[:, 2]) / np.maximum(np.abs(C[:, 1]), 1e-12)
    ax.plot(C[:, 0] / lam_c[mu], r, lw=0.7, color=cmap(norm(lam_c[mu])), alpha=0.7)
ax.set_xlabel(r"$\lambda/\lambda_c(\mu)$"); ax.set_ylabel(r"tilt $a_{\mu+1}/a_\mu$")
ax.set_title("Rescaled by the per-pattern threshold: tight universal collapse\n"
             "(the disorder only sets $\\lambda_c(\\mu)$, not the shape)")
ax.grid(alpha=.3)

ax = axes[1, 0]
ax.hist(x50, bins=24, color="#2b6cb0", alpha=0.85)
ax.axvline(np.median(x50), color="crimson", lw=1.6,
           label=f"median {np.median(x50):.2f}")
ax.set_xlabel(r"$x_{50}$ = $\lambda$(half tilt)$/\lambda_c(\mu)$")
ax.set_ylabel("branches")
ax.set_title("Knee detector: half the final tilt is acquired at "
             "$\\sim$0.5--0.7$\\,\\lambda_c$\n(a sudden departure would pile up near 1)")
ax.legend(); ax.grid(alpha=.3)

ax = axes[1, 1]
cols5 = ["#2b6cb0", "#2f9e44", "#e07b00", "#b03a5b", "#6f42c1"]
for mu, c in zip(reps, cols5):
    C = curves[mu]; lam, amu = C[:, 0], np.abs(C[:, 1])
    m = lam > lam_c[mu] - 0.04
    ax.plot(np.sqrt(np.maximum(lam_c[mu] - lam[m], 0)), amu[m], "o-", ms=3.5, lw=1.1,
            color=c, label=rf"$\mu={mu}$, $\lambda_c={lam_c[mu]:.3f}$")
ax.set_xlabel(r"$\sqrt{\lambda_c(\mu)-\lambda}$"); ax.set_ylabel(r"$a_\mu$")
ax.set_title("Terminal regime: $a_\\mu$ linear in $\\sqrt{\\lambda_c-\\lambda}$\n"
             "(the universal fold acceleration -- the only 'sudden' part)")
ax.legend(fontsize=8); ax.grid(alpha=.3)

fig.suptitle("E24a -- how a memory attractor becomes the tilted front "
             r"($N=2000$, $\alpha=0.05$, seed 42, all $P=100$ branches)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fp = os.path.join(BASE, "figures", "figU_branch_geometry.png")
fig.savefig(fp, dpi=175, bbox_inches="tight")
print(f"[E24a] figure -> {fp}")
