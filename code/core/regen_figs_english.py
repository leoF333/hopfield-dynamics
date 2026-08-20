"""
Regenerate figE_v2, figF, figG with ENGLISH titles/labels (project convention)
from the archived npz data. Pure rendering — no recomputation.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")
FIG = os.path.join(BASE, "figures")

# ------------------------------------------------------------------ figE v2 --
d = np.load(os.path.join(BASE, "E11_floquet_spectra_N500.npz"), allow_pickle=True)
print("E11 keys:", list(d.keys()))
lams = [0.90, 0.65, 0.45]
cols = {0.90: "teal", 0.65: "darkorange", 0.45: "purple"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for lam in lams:
    key = None
    for k in d.keys():
        if k.startswith("mu") and f"{lam}" in k:
            key = k
    if key is None:
        key = [k for k in d.keys() if str(lam) in k and "mu" in k][0]
    mu = np.asarray(d[key]).ravel()
    Tk = [k for k in d.keys() if str(lam) in k and ("T" in k and "mu" not in k)]
    T = float(np.asarray(d[Tk[0]]).ravel()[0]) if Tk else np.nan
    mod = np.sort(np.abs(mu))[::-1]
    a1.semilogy(range(len(mod)), mod, "o-", color=cols[lam], ms=6,
                label=rf"$\lambda={lam}$ (T={T:.0f})")
    z = np.log(np.clip(mod[1:], 1e-300, None)) / T
    a2.plot(range(1, len(mod)), z, "s-", color=cols[lam], ms=5,
            label=rf"$\lambda={lam}$")
a1.axhline(1.0, color="crimson", ls="--", lw=1.2)
a1.text(4.5, 1.6, r"unit circle ($\mu_0=1$, phase mode)", color="crimson", fontsize=9)
a1.axhline(2.7e-16, color="gray", ls=":", lw=1)
a1.text(7.5, 4e-16, "machine floor", color="gray", fontsize=9)
a1.set_xlabel("index k"); a1.set_ylabel(r"$|\mu_k|$ (log)")
a1.set_title("NO multiplier rises toward 1 as the cycle dies;\n"
             r"$|\mu_1|$ collapses: $8.8\times10^{-8}\to1.8\times10^{-10}\to\leq3\times10^{-16}$")
a1.legend(fontsize=9); a1.grid(alpha=.3, which="both")
a2.axhline(0, color="k", ls=":", lw=1)
a2.set_xlabel("index k (nontrivial)"); a2.set_ylabel(r"$\mathrm{Re}\,z_k=\ln|\mu_k|/T$")
a2.set_title("Band of front-displacement exponents\n"
             r"($\mathbb{Z}_P$ degeneracy lifted by disorder; $\lambda=0.45$: floor bound)")
a2.legend(fontsize=9); a2.grid(alpha=.3)
fig.suptitle("E11 — the cycle death is INVISIBLE on its Floquet spectrum "
             r"(SNIC signature) — $N=500$, $\tau=10$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG, "figE_floquet_spectrum_v2.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("figE_v2 regenerated (English)")

# --------------------------------------------------------------------- figF --
e = np.load(os.path.join(BASE, "E12_pinned_branch.npz"), allow_pickle=True)
print("E12 keys:", list(e.keys()))
rows = np.asarray(e["rows"])              # columns: lam, res, a91, a92, eig_max
lam_b, eig_b = rows[:, 0], rows[:, 4]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(lam_b, eig_b, "o-", color="navy", ms=6,
        label=r"$\mathrm{eig}_{\max}(M)$ at the pinned front (bond 91)")
ax.axhline(0, color="k", ls=":", lw=1)
ax.axvspan(0.327, 0.328, color="crimson", alpha=0.18,
           label=r"dynamic $\lambda^*$ (E8c/E9)")
ax.axvline(0.3262, color="green", ls="--", lw=1.4,
           label=r"$1/t_{\rm MAX}^2$ intercept (E9): 0.3262")
lf = np.linspace(lam_b.min(), 0.328, 200)
ax.plot(lf, -2.1 * np.sqrt(np.clip(0.328 - lf, 0, None)), color="gray", ls="-.",
        lw=1.2, label=r"saddle-node law $-c\sqrt{\lambda^*-\lambda}$ ($c=2.1$)")
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"dominant (real) eigenvalue of $M$")
ax.set_title("E12: SPECTRAL read-out of the pinned-front saddle-node\n"
             "(the SNIC read on the fixed point's local Jacobian)")
ax.legend(fontsize=9); ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figF_pinned_fold_spectral.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("figF regenerated (English)")

# --------------------------------------------------------------------- figG --
f = np.load(os.path.join(BASE, "E15_pinned_margins.npz"), allow_pickle=True)
print("E15 keys:", list(f.keys()))
LAMS = [0.300, 0.320, 0.3265]
cols3 = ["navy", "teal", "crimson"]
mem_tau, mem_m = f["mem_tau"], f["mem_margin"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for lam, c in zip(LAMS, cols3):
    R = f[f"rows_lam{lam}"]
    a1.loglog(R[:, 0], -R[:, 2], "o-", color=c, ms=6,
              label=rf"pinned front, $\lambda={lam}$")
    a2.semilogx(R[:, 0], -R[:, 2] * R[:, 0], "o-", color=c, ms=6,
                label=rf"pinned front, $\lambda={lam}$")
a1.loglog(mem_tau, mem_m, "k^--", ms=7, label=r"memory at $\lambda_c$ (static ref.)")
a2.semilogx(mem_tau, mem_m * mem_tau, "k^--", ms=7, label=r"memory at $\lambda_c$")
tt = np.array([10, 100.0])
a1.loglog(tt, 1.8 / tt, color="gray", ls=":", lw=1.2)
a1.text(38, 0.06, r"$\propto 1/\tau$", color="gray", fontsize=10)
a1.set_xlabel(r"$\tau$"); a1.set_ylabel(r"Hopf margin $|\mathrm{Re}\,z_{\rm cplx}|$")
a1.set_title("Delay erosion of the oscillatory margin\n(pinned front vs memory)")
a1.legend(fontsize=8.5); a1.grid(alpha=.3, which="both")
a2.set_xlabel(r"$\tau$"); a2.set_ylabel(r"margin $\times\ \tau$")
a2.set_title("Law test: margin$\\times\\tau$ constant = pure $1/\\tau$ (no crossing)\n"
             "pinned front: plateau; memory: decreasing (finite $\\tau^*$)")
a2.legend(fontsize=8.5); a2.grid(alpha=.3)
fig.suptitle("E15 — Hopf-margin erosion: pinned front vs memory "
             "(N=2000, seed 42, bond 91)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(os.path.join(FIG, "figG_pinned_hopf_margin.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("figG regenerated (English)")
