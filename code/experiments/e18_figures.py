"""
E18 figures (presentation quality, English).
  figJ_front_branch.png : the bond-91 pinned-front branch in lambda -
     (left) amplitudes a91, a92 and their sum along the branch, with lambda_c and
            the depinning fold marked; terminating fold at low lambda highlighted;
     (right) eig_max(M) (instantaneous, tau-independent) and the rightmost delayed
            root Re z (tau=10) vs lambda -> the front is linearly stable throughout.
  figK_front_basins.png : basin size of the pinned front vs lambda -
     (left) directional escape radii (toward xi^1, and the weakest eigendirection);
     (right) random-IC captured fraction vs lambda for each sampling radius.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIG = os.path.join(OUT, "figures")
LAM_C = 0.2822
LAM_DEPIN = 0.328

BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
rows = BR["rows"]                      # lam,res,a91,a92,eigmaxM, +overlaps
DS = np.load(os.path.join(OUT, "E18_delayed_stability.npz"))
dr = DS["rows"]                        # lam,res,eigM,reZ,imZ,nunst,...

# ---------------- figJ: the branch --------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
lam = rows[:, 0]
axL.plot(lam, rows[:, 2], "o-", color="navy", ms=4, label=r"$a_{91}$ (leading)")
axL.plot(lam, rows[:, 3], "s-", color="darkorange", ms=4, label=r"$a_{92}$ (trailing)")
axL.plot(lam, np.abs(rows[:, 2]) + np.abs(rows[:, 3]), "^-", color="green", ms=3,
         label=r"$|a_{91}|+|a_{92}|$")
axL.axvline(LAM_C, color="gray", ls="--", lw=1.2, label=r"$\lambda_c=0.2822$ (memory fold)")
axL.axvline(LAM_DEPIN, color="crimson", ls=":", lw=1.2, label=r"$\lambda^*\approx0.328$ (depinning)")
axL.axvline(lam.min(), color="purple", ls="-.", lw=1.2,
            label=rf"terminating fold $\lambda_f\approx{lam.min():.3f}$")
axL.axvspan(lam.min(), LAM_C, color="lightyellow", alpha=0.5, zorder=0)
axL.set_xlabel(r"$\lambda$"); axL.set_ylabel("front amplitudes")
axL.set_title("Bond-91 pinned-front branch\n(exists deep into the static-recall phase)")
axL.legend(fontsize=8, loc="center left"); axL.grid(alpha=0.3)

axR.plot(lam, rows[:, 4], "o-", color="teal", ms=4,
         label=r"$\mathrm{eig}_{\max}(M)$ (instantaneous, $\tau$-indep.)")
axR.plot(dr[:, 0], dr[:, 3], "D-", color="crimson", ms=5,
         label=r"rightmost $\mathrm{Re}\,z$ ($\tau=10$ delayed)")
axR.axhline(0.0, color="k", lw=1.0)
axR.axvline(LAM_C, color="gray", ls="--", lw=1.2)
axR.axvspan(lam.min(), LAM_C, color="lightyellow", alpha=0.5, zorder=0)
axR.set_xlabel(r"$\lambda$"); axR.set_ylabel(r"rightmost eigenvalue / root")
axR.set_title("Front is linearly STABLE throughout\n(no crossing of 0; born stable at the fold)")
axR.legend(fontsize=8.5, loc="lower left"); axR.grid(alpha=0.3)
fig.suptitle(r"E18a/b - Birth of the pinned-front attractor (N=2000, seed 42, $\tau=10$)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fpJ = os.path.join(FIG, "figJ_front_branch.png")
fig.savefig(fpJ, dpi=180, bbox_inches="tight")
print(f"[E18 fig] -> {fpJ}")

# ---------------- figK: basins ------------------------------------------------
try:
    BA = np.load(os.path.join(OUT, "E18_basins.npz"))
    rd = BA["rows_dir"]; rr = BA["rows_rand"]; radii = BA["radii"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    axL.plot(rd[:, 0], rd[:, 1], "o-", color="navy", ms=6, label=r"toward $\xi^1$ (memory)")
    axL.plot(rd[:, 0], rd[:, 2], "s-", color="crimson", ms=6,
             label="weakest stable eigendirection")
    axL.axvline(LAM_C, color="gray", ls="--", lw=1.2, label=r"$\lambda_c=0.2822$")
    axL.axvspan(rd[:, 0].min(), LAM_C, color="lightyellow", alpha=0.5, zorder=0)
    axL.set_xlabel(r"$\lambda$"); axL.set_ylabel("critical escape amplitude")
    axL.set_title("Directional basin radius of the pinned front")
    axL.legend(fontsize=9); axL.grid(alpha=0.3)

    lams = np.unique(rr[:, 0])
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(radii)))
    for r, c in zip(radii, cmap):
        sel = np.isclose(rr[:, 1], r)
        axR.plot(rr[sel, 0], rr[sel, 2], "o-", color=c, ms=6,
                 label=rf"random IC, $\|\delta a\|={r}$")
    axR.axvline(LAM_C, color="gray", ls="--", lw=1.2)
    axR.axvspan(lams.min(), LAM_C, color="lightyellow", alpha=0.5, zorder=0)
    axR.set_xlabel(r"$\lambda$"); axR.set_ylabel("captured fraction into the front")
    axR.set_ylim(-0.03, 1.03)
    axR.set_title(f"Random-IC capture vs $\\lambda$ (t={int(BA['t_int'])}, "
                  f"{int(BA['n_rand'])} ICs/point)")
    axR.legend(fontsize=9); axR.grid(alpha=0.3)
    fig.suptitle(r"E18d - Basin size of the pinned-front attractor ($\tau=10$)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fpK = os.path.join(FIG, "figK_front_basins.png")
    fig.savefig(fpK, dpi=180, bbox_inches="tight")
    print(f"[E18 fig] -> {fpK}")
except FileNotFoundError:
    print("[E18 fig] E18_basins.npz not found yet; skipped figK")
