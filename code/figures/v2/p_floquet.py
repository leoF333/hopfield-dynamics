
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys; sys.path.insert(0,'/tmp/p2')
from common2 import *
z=np.load(NC+'2_cycle_rappel_snic/data/floquet_scan_N2000_tau10.0.npz',allow_pickle=True)
lam=np.asarray(z['lam'],float); mus=np.abs(np.asarray(z['mu_star'],complex))
T1=np.asarray(z['T1'],float); o=np.argsort(lam)
A_bound=-np.log(np.maximum(mus,1e-17))            # Arnoldi floor -> LOWER bound on the contraction
# E11, N=500, true (above-floor) measurements
l500=np.array([0.90,0.65,0.45]); A500=np.array([16.2,22.4,35.7]); A500_bound=np.array([0,0,1],bool)
# E14, N=2000, floor-free Benettin/QR near death
l14=np.array([0.400,0.335,0.329]); A14=np.array([219.7,419.5,440.1]); AP14=np.array([2.20,4.20,4.40])
LSTAR=0.32763

fig=plt.figure(figsize=(8.0,6.0))
ax=fig.add_axes([0.135,0.125,0.775,0.775])
ax.axvspan(0.30,LSTAR,color="0.90",zorder=0)
ax.axvline(LSTAR,color=RED,lw=1.7,ls="--",zorder=2)
ax.axhspan(0,40,color="0.86",alpha=0.55,zorder=0)
ax.text(0.965,20,"not resolvable by Arnoldi\n(relative floor $2\\times10^{-16}$)",
        fontsize=11.5,color="0.35",ha="right",va="center")
ax.plot(lam[o],A_bound[o],"v",ms=8,mfc="none",mew=1.6,color=GREY,zorder=3,
        label=r"$N=2000$ scan: lower bound")
ax.plot(l500[~A500_bound],A500[~A500_bound],"s-",ms=9,lw=2.0,color=GRN,zorder=4,
        label=r"$N=500$, measured")
ax.plot(l500[A500_bound],A500[A500_bound],"s",ms=9,mfc="none",mew=1.8,color=GRN,zorder=4)
ax.plot(l14,A14,"o-",ms=10,lw=2.4,color=INK,zorder=5,
        label=r"$N=2000$, floor-free (Benettin–QR)")
for (x,y),off in zip(zip(l14,A14),[(10,-14),(14,4),(-6,16)]):
    ax.annotate("%.1f"%y,xy=(x,y),xytext=off,textcoords="offset points",
                fontsize=12.5,color=INK,ha="left",va="center")
ax.set_yscale("log")
ax.set_xlabel(r"mixing $\lambda$")
ax.set_ylabel(r"contraction per turn   $A=-\ln|\varrho_1|$")
ax.set_xlim(0.30,0.99); ax.set_ylim(4,1.1e3)
ax.legend(frameon=False,loc="upper right",fontsize=11.5)
ax.text(LSTAR+0.010,5.2,r"$\lambda^{*}$",color=RED,fontsize=15,ha="left",va="bottom")
ax.text(0.475,330,"the contraction GROWS on approach:\nthe multipliers flee the unit circle\n"
                  r"as $e^{-440}$, and $-\mathrm{Re}\,z_1$ stays finite at $0.23$",
        fontsize=12.5,color=INK,ha="left",va="center")
ax.text(0.965,4.9,r"phase-mode control $\ln|\varrho_0|=0.001$ at all three points",
        fontsize=11.5,color=DARK,ha="right",va="bottom")
L(ax,"the recall cycle is hyper-stable, and stays so up to its death",fs=14.5)
fig.savefig(OUTW+"component_floquet_hyperstability.png",dpi=300,bbox_inches="tight",
            pad_inches=0.08,facecolor="white")
print("saved")
