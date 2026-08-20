"""
E15: delayed spectrum of the PINNED FRONT vs tau — does its Hopf margin erode
like the static memory's?

Key facts exploited:
  * the pinned front is a FIXED POINT -> its state u_pin is the SAME for all tau
    (delay drops for constant states). We build the bond-91 branch states once
    (dynamic corrector, E12 recipe, tau=10) and scan tau purely in the spectrum.
  * exact reduced characteristic matrix T_P (P x P) gives the physical roots;
    internal check #1: the rightmost REAL root (fold mode) must be tau-INVARIANT.
  * internal check #2 / comparison: memory-state margins at alpha=0.05 from the
    static study (hopf_margin_vs_tau) are overlaid.

Outputs: results/cycle/figures/figG_pinned_hopf_margin.png + E15 npz + table.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from couplings import Couplings, make_patterns
from pacemaker_cycle import build_cycle
from cycle_reduced import ReducedDDE
from reduced_spectrum import physical_roots

N, ALPHA, BETA, T0, DT, SEED = 2000, 0.05, 20.0, 1.0, 0.01, 42
TAU_BUILD = 10.0
P = round(ALPHA * N)
TAUS = [10.0, 20.0, 30.0, 50.0, 70.0, 100.0]
LAM_TARGETS = [0.300, 0.320, 0.3265]          # on the bond-91 pinned branch
IM_TOL = 1e-3

# memory-state margins at lambda_c, alpha=0.05 (static study, hopf_margin data)
MEM_TAU = np.array([10, 20, 30, 50, 70, 100])
MEM_MARGIN = np.array([0.152, 0.065, 0.037, 0.017, 0.010, 0.005])

xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

# ---- 1) build the bond-91 pinned states once (tau=10 dynamics, E12 recipe) --
print("[E15] building bond-91 pinned states (quench 0.34 -> 0.318, then dyn. continuation)",
      flush=True)
cyc = build_cycle(xi, BETA, 0.34, TAU_BUILD, dt=DT, verbose=False, settle_turns=3.0)
hist = cyc["a_grid"][:cyc["L"] + 1].copy()
sysP = ReducedDDE(xi, BETA, 0.318, TAU_BUILD, T0)
sol = sysP.integrate(hist, 3000.0, DT)
state_hist = sol["hist"]; a318 = sol["a"][-1]

states = {}
def dyn_step(hist0, lam, tconv=500.0):
    s = ReducedDDE(xi, BETA, lam, TAU_BUILD, T0).integrate(hist0, tconv, DT)
    return s["a"][-1], s["hist"]

# down to 0.300
h = state_hist; a = a318
for lam in [0.312, 0.306, 0.300]:
    a, h = dyn_step(h, lam)
states[0.300] = a.copy()
# up to 0.320 and 0.3265
h = state_hist; a = a318
for lam in [0.320]:
    a, h = dyn_step(h, lam)
states[0.320] = a.copy()
for lam in [0.324, 0.3265]:
    a, h = dyn_step(h, lam, tconv=800.0)
states[0.3265] = a.copy()
for lam, a in states.items():
    u = xi.T @ a
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    top = np.argsort(np.abs(a))[::-1][:2]
    print(f"  lam={lam}: res={res:.1e}  site={sorted(int(k) for k in top)} "
          f"a={abs(a[top[0]]):.3f},{abs(a[top[1]]):.3f}", flush=True)

# ---- 2) tau-scan of the delayed spectrum at each state ----------------------
print(f"\n{'lam':>7} {'tau':>5} {'re_real (fold, cst?)':>20} {'re_cplx':>9} "
      f"{'marge':>7} {'omega':>7} {'T_osc/tau':>9}", flush=True)
out = {}
for lam in LAM_TARGETS:
    u = xi.T @ states[lam]
    rows = []
    for tau in TAUS:
        M = 32 if tau <= 20 else (48 if tau <= 50 else 64)
        t0w = time.time()
        ph, _, _ = physical_roots(coup, u, BETA, lam, tau, T0,
                                  M=M, n_cand=60, k=12, polish=True)
        isc = np.abs(ph.imag) >= IM_TOL
        rr = float(ph.real[~isc].max()) if np.any(~isc) else np.nan
        rc, om = np.nan, np.nan
        if np.any(isc):
            kk = int(np.argmax(np.where(isc, ph.real, -np.inf)))
            rc, om = float(ph.real[kk]), float(abs(ph.imag[kk]))
        rows.append((tau, rr, rc, om))
        print(f"{lam:>7.4f} {tau:>5.0f} {rr:>20.4f} {rc:>9.4f} {-rc:>7.4f} "
              f"{om:>7.4f} {2*np.pi/om/tau if om and om>0 else float('nan'):>9.2f}"
              f"   ({time.time()-t0w:.0f}s)", flush=True)
    out[lam] = np.array(rows)

# ---- 3) figure ---------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
cols = ["navy", "teal", "crimson"]
for lam, c in zip(LAM_TARGETS, cols):
    R = out[lam]
    a1.loglog(R[:, 0], -R[:, 2], "o-", color=c, ms=6,
              label=rf"front piégé, $\lambda={lam}$")
a1.loglog(MEM_TAU, MEM_MARGIN, "k^--", ms=7, label=r"mémoire à $\lambda_c$ (réf. statique)")
a1.set_xlabel(r"$\tau$"); a1.set_ylabel(r"marge de Hopf $|\mathrm{Re}\,z_{\rm cplx}|$")
a1.set_title("Érosion de la marge oscillatoire avec le délai\n(front piégé vs mémoire)")
a1.legend(fontsize=8.5); a1.grid(alpha=.3, which="both")
for lam, c in zip(LAM_TARGETS, cols):
    R = out[lam]
    a2.plot(R[:, 0], R[:, 1], "s-", color=c, ms=6,
            label=rf"$\lambda={lam}$ (mode réel/fold)")
a2.set_xlabel(r"$\tau$"); a2.set_ylabel(r"$\mathrm{Re}\,z_{\rm réel}$ dominant")
a2.set_title("Contrôle : le mode de fold est $\\tau$-INVARIANT\n(le fold ne voit pas le délai)")
a2.legend(fontsize=9); a2.grid(alpha=.3)
fig.suptitle("E15 — spectre retardé du front piégé vs $\\tau$ "
             f"(N={N}, seed {SEED}, liaison 91)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "results", "cycle", "figures", "figG_pinned_hopf_margin.png")
fig.savefig(fp, dpi=170, bbox_inches="tight")
np.savez(fp.replace("figures/figG_pinned_hopf_margin.png", "E15_pinned_margins.npz"),
         **{f"rows_lam{lam}": out[lam] for lam in LAM_TARGETS},
         mem_tau=MEM_TAU, mem_margin=MEM_MARGIN)
print(f"\n[E15] fig -> {fp}", flush=True)
