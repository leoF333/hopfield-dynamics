"""
E19d - Fixed-point cross-check at small tau.

The pinned front is a FIXED POINT: its state u_pin is tau-INDEPENDENT (delay drops
for constant states; Delta(0) = -M carries no tau). So:
  1. Rebuild the bond-91 pinned states (E12/E15 recipe: quench from a travelling
     state at lam=0.34, tau=10, then dynamic continuation in lambda) at 3 lambdas.
  2. Verify eig_max(M) at the pinned front == E12 rows to machine precision
     (tau-independent real fold mode).
  3. Compute the rightmost T_P(z) roots at the front for tau in {0.25, 1, 2}:
     does small tau move the DELAYED (complex) stability of the front, even though
     the fold (z=0 real mode) does not move?

Outputs: results/cycle/E19d_fixedpoint.npz + printed tables.
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

from couplings import Couplings, make_patterns
from cycle_reduced import ReducedDDE
from robust_branch import woodbury_newton, eigmax_M
from reduced_spectrum import physical_roots

N, ALPHA, BETA, T0, DT, SEED = 2000, 0.05, 20.0, 1.0, 0.01, 42
TAU_BUILD = 10.0
P = round(ALPHA * N)
LAM_TARGETS = [0.300, 0.320, 0.3265]        # bond-91 pinned branch (E12/E15)
TAUS = [0.25, 1.0, 2.0, 10.0]               # incl. tau=10 as reference
IM_TOL = 1e-3

xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")

# E12 reference rows: [lam, res, a_91, a_92, eig_max(M)]
E12 = np.load(os.path.join(OUT, "E12_pinned_branch.npz"))["rows"]

# ---- 1) rebuild the bond-91 pinned states (E15 recipe) -----------------------
print("[E19d] building bond-91 pinned states (quench 0.34->0.318, dyn. continuation)",
      flush=True)
from pacemaker_cycle import build_cycle
cyc = build_cycle(xi, BETA, 0.34, TAU_BUILD, dt=DT, verbose=False, settle_turns=3.0)
hist = cyc["a_grid"][:cyc["L"] + 1].copy()
sysP = ReducedDDE(xi, BETA, 0.318, TAU_BUILD, T0)
sol = sysP.integrate(hist, 3000.0, DT)
state_hist = sol["hist"]; a318 = sol["a"][-1]


def dyn_step(hist0, lam, tconv=500.0):
    s = ReducedDDE(xi, BETA, lam, TAU_BUILD, T0).integrate(hist0, tconv, DT)
    return s["a"][-1], s["hist"]


states = {}
h = state_hist; a = a318
for lam in [0.312, 0.306, 0.300]:
    a, h = dyn_step(h, lam)
states[0.300] = a.copy()
h = state_hist; a = a318
for lam in [0.320]:
    a, h = dyn_step(h, lam)
states[0.320] = a.copy()
for lam in [0.324, 0.3265]:
    a, h = dyn_step(h, lam, tconv=800.0)
states[0.3265] = a.copy()

# polish to an exact fixed point with Woodbury-Newton, then re-project to a-space
print("\n[E19d] pinned states + eig_max(M) cross-check vs E12:")
print(f"{'lam':>7} {'res':>9} {'a91':>8} {'a92':>8} {'eigM(now)':>11} "
      f"{'eigM(E12)':>11} {'|diff|':>9}")
eig_now = {}
for lam in LAM_TARGETS:
    u = xi.T @ states[lam]
    u, ok = woodbury_newton(coup, u, lam, BETA, tol=1e-12)
    a = np.linalg.lstsq(xi.T, u, rcond=None)[0]
    states[lam] = a
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    em = eigmax_M(coup, u, lam, BETA)
    eig_now[lam] = em
    # nearest E12 row
    j = int(np.argmin(np.abs(E12[:, 0] - lam)))
    em12 = E12[j, 4] if abs(E12[j, 0] - lam) < 1e-6 else np.nan
    top = np.argsort(np.abs(a))[::-1][:2]
    print(f"{lam:>7.4f} {res:>9.1e} {abs(a[top[0]]):>8.4f} {abs(a[top[1]]):>8.4f} "
          f"{em:>11.6f} {em12:>11.6f} {abs(em-em12):>9.1e}")

# ---- 2) delayed spectrum T_P rightmost roots at the front, per tau ------------
print(f"\n[E19d] rightmost T_P roots at the pinned front vs tau:")
print(f"{'lam':>7} {'tau':>6} {'re_real(fold)':>14} {'re_cplx':>9} "
      f"{'omega':>8} {'T_osc':>8} {'stable?':>8}")
out = {}
for lam in LAM_TARGETS:
    u = xi.T @ states[lam]
    rows = []
    for tau in TAUS:
        M = 24 if tau <= 2 else 32
        t0w = time.time()
        ph, _, _ = physical_roots(coup, u, BETA, lam, tau, T0,
                                  M=M, n_cand=60, k=12, polish=True)
        isc = np.abs(ph.imag) >= IM_TOL
        rr = float(ph.real[~isc].max()) if np.any(~isc) else np.nan
        rc, om = np.nan, np.nan
        if np.any(isc):
            kk = int(np.argmax(np.where(isc, ph.real, -np.inf)))
            rc, om = float(ph.real[kk]), float(abs(ph.imag[kk]))
        re_dom = np.nanmax([rr, rc])
        stable = re_dom < 0
        Tosc = (2 * np.pi / om) if (om and om > 0) else np.nan
        rows.append((tau, rr, rc, om, Tosc, re_dom))
        print(f"{lam:>7.4f} {tau:>6.2f} {rr:>14.5f} {rc:>9.4f} {om:>8.4f} "
              f"{Tosc:>8.2f} {str(stable):>8}  ({time.time()-t0w:.0f}s)",
              flush=True)
    out[lam] = np.array(rows)

np.savez(os.path.join(OUT, "E19d_fixedpoint.npz"),
         lam_targets=np.array(LAM_TARGETS), taus=np.array(TAUS),
         eig_now=np.array([eig_now[l] for l in LAM_TARGETS]),
         **{f"roots_lam{l}": out[l] for l in LAM_TARGETS})
print("\n[E19d] npz saved.")
print("  Columns of roots_lam*: [tau, re_real(fold), re_cplx, omega, T_osc, re_dom]")
