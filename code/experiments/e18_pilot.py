"""
E18 pilot: reconstruct the bond-91 pinned front at lambda=0.30, identify active
pattern indices, time one Woodbury-Newton continuation step, one T_P spectrum
evaluation, and one short reduced-DDE integration. Establishes timing before the
full E18 sweep.
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
from pacemaker_cycle import build_cycle
from cycle_reduced import ReducedDDE
from robust_branch import woodbury_newton, eigmax_M
from reduced_spectrum import physical_roots

N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.01, 42, 10.0
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

# --- Reconstruct bond-91 pinned front at lam=0.30 via E15/E12 recipe ----------
t0w = time.time()
print("[pilot] building bond-91 pinned front (quench 0.34 -> 0.318 -> 0.30)...", flush=True)
cyc = build_cycle(xi, BETA, 0.34, TAU, dt=DT, verbose=False, settle_turns=3.0)
hist = cyc["a_grid"][:cyc["L"] + 1].copy()
sysP = ReducedDDE(xi, BETA, 0.318, TAU, T0)
sol = sysP.integrate(hist, 3000.0, DT)
h = sol["hist"]; a = sol["a"][-1]
for lam in [0.312, 0.306, 0.300]:
    s = ReducedDDE(xi, BETA, lam, TAU, T0).integrate(h, 500.0, DT)
    a, h = s["a"][-1], s["hist"]
print(f"[pilot] dynamic build done ({time.time()-t0w:.0f}s)", flush=True)

u30 = xi.T @ a
# Newton-polish in u-space
u30, ok = woodbury_newton(coup, u30, 0.300, BETA, tol=1e-12)
res = np.linalg.norm(coup.field_F(u30, 0.300, BETA)) / np.sqrt(N)
ov = coup.overlap_raw(np.tanh(BETA * u30))
top = np.argsort(np.abs(ov))[::-1][:6]
print(f"[pilot] lam=0.30 front: polished res={res:.2e} ok={ok}")
print(f"        active indices (0-idx) = {sorted(int(k) for k in top[:2])}, "
      f"overlaps top6 = {[f'{ov[k]:+.3f}' for k in top]}")
em = eigmax_M(coup, u30, 0.300, BETA)
print(f"        eigmax_M = {em:+.5f}")

# --- Time one T_P spectrum at tau=10 ------------------------------------------
t1 = time.time()
ph, _, _ = physical_roots(coup, u30, BETA, 0.300, TAU, T0, M=32, n_cand=50, k=10)
print(f"[pilot] T_P spectrum ({time.time()-t1:.1f}s): rightmost roots =")
for z in ph[:5]:
    print(f"        {z.real:+.5f} {z.imag:+.5f}i")

# --- Time one continuation step down in lambda --------------------------------
t2 = time.time()
u_new, okc = woodbury_newton(coup, u30, 0.290, BETA, tol=1e-12)
resc = np.linalg.norm(coup.field_F(u_new, 0.290, BETA)) / np.sqrt(N)
ovn = coup.overlap_raw(np.tanh(BETA * u_new))
topn = np.argsort(np.abs(ovn))[::-1][:2]
print(f"[pilot] one warm-start Newton step to lam=0.29 ({time.time()-t2:.2f}s): "
      f"res={resc:.2e} a=({ovn[topn[0]]:+.3f},{ovn[topn[1]]:+.3f})")

# --- Time one short reduced-DDE integration -----------------------------------
t3 = time.time()
a30 = coup.overlap_raw(u30)   # raw overlap on u (a = (1/N) xi u)
sysP = ReducedDDE(xi, BETA, 0.300, TAU, T0)
solp = sysP.integrate(a30, 50.0, DT)
print(f"[pilot] reduced DDE integrate t=50 ({time.time()-t3:.1f}s), "
      f"drift ||a(50)-a0||={np.linalg.norm(solp['a'][-1]-a30):.2e}")

np.savez(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "results", "cycle", "E18_pilot_u30.npz"),
         u30=u30, a30=a30, active=np.array(sorted(int(k) for k in top[:2])))
print(f"[pilot] TOTAL {time.time()-t0w:.0f}s")
