"""
E18 pilot 6: diagnose the a (raw reduced overlap) vs m (condensed) mismatch at the
pinned front, and verify u = xi^T a is the true fixed point.
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

from couplings import Couplings, make_patterns
from pacemaker_cycle import build_cycle
from cycle_reduced import ReducedDDE

N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.01, 42, 10.0
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

cyc = build_cycle(xi, BETA, 0.34, TAU, dt=DT, verbose=False, settle_turns=3.0)
h = cyc["a_grid"][:cyc["L"] + 1].copy()
sysP = ReducedDDE(xi, BETA, 0.318, TAU, T0)
a_pin = None
for chunk in range(40):
    sol = sysP.integrate(h, 100.0, DT)
    h = sol["hist"]; a = sol["a"][-1]
    drift = np.linalg.norm(sol["a"][-1] - sol["a"][-2]) / DT
    top = np.argsort(np.abs(a))[::-1][:2]
    if sorted(int(k) for k in top) == [91, 92] and drift < 1e-8:
        a_pin = a.copy(); print(f"pinned t={(chunk+1)*100}"); break

a = a_pin
top_a = np.argsort(np.abs(a))[::-1][:4]
print("RAW reduced state a: top idx", [int(k) for k in top_a],
      "vals", [f"{a[k]:+.3f}" for k in top_a])
# condensed overlap m = (1/N) xi tanh(beta xi^T a)
u = xi.T @ a
m = coup.overlap_raw(np.tanh(BETA * u))
top_m = np.argsort(np.abs(m))[::-1][:4]
print("CONDENSED m = xi.tanh(bu)/N: top idx", [int(k) for k in top_m],
      "vals", [f"{m[k]:+.3f}" for k in top_m])
# reduced rhs residual (is a a fixed point of the reduced DDE?)
rhs = sysP.rhs(a, a)   # constant history -> a_del = a
print("reduced ||rhs(a,a)|| =", np.linalg.norm(rhs))
# full field residual
F = coup.field_F(u, 0.318, BETA)
print("full ||F(u)||/sqrt(N) =", np.linalg.norm(F)/np.sqrt(N))
# Is u in-span? u = xi^T a means a should equal (1/N) xi u only if xi xi^T/N = I (no)
a_from_u = coup.overlap_raw(u)   # (1/N) xi u
print("||a - (1/N)xi u|| =", np.linalg.norm(a - a_from_u),
      " (a is NOT the raw overlap of u unless Gram=I)")
print("m matches sysP.m(a)?", np.linalg.norm(m - sysP.m(a)))
