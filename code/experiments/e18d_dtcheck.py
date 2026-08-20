"""E18d dt-robustness: does dt=0.05 classify a borderline escape the same as
dt=0.01? Compare the escape radius toward xi^1 at lam=0.25 for both dt."""

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

N, ALPHA, BETA, T0, SEED, TAU = 2000, 0.05, 20.0, 1.0, 42, 10.0
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
a_front = np.load(os.path.join(OUT, "E18_front_branch.npz"))["a_0.2500"]
e1 = np.zeros(P); e1[0] = 1.0

def escaped(r, dt):
    sysP = ReducedDDE(xi, BETA, 0.25, TAU, T0)
    sol = sysP.integrate(a_front + r * e1, 300.0, dt, record_every=200)
    return np.linalg.norm(sol["a"][-1] - a_front) > 0.05

for dt in [0.01, 0.05]:
    lo, hi = 0.5, 0.8
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if escaped(mid, dt): hi = mid
        else: lo = mid
    print(f"dt={dt}: critical escape radius toward xi^1 ~ {0.5*(lo+hi):.4f}")
