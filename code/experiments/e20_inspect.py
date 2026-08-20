"""Inspect settled states: pinned front vs pure memory overlap structure."""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cycle_reduced import ReducedDDE
import e20_common as C

for lam in (0.15, 0.20, 0.25, 0.30):
    a_front = C.reach_front(lam)
    top = np.argsort(np.abs(a_front))[::-1][:4]
    print(f"lam={lam:.3f} FRONT  top idx={top}  vals={np.round(a_front[top],3)}")

print()
for lam in (0.15, 0.20, 0.25):
    a_mem = C.reach_memory(lam, 0)
    if a_mem is None:
        print(f"lam={lam:.3f} MEMORY unreachable"); continue
    top = np.argsort(np.abs(a_mem))[::-1][:4]
    print(f"lam={lam:.3f} MEMORY top idx={top}  vals={np.round(a_mem[top],3)}")
    # now integrate it as constant history and see settled structure
    sysP = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)
    sol = sysP.integrate(a_mem.copy(), 300.0, C.DT, record_every=40)
    a_f = sol["a"][-1]
    top2 = np.argsort(np.abs(a_f))[::-1][:4]
    drift = np.linalg.norm(sol["a"][-1]-sol["a"][-2])/np.linalg.norm(sol["a"][-1])
    print(f"          settled top idx={top2} vals={np.round(a_f[top2],3)} drift={drift:.2e} "
          f"consec={abs(int(top2[0])-int(top2[1]))in(1,C.P-1)}")
