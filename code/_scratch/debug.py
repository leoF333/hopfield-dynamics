
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import mlx.core as mx
import numpy as np
from scipy.sparse.linalg import eigs, LinearOperator
import sys
sys.path.append(CHICAGO_S + "/3_numerics/src")
from jacobian_spectrum import get_dominant_eigenvalue_cycle, get_dominant_eigenvalue_fixed_point

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
CHICAGO_S = str(CHICAGO)
# ---------------------------------------------------------------------------


N = 300
alpha = 0.05
p = max(1, int(N*alpha))
tau = 10
beta = 20.0

patterns_int = mx.random.randint(0, 2, [p, N])
patterns = patterns_int.astype(mx.float32) * 2.0 - 1.0
patterns_shifted = mx.concatenate([patterns[1:], patterns[:1]], axis=0)
mx.eval(patterns, patterns_shifted)

for lam in [0.0, 0.5, 1.0]:
    lam_mx = mx.array(lam, dtype=mx.float32)
    print(f"--- lambda={lam} ---")
    fp = get_dominant_eigenvalue_fixed_point(patterns, patterns_shifted, float(N), float(p), beta, lam_mx, tau)
    print("Fixed Point:", fp)
    cyc = get_dominant_eigenvalue_cycle(patterns, patterns_shifted, float(N), float(p), beta, lam_mx, tau)
    print("Cycle:", cyc)
