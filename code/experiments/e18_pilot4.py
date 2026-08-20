"""
E18 pilot 4: locate a genuine two-consecutive-pattern pinned front.
(1) scan the E7 census (lam=0.31 final states) for 2-pattern mixtures ~[0.68,0.32];
(2) take such an a-vector, project to u, Woodbury-Newton polish at lam=0.31 and
    check it stays a two-pattern front (res<1e-12, not a basin jump to memory).
This gives a clean warm-start for the E18a downward continuation.
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
from robust_branch import woodbury_newton, eigmax_M

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")

A = np.load(os.path.join(OUT, "E7_attractor_census_lam0.31.npz"))["A"]  # (40,100)
print("[pilot4] scanning E7 census (lam=0.31) for two-pattern fronts:")
fronts = []
for i in range(A.shape[0]):
    a = A[i]; top = np.argsort(np.abs(a))[::-1][:3]
    s = np.abs(a[top])
    consec = (abs(int(top[1]) - int(top[0])) == 1)
    if s[0] < 0.9 and s[1] > 0.15 and consec:
        fronts.append((i, sorted(int(k) for k in top[:2])))
        print(f"  IC {i}: top={sorted(int(k) for k in top[:2])} "
              f"ov={[f'{a[k]:+.3f}' for k in top]}")
print(f"[pilot4] found {len(fronts)} candidate two-pattern fronts")

# Take the first and polish at lam=0.31 (raw overlaps a; u = xi^T a IS in-span)
if fronts:
    i0 = fronts[0][0]
    a0 = A[i0]
    u = xi.T @ a0
    for lam in [0.31, 0.32]:
        up, ok = woodbury_newton(coup, u, lam, BETA, tol=1e-12)
        res = np.linalg.norm(coup.field_F(up, lam, BETA)) / np.sqrt(N)
        ov = coup.overlap_raw(np.tanh(BETA * up))
        top = np.argsort(np.abs(ov))[::-1][:3]
        em = eigmax_M(coup, up, lam, BETA)
        print(f"  polished lam={lam}: res={res:.1e} eigM={em:+.4f} "
              f"top={sorted(int(k) for k in top[:2])} "
              f"ov={[f'{ov[k]:+.3f}' for k in top]}")
        u = up
        np.savez(os.path.join(OUT, f"E18_front_seed_lam{lam}.npz"),
                 u=up, a=coup.overlap_raw(up))
