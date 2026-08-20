"""
E18 pilot 2: reconstruct the bond-91 pinned front DIRECTLY as a u-space ansatz
u0 = atanh-scale mixture of two consecutive patterns, then Newton-polish. Test a
few candidate consecutive bonds near index 90/91 and report which converge to a
genuine two-consecutive-pattern fixed point at lam=0.32 (where E12 says it lives).
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
from robust_branch import woodbury_newton, eigmax_M, anneal_newton

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

def try_bond(mu, lam, c0=0.68, c1=0.32, amp=0.5):
    """Ansatz u = amp*(sign convex mixture of xi[mu], xi[mu+1]); Newton polish."""
    xa, xb = xi[mu], xi[(mu + 1) % P]
    # field pre-image: want tanh(beta u) ~ mixture -> u ~ small; use signed combo
    u0 = amp * (c0 * xa + c1 * xb)
    u, ok = anneal_newton(coup, u0, lam, BETA, tol=1e-12)
    if not ok:
        u, ok = woodbury_newton(coup, u0, lam, BETA, tol=1e-12)
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    ov = coup.overlap_raw(np.tanh(BETA * u))
    top = np.argsort(np.abs(ov))[::-1][:3]
    return u, res, ov, sorted(int(k) for k in top[:2]), [float(ov[k]) for k in top]

lam = 0.32
print(f"[pilot2] testing consecutive-bond ansatze at lam={lam}", flush=True)
for mu in [89, 90, 91]:
    u, res, ov, top2, tops = try_bond(mu, lam)
    is_front = (top2[1] - top2[0] == 1) and abs(ov[top2[0]]) < 0.95
    print(f"  bond mu={mu}/{mu+1} (0-idx): res={res:.1e} top={top2} "
          f"ov={[f'{v:+.3f}' for v in tops]} {'<-- FRONT' if is_front else ''}",
          flush=True)
