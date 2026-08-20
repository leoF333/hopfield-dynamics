"""
E18 pilot 3: reproduce E15's exact build of the bond-91 pinned front at lam=0.318,
inspect the two-consecutive-pattern overlaps, then step DOWN in lambda by dynamic
continuation with small steps, Newton-polishing each and reporting overlaps, to
locate where the front collapses to memory (or a terminating fold). This is the
E18a mechanism in miniature.
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

N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.01, 42, 10.0
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

def diag(u, lam):
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    ov = coup.overlap_raw(np.tanh(BETA * u))
    top = np.argsort(np.abs(ov))[::-1][:3]
    return res, sorted(int(k) for k in top[:2]), [float(ov[k]) for k in top[:3]]

t0w = time.time()
print("[pilot3] E15 build: quench lam=0.34 -> settle at 0.318", flush=True)
cyc = build_cycle(xi, BETA, 0.34, TAU, dt=DT, verbose=False, settle_turns=3.0)
hist = cyc["a_grid"][:cyc["L"] + 1].copy()
sysP = ReducedDDE(xi, BETA, 0.318, TAU, T0)
sol = sysP.integrate(hist, 3000.0, DT)
h318 = sol["hist"]; a318 = sol["a"][-1]
u318 = xi.T @ a318
u318p, ok = woodbury_newton(coup, u318, 0.318, BETA, tol=1e-12)
res, top2, tops = diag(u318p, 0.318)
print(f"[pilot3] lam=0.318 state: res={res:.1e} top={top2} ov={[f'{v:+.3f}' for v in tops]}",
      flush=True)

# Dynamic continuation DOWN, small steps, polish each
h = h318; a = a318
prev_u = u318p
print("\n[pilot3] stepping DOWN via dynamic continuation + Newton polish:")
for lam in [0.315, 0.310, 0.305, 0.300, 0.295, 0.290, 0.285]:
    s = ReducedDDE(xi, BETA, lam, TAU, T0).integrate(h, 400.0, DT)
    a, h = s["a"][-1], s["hist"]
    u = xi.T @ a
    up, okn = woodbury_newton(coup, u, lam, BETA, tol=1e-12)
    res, top2, tops = diag(up, lam)
    em = eigmax_M(coup, up, lam, BETA)
    is_front = (top2[1] - top2[0] == 1) and abs(tops[0]) < 0.95
    print(f"  lam={lam:.3f}: res={res:.1e} eigmaxM={em:+.4f} top={top2} "
          f"ov={[f'{v:+.3f}' for v in tops]} {'FRONT' if is_front else 'memory?'}",
          flush=True)
print(f"[pilot3] TOTAL {time.time()-t0w:.0f}s")
