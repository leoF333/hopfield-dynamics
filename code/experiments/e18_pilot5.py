"""
E18 pilot 5: catch the pinned front during the quench settle.
Quench a travelling cycle (lam=0.34) down to lam=0.318 and record the sorted-overlap
signature vs settle time, to see whether the trajectory PASSES THROUGH / settles on
a [0.68,0.32] two-pattern front (pinned) before (if ever) collapsing to a memory.
Then Newton-polish the two-pattern state where the sorted top-2 = [~0.68,~0.32].
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
from robust_branch import woodbury_newton, eigmax_M

N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.01, 42, 10.0
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

print("[pilot5] quench cycle 0.34, then integrate at 0.318 recording signature",
      flush=True)
cyc = build_cycle(xi, BETA, 0.34, TAU, dt=DT, verbose=False, settle_turns=3.0)
hist = cyc["a_grid"][:cyc["L"] + 1].copy()
sysP = ReducedDDE(xi, BETA, 0.318, TAU, T0)

# integrate in chunks, inspect signature each chunk
h = hist; t_cum = 0.0
best = None
for chunk in range(30):
    sol = sysP.integrate(h, 100.0, DT)
    h = sol["hist"]; a = sol["a"][-1]; t_cum += 100.0
    s = np.sort(np.abs(a))[::-1]
    top = np.argsort(np.abs(a))[::-1][:2]
    consec = abs(int(top[1]) - int(top[0])) == 1
    drift = np.linalg.norm(sol["a"][-1] - sol["a"][-2]) / DT  # ||a'|| proxy
    tag = "FRONT" if (0.55 < s[0] < 0.85 and 0.2 < s[1] < 0.45 and consec) else ""
    print(f"  t={t_cum:5.0f}: sorted top3=[{s[0]:.3f},{s[1]:.3f},{s[2]:.3f}] "
          f"idx={sorted(int(k) for k in top)} ||a'||~{drift:.1e} {tag}", flush=True)
    if tag and best is None:
        best = a.copy()

if best is not None:
    u = xi.T @ best
    up, ok = woodbury_newton(coup, u, 0.318, BETA, tol=1e-12)
    res = np.linalg.norm(coup.field_F(up, 0.318, BETA)) / np.sqrt(N)
    ov = coup.overlap_raw(np.tanh(BETA * up))
    top = np.argsort(np.abs(ov))[::-1][:3]
    em = eigmax_M(coup, up, 0.318, BETA)
    print(f"[pilot5] polished front lam=0.318: res={res:.1e} eigM={em:+.4f} "
          f"top={sorted(int(k) for k in top[:2])} ov={[f'{ov[k]:+.3f}' for k in top]}")
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "cycle")
    np.savez(os.path.join(OUT, "E18_front_lam0318.npz"), u=up, a=coup.overlap_raw(up))
else:
    print("[pilot5] no two-pattern front caught during settle")
