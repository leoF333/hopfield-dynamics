"""
E18d pilot: time one directional-escape bisection and a handful of random-IC
integrations of the reduced DDE (dim P=100, tau=10), and validate the final-state
classifier, before the full 6-lambda basin scan.
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
from robust_branch import woodbury_newton

N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.05, 42, 10.0
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
GRAM = (xi @ xi.T) / N
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))

def amp(u): return np.linalg.solve(GRAM, coup.overlap_raw(u))

lam = 0.25
a_front = BR["a_0.2500"]
print(f"[pilot] lam={lam} front amplitudes a91={a_front[91]:.3f} a92={a_front[92]:.3f}")

sysP = ReducedDDE(xi, BETA, lam, TAU, T0)

# classifier: integrate from a-history, classify by overlap signature at the end
def classify(a_final, a_front):
    top = np.argsort(np.abs(a_final))[::-1][:2]
    top = sorted(int(k) for k in top)
    s = np.sort(np.abs(a_final))[::-1]
    d_front = np.linalg.norm(a_final - a_front)
    if d_front < 0.05:
        return "front", d_front
    # memory: one dominant amplitude ~0.9+, single pattern
    if s[0] > 0.7 and s[1] < 0.35 and (top[1]-top[0] != 1 or s[1] < 0.15):
        return "memory", d_front
    if s[0] < 0.6:
        return "chaos/mix", d_front
    return "other", d_front

# time one integration t=300
a0 = a_front.copy()
t0w = time.time()
sol = sysP.integrate(a0, 300.0, DT, record_every=20)
dt_one = time.time() - t0w
print(f"[pilot] one reduced-DDE integrate t=300 dt={DT}: {dt_one:.2f}s")
cls, d = classify(sol["a"][-1], a_front)
print(f"[pilot] from front itself: final class={cls} d_front={d:.2e} (should be front)")

# direction toward xi^1: in amplitude space, e_1 (pattern 0)
e1 = np.zeros(P); e1[0] = 1.0
for r in [0.1, 0.3, 0.5, 0.8]:
    a0 = a_front + r * e1
    sol = sysP.integrate(a0, 300.0, DT, record_every=50)
    cls, d = classify(sol["a"][-1], a_front)
    print(f"[pilot] toward xi^1, r={r}: class={cls} d_front={d:.3f}")

# a few random ICs at radius 0.3
rng = np.random.default_rng(7)
t1 = time.time()
for i in range(4):
    delta = rng.standard_normal(P); delta *= 0.3 / np.linalg.norm(delta)
    sol = sysP.integrate(a_front + delta, 300.0, DT, record_every=100)
    cls, d = classify(sol["a"][-1], a_front)
    print(f"[pilot] random r=0.3 #{i}: class={cls} d_front={d:.3f}")
print(f"[pilot] 4 random integrations: {time.time()-t1:.1f}s "
      f"=> ~{(time.time()-t1)/4:.2f}s each")
