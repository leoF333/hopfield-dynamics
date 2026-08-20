"""
E17b: identify the final state of the E17 test run (lam=0.20167, alpha=0.07,
tau=100) which saturated at a CONSTANT ||da||~1e-2 instead of oscillating.

Hypothesis: the trajectory converged onto a SECOND nearby fixed point (the fold
partner at delta_lam=4e-5 below fold=0.20171, branch separation ~ sqrt).

Protocol: reproduce the E17 perturbed IC exactly (same rng draws), integrate
t=1500 (plateau reached by t~300), Newton-polish the final state, compare to
the polished branch state a*, then compute the leading delayed roots (T_P) at
BOTH states to see which is stable at tau=100 and where the Hopf mode sits.
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
from robust_branch import trace_branch, woodbury_newton
from cycle_reduced import ReducedDDE
from reduced_spectrum import physical_roots

N, ALPHA, BETA, T0, SEED = 2000, 0.07, 20.0, 1.0, 42
TAU, DT = 100.0, 0.02
LAM = 0.20167
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
rng = np.random.default_rng(11)
_ = rng.standard_normal(P)              # burn draw #1 (E17 control run)
pert = 1e-3 * rng.standard_normal(P)    # draw #2 = E17 test perturbation

cfg = dict(lam_min=0.0, lam_max=0.6, ds=0.01, n_overlaps=5)
br, fold, st = trace_branch(coup, BETA, cfg)
i_pk = int(np.argmax(br["lam"]))
lam_up, u_up = br["lam"][:i_pk + 1], br["u_star"][:i_pk + 1]
j = int(np.argmin(np.abs(lam_up - LAM)))
u_star, ok = woodbury_newton(coup, u_up[j], LAM, BETA)
a_star = np.linalg.lstsq(xi.T, u_star, rcond=None)[0]
print(f"[E17b] fold={fold:.5f}  a*: |a| top = "
      f"{np.array2string(np.sort(np.abs(a_star))[::-1][:3], precision=4)}", flush=True)

t0w = time.time()
sysP = ReducedDDE(xi, BETA, LAM, TAU, T0)
sol = sysP.integrate(a_star + pert, 1500.0, DT, record_every=25)
a_end = sol["a"][-1]
print(f"[E17b] sim t=1500 done ({time.time()-t0w:.0f}s)  "
      f"||a_end-a*||={np.linalg.norm(a_end-a_star):.4e}", flush=True)

# Newton-polish the final state in full-N coordinates
u_end = xi.T @ a_end
u_fin, ok2 = woodbury_newton(coup, u_end, LAM, BETA)
a_fin = np.linalg.lstsq(xi.T, u_fin, rcond=None)[0]
res_fin = np.linalg.norm(coup.field_F(u_fin, LAM, BETA)) / np.sqrt(N)
m1_star = float(xi[0] @ np.tanh(BETA * u_star) / N)
m1_fin = float(xi[0] @ np.tanh(BETA * u_fin) / N)
print(f"[E17b] polished final state: res={res_fin:.1e}  ok={ok2}", flush=True)
print(f"[E17b] ||a_fin - a*|| = {np.linalg.norm(a_fin - a_star):.4e}   "
      f"||a_fin - a_end|| = {np.linalg.norm(a_fin - a_end):.4e}", flush=True)
print(f"[E17b] m1(a*)={m1_star:.6f}   m1(a_fin)={m1_fin:.6f}", flush=True)
print(f"[E17b] top-4 |a| of a*   : "
      f"{np.array2string(np.sort(np.abs(a_star))[::-1][:4], precision=4)}", flush=True)
print(f"[E17b] top-4 |a| of a_fin: "
      f"{np.array2string(np.sort(np.abs(a_fin))[::-1][:4], precision=4)}", flush=True)

# leading delayed roots at both states
for name, u in [("a*  (branch)", u_star), ("a_fin (final)", u_fin)]:
    t0w = time.time()
    ph, _, _ = physical_roots(coup, u, BETA, LAM, TAU, T0,
                              M=64, n_cand=60, k=12, polish=True)
    order = np.argsort(-ph.real)
    tops = ph[order][:4]
    print(f"[E17b] {name}: leading roots "
          + "  ".join(f"({z.real:+.5f},{z.imag:+.4f}i)" for z in tops)
          + f"   ({time.time()-t0w:.0f}s)", flush=True)
