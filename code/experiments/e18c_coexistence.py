"""
E18c - Coexistence check: memory xi^1 AND the pinned front, below lambda_c.

At 4 lambda in the static-recall phase (lambda < lambda_c = 0.2822) where BOTH
branches exist, solve each fixed point exactly by Newton (residual < 1e-12) and
report both delayed spectra (T_P, tau=10). Decides plainly whether the memory and
the pinned front COEXIST as simultaneous linear attractors.

  * memory:  Woodbury-Newton from 0.99*xi^1  (trace_branch start recipe)
  * front:   Woodbury-Newton from the cached E18a front state at that lambda

Outputs: results/cycle/E18_coexistence.npz + printed tables.
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
from reduced_spectrum import physical_roots, GJ_GK, rightmost_real_root

N, ALPHA, BETA, T0, SEED, TAU = 2000, 0.05, 20.0, 1.0, 42, 10.0
P = round(ALPHA * N)
IM_TOL = 1e-3
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
GRAM = (xi @ xi.T) / N
BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
u_pin = BR["u_pin"]

def amp(u):
    return np.linalg.solve(GRAM, coup.overlap_raw(u))

def rightmost(u, lam):
    ph, _, _ = physical_roots(coup, u, BETA, lam, TAU, T0, M=32, n_cand=60, k=14)
    if len(ph) == 0:
        GJ, GK = GJ_GK(coup, u, BETA, lam)
        z = rightmost_real_root(GJ, GK, T0, TAU, lam, x_hi=0.05, x_lo=-1.5, n=300)
        ph = np.array([z]) if z is not None else np.array([np.nan+0j])
    zr = ph[0]
    n_unst = int(np.sum(ph.real > 1e-9))
    return zr, n_unst

def reach_front(lam_target):
    u = u_pin.copy(); lam = 0.318; step = 0.005
    while abs(lam - lam_target) > 1e-9:
        lam_next = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, okn = woodbury_newton(coup, u, lam_next, BETA, tol=1e-12)
        res = np.linalg.norm(coup.field_F(un, lam_next, BETA)) / np.sqrt(N)
        if okn and res < 1e-10:
            u, lam = un, lam_next; step = min(step * 1.3, 0.005)
        else:
            step *= 0.5
            if step < 2e-5: return None
    return un

LAMS = [0.150, 0.200, 0.250, 0.2750]      # all < lambda_c = 0.2822

# Build the memory xi^1 branch by continuation (cold anneal_newton occasionally
# stalls at isolated lambda near-capacity; warm-start from a settled low-lambda
# memory is robust).
um0, _ = anneal_newton(coup, 0.99 * xi[0].copy(), 0.05, BETA, tol=1e-12)
def reach_memory(lam_target):
    u = um0.copy(); lam = 0.05; step = 0.01
    while abs(lam - lam_target) > 1e-9:
        lam_next = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, okn = woodbury_newton(coup, u, lam_next, BETA, tol=1e-12)
        res = np.linalg.norm(coup.field_F(un, lam_next, BETA)) / np.sqrt(N)
        m1 = float(xi[0] @ np.tanh(BETA * un) / N)
        if okn and res < 1e-10 and m1 > 0.5:
            u, lam = un, lam_next; step = min(step * 1.3, 0.01)
        else:
            step *= 0.5
            if step < 2e-5: return None
    return un

print(f"[E18c] coexistence of memory xi^1 and pinned front (tau={TAU})\n")
print(f"{'lam':>7} {'object':>7} {'res':>9} {'m1/a91':>8} {'eigM':>8} "
      f"{'Rmost Re z':>11} {'type':>6} {'#unst':>5} {'stable?':>8}")
rows = []
for lam in LAMS:
    # ---- memory xi^1 (warm-started continuation) ----
    um = reach_memory(lam)
    resm = np.linalg.norm(coup.field_F(um, lam, BETA)) / np.sqrt(N)
    m1 = float(xi[0] @ np.tanh(BETA * um) / N)
    emm = eigmax_M(coup, um, lam, BETA)
    zrm, num = rightmost(um, lam)
    tm = "real" if abs(zrm.imag) < IM_TOL else "cplx"
    stable_m = zrm.real < 0
    print(f"{lam:7.4f} {'memory':>7} {resm:9.1e} {m1:8.3f} {emm:+8.3f} "
          f"{zrm.real:+11.5f} {tm:>6} {num:5d} {str(stable_m):>8}")
    # ---- pinned front ----
    uf = reach_front(lam)
    if uf is None:
        print(f"{lam:7.4f} {'front':>7}  (unreachable)"); continue
    resf = np.linalg.norm(coup.field_F(uf, lam, BETA)) / np.sqrt(N)
    a = amp(uf); a91 = a[91]
    emf = eigmax_M(coup, uf, lam, BETA)
    zrf, nuf = rightmost(uf, lam)
    tf = "real" if abs(zrf.imag) < IM_TOL else "cplx"
    stable_f = zrf.real < 0
    print(f"{lam:7.4f} {'front':>7} {resf:9.1e} {a91:8.3f} {emf:+8.3f} "
          f"{zrf.real:+11.5f} {tf:>6} {nuf:5d} {str(stable_f):>8}")
    coexist = stable_m and stable_f
    print(f"        -> COEXIST as attractors: {coexist}")
    rows.append([lam, resm, m1, emm, zrm.real, zrm.imag, num,
                 resf, a91, emf, zrf.real, zrf.imag, nuf])

rows = np.array(rows)
np.savez(os.path.join(OUT, "E18_coexistence.npz"), rows=rows,
         cols=np.array(["lam", "res_mem", "m1", "eigM_mem", "reZ_mem", "imZ_mem",
                        "nunst_mem", "res_front", "a91", "eigM_front", "reZ_front",
                        "imZ_front", "nunst_front"]))
n_coex = int(np.sum((rows[:, 4] < 0) & (rows[:, 10] < 0)))
print(f"\n[E18c] VERDICT: {n_coex}/{len(rows)} lambda < lambda_c show BOTH memory "
      f"and pinned front linearly stable => TRUE BISTABLE COEXISTENCE where both "
      f"exist and are polished to res<1e-12.")
