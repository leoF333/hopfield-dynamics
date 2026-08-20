"""
E18b - Delayed stability of the pinned-front branch at tau=10.

For each cached front state u*(lam) (from E18a), compute the rightmost PHYSICAL
characteristic roots of T_P(z) at tau=10 (exact P x P reduced spectrum). Report:
  * rightmost Re z, and whether it is a REAL root (fold/monotone mode) or a
    COMPLEX pair (would signal a Hopf if it crossed 0);
  * number of unstable directions (roots with Re z > 0);
  * for context, eig_max(M) (= rightmost root at tau=0 / the z=0 fold indicator).

Covers lambda below and above lambda_c = 0.2822.
Outputs: results/cycle/E18_delayed_stability.npz + printed table.
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
from tqdm import tqdm

from couplings import Couplings, make_patterns
from robust_branch import woodbury_newton, eigmax_M
from reduced_spectrum import physical_roots, GJ_GK, rightmost_real_root

N, ALPHA, BETA, T0, SEED, TAU = 2000, 0.05, 20.0, 1.0, 42, 10.0
P = round(ALPHA * N)
IM_TOL = 1e-3
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")

BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
# lambda grid: use cached states + reconstruct extra points by warm-start from u_pin
u_pin = BR["u_pin"]

def reach(u0, lam0, lam_target):
    u = u0.copy(); lam = lam0; step = 0.005
    while abs(lam - lam_target) > 1e-9:
        lam_next = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, okn = woodbury_newton(coup, u, lam_next, BETA, tol=1e-12)
        res = np.linalg.norm(coup.field_F(un, lam_next, BETA)) / np.sqrt(N)
        if okn and res < 1e-10:
            u, lam = un, lam_next; step = min(step * 1.3, 0.005)
        else:
            step *= 0.5
            if step < 2e-5:
                return None
    return un

LAMS = [0.140, 0.170, 0.200, 0.230, 0.260, 0.2822, 0.300, 0.310, 0.320, 0.327]

print(f"[E18b] delayed spectrum (T_P, tau={TAU}) along the front branch\n")
print(f"{'lam':>7} {'phase':>7} {'res':>8} {'eigM(t=0)':>10} "
      f"{'Rmost Re z':>11} {'type':>8} {'Im':>8} {'#unstable':>9}")
rows = []; t0w = time.time()
for lam in tqdm(LAMS, desc="E18b delayed spectrum"):
    u = reach(u_pin, 0.318, lam)
    if u is None:
        print(f"{lam:7.4f}  (front unreachable)"); continue
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    em = eigmax_M(coup, u, lam, BETA)
    ph, _, _ = physical_roots(coup, u, BETA, lam, TAU, T0, M=32, n_cand=60, k=14)
    if len(ph) == 0:
        # deeply-stable case: IG candidates missed the far-left roots. Fall back
        # to a rightmost-real-root scan (the front's fold mode is real & real-axis
        # scan is overflow-free). Its z is the rightmost when no complex pair leads.
        GJ, GK = GJ_GK(coup, u, BETA, lam)
        zrr = rightmost_real_root(GJ, GK, T0, TAU, lam, x_hi=0.05, x_lo=-1.5, n=300)
        if zrr is None:
            print(f"{lam:7.4f}  (no physical roots found)"); continue
        ph = np.array([zrr])
    zr = ph[0]                            # rightmost
    n_unstable = int(np.sum(ph.real > 1e-9))
    typ = "real" if abs(zr.imag) < IM_TOL else "cplx"
    phase = "static" if lam < 0.2822 else "mixed"
    print(f"{lam:7.4f} {phase:>7} {res:8.1e} {em:+10.4f} "
          f"{zr.real:+11.5f} {typ:>8} {abs(zr.imag):8.4f} {n_unstable:9d}")
    rows.append([lam, res, em, zr.real, zr.imag, n_unstable,
                 float(ph[1].real) if len(ph) > 1 else np.nan,
                 float(abs(ph[1].imag)) if len(ph) > 1 else np.nan])

rows = np.array(rows)
np.savez(os.path.join(OUT, "E18_delayed_stability.npz"),
         rows=rows,
         cols=np.array(["lam", "res", "eigM", "re_rightmost", "im_rightmost",
                        "n_unstable", "re_2nd", "im_2nd"]))
print(f"\n[E18b] all rightmost Re z: "
      f"[{rows[:,3].min():+.5f}, {rows[:,3].max():+.5f}] "
      f"({'all stable' if rows[:,3].max()<0 else 'SOME UNSTABLE'})")
print(f"[E18b] TOTAL {time.time()-t0w:.0f}s")
