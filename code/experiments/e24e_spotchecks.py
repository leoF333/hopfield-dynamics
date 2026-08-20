"""
E24e -- delayed-stability (T_P, tau=10) spot-checks along 5 representative
memory->front branches, chosen at the percentiles {min, 25%, 50%, 75%, max} of the
threshold distribution lambda_c(mu) at N=2000, seed 42.

Purpose: confirm that what E18b/E21c established on the bond-91 branch (0 unstable
delayed directions all along; the branch dies at its fold, never by Hopf) holds
across the distribution -- i.e. that "the static attractor exists" (Newton) implies
"the static attractor is a DDE attractor" (rightmost Re z < 0) up to lambda_c(mu).

Output: printed table + results/cycle/E24e_spotchecks.npz.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import e21c_tau_nature as TC          # reuse coup, xi, rightmost_root, walk helpers
from robust_branch import woodbury_newton, eigmax_M

BETA, TAU = 20.0, 10.0
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")

d = np.load(os.path.join(BASE, "E24_curves_N2000_s42.npz"))
lam_c = d["lam_c"]; P = int(d["P"])
order = np.argsort(lam_c)
REPS = [int(order[0]), int(order[P // 4]), int(order[P // 2]),
        int(order[3 * P // 4]), int(order[-1])]
print(f"[E24e] representative branches (percentiles of lam_c): "
      f"{[(mu, round(float(lam_c[mu]), 4)) for mu in REPS]}")

coup, xi = TC.coup, TC.xi
rows = []
t0 = time.time()
for mu in REPS:
    # walk this branch with warm-started Newton, probing T_P at ~6 lambdas
    lams = np.linspace(0.10, lam_c[mu] - 0.0015, 6)
    u = 2.0 * xi[mu]
    lam_prev = 0.05
    un, ok = woodbury_newton(coup, u, lam_prev, BETA, tol=1e-12)
    u = un
    for lt in lams:
        # continue from lam_prev to lt in small steps
        lam = lam_prev
        good = True
        while lam < lt - 1e-9:
            ln = min(lam + 0.004, lt)
            un, ok = woodbury_newton(coup, u, ln, BETA, tol=1e-12)
            if not ok:
                good = False; break
            u, lam = un, ln
        if not good:
            print(f"  mu={mu} lam={lt:.4f}: continuation failed (too close to fold)")
            continue
        lam_prev = lt
        em = eigmax_M(coup, u, lt, BETA)
        re, im, isc = TC.rightmost_root(u, lt, TAU)
        n_unst = int(re > 0)
        rows.append([mu, lt, lam_c[mu], em, re, im, float(isc)])
        print(f"  mu={mu:3d} lam={lt:.4f} (lam_c={lam_c[mu]:.4f}): eigM={em:+.4f} "
              f"rightmost Re z={re:+.5f} Im={im:+.4f} {'CPLX' if isc else 'real'} "
              f"{'*** UNSTABLE ***' if n_unst else ''}", flush=True)

R = np.array(rows)
np.savez(os.path.join(BASE, "E24e_spotchecks.npz"), rows=R,
         cols=np.array(["mu", "lam", "lam_c", "eigM", "re_z", "im_z", "is_cplx"]))
n_unst = int((R[:, 4] > 0).sum())
print(f"\n[E24e] {len(R)} probes on 5 branches: {n_unst} unstable points "
      f"(expected 0 -> existence implies delayed stability up to the fold)")
print(f"[E24e] total {time.time()-t0:.0f}s ; npz saved")
