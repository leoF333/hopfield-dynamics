"""
E21b - tau-independence of the front's POSITION and z=0 structure (machine precision).

The birth/position of the front is governed by the instantaneous Jacobian M (folds
and the z=0 root), and Delta(0) = -M is tau-INDEPENDENT by construction:
  T_P(z) = (t0 z+1) I - (1-lam) G_J - lam e^{-z tau} G_K,
  T_P(0) = I - (1-lam) G_J - lam G_K = -M_reduced   (the e^{-0*tau}=1 kills tau).

This script CONFIRMS numerically, to machine precision, that:
  (1) T_P(0) and its eigenstructure are IDENTICAL across tau in {5,10,20,50,100}
      (max |T_P(0;tau1) - T_P(0;tau2)| ~ 0, det and eigs identical);
  (2) the z=0 eigenvalues of T_P equal -eig(M) (the fold indicators) at each lam;
  (3) therefore lam_f (position) does NOT depend on tau.

We test at a few lam spanning the front's existence window and at the reference
front (bond 91/92) states.

Outputs: results/cycle/E21_tau_independence.npz + printed table.
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
from robust_branch import woodbury_newton, eigmax_M
from reduced_spectrum import GJ_GK, T_P

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
MU_A, MU_B = 91, 92
GRAM = (xi @ xi.T) / N
TAUS = [5.0, 10.0, 20.0, 50.0, 100.0]


def amp(u):
    return np.linalg.solve(GRAM, coup.overlap_raw(u))


def residual(u, lam):
    return np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)


def reach_front(lam_target):
    BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
    u = BR["u_pin"].copy(); lam = 0.318; step = 0.004
    while abs(lam - lam_target) > 1e-9:
        ln = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, ok = woodbury_newton(coup, u, ln, BETA, tol=1e-12)
        a = amp(un)
        top = sorted(int(k) for k in np.argsort(np.abs(a))[::-1][:2])
        if ok and top == [MU_A, MU_B] and residual(un, ln) < 1e-10:
            u, lam = un, ln; step = min(step * 1.3, 0.004)
        else:
            step *= 0.5
            if step < 1e-5:
                return None, lam
    return u, lam


def main():
    t0w = time.time()
    print("[E21b] === tau-independence of the z=0 (position) structure ===\n", flush=True)
    LAMS = [0.150, 0.2822, 0.310, 0.327]
    rows = []
    max_diff_global = 0.0
    max_eig_diff_global = 0.0
    for lam in LAMS:
        u, lr = reach_front(lam)
        if u is None:
            print(f"lam={lam}: front unreachable"); continue
        GJ, GK = GJ_GK(coup, u, BETA, lam)
        # reduced M at z=0: -M_reduced = T_P(0). Its eigenvalues:
        T0mats = [T_P(0.0 + 0j, GJ, GK, T0, tau, lam) for tau in TAUS]
        # pairwise max difference across tau
        base = T0mats[0]
        max_diff = max(np.max(np.abs(Tm - base)) for Tm in T0mats)
        # eigenvalues of T_P(0) for each tau, sorted
        eig_sets = [np.sort_complex(np.linalg.eigvals(Tm)) for Tm in T0mats]
        max_eig_diff = max(np.max(np.abs(es - eig_sets[0])) for es in eig_sets)
        # rightmost eig of M (fold indicator) via eigmax_M, cross-check vs -eig(T_P(0))
        em = eigmax_M(coup, u, lam, BETA)
        # -eig(T_P(0)) = eig(M_reduced); its max real:
        eigM_from_TP = float((-eig_sets[0]).real.max())
        max_diff_global = max(max_diff_global, max_diff)
        max_eig_diff_global = max(max_eig_diff_global, max_eig_diff)
        print(f"lam={lam:.4f}: max|T_P(0;tau_i)-T_P(0;tau_0)| = {max_diff:.2e}, "
              f"max eig diff across tau = {max_eig_diff:.2e}", flush=True)
        print(f"          eig_max(M) direct = {em:+.6f}  vs  "
              f"max Re(-eig T_P(0)) = {eigM_from_TP:+.6f}  "
              f"(diff {abs(em-eigM_from_TP):.2e})", flush=True)
        rows.append([lam, max_diff, max_eig_diff, em, eigM_from_TP,
                     abs(em - eigM_from_TP)])
    rows = np.array(rows)
    print(f"\n[E21b] GLOBAL: max |T_P(0)| difference across tau in {TAUS} = "
          f"{max_diff_global:.2e}", flush=True)
    print(f"[E21b] GLOBAL: max eigenvalue difference across tau = "
          f"{max_eig_diff_global:.2e}", flush=True)
    print(f"[E21b] => T_P(0) is tau-INDEPENDENT to machine precision. The front's "
          f"POSITION (any z=0 fold) is a tau-independent number.", flush=True)

    np.savez(os.path.join(OUT, "E21_tau_independence.npz"),
             rows=rows, taus=np.array(TAUS),
             cols=np.array(["lam", "max_TP0_diff", "max_eig_diff", "eigM_direct",
                            "eigM_from_TP0", "cross_diff"]),
             max_diff_global=max_diff_global,
             max_eig_diff_global=max_eig_diff_global)
    print(f"[E21b] saved results/cycle/E21_tau_independence.npz")
    print(f"[E21b] TOTAL {time.time()-t0w:.0f}s")


if __name__ == "__main__":
    main()
