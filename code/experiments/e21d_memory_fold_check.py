"""
E21d -- HONEST re-verification of the static memory-fold analysis (user request).

Claims under test (from REPORT_static_bifurcation + E21 unification):
  (1) The xi^1 memory branch ends in a genuine SADDLE-NODE at lambda_c ~ 0.2822:
      the branch TURNS (arclength), and a single REAL eigenvalue of M crosses 0
      exactly at the turning point.
  (2) The P x P Sylvester eigenvalue machinery (eigmax_M) agrees with a fully
      INDEPENDENT dense N x N computation (scipy eigvals on the explicit Jacobian).
  (3) At its fold, the xi^1 memory is itself a tilted 2-pattern mixture with the
      same shape as the bond-91 "pinned front" at ITS fold (0.664, 0.324) -- i.e.
      memory and front are the same branch family, each dying at its own
      quenched threshold (lambda_c^mu; depinning lambda* = max_mu).

Reuses the pseudo-arclength machinery of e21a_front_birth (tangent, bordered
corrector, Sylvester spectrum) with the branch tracked on patterns (0,1).
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
from scipy.linalg import eigvals as dense_eigvals

import e21a_front_birth as FB
from robust_branch import woodbury_newton, eigmax_M

# track the xi^1 branch: leading pattern index 0, cyclic successor 1
FB.MU_A, FB.MU_B = 0, 1
coup, xi, xis, N, P, BETA = FB.coup, FB.xi, FB.xis, FB.N, FB.P, FB.BETA

OUT = FB.OUT


def arclength_up(u0, lam0, ds0=0.02, ds_min=1e-5, ds_max=0.04, s_max=6.0):
    """Arclength continuation going UP in lambda, through the fold and back down.
    Returns rows [s, lam, res, eig_max, eig_closest_re, eig_closest_im, a0, a1,
    dlam_tangent], turned flag, lam_max reached."""
    rows = []
    u = u0.copy(); lam = lam0
    prev_dir = np.zeros(N + 1); prev_dir[-1] = +1.0
    ds = ds0; s = 0.0
    turned = False; lam_max_seen = lam
    a = FB.amp(u)
    re0, im0, remax = FB.eig_closest_to_zero(u, lam)
    rows.append([s, lam, FB.residual(u, lam), remax, re0, im0, a[0], a[1], +1.0])
    while s < s_max:
        tang = FB.tangent(u, lam, prev_dir)
        u_p = u + ds * tang[:-1]; lam_p = lam + ds * tang[-1]
        un, ln, ok, res = FB.bordered_corrector(u_p, lam_p, u, lam, tang, ds)
        aa = FB.amp(un)
        top = sorted(int(k) for k in np.argsort(np.abs(aa))[::-1][:2])
        on_branch = (top == [0, 1]) and (0.30 < abs(aa[0]) < 1.05)
        if ok and res < 1e-9 and on_branch:
            s += ds; u, lam = un, ln; prev_dir = tang
            re0, im0, remax = FB.eig_closest_to_zero(u, lam)
            rows.append([s, lam, res, remax, re0, im0, aa[0], aa[1], tang[-1]])
            lam_max_seen = max(lam_max_seen, lam)
            if tang[-1] < 0 and lam < lam_max_seen - 1e-4:
                turned = True
            ds = min(ds * 1.25, ds_max)
            # once turned, follow the unstable side a bit then stop
            if turned and lam < lam_max_seen - 0.02:
                break
        else:
            ds *= 0.5
            if ds < ds_min:
                print(f"  corrector stalled at lam={lam:.5f} (res={res:.1e})")
                break
    return np.array(rows), turned, lam_max_seen, u, lam


def dense_check(u, lam, tag):
    """Independent dense N x N Jacobian eigenvalue vs P x P Sylvester."""
    t0 = time.time()
    g = coup.compute_gain(u, BETA)                       # (N,)
    Ut = (1.0 - lam) * xi + lam * xis                    # (P,N)
    C = (Ut.T @ xi) / N                                  # (N,N) explicit Jacobian part
    M = -np.eye(N) + C * g[None, :]                      # M = -I + C diag(gain)
    ev = dense_eigvals(M)                                # full complex spectrum, N=2000
    re_max_dense = float(ev.real.max())
    re_max_sylv = eigmax_M(coup, u, lam, BETA)
    print(f"  [{tag}] lam={lam:.5f}  dense eig_max={re_max_dense:+.10f}  "
          f"Sylvester={re_max_sylv:+.10f}  |diff|={abs(re_max_dense-re_max_sylv):.2e}"
          f"  [{time.time()-t0:.0f}s]")
    return re_max_dense, re_max_sylv


if __name__ == "__main__":
    print("[E21d] === honest re-verification of the xi^1 memory fold ===\n")

    # -- seed the memory branch at lam=0.15
    u0, ok = woodbury_newton(coup, 2.0 * xi[0], 0.15, BETA)
    a0 = FB.amp(u0)
    print(f"[E21d] seed @0.15: ok={ok} res={FB.residual(u0,0.15):.1e} "
          f"a1={a0[0]:+.4f} a2={a0[1]:+.4f} eigM={eigmax_M(coup,u0,0.15,BETA):+.4f}\n")

    # -- arclength UP through the presumed fold
    rows, turned, lam_max, u_end, lam_end = arclength_up(u0, 0.15)
    print(f"\n[E21d] arclength: {len(rows)} points, TURNED={turned}, "
          f"lam_max (fold) = {lam_max:.5f}  (claimed lambda_c = 0.2822)")
    print(f"{'s':>6} {'lam':>8} {'res':>9} {'eig_max':>9} {'eig_near0':>10} "
          f"{'a1':>7} {'a2':>7} {'dlam_t':>7}")
    keep = np.unique(np.linspace(0, len(rows) - 1, 18).astype(int))
    for i in keep:
        r = rows[i]
        print(f"{r[0]:6.2f} {r[1]:8.4f} {r[2]:9.1e} {r[3]:+9.4f} {r[4]:+10.4f} "
              f"{r[6]:7.3f} {r[7]:7.3f} {r[8]:+7.3f}")

    # shape and eigenvalue at the fold (row with max lam)
    j = int(np.argmax(rows[:, 1]))
    print(f"\n[E21d] AT THE FOLD: lam={rows[j,1]:.5f}  (a1,a2)=({rows[j,6]:.3f},{rows[j,7]:.3f})"
          f"  ratio a2/a1={rows[j,7]/rows[j,6]:.3f}")
    print(f"        eig closest to 0: {rows[j,4]:+.5f} + {rows[j,5]:.5f}i  (REAL crossing expected)")
    print(f"        front-91 at ITS fold (E18a): lam=0.3276 (a91,a92)=(0.664,0.324) ratio 0.488")

    # -- independent dense check at two states: mid-branch and near-fold
    print(f"\n[E21d] independent DENSE N x N check (scipy eigvals, N={N}):")
    jmid = int(np.argmin(np.abs(rows[:, 1] - 0.20)))
    # rebuild states by Newton at those lam (stable side)
    umid, _ = woodbury_newton(coup, 2.0 * xi[0], 0.20, BETA)
    dense_check(umid, 0.20, "mid-branch")
    # near-fold state: from the endpoint of the stable side before turning
    lam_nf = rows[j, 1] - 0.002
    unf, oknf = woodbury_newton(coup, u_end if abs(lam_end - lam_nf) < 0.05 else 2.0 * xi[0],
                                lam_nf, BETA)
    if oknf:
        dense_check(unf, lam_nf, "near-fold")
    else:
        print("  near-fold Newton failed (expected very close to fold); skipping")

    np.savez(os.path.join(OUT, "E21d_memory_fold_check.npz"),
             rows=rows, turned=turned, lam_fold=lam_max,
             cols=np.array(["s", "lam", "res", "eig_max", "eig_near0_re",
                            "eig_near0_im", "a1", "a2", "dlam_tangent"]))
    print(f"\n[E21d] saved {os.path.join(OUT, 'E21d_memory_fold_check.npz')}")
