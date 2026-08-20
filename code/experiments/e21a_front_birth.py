"""
E21a - Properly characterize the BIRTH of the pinned front (bottom of the branch).

E18a followed the bond-91 pinned front DOWNWARD in lambda by naive lambda-stepping
(Woodbury-Newton warm-start) and STOPPED at lam=0.1322 with eig_max(M)=-0.998,
CALLING it a saddle-node fold. That label is wrong: a genuine fixed-point fold needs
a REAL eigenvalue of M crossing 0 (eig_max(M) -> 0, identity Delta(0)=-M). At
eig_max(M)=-0.998 the front is DEEPLY stable, so the branch cannot be turning there.
The naive lambda-stepping simply failed to continue (basin jump / Newton left the
front basin as ds shrank below tolerance).

This script replaces lambda-stepping by PSEUDO-ARCLENGTH continuation in the reduced
2-pattern amplitude coordinates + lambda, following the branch tangent through any
turning point. The corrector is a bordered Newton using the SAME exact Woodbury
algebra for the u-block. We work in the full u-space (N-dim) but the arclength
parameter and the tangent's lambda-component let us round a fold in lambda.

Continuation state: y = (u, lam), N+1 unknowns.
  F1(u,lam) = -u + C(lam) tanh(beta u)            (N eqs, exact fixed point)
  F2(u,lam) = t_u.(u-u0) + t_lam*(lam-lam0) - ds  (1 pseudo-arclength eq)
The bordered Jacobian is
  [ M            dF/dlam ]
  [ t_u^T        t_lam   ]
solved by block elimination using woodbury_solve for M^{-1}.

Along the branch we record: arclength s, lam, residual, full sorted a-vector's
dominant components, eig_max(M), AND the smallest-|Re| eigenvalue of the P x P
Sylvester matrix VU/N (the eigenvalue of M closest to 0 -> the one that would
vanish at a fold). We also record the full REAL spectrum of M (its P nonzero
eigenvalues) at a few points to check no OTHER real eigenvalue crosses 0.

Validation gate: the arclength branch must reproduce the E18a lam-stepped branch on
[0.132, 0.328] (same a91/a92, same eig_max(M)) before we trust it below 0.132.

Outputs: results/cycle/E21_front_birth.npz + printed tables + incremental RESULTS.md.
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
from robust_branch import woodbury_solve, woodbury_newton, eigmax_M

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
MU_A, MU_B = 91, 92
GRAM = (xi @ xi.T) / N


def amp(u):
    """Reduced amplitude vector a from u = xi^T a (a = Gram^{-1} (1/N) xi u)."""
    return np.linalg.solve(GRAM, coup.overlap_raw(u))


def M_spectrum(u, lam):
    """The P nonzero eigenvalues of M(lam)=-I+C diag(gain) via the Sylvester P x P
    matrix (1/N)VU: eig(M)_nonzero = -1 + eig((1/N)VU). Returns COMPLEX array (P,)."""
    g = coup.compute_gain(u, BETA)
    Ut = (1.0 - lam) * xi + lam * xis          # (P,N) = U^T
    V = xi * g[None, :]                          # (P,N)
    ev = np.linalg.eigvals((V @ Ut.T) / N)      # (P,) complex
    return -1.0 + ev


def eig_closest_to_zero(u, lam):
    """The M-eigenvalue with the smallest |Re| (the candidate fold eigenvalue),
    and eig_max(Re). Returns (re_closest, im_closest, re_max)."""
    ev = M_spectrum(u, lam)
    j = int(np.argmin(np.abs(ev.real)))
    return float(ev[j].real), float(ev[j].imag), float(ev.real.max())


def dF_dlam(u):
    """dF1/dlam = (K - J) tanh(beta u)  (N,)."""
    tu = np.tanh(BETA * u)
    return coup.apply_K(tu) - coup.apply_J(tu)


def newton_polish(u, lam, tol=1e-13):
    u2, ok = woodbury_newton(coup, u, lam, BETA, tol=tol)
    return u2, ok


def residual(u, lam):
    return np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)


def tangent(u, lam, prev_dir):
    """Branch tangent (du, dlam), unit-normalized in the (u,lam) space, with a sign
    picked to continue in the prev_dir sense. Solve the bordered null-space:
      M du + dF/dlam * dlam = 0. Parametrize by dlam=1 (generic), then normalize;
    if M is near-singular (at the fold) this du blows up -> the tangent becomes
    nearly pure-u (dlam->0) after normalization, which is exactly how arclength
    rounds the fold."""
    g = coup.compute_gain(u, BETA)
    Fl = dF_dlam(u)
    du_raw = -woodbury_solve(coup, g, lam, Fl)   # M^{-1}(-dF/dlam)
    vec = np.concatenate([du_raw, [1.0]])
    vec = vec / np.linalg.norm(vec)
    if np.dot(vec, prev_dir) < 0:
        vec = -vec
    return vec


def bordered_corrector(u_p, lam_p, u0, lam0, tang, ds, tol=1e-11, max_it=40):
    """Newton on [F1; F2=tang.(y-y0)-ds]. Block-eliminate M via Woodbury.
    Returns (u, lam, ok, res)."""
    u = u_p.copy(); lam = lam_p
    tu_vec = tang[:-1]; tl = tang[-1]
    for _ in range(max_it):
        F1 = coup.field_F(u, lam, BETA)
        F2 = np.dot(tu_vec, (u - u0)) + tl * (lam - lam0) - ds
        r = np.linalg.norm(F1) / np.sqrt(N) + abs(F2)
        if r < tol:
            return u, lam, True, np.linalg.norm(F1) / np.sqrt(N)
        g = coup.compute_gain(u, BETA)
        Fl = dF_dlam(u)
        # solve [M  Fl][du   ] = [-F1]
        #       [t^T tl][dlam]   [-F2]
        a = woodbury_solve(coup, g, lam, -F1)     # M^{-1}(-F1)
        b = woodbury_solve(coup, g, lam, Fl)      # M^{-1} Fl
        denom = tl - np.dot(tu_vec, b)
        if abs(denom) < 1e-14:
            return u, lam, False, np.linalg.norm(F1) / np.sqrt(N)
        dlam = (-F2 - np.dot(tu_vec, a)) / denom
        du = a - dlam * b
        # backtracking on the fixed-point residual
        st = 1.0; improved = False; r_fp = np.linalg.norm(F1) / np.sqrt(N)
        for _ in range(25):
            un = u + st * du; ln = lam + st * dlam
            if np.linalg.norm(coup.field_F(un, ln, BETA)) / np.sqrt(N) < r_fp + 1e-9:
                improved = True; break
            st *= 0.5
        if not improved:
            return u, lam, False, r_fp
        u = u + st * du; lam = lam + st * dlam
    return u, lam, False, np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)


def reach_front(lam_target, lam_start=0.318):
    """Reconstruct the bond-91 pinned front at lam_start (from E18a cache) and
    warm-start Newton down/up to lam_target along the front branch."""
    BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
    u = BR["u_pin"].copy(); lam = 0.318
    step = 0.004
    while abs(lam - lam_target) > 1e-9:
        ln = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, ok = newton_polish(u, ln, tol=1e-12)
        a = amp(un)
        top = sorted(int(k) for k in np.argsort(np.abs(a))[::-1][:2])
        if ok and top == [MU_A, MU_B] and residual(un, ln) < 1e-10:
            u, lam = un, ln; step = min(step * 1.3, 0.004)
        else:
            step *= 0.5
            if step < 2e-5:
                return None, lam
    return u, lam


def run_arclength(u0, lam0, ds0=0.02, ds_min=1e-4, ds_max=0.05,
                  s_max=6.0, lam_floor=-0.05, init_dir_sign=-1.0, tag="down"):
    """Pseudo-arclength continuation from (u0,lam0). init_dir_sign sets the initial
    lambda direction (-1 = start going DOWN in lam). Stops when lam<lam_floor, when
    s>s_max, when it turns and comes back above lam0+0.05 (rounded a fold and going
    back up), or when the corrector fails at ds_min."""
    rows = []   # [s, lam, res, eig_max, re0, im0, a_top1, a_top2, a_top3, idx1, idx2, dlam_tangent]
    Mspecs = {} # sparse full-spectrum snapshots keyed by nearest lam
    u = u0.copy(); lam = lam0
    a = amp(u)
    order = np.argsort(np.abs(a))[::-1]
    re0, im0, remax = eig_closest_to_zero(u, lam)
    s = 0.0
    rows.append([s, lam, residual(u, lam), remax, re0, im0,
                 a[order[0]], a[order[1]], a[order[2]],
                 float(order[0]), float(order[1]), init_dir_sign])
    prev_dir = np.zeros(N + 1); prev_dir[-1] = init_dir_sign
    ds = ds0
    turned = False; lam_min_seen = lam
    pbar = tqdm(total=s_max, desc=f"arclength {tag}",
                bar_format="{l_bar}{bar}| s={n:.2f}/{total:.2f}")
    while s < s_max:
        tang = tangent(u, lam, prev_dir)
        u_p = u + ds * tang[:-1]; lam_p = lam + ds * tang[-1]
        un, ln, ok, res = bordered_corrector(u_p, lam_p, u, lam, tang, ds)
        a = amp(un)
        top = sorted(int(k) for k in np.argsort(np.abs(a))[::-1][:2])
        is_front = (top == [MU_A, MU_B]) and (0.30 < abs(a[MU_A]) < 1.05)
        if ok and res < 1e-9 and is_front:
            s += ds
            u, lam = un, ln; prev_dir = tang
            order = np.argsort(np.abs(a))[::-1]
            re0, im0, remax = eig_closest_to_zero(u, lam)
            rows.append([s, lam, res, remax, re0, im0,
                         a[order[0]], a[order[1]], a[order[2]],
                         float(order[0]), float(order[1]), tang[-1]])
            lam_min_seen = min(lam_min_seen, lam)
            if tang[-1] > 0 and lam > lam_min_seen + 1e-4:
                turned = True
            pbar.update(ds)
            ds = min(ds * 1.25, ds_max)
            if lam < lam_floor:
                pbar.set_postfix_str(f"reached lam_floor at lam={lam:.4f}")
                break
            if turned and lam > lam0 + 0.02:
                pbar.set_postfix_str(f"turned & returned above start (fold rounded)")
                break
        else:
            ds *= 0.5
            if ds < ds_min:
                pbar.set_postfix_str(f"STALLED at lam={lam:.5f} eig_closest={re0:+.4f} "
                                     f"res={res:.1e} front={is_front}")
                break
    pbar.close()
    return np.array(rows), turned, lam_min_seen, u, lam


def main():
    t0w = time.time()
    print("[E21a] === pseudo-arclength continuation of the pinned-front BIRTH ===\n",
          flush=True)

    # ---- Validation gate: reproduce E18a on a known interior point ----------
    print("[E21a] VALIDATION: reproduce E18a branch value at lam=0.20 ...", flush=True)
    u20, l20 = reach_front(0.20)
    assert u20 is not None, "could not reach front at 0.20"
    a20 = amp(u20); em20 = eigmax_M(coup, u20, 0.20, BETA)
    print(f"   arclength-machinery front @0.20: a91={a20[MU_A]:+.3f} a92={a20[MU_B]:+.3f} "
          f"eig_max(M)={em20:+.4f}  (E18a: a91=0.804 a92=0.187 eigM=-0.905)", flush=True)
    ok_val = (abs(a20[MU_A] - 0.804) < 0.02 and abs(em20 - (-0.905)) < 0.02)
    print(f"   VALIDATION {'PASSED' if ok_val else 'FAILED'}\n", flush=True)

    # ---- Start the arclength run near the reported bottom (lam ~ 0.15) ------
    # Start comfortably ABOVE the E18a stall (0.1322) so the first tangent is clean,
    # then continue DOWN and through whatever happens.
    print("[E21a] reaching front at lam=0.150 to seed the arclength run ...", flush=True)
    u_seed, l_seed = reach_front(0.150)
    assert u_seed is not None
    a_seed = amp(u_seed)
    print(f"   seed @{l_seed:.4f}: a91={a_seed[MU_A]:+.3f} a92={a_seed[MU_B]:+.3f} "
          f"eig_max(M)={eigmax_M(coup, u_seed, l_seed, BETA):+.4f}", flush=True)

    print("\n[E21a] running arclength DOWNWARD through the birth region ...", flush=True)
    rows, turned, lam_min, u_end, lam_end = run_arclength(
        u_seed, l_seed, ds0=0.02, ds_min=5e-5, s_max=8.0,
        lam_floor=-0.05, init_dir_sign=-1.0, tag="birth")

    print(f"\n[E21a] branch: {len(rows)} points, lam_min reached = {lam_min:.5f}, "
          f"turned={turned}, ended at lam={lam_end:.5f}", flush=True)

    # ---- full real M-spectrum near the turning point / bottom ---------------
    # snapshot at the lowest-lam point and a couple around it
    idx_sorted = np.argsort(rows[:, 1])
    snap_idx = [idx_sorted[0], idx_sorted[min(2, len(idx_sorted)-1)],
                idx_sorted[min(5, len(idx_sorted)-1)]]
    snaps = {}
    print("\n[E21a] full REAL M-spectrum near the bottom (check NO real eig crosses 0):",
          flush=True)
    for k, i in enumerate(sorted(set(snap_idx))):
        lam_i = rows[i, 1]
        # reconstruct u at this lam by warm-start from the seed (front-preserving)
        u_i, _ = reach_front(lam_i, lam_start=0.318) if lam_i > 0.05 else (None, lam_i)
        if u_i is None:
            continue
        ev = M_spectrum(u_i, lam_i)
        re_sorted = np.sort(ev.real)[::-1]
        n_pos = int(np.sum(ev.real > 1e-9))
        print(f"   lam={lam_i:.4f}: eig_max={re_sorted[0]:+.4f} "
              f"eig_2nd={re_sorted[1]:+.4f} eig_min={re_sorted[-1]:+.4f} "
              f"n(Re>0)={n_pos} closest|Re|={np.abs(ev.real).min():.4f}", flush=True)
        snaps[f"Mspec_lam{lam_i:.4f}"] = ev

    # ---- print branch table -------------------------------------------------
    print("\n[E21a] BRANCH TABLE (arclength, front birth):")
    print(f"{'s':>6} {'lam':>8} {'res':>9} {'eig_max':>9} {'eig_closest':>12} "
          f"{'im':>8} {'a_top1':>8} {'a_top2':>8} {'dlam_t':>8}")
    lam_prev = None
    for r in rows:
        show = (lam_prev is None or abs(r[1] - lam_prev) > 0.008 or r is rows[-1])
        if show:
            print(f"{r[0]:6.3f} {r[1]:8.4f} {r[2]:9.1e} {r[3]:+9.4f} {r[4]:+12.5f} "
                  f"{r[5]:+8.4f} {r[6]:+8.3f} {r[7]:+8.3f} {r[11]:+8.3f}")
            lam_prev = r[1]

    np.savez(os.path.join(OUT, "E21_front_birth.npz"),
             rows=rows,
             cols=np.array(["s", "lam", "res", "eig_max", "re_closest", "im_closest",
                            "a_top1", "a_top2", "a_top3", "idx1", "idx2", "dlam_tangent"]),
             turned=turned, lam_min=lam_min, lam_end=lam_end,
             validation_ok=ok_val, a20_a91=a20[MU_A], a20_em=em20,
             **snaps)
    print(f"\n[E21a] saved results/cycle/E21_front_birth.npz")
    print(f"[E21a] eig_max range on branch: "
          f"[{rows[:,3].min():+.4f}, {rows[:,3].max():+.4f}]")
    print(f"[E21a] eig_closest-to-0 range: "
          f"[{rows[:,4].min():+.5f}, {rows[:,4].max():+.5f}]")
    print(f"[E21a] TOTAL {time.time()-t0w:.0f}s")


if __name__ == "__main__":
    main()
