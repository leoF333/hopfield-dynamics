"""
E21c - tau-dependence of the front's NATURE / stability boundary (medium & large tau).

The front's POSITION (any z=0 fold) is tau-independent (E21b). The NATURE of its
stability boundary can still depend on tau: at large tau a COMPLEX pair (Hopf) may
cross Re z=0 at some lam INSIDE the existence window, before the real fold at
lam*~0.328 (just as the memory Hopf-destabilizes at tau*~52 for alpha=0.07).

For each tau in {5,10,20,50,100} and a grid of lam spanning [~0.15, 0.327], we take
the EXACT front fixed point u*(lam) (bond 91/92, tau-independent) and compute the
rightmost PHYSICAL root of T_P(z) (exact P x P reduced spectrum). We record its
real part and whether it is REAL (fold mode) or COMPLEX (Hopf mode), and count
unstable directions. This maps, per tau, the front's stability boundary and its
type.

Method: rightmost real root via overflow-free sigma_min real-axis scan; rightmost
complex pair via Beyn contour (a disk covering the low-|Im| region). We report
max(re_real, re_cplx) as the rightmost, tagged by which won.

Outputs: results/cycle/E21_tau_nature.npz + printed table per tau.
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
from reduced_spectrum import (GJ_GK, rightmost_real_root, physical_roots,
                              sigmin_TP, T_P as TP_mat)

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
MU_A, MU_B = 91, 92
GRAM = (xi @ xi.T) / N
TAUS = [5.0, 10.0, 20.0, 50.0]   # tau=100 dropped: exp(-z*tau) overflows the
# root polisher; the erosion is already monotone (no crossing) over 5..50, and
# E15 separately probes the front at tau=100 (delay-robust, no Hopf).
IM_TOL = 1e-3


def amp(u):
    return np.linalg.solve(GRAM, coup.overlap_raw(u))


def residual(u, lam):
    return np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)


def reach_front(u0, lam0, lam_target):
    u = u0.copy(); lam = lam0; step = 0.004
    if abs(lam0 - lam_target) < 1e-9:
        un, ok = woodbury_newton(coup, u, lam_target, BETA, tol=1e-12)
        return un if ok else None
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
                return None
    return un


def rightmost_root(u, lam, tau):
    """Rightmost PHYSICAL characteristic root of T_P(z), using the SAME validated
    machinery as E18b: reduced-IG operator candidates + sigma_min physicality
    filter (physical_roots), with the overflow-free real-axis sigma_min scan as a
    fallback when the IG candidates miss the deeply-stable real fold mode.
    Returns (re, im, is_complex). Every returned root is verified physical
    (sigma_min(T_P) << |t0 z+1|)."""
    GJ, GK = GJ_GK(coup, u, BETA, lam)
    ph, cand, sig = physical_roots(coup, u, BETA, lam, tau, T0,
                                   M=40, n_cand=80, thresh=0.05, k=16)
    # bound the lower search so exp(-z*tau) cannot overflow at large tau
    # (physical real roots sit near 0 for large tau; -30/tau brackets them safely)
    x_lo = max(-1.5, -30.0 / tau)
    zr = rightmost_real_root(GJ, GK, T0, tau, lam, x_hi=0.05, x_lo=x_lo, n=300)
    cands = list(ph)
    if zr is not None:
        cands.append(zr)
    if not cands:
        return -np.inf, 0.0, False
    cands = np.array(cands, complex)
    # keep only verified-physical roots (guard against IG spurious that slipped through)
    keep = []
    for z in cands:
        scale = max(abs(T0 * z + 1.0), 1e-6)
        if sigmin_TP(z, GJ, GK, T0, tau, lam) < 1e-3 * scale:
            keep.append(z)
    if not keep:
        return -np.inf, 0.0, False
    keep = np.array(keep, complex)
    j = int(np.argmax(keep.real))
    zbest = keep[j]
    isc = abs(zbest.imag) >= IM_TOL
    return float(zbest.real), float(abs(zbest.imag)), bool(isc)


def main():
    t0w = time.time()
    print("[E21c] === tau-dependence of the front's stability nature ===\n", flush=True)

    LAMS = np.array([0.150, 0.200, 0.250, 0.2822, 0.300, 0.310, 0.318,
                     0.322, 0.325, 0.327])
    # cache front states once (tau-independent)
    print("[E21c] caching tau-independent front states along lam ...", flush=True)
    BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
    u_pin = BR["u_pin"]
    ustates = {}
    for lam in tqdm(LAMS, desc="front states"):
        u = reach_front(u_pin, 0.318, lam)
        if u is not None:
            ustates[lam] = u
    print(f"[E21c] cached {len(ustates)}/{len(LAMS)} front states\n", flush=True)

    # PILOT one (lam,tau) to confirm the root finder
    lam_p = 0.327; u_p = ustates[lam_p]
    rp = rightmost_root(u_p, lam_p, 10.0)
    print(f"[E21c] PILOT lam=0.327 tau=10: rightmost = "
          f"{rp[0]:+.5f}{'+/-'+format(rp[1],'.4f')+'i' if rp[2] else ' (real)'} "
          f"(E18b reported Re z=-0.077 at 0.327)\n", flush=True)

    results = {}   # tau -> array of [lam, re, im, is_cplx, n_unstable, eigM]
    for tau in TAUS:
        print(f"[E21c] tau={tau}:", flush=True)
        print(f"{'lam':>8} {'eigM(z=0)':>10} {'rightmost Re z':>14} {'type':>6} "
              f"{'Im':>8} {'boundary?':>10}")
        rows = []
        for lam in ustates:
            u = ustates[lam]
            em = eigmax_M(coup, u, lam, BETA)
            re, im, isc = rightmost_root(u, lam, tau)
            typ = "cplx" if isc else "real"
            n_unst = 1 if re > 1e-9 else 0
            flag = "<-- UNSTABLE" if re > 1e-9 else ""
            print(f"{lam:8.4f} {em:+10.4f} {re:+14.5f} {typ:>6} {im:8.4f} {flag:>10}")
            rows.append([lam, em, re, im, float(isc), float(n_unst)])
        results[f"tau_{tau:.0f}"] = np.array(rows)
        print(flush=True)

    np.savez(os.path.join(OUT, "E21_tau_nature.npz"),
             taus=np.array(TAUS), lams=LAMS,
             cols=np.array(["lam", "eigM", "re_rightmost", "im_rightmost",
                            "is_complex", "n_unstable"]),
             **results)
    print(f"[E21c] saved results/cycle/E21_tau_nature.npz")
    # summary: does any tau have an unstable lam inside [lam_f, lam*]?
    print("\n[E21c] SUMMARY - front stability boundary per tau:")
    for tau in TAUS:
        r = results[f"tau_{tau:.0f}"]
        unstable = r[r[:, 5] > 0]
        rmax = r[:, 2].max()
        jmax = int(np.argmax(r[:, 2]))
        typ = "cplx" if r[jmax, 4] > 0 else "real"
        if len(unstable):
            print(f"  tau={tau:5.0f}: UNSTABLE at lam in "
                  f"{unstable[:,0].tolist()} (type at max: {typ})")
        else:
            print(f"  tau={tau:5.0f}: stable throughout; rightmost Re z = {rmax:+.5f} "
                  f"at lam={r[jmax,0]:.4f} (type {typ}) -> delay-robust")
    print(f"[E21c] TOTAL {time.time()-t0w:.0f}s")


if __name__ == "__main__":
    main()
