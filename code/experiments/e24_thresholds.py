"""
E24 -- per-pattern saddle-node thresholds lambda_c(mu) of the memory->front branches.

For every pattern mu, the fixed-point branch starting at xi^mu (lam=0) is continued
UPWARD in lambda by warm-started Newton (Woodbury); lambda_c(mu) is the last lambda
of on-branch convergence, refined by bisection to +-2e-4 and cross-checked by the
spectral extrapolation of eig_max(M) = -c sqrt(lambda_c - lam) (eig^2 linear in lam).

CLI:
  conda run -n mcmc_env python src/e24_thresholds.py --N 2000 --P 100 --seed 42 \
      [--store-curves] [--nproc 6] [--out results/cycle/E24_thr_N2000_P100_s42.npz]

Caveat (documented in E21d): the fold region carries twin micro-structure of width
~1.5e-3; lambda_c is defined operationally as the continuous-warm-start Newton limit.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from couplings import Couplings, make_patterns
from robust_branch import woodbury_newton, eigmax_M

BETA = 20.0
LAM0, DLAM0 = 0.05, 0.02
BISECT_TOL = 2e-4

_CTX = {}


def _init(N, P, seed):
    os.environ["OMP_NUM_THREADS"] = "2"
    xi, xis = make_patterns(N, P, seed)
    coup = Couplings(xi, xis); coup._use_np = True
    _CTX.update(N=N, P=P, xi=xi, coup=coup, gram=(xi @ xi.T) / N)


def _amp(u):
    return np.linalg.solve(_CTX["gram"], _CTX["coup"].overlap_raw(u))


def _on_branch(u, mu):
    """Converged state still belongs to pattern mu's branch: mu leads, and the only
    other significant overlap is the cyclic successor mu+1.
    RELAXED mode (_CTX['relaxed'], set by the correlated-pattern experiments E27/E28):
    only require that mu still leads -- correlated patterns dress each memory with
    its neighbors (a_{mu+-1} ~ f(c)), so the iid successor/third-overlap criteria
    would reject perfectly healthy branches. Branch identity is then guaranteed by
    warm-start continuity + the leading index."""
    a = _amp(u)
    P = _CTX["P"]
    order = np.argsort(np.abs(a))[::-1]
    if int(order[0]) != mu:
        return False, a
    if _CTX.get("relaxed", False):
        return True, a
    third = np.abs(a[order[2]]) if len(order) > 2 else 0.0
    return (int(order[1]) == (mu + 1) % P or np.abs(a[order[1]]) < 0.1) and third < 0.15, a


def _try(u_start, mu, lam):
    coup = _CTX["coup"]
    un, ok = woodbury_newton(coup, u_start, lam, BETA)
    if not ok:
        return None
    onb, a = _on_branch(un, mu)
    if not onb:
        return None
    res = np.linalg.norm(coup.field_F(un, lam, BETA)) / np.sqrt(_CTX["N"])
    if res > 1e-9:
        return None
    return un, a


def trace_one(mu, store_curve=False, no_eig=False):
    """Walk pattern mu's branch upward; return (lambda_c, lambda_c_extrap, tilt rows).
    no_eig: skip the P x P spectral calls (O(P^2 N) each) -- step adaptation by
    Newton failures only; lambda_c from bisection alone (no extrapolation check).
    Use for large P (e.g. N=10^4, alpha >= 0.05)."""
    xi, P = _CTX["xi"], _CTX["P"]
    r = _try(2.0 * xi[mu], mu, LAM0)
    if r is None:
        return dict(mu=mu, lam_c=np.nan, lam_extrap=np.nan, rows=None, fail="seed")
    u, a = r
    lam, dlam = LAM0, DLAM0
    rows = []          # lam, a_mu, a_mu1, eig_max
    tail = []          # (lam, eig^2) for the spectral extrapolation

    def record(lam):
        em = np.nan if no_eig else eigmax_M(_CTX["coup"], u, lam, BETA)
        aa = _amp(u)
        rows.append([lam, aa[mu], aa[(mu + 1) % P], em])
        if not no_eig:
            tail.append((lam, em * em))

    record(lam)
    while True:
        r = _try(u, mu, lam + dlam)
        if r is not None:
            lam += dlam; u, a = r
            record(lam)
            if no_eig:
                dlam = min(dlam * 1.2, DLAM0)
            else:
                # adaptive: shrink as the fold nears (eig^2 ~ c^2 (lam_c - lam))
                em2 = tail[-1][1]
                dlam = float(np.clip(0.25 * em2 / 4.0, BISECT_TOL / 2, DLAM0))
        else:
            if dlam <= BISECT_TOL:
                break
            dlam *= 0.5
    # bisection refine between lam (ok) and lam + 2*dlam (fail)
    lo, hi = lam, lam + 4 * BISECT_TOL
    u_lo = u
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        r = _try(u_lo, mu, mid)
        if r is not None:
            lo, u_lo = mid, r[0]
        else:
            hi = mid
        if hi - lo < BISECT_TOL / 2:
            break
    lam_c = lo
    # spectral extrapolation: fit eig^2 = c2*(lam_c_ext - lam) on the last points
    t = np.array([p for p in tail if p[1] < 0.09][-8:])   # |eig|<0.3 region
    lam_ext = np.nan
    if len(t) >= 3:
        A = np.vstack([t[:, 0], np.ones(len(t))]).T
        slope, icpt = np.linalg.lstsq(A, t[:, 1], rcond=None)[0]
        if slope < 0:
            lam_ext = -icpt / slope
    return dict(mu=mu, lam_c=lam_c, lam_extrap=lam_ext,
                rows=(np.array(rows) if store_curve else None), fail="")


def _worker(args):
    mu, store, no_eig = args
    return trace_one(mu, store, no_eig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--P", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store-curves", action="store_true")
    ap.add_argument("--no-eig", action="store_true",
                    help="skip O(P^2 N) spectral calls (large P); bisection only")
    ap.add_argument("--subset", type=int, default=0,
                    help="trace only a random subset of this many patterns (0 = all)")
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cycle",
        f"E24_thr_N{args.N}_P{args.P}_s{args.seed}.npz")

    mus = list(range(args.P))
    if args.subset and args.subset < args.P:
        mus = sorted(np.random.default_rng(1234).choice(args.P, args.subset,
                                                        replace=False).tolist())
    t0 = time.time()
    results = []
    with Pool(args.nproc, initializer=_init, initargs=(args.N, args.P, args.seed)) as pool:
        for r in tqdm(pool.imap_unordered(_worker, [(mu, args.store_curves, args.no_eig)
                                                    for mu in mus]),
                      total=len(mus), desc=f"N={args.N} P={args.P} s{args.seed}"):
            results.append(r)
    results.sort(key=lambda d: d["mu"])
    lam_c = np.array([d["lam_c"] for d in results])
    lam_ext = np.array([d["lam_extrap"] for d in results])
    fails = [d["mu"] for d in results if d["fail"]]

    save = dict(lam_c=lam_c, lam_extrap=lam_ext, N=args.N, P=args.P, seed=args.seed,
                fails=np.array(fails, dtype=int), mus=np.array(mus, dtype=int))
    if args.store_curves:
        for d in results:
            if d["rows"] is not None:
                save[f"curve_{d['mu']}"] = d["rows"]
    np.savez(out, **save)

    ok = np.isfinite(lam_c)
    print(f"\n[E24] N={args.N} P={args.P} seed={args.seed}: {ok.sum()}/{args.P} branches"
          f"  ({time.time()-t0:.0f}s)")
    print(f"  lam_c: min={np.nanmin(lam_c):.4f} median={np.nanmedian(lam_c):.4f} "
          f"max={np.nanmax(lam_c):.4f} std={np.nanstd(lam_c):.4f} "
          f"gap={np.nanmax(lam_c)-np.nanmin(lam_c):.4f}")
    d_ext = np.abs(lam_c - lam_ext)[np.isfinite(lam_ext) & ok]
    if d_ext.size:
        print(f"  |bisect - spectral extrapolation|: median={np.median(d_ext):.1e} "
              f"max={d_ext.max():.1e}")
    if fails:
        print(f"  FAILED branches: {fails}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
