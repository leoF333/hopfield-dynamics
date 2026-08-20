"""
Critical-lambda grid  lambda_c(alpha, tau)  for the static memory destabilization.

Key efficiency: the fixed-point (memory) branch is TAU-INDEPENDENT (sec 1.1 of the
report), so for each alpha we build it ONCE by continuation, then reuse it for every
tau and only redo the (tau-dependent) pseudospectral IG stability sweep.

For each (alpha, tau) we sweep lambda along the branch [0, lambda_fold], compute the
rightmost characteristic roots with the IG operator, and split them by class:
    re_real(lambda) = max Re z over roots with |Im z| <  IM_TOL   (real modes)
    re_cplx(lambda) = max Re z over roots with |Im z| >= IM_TOL   (complex pairs)
The first lambda at which EITHER class reaches 0 is lambda_c:
    - if the complex class crosses first (strictly below the fold) -> Hopf, record omega_c
    - else the real class crosses at the fold                       -> saddle-node / SNIC

Because the fold is the z=0 crossing (identity Delta(0) = -M(lambda), report 3.5) and
e^{-z tau}=1 at z=0, lambda_fold is tau-independent; tau only matters if a delay-induced
Hopf undercuts it.

Outputs (results/stability/):
    lambda_c_grid.png   lambda_c vs tau, one curve per alpha, fold vs Hopf markers
    lambda_c_grid.npz   raw table
plus a printed table.
"""
from __future__ import annotations

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg_mod
from couplings import Couplings, make_patterns
from fixed_point import continuation
from dde_stability import rightmost_eigenvalues

IM_TOL      = 1e-3     # |Im z| >= IM_TOL  => complex pair
FOLD_MARGIN = 0.01     # a Hopf must cross at least this far below the fold to count


def _first_upcross(lam, re, im):
    """First lambda where re crosses <0 -> >=0 (linear interp). (lam_c, omega) or (None,None)."""
    for i in range(1, len(lam)):
        if np.isfinite(re[i - 1]) and re[i - 1] < 0 <= re[i]:
            t = -re[i - 1] / (re[i] - re[i - 1] + 1e-14)
            return (float(lam[i - 1] + t * (lam[i] - lam[i - 1])),
                    float(im[i - 1] + t * (im[i] - im[i - 1])))
    return None, None


def build_branch(alpha, cfg):
    """Continuation of the (tau-independent) memory branch at this alpha."""
    cfg = dict(cfg); cfg["alpha"] = alpha; cfg["P"] = round(alpha * cfg["N"])
    xi, xis = make_patterns(cfg["N"], cfg["P"], cfg["seed"])
    coup = Couplings(xi, xis)
    coup._use_np = True          # float64 CPU path (float32 corrupts branch tracing at large N)
    u0 = 0.99 * coup._xi_np[0].copy()
    t0 = time.time()
    branch = continuation(coup, cfg["beta"], cfg, u0=u0)
    lam_fold = float(branch["lam"].max())
    ok = len(branch["lam"]) >= 10 and lam_fold >= cfg["lam_min"] + 0.02
    print(f"  [alpha={alpha}] branch: {len(branch['lam'])} pts, "
          f"lam_fold={lam_fold:.4f}, t={time.time()-t0:.1f}s"
          f"{'' if ok else '  *** DEGENERATE BRANCH ***'}", flush=True)
    return coup, branch, ok


def sweep_tau(coup, branch, alpha, tau, cfg, n_lam):
    """IG stability sweep along the branch at this tau; return lambda_c + nature."""
    cfg = dict(cfg); cfg["alpha"] = alpha; cfg["tau"] = tau
    cfg["P"] = round(alpha * cfg["N"])
    # restrict to the stable UPPER ARM (up to the fold peak); the stored branch
    # includes a few unstable lower-arm points just past the fold.
    i_peak = int(np.argmax(branch["lam"]))
    lam_branch = branch["lam"][:i_peak + 1]
    u_arr      = branch["u_star"][:i_peak + 1]
    lam_fold = float(lam_branch.max())

    lam_vals = np.linspace(lam_branch.min(), lam_fold, n_lam)
    re_real = np.full(n_lam, -np.inf)
    re_cplx = np.full(n_lam, -np.inf)
    im_cplx = np.zeros(n_lam)

    t0 = time.time()
    for i, lam in enumerate(lam_vals):
        j = int(np.argmin(np.abs(lam_branch - lam)))
        ev = rightmost_eigenvalues(u_arr[j], float(lam), coup, cfg)
        isc = np.abs(ev.imag) >= IM_TOL
        if np.any(~isc):
            re_real[i] = float(ev.real[~isc].max())
        if np.any(isc):
            k = int(np.argmax(np.where(isc, ev.real, -np.inf)))
            re_cplx[i] = float(ev.real[k]); im_cplx[i] = float(abs(ev.imag[k]))

    lam_real, _      = _first_upcross(lam_vals, re_real, np.zeros(n_lam))
    lam_hopf, w_hopf = _first_upcross(lam_vals, re_cplx, im_cplx)

    # saddle-node confirmed when the continuation actually folded below lam_max
    # (the fold IS the z=0 crossing via Delta(0)=-M); robust to the sign of the
    # near-zero eigenvalue at the endpoint.
    folded = lam_fold < cfg.get("lam_max", 1.0) - 0.02
    real_confirmed = (lam_real is not None) or folded
    hopf = (lam_hopf is not None
            and lam_hopf < lam_fold - FOLD_MARGIN
            and (lam_real is None or lam_hopf < lam_real))
    if hopf:
        lam_c, nature, omega = lam_hopf, "Hopf", w_hopf
    elif real_confirmed:
        lam_c = lam_real if lam_real is not None else lam_fold
        nature, omega = "saddle-node / SNIC", None
    else:
        lam_c, nature, omega = lam_fold, "UNCONFIRMED (no crossing in range)", None
        print(f"    WARNING tau={tau}: rightmost mode stays Re<0 to the branch "
              f"end (lam_fold={lam_fold:.4f}); lam_c is the endpoint, not a "
              f"verified crossing.", flush=True)

    print(f"    tau={tau:>4}: lam_c={lam_c:.4f}  {nature}"
          f"{'  omega_c=%.3f' % omega if omega else ''}  ({time.time()-t0:.0f}s)",
          flush=True)
    return dict(alpha=alpha, tau=tau, lam_c=lam_c, lam_fold=lam_fold,
                nature=nature, omega_c=omega,
                lam_vals=lam_vals, re_real=re_real, re_cplx=re_cplx)


def main():
    p = cfg_mod.make_parser("lambda_c(alpha, tau) grid")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--taus",   type=int,   nargs="+", default=[1, 2, 5, 10, 15, 20])
    p.add_argument("--n-lam",  type=int,   default=10, dest="n_lam")
    p.add_argument("--Ngrid",  type=int,   default=2000, dest="Ngrid")
    p.add_argument("--outdir", type=str,   default=None,
                   help="directory for figure/npz output (default: results/stability)")
    args = p.parse_args()

    cfg = cfg_mod.resolve(args)
    cfg.update(dict(N=args.Ngrid, M_cheb=24, n_eigs=8, arnoldi_ncv=40,
                    arnoldi_maxiter=150, lam_max=0.6, ds=0.05))

    print(f"lambda_c grid: alphas={args.alphas} taus={args.taus} "
          f"N={cfg['N']} beta={cfg['beta']} M={cfg['M_cheb']}", flush=True)

    rows = []
    t_all = time.time()
    for alpha in args.alphas:
        coup, branch, ok = build_branch(alpha, cfg)
        if not ok:
            print(f"  SKIP alpha={alpha}: degenerate branch (initial Newton "
                  f"failed?); no sweeps run.", flush=True)
            continue
        for tau in args.taus:
            rows.append(sweep_tau(coup, branch, alpha, tau, cfg, args.n_lam))
    print(f"\nTotal wall time: {time.time()-t_all:.1f}s", flush=True)
    if not rows:
        print("ERROR: no valid branch produced; aborting.", flush=True); return

    # ---- table ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'alpha':>6} {'tau':>5} {'lam_c':>8} {'lam_fold':>9} {'omega_c':>8}  nature")
    print("-" * 60)
    for r in rows:
        wc = f"{r['omega_c']:.3f}" if r["omega_c"] else "   --"
        print(f"{r['alpha']:>6.2f} {r['tau']:>5} {r['lam_c']:>8.4f} "
              f"{r['lam_fold']:>9.4f} {wc:>8}  {r['nature']}")
    print("=" * 60)

    # ---- figure ---------------------------------------------------------
    alphas = sorted(set(r["alpha"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(alphas)))
    for a, col in zip(alphas, colors):
        rs = sorted([r for r in rows if r["alpha"] == a], key=lambda r: r["tau"])
        taus = [r["tau"] for r in rs]
        lamc = [r["lam_c"] for r in rs]
        ax.plot(taus, lamc, "-", color=col, lw=1.5, label=f"$\\alpha={a}$")
        # markers by nature: circle=fold, triangle=Hopf
        for r in rs:
            mk = "^" if r["nature"] == "Hopf" else "o"
            ax.plot(r["tau"], r["lam_c"], mk, color=col, ms=9,
                    mec="k", mew=0.6, zorder=3)
        # tau-independent fold reference (mean over taus)
        lf = np.mean([r["lam_fold"] for r in rs])
        ax.axhline(lf, color=col, ls=":", lw=1, alpha=0.6)
    ax.plot([], [], "ko", label="fold (saddle-node)")
    ax.plot([], [], "k^", label="Hopf")
    ax.plot([], [], "k:", label=r"$\lambda_{\rm fold}$ (τ-indep.)")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\lambda_c$")
    ax.set_title(f"Critical $\\lambda_c(\\alpha,\\tau)$ for static-memory loss  "
                 f"($\\beta={cfg['beta']},N={cfg['N']}$)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    tag = f"N{cfg['N']}"
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, f"lambda_c_grid_{tag}.png")
        npz  = os.path.join(args.outdir, f"lambda_c_grid_{tag}.npz")
    else:
        path = cfg_mod.out_path(cfg, "stability", f"lambda_c_grid_{tag}.png")
        npz  = cfg_mod.out_path(cfg, "stability", f"lambda_c_grid_{tag}.npz")
    fig.savefig(path, dpi=cfg.get("dpi", 150), bbox_inches="tight")
    print(f"\n[lambda_c_grid] figure -> {path}")

    np.savez(npz,
             alpha=[r["alpha"] for r in rows], tau=[r["tau"] for r in rows],
             lam_c=[r["lam_c"] for r in rows], lam_fold=[r["lam_fold"] for r in rows],
             omega_c=[(r["omega_c"] if r["omega_c"] else np.nan) for r in rows],
             nature=[r["nature"] for r in rows])
    print(f"[lambda_c_grid] data   -> {npz}")


if __name__ == "__main__":
    main()
