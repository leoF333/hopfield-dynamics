"""
Static fixed-point bifurcation at N=10000, averaged over several seeds.

Runs the full memory-branch + pseudospectral-IG stability analysis at the target
size N=10000 for a few pattern realizations (seeds), to suppress finite-size
randomness. Produces, in results/N=10000/:

  A1_branch_overlay.png    m1(lambda) for each seed, fold marked
  A2_stability_overlay.png max Re(z) real/complex classes vs lambda, each seed,
                           with per-seed lambda_c and mean +/- std
  A3_roots_panel.png       characteristic roots in C at 6 lambda (first seed)
  N10000_summary.npz       raw arrays + lambda_c per seed

Default: alpha=0.05, tau=10, beta=20, seeds 42/43/44.
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

IM_TOL      = 1e-3
FOLD_MARGIN = 0.01
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def _first_upcross(lam, re, im):
    for i in range(1, len(lam)):
        if np.isfinite(re[i - 1]) and re[i - 1] < 0 <= re[i]:
            t = -re[i - 1] / (re[i] - re[i - 1] + 1e-14)
            return (float(lam[i - 1] + t * (lam[i] - lam[i - 1])),
                    float(im[i - 1] + t * (im[i] - im[i - 1])))
    return None, None


def analyze_seed(seed, cfg, n_lam):
    cfg = dict(cfg); cfg["seed"] = seed
    cfg["P"] = round(cfg["alpha"] * cfg["N"])
    print(f"\n=== seed {seed}  (N={cfg['N']}, P={cfg['P']}, "
          f"alpha={cfg['alpha']}, tau={cfg['tau']}) ===", flush=True)

    xi, xis = make_patterns(cfg["N"], cfg["P"], seed)
    coup = Couplings(xi, xis)
    coup._use_np = True          # float64 CPU path: float32 corrupts branch tracing at large N
    u0 = 0.99 * coup._xi_np[0].copy()

    t0 = time.time()
    branch = continuation(coup, cfg["beta"], cfg, u0=u0)
    lam_fold = float(branch["lam"].max())
    if len(branch["lam"]) < 10 or lam_fold < cfg["lam_min"] + 0.02:
        print(f"  ERROR: degenerate branch ({len(branch['lam'])} pts, "
              f"lam_fold={lam_fold:.4f}); seed {seed} skipped.", flush=True)
        return None
    print(f"  branch: {len(branch['lam'])} pts, lam_fold={lam_fold:.4f}, "
          f"t={time.time()-t0:.0f}s", flush=True)

    # Restrict the stability sweep to the STABLE UPPER ARM (indices up to the
    # fold peak). The stored branch also holds a few lower/unstable-arm points
    # just past the fold; excluding them prevents argmin() from ever selecting
    # an unstable saddle point near the fold.
    i_peak = int(np.argmax(branch["lam"]))
    lam_branch = branch["lam"][:i_peak + 1]
    u_arr      = branch["u_star"][:i_peak + 1]
    # sqrt-spacing: denser toward the fold where the physical mode matters
    s = np.linspace(0, 1, n_lam) ** 0.7
    lam_vals = lam_branch.min() + s * (lam_fold - lam_branch.min())
    re_real = np.full(n_lam, -np.inf); re_cplx = np.full(n_lam, -np.inf)
    im_cplx = np.zeros(n_lam); evals = []

    t0 = time.time()
    for i, lam in enumerate(lam_vals):
        j = int(np.argmin(np.abs(lam_branch - lam)))
        ev = rightmost_eigenvalues(u_arr[j], float(lam), coup, cfg)
        evals.append(ev)
        isc = np.abs(ev.imag) >= IM_TOL
        if np.any(~isc): re_real[i] = float(ev.real[~isc].max())
        if np.any(isc):
            k = int(np.argmax(np.where(isc, ev.real, -np.inf)))
            re_cplx[i] = float(ev.real[k]); im_cplx[i] = float(abs(ev.imag[k]))
        print(f"    lam={lam:.4f} re_real={re_real[i]:+.4f} "
              f"re_cplx={re_cplx[i]:+.4f}", flush=True)
    print(f"  sweep t={time.time()-t0:.0f}s", flush=True)

    lam_real, _      = _first_upcross(lam_vals, re_real, np.zeros(n_lam))
    lam_hopf, w_hopf = _first_upcross(lam_vals, re_cplx, im_cplx)
    # The saddle-node is confirmed when the continuation actually turned (the
    # branch folded strictly below lam_max): by the identity Delta(0)=-M, the
    # fold IS the z=0 real crossing. Keying on the fold (not on the sign of the
    # near-zero eigenvalue, which flips with M/sampling) is robust; it only
    # fails to confirm when the branch ran into lam_max without folding.
    folded = lam_fold < cfg.get("lam_max", 1.0) - 0.02
    real_confirmed = (lam_real is not None) or folded
    if (lam_hopf is not None and lam_hopf < lam_fold - FOLD_MARGIN
            and (lam_real is None or lam_hopf < lam_real)):
        lam_c, nature, omega = lam_hopf, "Hopf", w_hopf
    elif real_confirmed:
        lam_c = lam_real if lam_real is not None else lam_fold
        nature, omega = "saddle-node / SNIC", None
    else:
        lam_c, nature, omega = lam_fold, "UNCONFIRMED (no crossing in range)", None
        print(f"  WARNING: rightmost mode stays Re<0 up to the branch end "
              f"(lam_fold={lam_fold:.4f}); memory may not destabilize below "
              f"lam_max -- lam_c reported as endpoint, NOT a verified crossing.",
              flush=True)
    print(f"  -> lam_c={lam_c:.4f}  {nature}", flush=True)

    # NOTE: return the FULL branch (incl. the small lower-arm tail) for the A1
    # overlay so the fold curl is visible; the sweep above used only the arm.
    return dict(seed=seed, lam_fold=lam_fold, lam_c=lam_c, nature=nature,
                omega_c=omega, lam_vals=lam_vals, re_real=re_real,
                re_cplx=re_cplx, evals=evals, branch_lam=branch["lam"],
                m1=branch["m_nu"][:, 0])


def main():
    p = cfg_mod.make_parser("N=10000 static bifurcation, multi-seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-lam", type=int, default=10, dest="n_lam")
    args = p.parse_args()

    # resolve() defaults are exactly N=10000, alpha=0.05, tau=10, beta=20;
    # CLI flags (--N, --alpha, --tau, --beta, --M) override for smoke-testing.
    # NOTE: resolve() also fills lam_max/ds/... from DEFAULTS, so the analysis
    # knobs below must be ASSIGNED (not setdefault) to actually take effect.
    cfg = cfg_mod.resolve(args)
    cfg["t0"]             = cfg.get("t0", 1.0)
    cfg["n_eigs"]         = 6
    cfg["arnoldi_ncv"]    = 40
    cfg["arnoldi_maxiter"] = 150
    cfg["lam_max"]        = 0.6
    cfg["ds"]             = 0.05
    cfg["M_cheb"]         = 16 if args.M_cheb is None else args.M_cheb
    OUTDIR = os.path.join(RESULTS, f"N={cfg['N']}")
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"N={cfg['N']} study: seeds={args.seeds}  alpha={cfg['alpha']} "
          f"tau={cfg['tau']} beta={cfg['beta']} M={cfg['M_cheb']}", flush=True)

    t_all = time.time()
    res = [analyze_seed(s, cfg, args.n_lam) for s in args.seeds]
    res = [r for r in res if r is not None]        # drop degenerate/failed seeds
    print(f"\nTotal wall time: {time.time()-t_all:.0f}s", flush=True)
    if not res:
        print("ERROR: no seed produced a valid branch; aborting.", flush=True)
        return

    lam_cs = np.array([r["lam_c"] for r in res])
    folds  = np.array([r["lam_fold"] for r in res])
    print("\n" + "=" * 52)
    print(f"{'seed':>6} {'lam_c':>9} {'lam_fold':>9}  nature")
    for r in res:
        print(f"{r['seed']:>6} {r['lam_c']:>9.4f} {r['lam_fold']:>9.4f}  {r['nature']}")
    print("-" * 52)
    print(f"lambda_c = {lam_cs.mean():.4f} +/- {lam_cs.std():.4f}  "
          f"(mean +/- std over {len(res)} seeds)")
    print("=" * 52)

    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(res)))

    # --- A1: branch overlay -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, c in zip(res, colors):
        ax.plot(r["branch_lam"], r["m1"], color=c, lw=1.4, label=f"seed {r['seed']}")
        ax.plot(r["lam_fold"], r["m1"][int(np.argmax(r["branch_lam"]))],
                "o", color=c, ms=7, mec="k", mew=.5)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$m_1$ (memory overlap)")
    ax.set_title(f"Memory branch, N=10000  ($\\alpha=0.05,\\tau=10,\\beta=20$)")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "A1_branch_overlay.png"), dpi=150)

    # --- A2: stability overlay ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r, c in zip(res, colors):
        ax.plot(r["lam_vals"], r["re_real"], "o-", color=c, ms=4,
                label=f"seed {r['seed']} real")
        ax.plot(r["lam_vals"], r["re_cplx"], "s--", color=c, ms=4, alpha=.6,
                label=f"seed {r['seed']} cplx")
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.axvspan(lam_cs.mean()-lam_cs.std(), lam_cs.mean()+lam_cs.std(),
               color="red", alpha=.12)
    ax.axvline(lam_cs.mean(), color="red", lw=1.5, ls="--",
               label=f"$\\lambda_c={lam_cs.mean():.3f}\\pm{lam_cs.std():.3f}$")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\max\,\mathrm{Re}\,z$ (real / complex)")
    ax.set_title("Stability: real vs complex rightmost mode, N=10000 (3 seeds)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "A2_stability_overlay.png"), dpi=150)

    # --- A3: roots panel (first seed) --------------------------------------
    r0 = res[0]; nshow = min(6, len(r0["lam_vals"]))
    idx = np.linspace(0, len(r0["lam_vals"])-1, nshow, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8)); axes = axes.ravel()
    for ii, k in enumerate(idx):
        ax = axes[ii]; ev = r0["evals"][k]
        ax.scatter(ev.real, ev.imag, s=22, c="steelblue", zorder=3)
        ax.axvline(0, color="k", lw=.7, ls="--"); ax.axhline(0, color="k", lw=.7, ls="--")
        ax.set_title(f"$\\lambda={r0['lam_vals'][k]:.3f}$")
        ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)"); ax.grid(True, alpha=.2)
    fig.suptitle(f"Characteristic roots, N=10000 seed {r0['seed']} "
                 f"($\\lambda_c\\approx{r0['lam_c']:.3f}$)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "A3_roots_panel.png"), dpi=150)

    np.savez(os.path.join(OUTDIR, "N10000_summary.npz"),
             seeds=[r["seed"] for r in res], lam_c=lam_cs, lam_fold=folds,
             nature=[r["nature"] for r in res],
             lam_c_mean=lam_cs.mean(), lam_c_std=lam_cs.std())
    print(f"\n[run_N10000] figures + data -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
