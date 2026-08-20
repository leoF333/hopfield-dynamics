"""
lambda_c(alpha, tau) grid v2 — corrected machinery, artefact-free bifurcation analysis.

Improvements over lambda_c_grid.py (v1):
  * branch traced by the ROBUST Woodbury tracer (robust_branch.py, float64, exact
    Newton steps) — no GMRES false folds, so the full memory branch is examined;
  * Hopf test uses the EXACT reduced spectrum (reduced_spectrum.py): rightmost
    physical roots of det T_P(z)=0, certified by sigma_min — no spurious
    pseudospectral modes to confuse the real/complex split;
  * tau=0 included exactly (ODE limit): roots z = (eig[(1-lam)G_J + lam G_K] - 1)/t0.

For each alpha the branch is traced ONCE (tau-independent). For each tau, the
rightmost real and complex physical roots are computed at n_lam points on the
stable upper arm (weighted toward the fold). Verdict per (alpha, tau):
  Hopf              iff the complex class crosses Re=0 strictly below the fold;
  saddle-node/SNIC  otherwise (the real z=0 crossing AT the fold, tau-independent
                    by Delta(0) = -M).

Outputs: results/stability/lambda_c_grid_v2.png / .npz + printed table.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from couplings import Couplings, make_patterns
from robust_branch import trace_branch
from reduced_spectrum import GJ_GK, physical_roots

IM_TOL      = 1e-3
FOLD_MARGIN = 0.01
BETA, T0 = 20.0, 1.0


def roots_tau0(coup, u, lam):
    """Exact ODE-limit (tau=0) spectrum: z = (eig[(1-lam)G_J + lam G_K] - 1)/t0."""
    GJ, GK = GJ_GK(coup, u, BETA, lam)
    mu = np.linalg.eigvals((1.0 - lam) * GJ + lam * GK)
    z = (mu - 1.0) / T0
    return z[np.argsort(-z.real)]


def split(roots):
    """(re_real, re_cplx, im_cplx) from a root set."""
    if len(roots) == 0:
        return -np.inf, -np.inf, 0.0
    isc = np.abs(roots.imag) >= IM_TOL
    rr = float(roots.real[~isc].max()) if np.any(~isc) else -np.inf
    rc, ic = -np.inf, 0.0
    if np.any(isc):
        j = int(np.argmax(np.where(isc, roots.real, -np.inf)))
        rc, ic = float(roots.real[j]), float(abs(roots.imag[j]))
    return rr, rc, ic


def _first_upcross(lam, re, im):
    for i in range(1, len(lam)):
        if np.isfinite(re[i - 1]) and re[i - 1] < 0 <= re[i]:
            t = -re[i - 1] / (re[i] - re[i - 1] + 1e-14)
            return (float(lam[i - 1] + t * (lam[i] - lam[i - 1])),
                    float(im[i - 1] + t * (im[i] - im[i - 1])))
    return None, None


def sweep_tau(coup, branch, fold, alpha, tau, n_lam, M_cheb):
    """Real/complex Hopf test along the stable upper arm at this tau."""
    i_peak = int(np.argmax(branch["lam"]))
    lam_up, u_up = branch["lam"][:i_peak + 1], branch["u_star"][:i_peak + 1]
    s = np.linspace(0.15, 1.0, n_lam) ** 0.7          # weighted toward the fold
    lam_vals = lam_up.min() + s * (fold - lam_up.min())

    rr = np.full(n_lam, -np.inf); rc = np.full(n_lam, -np.inf); ic = np.zeros(n_lam)
    t0w = time.time()
    for i, lam in enumerate(lam_vals):
        j = int(np.argmin(np.abs(lam_up - lam)))
        lam = float(lam_up[j]); u = u_up[j]; lam_vals[i] = lam
        if tau == 0:
            roots = roots_tau0(coup, u, lam)[:12]
        else:
            roots, _, _ = physical_roots(coup, u, BETA, lam, float(tau), T0,
                                         M=M_cheb, n_cand=60, thresh=0.05,
                                         k=12, polish=True)
        rr[i], rc[i], ic[i] = split(np.asarray(roots, complex))

    lam_hopf, w_hopf = _first_upcross(lam_vals, rc, ic)
    hopf = lam_hopf is not None and lam_hopf < fold - FOLD_MARGIN
    if hopf:
        lam_c, nature, omega = lam_hopf, "Hopf", w_hopf
    else:
        lam_c, nature, omega = fold, "saddle-node / SNIC", None
    print(f"    tau={tau:>5}: lam_c={lam_c:.4f}  {nature}"
          f"{'  omega_c=%.3f' % omega if omega else ''}"
          f"  [max re_cplx={np.max(rc[np.isfinite(rc)]) if np.any(np.isfinite(rc)) else float('-inf'):+.3f}]"
          f"  ({time.time()-t0w:.0f}s)", flush=True)
    return dict(alpha=alpha, tau=tau, lam_c=lam_c, lam_fold=fold, nature=nature,
                omega_c=omega, lam_vals=lam_vals, re_real=rr, re_cplx=rc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.01, 0.03, 0.05, 0.07, 0.10])
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0, 0.5, 1, 2, 4, 6, 8, 10, 12, 15])
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--n-lam", type=int, default=6, dest="n_lam")
    p.add_argument("--M", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "stability")
    os.makedirs(outdir, exist_ok=True)
    print(f"grid v2: alphas={args.alphas} taus={args.taus} N={args.N} "
          f"beta={BETA} M={args.M} seed={args.seed}", flush=True)

    rows = []; t_all = time.time()
    for alpha in args.alphas:
        P = round(alpha * args.N)
        cfg = dict(lam_min=0.0, lam_max=0.6, ds=0.01, n_overlaps=5)
        xi, xis = make_patterns(args.N, P, args.seed)
        coup = Couplings(xi, xis); coup._use_np = True
        t0w = time.time()
        br, fold, status = trace_branch(coup, BETA, cfg)
        near_cap = alpha >= 0.08
        print(f"  [alpha={alpha}] branch: {len(br['lam'])} pts, "
              f"lam_fold={fold:.4f} ({status}), t={time.time()-t0w:.0f}s"
              f"{'   *** NEAR-CAPACITY: single-seed fold unreliable ***' if near_cap else ''}",
              flush=True)
        for tau in args.taus:
            r = sweep_tau(coup, br, fold, alpha, tau, args.n_lam, args.M)
            r["near_capacity"] = near_cap
            rows.append(r)
    print(f"\nTotal wall time: {time.time()-t_all:.0f}s", flush=True)

    # ---- table -----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'alpha':>6} {'tau':>6} {'lam_c':>8} {'lam_fold':>9} {'omega_c':>8}  nature")
    print("-" * 72)
    for r in rows:
        wc = f"{r['omega_c']:.3f}" if r["omega_c"] else "   --"
        flag = " (near-cap.)" if r["near_capacity"] else ""
        print(f"{r['alpha']:>6.2f} {r['tau']:>6} {r['lam_c']:>8.4f} "
              f"{r['lam_fold']:>9.4f} {wc:>8}  {r['nature']}{flag}")
    print("=" * 72)

    # ---- figure ----------------------------------------------------------
    alphas = sorted(set(r["alpha"] for r in rows))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(alphas)))
    for a, col in zip(alphas, colors):
        rs = sorted([r for r in rows if r["alpha"] == a], key=lambda r: r["tau"])
        taus = [r["tau"] for r in rs]; lamc = [r["lam_c"] for r in rs]
        ls = "--" if rs[0]["near_capacity"] else "-"
        lab = f"$\\alpha={a}$" + (" (near-cap., indicative)" if rs[0]["near_capacity"] else "")
        ax.plot(taus, lamc, ls, color=col, lw=1.6, label=lab)
        for r in rs:
            mk = "^" if r["nature"] == "Hopf" else "o"
            ax.plot(r["tau"], r["lam_c"], mk, color=col, ms=8, mec="k", mew=.5, zorder=3)
    ax.plot([], [], "ko", label="fold (saddle-node/SNIC)")
    ax.plot([], [], "k^", label="Hopf")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\lambda_c$")
    ax.set_title(f"Critical $\\lambda_c(\\alpha,\\tau)$ — robust branch + exact "
                 f"reduced spectrum  ($N={args.N},\\ \\beta={BETA}$)")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fp = os.path.join(outdir, f"lambda_c_grid_v2_N{args.N}.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"\n[grid v2] figure -> {fp}")
    def _margin(r):
        """Hopf margin: rightmost complex Re at the fold-adjacent sample (== max
        over samples, since complex roots rise monotonically toward the fold)."""
        finite = r["re_cplx"][np.isfinite(r["re_cplx"])]
        return float(finite.max()) if len(finite) else np.nan

    np.savez(os.path.join(outdir, f"lambda_c_grid_v2_N{args.N}.npz"),
             alpha=[r["alpha"] for r in rows], tau=[r["tau"] for r in rows],
             lam_c=[r["lam_c"] for r in rows],
             lam_fold=[r["lam_fold"] for r in rows],
             omega_c=[(r["omega_c"] if r["omega_c"] else np.nan) for r in rows],
             nature=[r["nature"] for r in rows],
             near_capacity=[r["near_capacity"] for r in rows],
             hopf_margin=[_margin(r) for r in rows])
    print(f"[grid v2] data   -> {os.path.join(outdir, f'lambda_c_grid_v2_N{args.N}.npz')}")


if __name__ == "__main__":
    main()
