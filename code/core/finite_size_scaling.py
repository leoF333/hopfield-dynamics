"""
Finite-size scaling of the memory-branch destabilization.

Goal
----
At finite N the memory branch F(u,lambda)=0 ends in a saddle-node FOLD.
At a fold the Jacobian M(lambda) is singular, so the rightmost characteristic
root always touches Re(z)=0 *at the fold* — this is true for ANY mechanism and
is NOT by itself evidence of a saddle-node bifurcation of the dynamics.

The physical question (Hopf vs saddle-node/SNIC) is:
    Does a COMPLEX-CONJUGATE pair cross Re(z)=0 at some lambda STRICTLY BELOW
    the fold (a dynamic Hopf instability of the still-existing memory state)?
      - YES  -> Hopf precedes the fold.
      - NO   -> the only zero crossing is a real mode AT the fold -> saddle-node.

We sweep N in {2000, 4000, 8000}, and for each N we separately track:
    re_real_max(lambda) = max Re over eigenvalues with |Im| <  IM_TOL  (real)
    re_cplx_max(lambda) = max Re over eigenvalues with |Im| >= IM_TOL  (complex)
Then we report lambda_fold(N), and lambda_hopf(N) if a complex pair crosses 0
below the fold.  Extrapolation of lambda_fold and lambda_hopf vs 1/N indicates
the thermodynamic-limit behaviour.

Outputs
-------
  results/stability/finite_size_scaling_a{alpha}_tau{tau}.png
  results/stability/finite_size_scaling_a{alpha}_tau{tau}.npz
  printed table.
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

IM_TOL = 1e-3      # |Im(z)| >= IM_TOL  => count as complex pair
FOLD_MARGIN = 0.01  # a Hopf crossing must be at least this far below the fold


def analyze_one_N(N: int, cfg: dict, n_lam: int = 10) -> dict:
    """
    Build the memory branch at this N, sweep lambda, and split the rightmost
    spectrum into real vs complex parts.
    """
    cfg = dict(cfg)
    cfg["N"] = N
    cfg["P"] = round(cfg["alpha"] * N)

    print(f"\n{'='*56}\n N = {N}   P = {cfg['P']}\n{'='*56}", flush=True)
    xi, xis = make_patterns(N, cfg["P"], cfg["seed"])
    coup = Couplings(xi, xis)

    # ---- memory branch (stops at the fold) -----------------------------
    t0 = time.time()
    u0 = 0.99 * coup._xi_np[0].copy()
    branch = continuation(coup, cfg["beta"], cfg, u0=u0)
    lam_fold = float(branch["lam"].max())
    t_branch = time.time() - t0
    print(f"  branch: {len(branch['lam'])} pts, "
          f"lam_fold = {lam_fold:.4f},  t = {t_branch:.1f}s", flush=True)

    # ---- lambda sweep: full spectrum split real/complex ----------------
    lam_branch = branch["lam"]
    u_arr      = branch["u_star"]
    # sample lambda along the branch, weighted toward the fold (where the
    # physical mode matters); use a sqrt spacing.
    s        = np.linspace(0.0, 1.0, n_lam) ** 0.7
    lam_vals = lam_branch.min() + s * (lam_branch.max() - lam_branch.min())

    re_real = np.full(n_lam, -np.inf)
    re_cplx = np.full(n_lam, -np.inf)
    im_at_cplx = np.zeros(n_lam)
    evals_all = []

    t0 = time.time()
    for i, lam in enumerate(lam_vals):
        j = int(np.argmin(np.abs(lam_branch - lam)))
        u = u_arr[j]
        ev = rightmost_eigenvalues(u, float(lam), coup, cfg)
        evals_all.append(ev)
        is_cplx = np.abs(ev.imag) >= IM_TOL
        if np.any(~is_cplx):
            re_real[i] = float(ev.real[~is_cplx].max())
        if np.any(is_cplx):
            k = int(np.argmax(np.where(is_cplx, ev.real, -np.inf)))
            re_cplx[i]    = float(ev.real[k])
            im_at_cplx[i] = float(abs(ev.imag[k]))
        print(f"    lam={lam:.4f}  re_real={re_real[i]:+.4f}  "
              f"re_cplx={re_cplx[i]:+.4f}  (|Im|={im_at_cplx[i]:.3f})", flush=True)
    t_sweep = time.time() - t0
    print(f"  sweep t = {t_sweep:.1f}s", flush=True)

    # ---- classify ------------------------------------------------------
    # Hopf: complex pair crosses 0 strictly below the fold.
    lam_hopf, omega_hopf = _first_upcross(lam_vals, re_cplx, im_at_cplx)
    if lam_hopf is not None and lam_hopf < lam_fold - FOLD_MARGIN:
        nature = "Hopf"
    else:
        lam_hopf, omega_hopf = None, None
        nature = "saddle-node / SNIC"

    return dict(
        N=N, lam_fold=lam_fold, lam_hopf=lam_hopf, omega_hopf=omega_hopf,
        nature=nature, lam_vals=lam_vals, re_real=re_real, re_cplx=re_cplx,
        im_at_cplx=im_at_cplx, t_branch=t_branch, t_sweep=t_sweep,
    )


def _first_upcross(lam, re, im):
    """First lambda where re crosses from <0 to >=0; linear interp. Returns
    (lam_c, omega_c) or (None, None)."""
    for i in range(1, len(lam)):
        if re[i - 1] < 0 <= re[i] and np.isfinite(re[i - 1]):
            t = -re[i - 1] / (re[i] - re[i - 1] + 1e-14)
            lam_c = lam[i - 1] + t * (lam[i] - lam[i - 1])
            omega = im[i - 1] + t * (im[i] - im[i - 1])
            return float(lam_c), float(omega)
    return None, None


def main():
    p = cfg_mod.make_parser("Finite-size scaling of memory-branch destabilization")
    p.add_argument("--Ns", type=int, nargs="+", default=[2000, 4000, 8000])
    p.add_argument("--n-lam", type=int, default=10, dest="n_lam")
    args = p.parse_args()
    cfg = cfg_mod.resolve(args)

    # Stability/continuation knobs tuned for speed at large N
    cfg.update(dict(M_cheb=20, n_eigs=6, arnoldi_ncv=40, arnoldi_maxiter=300,
                    lam_max=0.6, ds=0.05))
    cfg["alpha"] = cfg["alpha"] or 0.05

    print(f"Finite-size scaling: Ns={args.Ns}  alpha={cfg['alpha']}  "
          f"tau={cfg['tau']}  beta={cfg['beta']}  M={cfg['M_cheb']}")

    results = []
    t_all = time.time()
    for N in args.Ns:
        results.append(analyze_one_N(N, cfg, n_lam=args.n_lam))
    print(f"\nTotal wall time: {time.time()-t_all:.1f}s")

    # ---- table ---------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"{'N':>7} {'lam_fold':>10} {'lam_hopf':>10} {'omega_c':>9} {'nature':>20}")
    print("-" * 64)
    for r in results:
        lh = f"{r['lam_hopf']:.4f}" if r["lam_hopf"] is not None else "   --   "
        wc = f"{r['omega_hopf']:.4f}" if r["omega_hopf"] is not None else "  --  "
        print(f"{r['N']:>7} {r['lam_fold']:>10.4f} {lh:>10} {wc:>9} {r['nature']:>20}")
    print("=" * 64)

    # ---- figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for r in results:
        ax.plot(r["lam_vals"], r["re_real"], "o-", ms=4,
                label=f"N={r['N']} real")
        ax.plot(r["lam_vals"], r["re_cplx"], "s--", ms=4, alpha=0.7,
                label=f"N={r['N']} cplx")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\max\,\mathrm{Re}\,z$ (real vs complex modes)")
    ax.set_title("Rightmost real / complex eigenvalue along the branch")
    ax.legend(fontsize=7, ncol=len(results))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    invN = np.array([1.0 / r["N"] for r in results])
    fold = np.array([r["lam_fold"] for r in results])
    ax.plot(invN, fold, "ko-", ms=7, label=r"$\lambda_{\mathrm{fold}}(N)$")
    hopf_pts = [(1.0 / r["N"], r["lam_hopf"]) for r in results
                if r["lam_hopf"] is not None]
    if hopf_pts:
        hx, hy = zip(*hopf_pts)
        ax.plot(hx, hy, "r^-", ms=8, label=r"$\lambda_{\mathrm{Hopf}}(N)$")
    # linear fit of fold vs 1/N -> extrapolate to N->inf (1/N=0)
    if len(results) >= 2:
        a, b = np.polyfit(invN, fold, 1)
        xx = np.linspace(0, invN.max() * 1.05, 50)
        ax.plot(xx, a * xx + b, "b:", lw=1.2,
                label=f"fold $\\to$ {b:.3f} as $N\\to\\infty$")
        ax.plot(0, b, "b*", ms=12)
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(r"$\lambda_c$")
    ax.set_title("Finite-size extrapolation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Finite-size scaling  ($\\alpha={cfg['alpha']},\\,\\tau={cfg['tau']},"
        f"\\,\\beta={cfg['beta']}$)", fontsize=12)
    fig.tight_layout()

    tag  = f"a{cfg['alpha']:.3f}_tau{cfg['tau']}_b{cfg['beta']:.1f}"
    path = cfg_mod.out_path(cfg, "stability", f"finite_size_scaling_{tag}.png")
    fig.savefig(path, dpi=cfg.get("dpi", 150), bbox_inches="tight")
    print(f"\n[finite_size_scaling] figure -> {path}")

    npz = cfg_mod.out_path(cfg, "stability", f"finite_size_scaling_{tag}.npz")
    np.savez(npz,
             Ns=[r["N"] for r in results],
             lam_fold=[r["lam_fold"] for r in results],
             lam_hopf=[(r["lam_hopf"] if r["lam_hopf"] is not None else np.nan)
                       for r in results],
             nature=[r["nature"] for r in results])
    print(f"[finite_size_scaling] data   -> {npz}")


if __name__ == "__main__":
    main()
