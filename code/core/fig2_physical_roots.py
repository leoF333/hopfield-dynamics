"""
Figure 2 (redone): the 10 rightmost PHYSICAL characteristic roots in the complex
plane at six lambda clustered just below lambda_c, for alpha=0.05, tau=10, N=10000.

Uses the hybrid exact spectrum (IG candidates filtered/certified by the reduced
P x P equation T_P, then polished) -> NO spurious pseudospectral modes. Shows the
genuine evolution of the physical eigenvalues as lambda -> fold: the real fold-mode
climbs (square-root law) and crosses 0 at the fold, overtaking the complex modes.

Also prints, per lambda, the rightmost real and rightmost complex Re(z) -> an exact
saddle-node-vs-Hopf check (real reaches 0 first => saddle-node; a complex pair
reaching 0 below the fold => Hopf).
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from couplings import Couplings, make_patterns
from robust_branch import trace_branch, woodbury_newton, eigmax_M
from reduced_spectrum import physical_roots, rightmost_real_root, GJ_GK, sigmin_TP
import config as cfg_mod

N, alpha, tau, t0, beta, seed = 10000, 0.05, 10, 1.0, 20.0, 42
IM_TOL = 1e-3
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "N=10000")


def main():
    cfg = cfg_mod.resolve()
    cfg.update(N=N, alpha=alpha, beta=beta, t0=t0, tau=tau, lam_max=0.6, ds=0.01,
               n_overlaps=5, M_cheb=20, n_eigs=8, arnoldi_ncv=50, arnoldi_maxiter=300)
    cfg["P"] = round(alpha * N)
    xi, xis = make_patterns(N, cfg["P"], seed)
    coup = Couplings(xi, xis); coup._use_np = True     # float64 exact path

    t0w = time.time()
    br, fold, st = trace_branch(coup, beta, cfg)
    print(f"branch: fold={fold:.4f} ({st}), {time.time()-t0w:.0f}s", flush=True)

    # six DISTINCT lambda clustered just below the fold; solve u* at each by
    # annealed Newton (exact, distinct) rather than snapping to a sparse branch point
    # pick six DISTINCT stable branch points at target lambda near the fold, on the
    # UPPER (stable) arm; use the branch states directly (no re-solving / drift).
    i_peak = int(np.argmax(br["lam"]))
    lam_up = br["lam"][:i_peak + 1]
    targets = [0.20, 0.235, 0.25, 0.258, 0.263, float(fold)]
    idxs = []
    for t in targets:
        j = int(np.argmin(np.abs(lam_up - t)))
        if j not in idxs:
            idxs.append(j)

    results = []
    for j in idxs:
        lam = float(br["lam"][j]); u = br["u_star"][j]
        em = eigmax_M(coup, u, lam, beta)          # stability sanity (must be < 0)
        t1 = time.time()
        phys, cand, sigs = physical_roots(coup, u, beta, lam, tau, t0, M=cfg["M_cheb"],
                                          n_cand=100, thresh=0.05, k=10, polish=True)
        # supplement: guarantee the rightmost REAL root (correct branch u*), in case
        # the reduced-IG candidates missed it; merge and re-take the 10 rightmost
        GJ, GK = GJ_GK(coup, u, beta, lam)
        rr = rightmost_real_root(GJ, GK, t0, tau, lam)
        rl = list(phys)
        if rr is not None and not any(abs(rr - w) < 1e-4 for w in rl):
            rl.append(rr)
        phys = np.array(rl, complex)
        phys = phys[np.argsort(-phys.real)][:10]
        isc = np.abs(phys.imag) >= IM_TOL
        re_real = phys.real[~isc].max() if np.any(~isc) else -np.inf
        re_cplx = phys.real[isc].max() if np.any(isc) else -np.inf
        results.append(dict(lam=lam, roots=phys, re_real=re_real, re_cplx=re_cplx))
        print(f"  lam={lam:.4f}: m1={float(coup._xi_np[0]@np.tanh(beta*u)/N):.4f} "
              f"eigM={em:+.3f} | {len(phys)} roots  "
              f"re_real={re_real:+.4f} re_cplx={re_cplx:+.4f}  ({time.time()-t1:.0f}s)",
              flush=True)

    # ---- figure: six complex-plane panels ---------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8)); axes = axes.ravel()
    xall = np.concatenate([r["roots"].real for r in results])
    xmin = min(-0.05, xall.min() - 0.03); xmax = 0.05
    for ax, r in zip(axes, results):
        z = r["roots"]
        isc = np.abs(z.imag) >= IM_TOL
        ax.scatter(z.real[~isc], z.imag[~isc], s=55, c="crimson", marker="o",
                   zorder=3, label="real modes", edgecolors="k", linewidths=.4)
        ax.scatter(z.real[isc], z.imag[isc], s=45, c="steelblue", marker="s",
                   zorder=3, label="complex modes", edgecolors="k", linewidths=.4)
        ax.axvline(0, color="k", lw=1.0, ls="--"); ax.axhline(0, color="k", lw=.6, ls=":")
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"$\\lambda={r['lam']:.4f}$  "
                     f"($\\lambda/\\lambda_c={r['lam']/fold:.3f}$)", fontsize=11)
        ax.set_xlabel("Re$(z)$"); ax.set_ylabel("Im$(z)$"); ax.grid(True, alpha=.25)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"10 rightmost PHYSICAL characteristic roots (exact, no artefacts) "
                 f"near $\\lambda_c={fold:.4f}$\n"
                 f"$N={N},\\ \\alpha={alpha},\\ \\tau={tau},\\ \\beta={beta}$ — "
                 f"the real fold-mode climbs to 0; complex modes stay left",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT, "fig2_physical_roots_near_lc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[fig2] -> {path}", flush=True)

    print("\nExact saddle-node-vs-Hopf check (rightmost real vs complex):")
    for r in results:
        tag = "REAL leads" if r["re_real"] >= r["re_cplx"] else "complex leads"
        print(f"  lam/lc={r['lam']/fold:.3f}: re_real={r['re_real']:+.4f} "
              f"re_cplx={r['re_cplx']:+.4f}   -> {tag}")


if __name__ == "__main__":
    main()
