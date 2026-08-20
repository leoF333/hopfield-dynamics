"""
Quick validation run at small N=2000.

Checks performed
----------------
1. Pattern generation and matrix-free ops (apply_M vs dense).
2. Chebyshev differentiation matrix accuracy.
3. Newton solve at lambda=0 and lambda=0.5.
4. IG eigenvalue computation at the reference tau.
5. tau=0 reduction: IG spectrum vs ODE spectrum.
6. Short DDE simulation.
7. M-convergence diagnostic.

Usage
-----
    conda run -n mcmc_env python run_validation.py [--tau 10] [--beta 20]

Outputs: numerics/results/validation/
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse

import config as cfg_mod
from couplings import Couplings, make_patterns
from fixed_point import newton_solve, continuation, save_branch
from dde_stability import (cheb_diff_matrix, IGOperator, rightmost_eigenvalues,
                            m_convergence, lambda_sweep)
from simulate_dde import DDEIntegrator, measure_oscillation
from analytic_check import locus_analytic
import plots


def main():
    p = cfg_mod.make_parser("Validation run at N=2000")
    args = p.parse_args()
    cfg  = cfg_mod.resolve(args)
    # Force small N
    cfg["N"]   = cfg["small_N"]
    cfg["P"]   = round(cfg["alpha"] * cfg["N"])
    cfg["M_cheb"] = 24

    print("=" * 60)
    print(f" VALIDATION  N={cfg['N']}  alpha={cfg['alpha']}  "
          f"tau={cfg['tau']}  beta={cfg['beta']}")
    print("=" * 60)

    # ---- 1. Patterns -------------------------------------------------------
    print("\n[1] Generating patterns...")
    xi, xis = make_patterns(cfg["N"], cfg["P"], cfg["seed"])
    coup    = Couplings(xi, xis)
    print(f"    xi shape: {xi.shape},  xi_shift shape: {xis.shape}")

    # ---- 2. Chebyshev -------------------------------------------------------
    print("\n[2] Chebyshev differentiation accuracy...")
    for M in [8, 16, 32]:
        D, x = cheb_diff_matrix(M)
        f    = np.cos(x)
        fp   = D @ f
        err  = np.max(np.abs(fp - (-np.sin(x))))
        print(f"    M={M:3d}:  max|D cos - (-sin)| = {err:.2e}")

    # ---- 3. Matrix-free correctness ----------------------------------------
    print("\n[3] Matrix-free apply_M vs dense (N=100 subset)...")
    xi_s, xis_s = make_patterns(100, 5, 42)
    coup_s = Couplings(xi_s, xis_s)
    u0_s   = 0.99 * xi_s[0]
    u_s, _ = newton_solve(u0_s, 0.3, coup_s, cfg["beta"])
    gain_s = coup_s.compute_gain(u_s, cfg["beta"])
    M_d    = coup_s.dense_M(gain_s, 0.3)
    rng    = np.random.default_rng(0)
    v      = rng.standard_normal(100)
    Mv_mf  = coup_s.apply_M(v, gain_s, 0.3)
    Mv_d   = M_d @ v
    rel    = np.linalg.norm(Mv_mf - Mv_d) / (np.linalg.norm(Mv_d) + 1e-14)
    print(f"    Relative error apply_M vs dense: {rel:.2e}  "
          f"({'OK' if rel < 1e-4 else 'FAIL'})")

    # ---- 4. Newton at lambda=0 and lambda=0.5 ------------------------------
    print("\n[4] Newton solve at lambda=0 and 0.5...")
    u0 = 0.99 * coup._xi_np[0].copy()
    for lam in [0.0, 0.3, 0.5]:
        u, ok = newton_solve(u0, lam, coup, cfg["beta"])
        F_res = coup.field_F(u, lam, cfg["beta"])
        res   = np.linalg.norm(F_res) / np.sqrt(cfg["N"])
        m_nu  = coup.condensed_overlaps(u, cfg["beta"])
        print(f"    lam={lam:.2f}: converged={ok}  ||F||/sqrtN={res:.2e}  "
              f"m1={m_nu[0]:.4f}  m2={m_nu[1]:.4f}")
        u0 = u   # warm start

    # ---- 5. Short fixed-point branch ----------------------------------------
    print("\n[5] Short pseudo-arclength continuation (lam_max=0.6)...")
    cfg_short = dict(cfg)
    cfg_short["lam_max"] = 0.6
    cfg_short["ds"]      = 0.05
    u0 = 0.99 * coup._xi_np[0].copy()
    branch = continuation(coup, cfg["beta"], cfg_short, u0=u0)
    print(f"    Branch: {len(branch['lam'])} points, "
          f"lam in [{branch['lam'].min():.3f}, {branch['lam'].max():.3f}]")
    print(f"    m1 range: [{branch['m_nu'][:,0].min():.3f}, {branch['m_nu'][:,0].max():.3f}]")

    # ---- 6. IG eigenvalues at reference tau --------------------------------
    # Reference lambda MUST sit on the memory branch (below any finite-size
    # fold). The branch folds near lam_max(branch); pick 90% of that so u_ref
    # is a genuine memory fixed point, not a diverged Newton iterate.
    lam_ref = 0.9 * float(branch["lam"].max())
    i_ref   = int(np.argmin(np.abs(branch["lam"] - lam_ref)))
    lam_ref = float(branch["lam"][i_ref])
    u_ref   = branch["u_star"][i_ref].copy()
    print(f"\n[6] IG eigenvalues at tau={cfg['tau']}, "
          f"lambda={lam_ref:.3f} (on memory branch, below fold) ...")
    ev = rightmost_eigenvalues(u_ref, lam_ref, coup, cfg, M=cfg["M_cheb"])
    print(f"    Top 5 characteristic roots (by Re):")
    for z in ev[:5]:
        print(f"      z = {z.real:+.4f} + {z.imag:+.4f}i")

    # ---- 7. tau=0 reduction -----------------------------------------------
    print("\n[7] tau=0 reduction check (N=100, dense ODE vs IG)...")
    from scipy.linalg import eigvals as sp_eigvals
    N_s   = 100
    xi_t, xis_t = make_patterns(N_s, 5, 42)
    coup_t = Couplings(xi_t, xis_t)
    u_t, _ = newton_solve(0.99 * xi_t[0], 0.3, coup_t, cfg["beta"])
    gain_t = coup_t.compute_gain(u_t, cfg["beta"])
    M_d    = coup_t.dense_M(gain_t, 0.3)
    eig_ode = sp_eigvals(M_d)
    max_re_ode = float(eig_ode.real.max())
    # IG at tau=2 (small, proxy for tau->0)
    cfg_t2 = dict(cfg)
    cfg_t2["tau"] = 2
    n2 = max(1, round(2 / cfg["dt"]))
    cfg_t2["dt"]    = 2 / n2
    cfg_t2["L_buf"] = n2
    ev_ig = rightmost_eigenvalues(u_t, 0.3, coup_t, cfg_t2, M=32)
    max_re_ig = float(ev_ig.real.max()) if len(ev_ig) else np.nan
    print(f"    ODE max Re(z)/t0 = {max_re_ode:.4f},  IG max Re(z) = {max_re_ig:.4f}")
    print(f"    Difference: {abs(max_re_ig - max_re_ode):.4f}  "
          f"({'OK' if abs(max_re_ig - max_re_ode) < 0.3 else 'check'})")

    # ---- 8. M-convergence --------------------------------------------------
    print(f"\n[8] M-convergence at tau={cfg['tau']}, lambda={lam_ref:.3f} ...")
    conv = m_convergence(u_ref, lam_ref, coup, cfg, M_values=[12, 16, 24, 32])
    for M_val, ev_m in sorted(conv.items()):
        z = ev_m[0]
        print(f"    M={M_val:3d}: z1 = {z.real:+.5f} + {z.imag:+.5f}i")

    # ---- 9. Short DDE simulation -------------------------------------------
    print(f"\n[9] Short DDE simulation at lambda=0.6 ...")
    intgr  = DDEIntegrator(coup, cfg)
    u0_dde = 0.99 * coup._xi_np[0].copy()
    result = intgr.run(0.6, u0_dde, t_total=cfg["t_transient"] + 50.0)
    meas   = measure_oscillation(result["m1"], cfg["dt"],
                                  t_transient=cfg["t_transient"])
    print(f"    A={meas['A']:.4f}  T_osc={meas['T_osc']}  fp={meas['fixed_point']}")

    # ---- 10. Fig 1 from the short branch -----------------------------------
    print("\n[10] Saving fig1 (fixed-point branch)...")
    plots.fig1_fixed_point_branch(branch, cfg)

    # ---- 11. Lambda sweep and fig 2, 3 ------------------------------------
    print("\n[11] Lambda sweep for figs 2, 3 (n_lam=12)...")
    branch_full_cfg = dict(cfg)
    branch_full_cfg["lam_max"] = 0.6
    sweep = lambda_sweep(branch, coup, branch_full_cfg, n_lam=12)
    plots.fig2_rightmost_roots_panel(sweep, cfg)
    plots.fig3_max_re_vs_lambda(sweep, cfg)

    # ---- 12. M-convergence figure -----------------------------------------
    print("\n[12] Saving fig9 (M-convergence)...")
    plots.fig9_m_convergence(conv, cfg)

    print("\n" + "=" * 60)
    print(" VALIDATION COMPLETE")
    crossing = sweep.get("crossing", {})
    lam_c    = crossing.get("lam_c", None)
    nature   = crossing.get("nature", "unknown")
    print(f"  lambda_c ~ {lam_c}   nature: {nature}")
    print(f"  Outputs in: {cfg['out_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
