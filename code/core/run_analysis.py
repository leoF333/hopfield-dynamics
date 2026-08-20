"""
Full analysis run at N=10000 (or --small for N=2000).

Steps
-----
1. Fixed-point branch (pseudo-arclength continuation in lambda).
2. DDE spectral analysis: lambda sweep at reference tau.
3. M-convergence diagnostic.
4. Hopf locus: sweep over tau.
5. DDE simulation: up/down lambda sweeps (hysteresis).
6. Amplitude and period scaling fits near lambda_c.
7. Phase portrait and space-time raster.
8. Floquet multipliers at a lambda above the Hopf.
9. Analytic (condensed-subspace) estimate overlay.
10. Discrete bridge (optional, tau<=20 only).
11. Print verdict and save NOTES entry.

Usage
-----
    conda run -n mcmc_env python run_analysis.py [--small] [--alpha 0.05] [--tau 10]

Outputs: numerics/results/  (see config.py for subdirectories)
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
from fixed_point import continuation, save_branch, load_branch
from dde_stability import lambda_sweep, m_convergence, rightmost_eigenvalues
from hopf_locus import compute_locus, save_locus
from simulate_dde import (DDEIntegrator, hysteresis_sweep, run_trajectory,
                           save_sweep)
from floquet import floquet_from_simulation, store_orbit, save_floquet
from analytic_check import locus_analytic, save_analytic
from discrete_bridge import find_discrete_lc
import plots


# ----------------------------------------------------------------- verdict

def _print_verdict(cfg, lam_c_linear, lam_c_sim, lam_c_fold,
                   nature, outlier, alpha_exp, gamma_exp,
                   crossing_details):
    """Print and return the final scientific verdict."""
    lines = [
        "=" * 65,
        " BIFURCATION VERDICT",
        "=" * 65,
        f"  Parameters: N={cfg['N']}, alpha={cfg['alpha']}, "
        f"tau={cfg['tau']}, beta={cfg['beta']}",
        "",
        f"  Bifurcation nature    : {nature}",
        f"  Destabilizing mode    : {'outlier (discrete)' if outlier else 'bulk edge (glassy)' if outlier is False else 'unknown'}",
        "",
        "  lambda_c estimates:",
        f"    Linear stability    : {lam_c_linear}",
        f"    Simulation onset    : {lam_c_sim}",
        f"    Continuation fold   : {lam_c_fold}",
    ]

    # Agreement check
    vals = [v for v in [lam_c_linear, lam_c_sim, lam_c_fold] if v is not None]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        agree  = spread < 0.05
        lines.append(f"    Spread (max-min)    : {spread:.4f}  "
                     f"{'(AGREE)' if agree else '(DISAGREE — check)'}")

    lines += [
        "",
        f"  omega_c at onset      : {crossing_details.get('omega_c')}",
        f"  T_c = 2pi/omega_c     : {crossing_details.get('T_c')}",
        "",
        "  Criticality (from simulation):",
        f"    Amplitude exponent  : {alpha_exp} "
        f"(supercritical ~ 0.5; subcritical: finite jump)",
        f"    Period exponent     : {gamma_exp} "
        f"(Hopf ~ 0; SNIC ~ -0.5)",
        "",
    ]

    # Classification
    if nature == "Hopf":
        if alpha_exp is not None and not np.isnan(alpha_exp) and 0.3 < alpha_exp < 0.8:
            criticality = "supercritical Hopf (no hysteresis expected)"
        elif alpha_exp is not None and not np.isnan(alpha_exp) and alpha_exp < 0.1:
            criticality = "subcritical Hopf (possible hysteresis)"
        else:
            criticality = "Hopf (criticality TBD from scaling)"
    elif nature.startswith("saddle"):
        criticality = "SNIC / saddle-node (period diverges at onset)"
    else:
        criticality = "undetermined"

    lines.append(f"  Classification       : {criticality}")
    lines.append("=" * 65)
    verdict = "\n".join(lines)
    print(verdict)
    return verdict


def _save_verdict(verdict: str, cfg: dict):
    path = cfg_mod.out_path(cfg, "", "verdict.txt")
    with open(path, "w") as f:
        f.write(verdict + "\n")
    print(f"[run_analysis] Verdict saved → {path}")


# ----------------------------------------------------------------- main

def main():
    p = cfg_mod.make_parser("Full analysis: delayed mixed Hopfield")
    p.add_argument("--skip-floquet",   action="store_true")
    p.add_argument("--skip-sim",       action="store_true")
    p.add_argument("--skip-locus",     action="store_true")
    p.add_argument("--skip-discrete",  action="store_true")
    p.add_argument("--reload-branch",  type=str, default=None,
                   help="Path to existing branch NPZ (skip recomputing)")
    args = p.parse_args()
    cfg  = cfg_mod.resolve(args)

    print("=" * 65)
    print(f" FULL ANALYSIS  N={cfg['N']}  alpha={cfg['alpha']}  "
          f"tau={cfg['tau']}  beta={cfg['beta']}")
    print("=" * 65)

    # ---- 1. Patterns -------------------------------------------------------
    print("\n--- Step 1: Pattern generation ---")
    xi, xis = make_patterns(cfg["N"], cfg["P"], cfg["seed"])
    coup    = Couplings(xi, xis)

    # ---- 2. Fixed-point branch ---------------------------------------------
    print("\n--- Step 2: Fixed-point branch (pseudo-arclength) ---")
    if args.reload_branch:
        print(f"  Loading branch from {args.reload_branch}")
        branch = load_branch(args.reload_branch)
    else:
        u0     = 0.99 * coup._xi_np[0].copy()
        branch = continuation(coup, cfg["beta"], cfg, u0=u0)
        save_branch(branch, coup, cfg)

    lam_branch = branch["lam"]
    m_nu_branch = branch["m_nu"]
    print(f"  Branch: {len(lam_branch)} pts, "
          f"lam in [{lam_branch.min():.3f}, {lam_branch.max():.3f}]")
    print(f"  m1 at max lam: {m_nu_branch[-1, 0]:.4f}")

    # Check for fold
    lam_c_fold = None
    if lam_branch[-1] < lam_branch.max() - 0.01:
        i_fold = int(np.argmax(lam_branch))
        lam_c_fold = float(lam_branch[i_fold])
        print(f"  FOLD detected at lambda ~ {lam_c_fold:.4f}")

    # Fig 1
    plots.fig1_fixed_point_branch(branch, cfg)

    # ---- 3. DDE spectral analysis at reference tau -------------------------
    print(f"\n--- Step 3: DDE spectral analysis (tau={cfg['tau']}) ---")
    sweep = lambda_sweep(branch, coup, cfg, n_lam=30)
    crossing   = sweep["crossing"]
    lam_c_lin  = crossing.get("lam_c", None)
    nature     = crossing.get("nature", "unknown")
    outlier    = crossing.get("outlier", None)

    print(f"  lambda_c (linear stability) = {lam_c_lin}")
    print(f"  Nature: {nature}")
    print(f"  Outlier mode: {outlier}")

    plots.fig2_rightmost_roots_panel(sweep, cfg)
    plots.fig3_max_re_vs_lambda(sweep, cfg)

    # M-convergence at lambda near lambda_c
    print(f"\n--- Step 3b: M-convergence ---")
    from dde_stability import _interp_u
    lam_test  = lam_c_lin if lam_c_lin else 0.4
    u_test    = _interp_u(lam_test, lam_branch, branch["u_star"])
    conv_res  = m_convergence(u_test, lam_test, coup, cfg,
                               M_values=[16, 24, 32, 48])
    plots.fig9_m_convergence(conv_res, cfg)

    # Save sweep
    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "stability", f"sweep_{tag}.npz")
    np.savez(path,
             lam_vals = sweep["lam_vals"],
             max_re   = sweep["max_re"])
    print(f"  Sweep saved → {path}")

    # ---- 4. Analytic estimate ----------------------------------------------
    print("\n--- Step 4: Analytic condensed-subspace estimate ---")
    m1_est  = float(branch["m_nu"][len(branch["lam"]) // 2, 0])
    tau_vals_analytic = np.array(cfg["tau_list"], dtype=float)
    analytic = locus_analytic(tau_vals_analytic, m1_est, cfg, k=1)
    save_analytic(analytic, cfg)
    print(f"  Analytic lam_c at tau={cfg['tau']}: "
          f"{analytic['lam_c_analytic'][tau_vals_analytic == cfg['tau']]}")

    # ---- 5. Hopf locus (tau sweep) ----------------------------------------
    if not args.skip_locus:
        print("\n--- Step 5: Hopf locus (tau sweep) ---")
        locus = compute_locus(branch, coup, cfg, tau_values=cfg["tau_list"])
        save_locus(locus, cfg)
        plots.fig4_hopf_locus(locus, analytic, cfg)
    else:
        locus = None
        print("  Skipped.")

    # ---- 6. Simulation (hysteresis) ----------------------------------------
    alpha_exp = gamma_exp = np.nan
    lam_c_sim = None
    sweep_up  = sweep_down = None

    if not args.skip_sim:
        print(f"\n--- Step 6: DDE simulation sweep (tau={cfg['tau']}) ---")
        lam_lo = max(0.05, (lam_c_lin or 0.3) - 0.25)
        lam_hi = min(0.95, (lam_c_lin or 0.5) + 0.25)
        sweep_up, sweep_down = hysteresis_sweep(
            coup, cfg, lam_lo=lam_lo, lam_hi=lam_hi, n_pts=16)

        save_sweep(sweep_up,   "up",   cfg)
        save_sweep(sweep_down, "down", cfg)

        plots.fig5_bifurcation_diagram(sweep_up, sweep_down, cfg, lam_c_lin)

        # Detect lambda_c from simulation: first lambda where A > threshold
        mask = (~sweep_up["fixed_point"]) & (sweep_up["A"] > 0.02)
        if mask.any():
            lam_c_sim = float(sweep_up["lam"][mask].min())
            print(f"  lambda_c (simulation) ~ {lam_c_sim:.4f}")

        # Scaling
        if lam_c_lin is not None:
            _, scaling = plots.fig6_scaling(sweep_up, lam_c_lin, cfg)
            alpha_exp  = scaling.get("alpha_exp", np.nan)
            gamma_exp  = scaling.get("gamma_exp", np.nan)
            print(f"  Amplitude exponent: {alpha_exp:.3f}")
            print(f"  Period exponent:    {gamma_exp:.3f}")

        # Phase portrait and raster
        print(f"\n--- Step 6b: Trajectory for phase portrait ---")
        lam_below = max(0.05, (lam_c_lin or 0.3) - 0.1)
        lam_above = min(0.95, (lam_c_lin or 0.5) + 0.1)
        traj_below = run_trajectory(lam_below, coup, cfg, save_every=5)
        traj_above = run_trajectory(lam_above, coup, cfg, save_every=5)
        plots.fig7_phase_portrait_raster(traj_below, traj_above, cfg)
    else:
        print("  Simulation skipped.")

    # ---- 7. Floquet --------------------------------------------------------
    if not args.skip_floquet and sweep_up is not None:
        print(f"\n--- Step 7: Floquet multipliers ---")
        # Find a lambda where oscillation exists
        mask_osc = (~sweep_up["fixed_point"]) & (sweep_up["A"] > 0.05)
        if mask_osc.any():
            lam_floq = float(sweep_up["lam"][mask_osc].mean())
            print(f"  Computing Floquet at lambda={lam_floq:.3f} ...")
            # Get orbit from simulation
            intgr   = DDEIntegrator(coup, cfg)
            u0_f    = 0.99 * coup._xi_np[0].copy()
            res_tr  = intgr.run(lam_floq, u0_f, t_total=cfg["t_transient"])
            T_osc   = float(sweep_up["T_osc"][mask_osc][0])
            if np.isfinite(T_osc):
                fl_res = floquet_from_simulation(
                    lam_floq, coup, cfg, T_osc, res_tr["buf_final"], k=8)
                save_floquet(fl_res, cfg)
                plots.fig8_floquet(fl_res, cfg)
            else:
                print("  T_osc not available; skipping Floquet.")
        else:
            print("  No oscillation found; skipping Floquet.")
    else:
        print("  Floquet skipped.")

    # ---- 8. Discrete bridge ------------------------------------------------
    if not args.skip_discrete and cfg["tau"] <= 20:
        print(f"\n--- Step 8: Discrete bridge (Neimark-Sacker) ---")
        from dde_stability import _interp_u as _iu
        disc_result = find_discrete_lc(coup, cfg, n_eig=6)
        lam_c_disc  = disc_result.get("lam_c_disc")
        print(f"  Neimark-Sacker crossing: lambda_c_disc = {lam_c_disc}")
        # Compare to continuous Hopf
        if lam_c_disc is not None and lam_c_lin is not None:
            diff = abs(lam_c_disc - lam_c_lin)
            print(f"  |lambda_c_disc - lambda_c_continuous| = {diff:.4f}")
    else:
        print("  Discrete bridge skipped.")

    # ---- 9. Verdict --------------------------------------------------------
    print("\n--- Step 9: Final verdict ---")
    verdict = _print_verdict(
        cfg,
        lam_c_linear  = lam_c_lin,
        lam_c_sim     = lam_c_sim,
        lam_c_fold    = lam_c_fold,
        nature        = nature,
        outlier       = outlier,
        alpha_exp     = float(alpha_exp) if not np.isnan(alpha_exp) else None,
        gamma_exp     = float(gamma_exp) if not np.isnan(gamma_exp) else None,
        crossing_details = crossing,
    )
    _save_verdict(verdict, cfg)

    # ---- 10. Notes entry ---------------------------------------------------
    notes_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "NOTES_numerics.md")
    from datetime import date
    entry = f"""
## {date.today().isoformat()} — DDE bifurcation analysis (continuous-time)
**Parameters:** N={cfg['N']}, alpha={cfg['alpha']}, tau={cfg['tau']}, beta={cfg['beta']}, seed={cfg['seed']}
**Tested:** Full pipeline — fixed-point branch, IG spectrum, Hopf locus, simulation, Floquet.
**Result:**
- Nature: {nature}
- lambda_c (linear): {lam_c_lin}
- lambda_c (simulation): {lam_c_sim}
- lambda_c (fold): {lam_c_fold}
- Amplitude exponent: {alpha_exp:.3f}
- Period exponent: {gamma_exp:.3f}
- Outlier mode: {outlier}
**Verdict:** {nature} bifurcation — see {cfg['out_dir']}/verdict.txt
"""
    try:
        with open(notes_path, "a") as f:
            f.write(entry)
        print(f"[run_analysis] NOTES appended → {notes_path}")
    except Exception as e:
        print(f"[run_analysis] Could not write NOTES: {e}")


if __name__ == "__main__":
    main()
