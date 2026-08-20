"""
Figure-generating functions for the delayed mixed Hopfield analysis.
All figures saved as PNG at dpi=150.  All labels in English.

Figure inventory (matching §5 of the spec)
------------------------------------------
fig1 : Fixed-point branch m1, m2, m3 vs lambda
fig2 : Rightmost roots in complex plane at multiple lambda (panel)
fig3 : max Re(z) vs lambda with lambda_c marked
fig4 : Hopf locus lambda_c(tau) and omega_c(tau) / T_c(tau)
fig5 : Bifurcation diagram A(lambda) — up and down sweeps
fig6 : Log-log scaling plots A ~ (lam-lam_c)^exponent and T(lam)
fig7 : Phase portrait in (m1, m2, m3) and space-time raster of m_nu(t)
fig8 : Floquet multipliers in complex plane
fig9 : M-convergence diagnostic
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress

import config as cfg_mod


# ----------------------------------------------------------------- helpers

def _save(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")


def _unit_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5)


# ----------------------------------------------------------------- fig 1

def fig1_fixed_point_branch(branch: dict, cfg: dict):
    """Fixed-point overlaps m1, m2, m3 vs lambda.  Mark any fold."""
    lam   = branch["lam"]
    m_nu  = branch["m_nu"]        # (n_pts, n_ov)
    n_ov  = min(m_nu.shape[1], 3)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["$m_1$", "$m_2$", "$m_3$"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for nu in range(n_ov):
        ax.plot(lam, m_nu[:, nu], color=colors[nu], label=labels[nu])

    # Mark fold (max lambda along branch)
    if lam[-1] < lam.max() - 0.01:   # branch turned back
        i_fold = int(np.argmax(lam))
        ax.axvline(lam[i_fold], color="red", ls="--", lw=1, label="fold")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Condensed overlap $m_\nu$")
    ax.set_title(f"Fixed-point branch  "
                 f"($N={cfg['N']},\\,\\alpha={cfg['alpha']},\\,\\beta={cfg['beta']}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "fixed_point", f"fig1_branch_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path


# ----------------------------------------------------------------- fig 2

def fig2_rightmost_roots_panel(sweep_result: dict, cfg: dict):
    """
    Panel: characteristic roots in the complex plane at several lambda values.
    Color encodes lambda (cold=stable, warm=unstable).
    """
    lam_vals   = sweep_result["lam_vals"]
    evals_list = sweep_result["evals"]
    crossing   = sweep_result.get("crossing", {})
    lam_c      = crossing.get("lam_c", None)

    n_show = min(len(lam_vals), 6)
    idx    = np.linspace(0, len(lam_vals) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ii, i in enumerate(idx):
        ax   = axes[ii]
        ev   = evals_list[i]
        lam  = lam_vals[i]
        ax.scatter(ev.real, ev.imag, s=20, c="steelblue", alpha=0.8, zorder=3)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_title(f"$\\lambda={lam:.3f}$")
        ax.set_xlabel("Re$(z)$")
        ax.set_ylabel("Im$(z)$")
        ax.grid(True, alpha=0.2)

    if lam_c is not None:
        fig.suptitle(
            f"Characteristic roots  ($\\tau={cfg['tau']},\\,\\lambda_c\\approx{lam_c:.3f}$)",
            fontsize=12)
    else:
        fig.suptitle(f"Characteristic roots  ($\\tau={cfg['tau']}$)", fontsize=12)

    fig.tight_layout()
    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "stability", f"fig2_roots_panel_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path


# ----------------------------------------------------------------- fig 3

def fig3_max_re_vs_lambda(sweep_result: dict, cfg: dict):
    """max Re(z) vs lambda with lambda_c marked; annotate nature and outlier."""
    lam_vals = sweep_result["lam_vals"]
    max_re   = sweep_result["max_re"]
    crossing = sweep_result.get("crossing", {})
    lam_c    = crossing.get("lam_c", None)
    nature   = crossing.get("nature", "unknown")
    outlier  = crossing.get("outlier", None)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_vals, max_re, "b-o", ms=4, label=r"$\max_k \mathrm{Re}\,z_k$")
    ax.axhline(0, color="k", lw=0.8, ls="--")

    if lam_c is not None:
        ax.axvline(lam_c, color="red", lw=1.5, ls="--",
                   label=f"$\\lambda_c\\approx{lam_c:.3f}$")

    annot = f"Nature: {nature}"
    if outlier is not None:
        annot += f"\nMode: {'outlier' if outlier else 'bulk edge'}"
    ax.text(0.02, 0.95, annot, transform=ax.transAxes, va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\max_k\,\mathrm{Re}\,z_k(\lambda)$")
    ax.set_title(f"Stability margin  ($\\tau={cfg['tau']},\\,N={cfg['N']}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "stability", f"fig3_maxre_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path


# ----------------------------------------------------------------- fig 4

def fig4_hopf_locus(locus: dict, analytic: dict | None, cfg: dict):
    """Hopf locus lambda_c(tau) and omega_c(tau) / T_c(tau)."""
    tau  = locus["tau_vals"]
    lc   = locus["lam_c"]
    wc   = locus["omega_c"]
    Tc   = locus["T_c"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: lambda_c vs tau
    ax = axes[0]
    ax.plot(tau, lc, "b-o", ms=5, label="Numerical $\\lambda_c$")
    if analytic is not None:
        lca = analytic["lam_c_analytic"]
        ax.plot(tau, lca, "r--", lw=1.5,
                label="Analytic (condensed, $\\alpha=0$)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\lambda_c(\tau)$")
    ax.set_title("Hopf locus")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: omega_c vs tau
    ax = axes[1]
    ax.plot(tau, wc, "b-o", ms=5, label="$\\omega_c$ (numerical)")
    if analytic is not None:
        wca = analytic["omega_c_analytic"]
        ax.plot(tau, wca, "r--", lw=1.5, label="$\\omega_c$ (analytic)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\omega_c(\tau)$")
    ax.set_title("Onset frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: T_c vs tau
    ax = axes[2]
    ax.plot(tau, Tc, "b-o", ms=5, label="$T_c$ (numerical)")
    if analytic is not None:
        Tca = analytic["T_c_analytic"]
        ax.plot(tau, Tca, "r--", lw=1.5, label="$T_c$ (analytic)")
    # Overlay expected pacemaker scale tau+1
    ax.plot(tau, tau + 1, "g:", lw=1.5, label=r"$\tau+1$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$T_c(\tau) = 2\pi/\omega_c$")
    ax.set_title("Onset period")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Hopf locus  ($N={cfg['N']},\\,\\alpha={cfg['alpha']},\\,"
                 f"\\beta={cfg['beta']}$)", fontsize=12)
    fig.tight_layout()

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "hopf_locus", f"fig4_locus_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path


# ----------------------------------------------------------------- fig 5

def fig5_bifurcation_diagram(sweep_up: dict, sweep_down: dict, cfg: dict,
                              lam_c_linear: float | None = None):
    """Amplitude A(lambda) — up and down sweeps for hysteresis."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(sweep_up["lam"],   sweep_up["A"],   "b-o", ms=4,
            label="Increasing $\\lambda$")
    ax.plot(sweep_down["lam"], sweep_down["A"], "r-s", ms=4,
            label="Decreasing $\\lambda$")

    if lam_c_linear is not None:
        ax.axvline(lam_c_linear, color="green", ls="--", lw=1.5,
                   label=f"$\\lambda_c^{{\\rm linear}}\\approx{lam_c_linear:.3f}$")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Amplitude $A(\lambda)$")
    ax.set_title(f"Bifurcation diagram  ($\\tau={cfg['tau']},\\,N={cfg['N']}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "simulation", f"fig5_bifdiag_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path


# ----------------------------------------------------------------- fig 6

def fig6_scaling(sweep: dict, lam_c: float, cfg: dict) -> dict:
    """
    Log-log scaling plots of A and T near lambda_c.
    Fit A ~ (lam - lam_c)^alpha_exp and T ~ (lam - lam_c)^gamma_exp.
    Returns fitted exponents.
    """
    lam  = sweep["lam"]
    A    = sweep["A"]
    T    = sweep["T_osc"]
    fp   = sweep["fixed_point"]

    # Filter: above lambda_c and not fixed point
    mask = (lam > lam_c + 0.005) & (~fp) & np.isfinite(A) & (A > 0.01)
    lam_a = lam[mask]
    A_a   = A[mask]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Amplitude scaling
    ax = axes[0]
    if mask.sum() >= 3:
        dl = lam_a - lam_c
        log_dl = np.log10(dl)
        log_A  = np.log10(A_a)
        slope, intercept, r, *_ = linregress(log_dl, log_A)
        ax.loglog(dl, A_a, "bo", ms=5, label="Simulation")
        ax.loglog(dl, 10**intercept * dl**slope, "b--",
                  label=f"fit: $A\\propto\\Delta\\lambda^{{{slope:.2f}}}$")
        alpha_exp = float(slope)
    else:
        ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center")
        alpha_exp = np.nan

    ax.set_xlabel(r"$\lambda - \lambda_c$")
    ax.set_ylabel(r"$A(\lambda)$")
    ax.set_title("Amplitude scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Period scaling
    ax = axes[1]
    mask_T = mask & np.isfinite(T) & (T > 0)
    lam_t  = lam[mask_T]
    T_t    = T[mask_T]
    if mask_T.sum() >= 3:
        dl_t   = lam_t - lam_c
        log_dl = np.log10(dl_t)
        log_T  = np.log10(T_t)
        slope_T, intercept_T, *_ = linregress(log_dl, log_T)
        ax.loglog(dl_t, T_t, "ro", ms=5, label="Simulation")
        ax.loglog(dl_t, 10**intercept_T * dl_t**slope_T, "r--",
                  label=f"fit: $T\\propto\\Delta\\lambda^{{{slope_T:.2f}}}$")
        gamma_exp = float(slope_T)
    else:
        ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center")
        gamma_exp = np.nan

    ax.set_xlabel(r"$\lambda - \lambda_c$")
    ax.set_ylabel(r"$T(\lambda)$")
    ax.set_title("Period scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Scaling near $\\lambda_c\\approx{lam_c:.3f}$  "
                 f"($\\tau={cfg['tau']}$)", fontsize=11)
    fig.tight_layout()

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "simulation", f"fig6_scaling_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path, dict(alpha_exp=alpha_exp, gamma_exp=gamma_exp)


# ----------------------------------------------------------------- fig 7

def fig7_phase_portrait_raster(traj_below: dict, traj_above: dict,
                                cfg: dict):
    """Phase portrait in (m1, m2, m3) and space-time raster of m_nu(t)."""
    fig = plt.figure(figsize=(14, 8))

    # --- Phase portraits ---
    for ii, (traj, label) in enumerate([(traj_below, "below $\\lambda_c$"),
                                         (traj_above, "above $\\lambda_c$")]):
        m = traj["m_nu"]
        if m.shape[1] < 3:
            continue
        ax = fig.add_subplot(2, 3, ii + 1, projection="3d")
        ax.plot(m[:, 0], m[:, 1], m[:, 2], lw=0.5, alpha=0.8)
        ax.set_xlabel("$m_1$", fontsize=8)
        ax.set_ylabel("$m_2$", fontsize=8)
        ax.set_zlabel("$m_3$", fontsize=8)
        ax.set_title(f"Phase portrait\n{label}", fontsize=9)

    # --- Space-time rasters ---
    for ii, (traj, label) in enumerate([(traj_below, "below $\\lambda_c$"),
                                         (traj_above, "above $\\lambda_c$")]):
        m  = traj["m_nu"]
        t  = traj["t_arr"]
        ax = fig.add_subplot(2, 3, ii + 4)
        im = ax.imshow(m.T, aspect="auto",
                       extent=[t[0], t[-1], 0.5, m.shape[1] + 0.5],
                       origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Pattern index $\\nu$", fontsize=8)
        ax.set_title(f"$m_\\nu(t)$ — {label}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Phase portrait and space-time raster  "
                 f"($\\tau={cfg['tau']},\\,N={cfg['N']}$)", fontsize=11)
    fig.tight_layout()

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "simulation", f"fig7_portrait_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path


# ----------------------------------------------------------------- fig 8

def fig8_floquet(floquet_result: dict, cfg: dict):
    """Floquet multipliers in complex plane with unit circle."""
    mults = floquet_result["multipliers"]
    lam   = floquet_result.get("lam", cfg.get("lam_max", 0.8))
    T_per = floquet_result.get("T_period", None)

    fig, ax = plt.subplots(figsize=(5, 5))
    _unit_circle(ax)
    ax.scatter(mults.real, mults.imag, s=40, c="steelblue", zorder=3,
               label="Floquet multipliers")
    # Highlight trivial multiplier (nearest to 1)
    idx_triv = np.argmin(np.abs(mults - 1.0))
    ax.scatter(mults[idx_triv].real, mults[idx_triv].imag,
               s=100, c="red", marker="*", zorder=4, label="|$\\mu_{\\rm triv}$|")

    ax.set_xlabel(r"Re$(\mu)$")
    ax.set_ylabel(r"Im$(\mu)$")
    title = f"Floquet multipliers  ($\\lambda={lam:.3f}$"
    if T_per is not None:
        title += f", $T\\approx{T_per:.1f}$"
    title += ")"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "floquet", f"fig8_floquet_{tag}_lam{lam:.3f}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path


# ----------------------------------------------------------------- fig 9

def fig9_m_convergence(conv_result: dict, cfg: dict):
    """M-convergence diagnostic: leading roots vs M."""
    M_vals = sorted(conv_result.keys())

    # Show real and imaginary parts of the top eigenvalue vs M
    re_vals = [conv_result[M][0].real for M in M_vals]
    im_vals = [conv_result[M][0].imag for M in M_vals]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(M_vals, re_vals, "b-o", ms=6)
    ax.set_xlabel("$M$ (Chebyshev intervals)")
    ax.set_ylabel(r"Re$(z_1)$ (leading root)")
    ax.set_title("M-convergence: real part")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(M_vals, im_vals, "r-o", ms=6)
    ax.set_xlabel("$M$ (Chebyshev intervals)")
    ax.set_ylabel(r"Im$(z_1)$ (leading root)")
    ax.set_title("M-convergence: imaginary part")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Pseudospectral M-convergence  ($\\tau={cfg['tau']}$)",
                 fontsize=11)
    fig.tight_layout()

    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "stability", f"fig9_mconv_{tag}.png")
    _save(fig, path, cfg.get("dpi", 150))
    return path
