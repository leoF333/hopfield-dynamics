"""
Condensed-subspace analytic estimate of the Hopf locus (§3.7).

Approximation
-------------
In the condensed subspace (alpha -> 0, ignore noise):
  J acts as the identity I on patterns.
  K acts as the cyclic forward shift S with eigenvalues nu_k = exp(2*pi*i*k/p).

Under a homogeneous gain D ~ d*I (d = beta*(1 - m1^2)):
  The Hopf condition Delta(i*omega) w = 0 for mode k reduces to:

    Real: 1 = (1-lam)*d + lam*d * cos(2*pi*k/p - omega*tau)
    Imag: t0*omega = lam*d * sin(2*pi*k/p - omega*tau)

Solve these two equations for (lam, omega) at fixed (tau, d, k, p).

This estimate is labeled "condensed-subspace (alpha=0)" in all figures.

Usage
-----
For each tau, compute d = beta*(1 - m1_est^2) using m1 from the fixed-point
branch at the estimated lambda_c (iterated once), then solve for (lam_c, omega_c).
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import brentq
import config as cfg_mod


def _hopf_residual(phi: float, lam: float, d: float, t0: float,
                   two_pi_k_over_p: float, tau: float) -> tuple[float, float]:
    """
    Given phi = 2*pi*k/p - omega*tau, return (residual_re, residual_im)
    from the Hopf conditions at (lam, omega).
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    # Real: 1 - (1-lam)*d - lam*d*cos_phi = 0
    r_re = 1.0 - (1.0 - lam) * d - lam * d * cos_phi
    # omega = (two_pi_k_over_p - phi) / tau
    omega = (two_pi_k_over_p - phi) / tau
    # Imag: t0*omega - lam*d*sin_phi = 0
    r_im  = t0 * omega - lam * d * sin_phi
    return r_re, r_im


def solve_hopf_condensed(tau: float, d: float, p: int, k: int,
                         t0: float = 1.0) -> tuple[float | None, float | None]:
    """
    Solve the homogeneous-gain Hopf conditions for mode k of the cyclic shift.

    Parameters
    ----------
    tau  : delay
    d    : homogeneous gain estimate beta*(1-m1^2)
    p    : number of patterns
    k    : mode index (k=1 is the first non-trivial mode)
    t0   : time constant

    Returns
    -------
    (lam_c, omega_c) or (None, None) if no solution found.
    """
    two_pi_k_over_p = 2.0 * np.pi * k / p

    def system(phi_omega):
        """Jointly zero phi and omega via substitution."""
        phi   = phi_omega
        omega = (two_pi_k_over_p - phi) / tau
        if omega <= 0:
            return 1e10
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        # From imaginary: lam = t0*omega / (d*sin_phi) (need sin_phi > 0)
        if abs(sin_phi) < 1e-10:
            return 1e10
        lam = t0 * omega / (d * sin_phi)
        if lam <= 0 or lam > 1:
            return 1e10
        # Substitute into real condition
        return 1.0 - (1.0 - lam) * d - lam * d * cos_phi

    # Scan phi in (0, 2*pi*k/p) to find sign changes
    phi_vals = np.linspace(1e-4, two_pi_k_over_p - 1e-4, 200)
    f_vals   = np.array([system(ph) for ph in phi_vals])

    lam_c = omega_c = None
    for i in range(len(phi_vals) - 1):
        if np.isfinite(f_vals[i]) and np.isfinite(f_vals[i + 1]):
            if f_vals[i] * f_vals[i + 1] < 0:
                try:
                    phi_sol = brentq(system, phi_vals[i], phi_vals[i + 1],
                                     xtol=1e-10, rtol=1e-10)
                    omega_sol = (two_pi_k_over_p - phi_sol) / tau
                    sin_phi   = np.sin(phi_sol)
                    if abs(sin_phi) > 1e-10 and omega_sol > 0:
                        lam_sol = t0 * omega_sol / (d * sin_phi)
                        if 0 < lam_sol <= 1:
                            lam_c   = lam_sol
                            omega_c = omega_sol
                            break
                except Exception:
                    continue

    return lam_c, omega_c


def locus_analytic(tau_values: np.ndarray, m1_at_lambda_c: float,
                   cfg: dict, k: int = 1) -> dict:
    """
    Compute the analytic Hopf locus for each tau in tau_values.

    Parameters
    ----------
    tau_values      : array of tau values
    m1_at_lambda_c  : estimated m1 at the Hopf bifurcation (from fixed-point branch)
                      Used to compute the homogeneous gain d = beta*(1 - m1^2).
    cfg             : config dict
    k               : Fourier mode of the cyclic shift (k=1 is the most unstable)

    Returns
    -------
    dict: tau_vals, lam_c_analytic, omega_c_analytic, T_c_analytic
    """
    beta = cfg["beta"]
    t0   = cfg["t0"]
    P    = cfg["P"]

    d    = beta * (1.0 - m1_at_lambda_c ** 2)

    lam_list   = []
    omega_list = []
    T_list     = []

    for tau in tau_values:
        lc, wc = solve_hopf_condensed(float(tau), d, P, k, t0)
        lam_list.append(lc)
        omega_list.append(wc)
        if wc is not None and wc > 0:
            T_list.append(2.0 * np.pi / wc)
        else:
            T_list.append(None)

    to_arr = lambda lst: np.array([x if x is not None else np.nan for x in lst])
    return dict(
        tau_vals        = np.array(tau_values, dtype=float),
        lam_c_analytic  = to_arr(lam_list),
        omega_c_analytic= to_arr(omega_list),
        T_c_analytic    = to_arr(T_list),
        d               = float(d),
        k               = k,
        P               = P,
    )


def save_analytic(result: dict, cfg: dict):
    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "analytic", f"analytic_{tag}.npz")
    np.savez(path, **{kk: vv for kk, vv in result.items()
                      if isinstance(vv, (np.ndarray, float, int))})
    print(f"[analytic_check] Saved → {path}")
    return path
