"""
Sweep over tau values and trace the Hopf locus lambda_c(tau), omega_c(tau).

For each tau:
  1. Reuse the tau-independent fixed-point branch.
  2. Interpolate u*(lambda) at a coarse grid of lambda values.
  3. Find lambda_c via bisection on max Re(z(lambda)).
  4. Record omega_c = Im(z) at lambda_c.

Outputs NPZ:
  tau_vals   : (n_tau,)
  lam_c      : (n_tau,)  lambda_c(tau)
  omega_c    : (n_tau,)  imaginary part of leading mode at crossing
  T_c        : (n_tau,)  2*pi/omega_c (onset period)
  nature     : (n_tau,)  'Hopf' or 'saddle-node / SNIC' or 'unknown'
  outlier    : (n_tau,)  bool, True if discrete outlier (vs bulk edge)
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
from tqdm import tqdm

import config as cfg_mod
from couplings import Couplings
from dde_stability import (IGOperator, rightmost_eigenvalues,
                            _interp_u, _detect_crossing)


def _max_re_at(lam: float, u_star: np.ndarray, coup: Couplings,
               cfg: dict) -> float:
    """max Re(z) of the DDE spectrum at (u*(lam), lam, tau=cfg['tau'])."""
    ev = rightmost_eigenvalues(u_star, lam, coup, cfg, M=cfg["M_cheb"])
    return float(ev.real.max())


def find_lambda_c(branch: dict, coup: Couplings, cfg: dict,
                  n_coarse: int = 20, bisect_tol: float = 1e-4) -> dict:
    """
    Find lambda_c(tau) for the current cfg['tau'] by bisection on max Re(z).

    Returns dict with lam_c, omega_c, T_c, nature, outlier.
    """
    lam_branch = branch["lam"]
    u_arr      = branch["u_star"]

    # Coarse scan
    lam_scan   = np.linspace(lam_branch.min(), lam_branch.max(), n_coarse)
    max_re_arr = np.empty(n_coarse)

    for i, lv in enumerate(lam_scan):
        u = _interp_u(lv, lam_branch, u_arr)
        max_re_arr[i] = _max_re_at(lv, u, coup, cfg)

    # Find sign-change bracket
    lam_lo = lam_hi = None
    for i in range(1, n_coarse):
        if max_re_arr[i - 1] < 0 <= max_re_arr[i]:
            lam_lo, lam_hi = lam_scan[i - 1], lam_scan[i]
            break

    if lam_lo is None:
        if max_re_arr.max() < 0:
            return dict(lam_c=None, omega_c=None, T_c=None,
                        nature="stable (no crossing)", outlier=None)
        if max_re_arr.min() >= 0:
            return dict(lam_c=None, omega_c=None, T_c=None,
                        nature="unstable throughout", outlier=None)
        return dict(lam_c=None, omega_c=None, T_c=None,
                    nature="no clear crossing", outlier=None)

    # Bisection
    for _ in range(20):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        u_mid   = _interp_u(lam_mid, lam_branch, u_arr)
        re_mid  = _max_re_at(lam_mid, u_mid, coup, cfg)
        if re_mid < 0:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        if lam_hi - lam_lo < bisect_tol:
            break

    lam_c = 0.5 * (lam_lo + lam_hi)
    u_c   = _interp_u(lam_c, lam_branch, u_arr)
    ev    = rightmost_eigenvalues(u_c, lam_c, coup, cfg, M=cfg["M_cheb"])

    omega_c  = abs(ev[0].imag)
    T_c      = float(2 * np.pi / omega_c) if omega_c > 0.5 else None
    nature   = "Hopf" if omega_c > 0.5 else "saddle-node / SNIC"

    outlier  = None
    if len(ev) >= 4:
        gap     = ev[:2].real.mean() - ev[2].real
        outlier = bool(gap > 0.1)

    return dict(lam_c=float(lam_c), omega_c=float(omega_c),
                T_c=T_c, nature=nature, outlier=outlier)


def compute_locus(branch: dict, coup: Couplings, cfg: dict,
                  tau_values: list | None = None) -> dict:
    """
    Sweep over tau_values and compute the Hopf locus lambda_c(tau).

    Parameters
    ----------
    branch    : fixed-point branch (tau-independent, computed once)
    coup      : Couplings
    cfg       : base config dict (tau will be overridden per sweep point)
    tau_values: list of integer tau values

    Returns
    -------
    locus : dict with arrays tau_vals, lam_c, omega_c, T_c, nature, outlier
    """
    if tau_values is None:
        tau_values = cfg.get("tau_list", list(range(1, 21)))

    lam_c_list   = []
    omega_c_list = []
    T_c_list     = []
    nature_list  = []
    outlier_list = []

    for tau_val in tqdm(tau_values, desc="Hopf locus tau sweep"):
        # Override tau in a copy of cfg
        cfg_tau        = dict(cfg)
        cfg_tau["tau"] = int(tau_val)
        n   = max(1, round(tau_val / cfg["dt"]))
        cfg_tau["dt"]    = tau_val / n
        cfg_tau["L_buf"] = n

        res = find_lambda_c(branch, coup, cfg_tau)
        lam_c_list.append(res["lam_c"])
        omega_c_list.append(res["omega_c"])
        T_c_list.append(res["T_c"])
        nature_list.append(res["nature"])
        outlier_list.append(res["outlier"])

        lc  = res["lam_c"]
        wc  = res["omega_c"]
        nat = res["nature"]
        print(f"  tau={tau_val:3d}  lam_c={lc!s:>8}  omega_c={wc!s:>8}  {nat}")

    locus = dict(
        tau_vals  = np.array(tau_values, dtype=float),
        lam_c     = np.array([x if x is not None else np.nan for x in lam_c_list]),
        omega_c   = np.array([x if x is not None else np.nan for x in omega_c_list]),
        T_c       = np.array([x if x is not None else np.nan for x in T_c_list]),
        nature    = np.array(nature_list),
        outlier   = np.array([x if x is not None else False for x in outlier_list]),
    )
    return locus


def save_locus(locus: dict, cfg: dict):
    """Save locus to NPZ."""
    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "hopf_locus", f"locus_{tag}.npz")
    np.savez(path, **{k: v for k, v in locus.items()
                      if isinstance(v, np.ndarray)})
    print(f"[hopf_locus] Locus saved → {path}")
    return path
