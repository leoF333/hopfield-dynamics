"""
Configuration and parameter management for the delayed mixed Hopfield analysis.

All parameters are overridable from the CLI or by importing and modifying DEFAULTS.
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
import argparse
import numpy as np

DEFAULTS: dict = dict(
    # Network
    N       = 10_000,
    alpha   = 0.05,
    seed    = 42,
    # Physics
    beta    = 20.0,
    t0      = 1.0,
    tau     = 10,
    # Continuation / Newton
    lam_min = 0.0,
    lam_max = 0.95,
    ds      = 0.02,          # pseudo-arclength step
    newton_tol  = 1e-8,      # ||F|| / sqrt(N) < newton_tol
    gmres_tol   = 1e-6,
    gmres_restart = 30,
    n_overlaps  = 5,         # how many m_nu to record along branch
    # DDE stability (pseudospectral IG)
    M_cheb  = 32,            # CGL nodes minus 1
    n_eigs  = 20,            # Arnoldi eigenvalues
    arnoldi_ncv = 60,        # Krylov subspace size
    arnoldi_maxiter = 500,
    M_values = None,         # M-convergence sweep; default set in dde_stability.py
    # DDE integration
    dt      = 0.1,           # step size; must divide tau
    t_transient = 300.0,
    t_measure   = 200.0,
    # Tau sweep
    tau_list = None,         # default set in hopf_locus.py
    # Misc
    small_N = 2000,          # for fast validation
    out_dir = None,
    dpi     = 150,
)


def make_parser(description: str = "Delayed mixed Hopfield") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--N",     type=int,   default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--tau",   type=int,   default=None)
    p.add_argument("--beta",  type=float, default=None)
    p.add_argument("--t0",    type=float, default=None)
    p.add_argument("--seed",  type=int,   default=None)
    p.add_argument("--M",     type=int,   default=None, dest="M_cheb")
    p.add_argument("--lam-max", type=float, default=None, dest="lam_max")
    p.add_argument("--ds",    type=float, default=None)
    p.add_argument("--small", action="store_true",
                   help=f"Use N={DEFAULTS['small_N']} for fast validation")
    p.add_argument("--out-dir", type=str, default=None, dest="out_dir")
    return p


def resolve(args=None) -> dict:
    """
    Build a concrete config dict from argparse Namespace (or None for defaults).
    Derives P, dt, L_buf and out_dir. Creates output directories.
    """
    cfg = dict(DEFAULTS)

    if args is not None:
        for k, v in vars(args).items():
            if v is not None and k in cfg:
                cfg[k] = v
        if getattr(args, "small", False):
            cfg["N"] = DEFAULTS["small_N"]

    cfg["P"] = round(cfg["alpha"] * cfg["N"])

    # Ensure dt divides tau
    tau = cfg["tau"]
    dt  = cfg["dt"]
    n   = max(1, round(tau / dt))
    cfg["dt"]    = tau / n
    cfg["L_buf"] = n           # delay in samples: tau / dt

    # Defaults for lists
    if cfg["M_values"] is None:
        cfg["M_values"] = [16, 24, 32, 48]
    if cfg["tau_list"] is None:
        cfg["tau_list"] = list(range(1, 21))

    # Output directory
    if cfg["out_dir"] is None:
        src_dir  = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(src_dir)
        cfg["out_dir"] = os.path.join(root_dir, "results")

    for sub in ("fixed_point", "stability", "hopf_locus",
                "simulation", "floquet", "analytic", "validation"):
        os.makedirs(os.path.join(cfg["out_dir"], sub), exist_ok=True)

    return cfg


def out_path(cfg: dict, subdir: str, filename: str) -> str:
    """Return absolute path for an output file; create parent directory."""
    d = os.path.join(cfg["out_dir"], subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def param_tag(cfg: dict) -> str:
    """Short string encoding key parameters for filenames."""
    return f"N{cfg['N']}_a{cfg['alpha']:.3f}_tau{cfg['tau']}_b{cfg['beta']:.1f}_s{cfg['seed']}"
