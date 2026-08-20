"""Low-rank pseudo-arclength continuation for one memory-connected branch."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import numpy as np

from v5_paths import activate_reference_modules

activate_reference_modules()
from robust_branch import woodbury_solve  # noqa: E402


def tangent(coup, u: np.ndarray, lam: float, beta: float,
            previous: np.ndarray) -> np.ndarray:
    gain = coup.compute_gain(u, beta)
    derivative = coup.dF_dlam(u, beta)
    du = -woodbury_solve(coup, gain, lam, derivative)
    vector = np.r_[du, 1.0]
    vector /= np.linalg.norm(vector)
    if vector @ previous < 0:
        vector *= -1.0
    return vector


def corrector(coup, predictor_u: np.ndarray, predictor_lam: float,
              previous_u: np.ndarray, previous_lam: float,
              direction: np.ndarray, ds: float, beta: float,
              tolerance: float = 1e-11, max_iterations: int = 40):
    u = predictor_u.copy()
    lam = float(predictor_lam)
    tu, tl = direction[:-1], direction[-1]
    N = coup.N
    for _ in range(max_iterations):
        F = coup.field_F(u, lam, beta)
        constraint = tu @ (u - previous_u) + tl * (lam - previous_lam) - ds
        residual = np.linalg.norm(F) / np.sqrt(N)
        if residual + abs(constraint) < tolerance:
            return u, lam, True, residual
        gain = coup.compute_gain(u, beta)
        dlam_F = coup.dF_dlam(u, beta)
        a = woodbury_solve(coup, gain, lam, -F)
        b = woodbury_solve(coup, gain, lam, dlam_F)
        denominator = tl - tu @ b
        if abs(denominator) < 1e-14:
            return u, lam, False, residual
        delta_lam = (-constraint - tu @ a) / denominator
        delta_u = a - delta_lam * b
        step = 1.0
        accepted = False
        for _ in range(25):
            un = u + step * delta_u
            ln = lam + step * delta_lam
            new_residual = np.linalg.norm(coup.field_F(un, ln, beta)) / np.sqrt(N)
            if new_residual <= residual + 1e-9:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return u, lam, False, residual
        u, lam = un, float(ln)
    residual = np.linalg.norm(coup.field_F(u, lam, beta)) / np.sqrt(N)
    return u, lam, False, residual


def trace_through_fold(coup, u0: np.ndarray, lam0: float, beta: float,
                       identity, ds0: float = 0.005, ds_min: float = 1e-7,
                       ds_max: float = 0.01, max_arclength: float = 3.0) -> dict:
    previous_direction = np.zeros(coup.N + 1)
    previous_direction[-1] = 1.0
    u, lam = u0.copy(), float(lam0)
    ds = ds0
    arclength = 0.0
    rows = [(arclength, lam, u.copy(), 1.0)]
    max_lam = lam
    turned = False
    while arclength < max_arclength:
        direction = tangent(coup, u, lam, beta, previous_direction)
        predictor_u = u + ds * direction[:-1]
        predictor_lam = lam + ds * direction[-1]
        un, ln, ok, residual = corrector(
            coup, predictor_u, predictor_lam, u, lam,
            direction, ds, beta,
        )
        if ok and residual < 1e-10 and identity(un):
            arclength += ds
            u, lam = un, ln
            previous_direction = direction
            rows.append((arclength, lam, u.copy(), direction[-1]))
            max_lam = max(max_lam, lam)
            if direction[-1] < 0 and lam < max_lam - 1e-5:
                turned = True
            ds = min(ds * 1.2, ds_max)
            if turned and lam < max_lam - 0.01:
                break
        else:
            ds *= 0.5
            if ds < ds_min:
                break
    return {
        "s": np.array([row[0] for row in rows]),
        "lambda": np.array([row[1] for row in rows]),
        "state": np.stack([row[2] for row in rows]),
        "tangent_lambda": np.array([row[3] for row in rows]),
        "turned": turned,
        "fold_lambda": float(max(row[1] for row in rows)),
    }

