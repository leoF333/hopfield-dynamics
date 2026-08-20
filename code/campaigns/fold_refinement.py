"""High-precision reduced augmented solve for a stationary saddle-node fold."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


from dataclasses import dataclass

import numpy as np
from scipy import optimize

from v5_paths import activate_reference_modules

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402


@dataclass
class FoldResult:
    converged: bool
    lambda_fold: float
    fixed_residual: float
    null_residual: float
    augmented_residual: float
    nfev: int
    message: str
    coefficients: np.ndarray
    null_vector: np.ndarray


def static_jacobian(system: ReducedDDE, a: np.ndarray, lam: float) -> np.ndarray:
    """Jacobian of the stationary reduced field at ``(a, lam)``."""
    shift = np.roll(np.eye(system.P), 1, axis=0)
    return (
        -np.eye(system.P)
        + ((1.0 - lam) * np.eye(system.P) + lam * shift) @ system.Dm(a)
    )


def refine_fold_from_coefficients(
    xi: np.ndarray,
    coefficients: np.ndarray,
    lambda_guess: float,
    *,
    beta: float = 20.0,
    tau: float = 10.0,
    t0: float = 1.0,
    xtol: float = 1e-10,
    maxfev: int = 5000,
) -> FoldResult:
    """Solve ``F=0, Jv=0, ||v||=1`` near a stationary branch endpoint."""
    a0 = np.asarray(coefficients, dtype=float)
    system = ReducedDDE(xi, beta, float(lambda_guess), tau, t0)
    eigenvalues, eigenvectors = np.linalg.eig(
        static_jacobian(system, a0, float(lambda_guess))
    )
    leading = int(np.argmax(eigenvalues.real))
    v0 = eigenvectors[:, leading].real
    v0 /= max(float(np.linalg.norm(v0)), np.finfo(float).tiny)

    def residual(y: np.ndarray) -> np.ndarray:
        a = y[: system.P]
        v = y[system.P : 2 * system.P]
        lam = float(y[-1])
        system.lam = lam
        fixed = system.rhs(a, a)
        null = static_jacobian(system, a, lam) @ v
        normalization = 0.5 * (v @ v - 1.0)
        return np.r_[fixed, null, normalization]

    initial = np.r_[a0, v0, float(lambda_guess)]
    solution = optimize.root(
        residual,
        initial,
        method="hybr",
        options={"xtol": xtol, "maxfev": maxfev},
    )
    y = np.asarray(solution.x, dtype=float)
    a = y[: system.P]
    v = y[system.P : 2 * system.P]
    lam = float(y[-1])
    system.lam = lam
    fixed = system.rhs(a, a)
    null = static_jacobian(system, a, lam) @ v
    augmented = residual(y)
    return FoldResult(
        converged=bool(solution.success),
        lambda_fold=lam,
        fixed_residual=float(np.linalg.norm(fixed) / np.sqrt(system.P)),
        null_residual=float(np.linalg.norm(null) / np.sqrt(system.P)),
        augmented_residual=float(np.linalg.norm(augmented)),
        nfev=int(solution.nfev),
        message=str(solution.message),
        coefficients=a,
        null_vector=v,
    )


def refine_fold_from_full_state(
    xi: np.ndarray,
    state: np.ndarray,
    lambda_guess: float,
    **kwargs,
) -> FoldResult:
    coefficients = np.linalg.lstsq(
        np.asarray(xi, dtype=float).T,
        np.asarray(state, dtype=float),
        rcond=1e-12,
    )[0]
    return refine_fold_from_coefficients(
        xi, coefficients, lambda_guess, **kwargs
    )
