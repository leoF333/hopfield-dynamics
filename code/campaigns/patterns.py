"""Pattern ensembles used by N1, N7, N8 and N9."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import numpy as np

from couplings import make_patterns


def iid(N: int, P: int, seed: int) -> np.ndarray:
    return make_patterns(N, P, seed)[0]


def markov_flip(N: int, P: int, c: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_flip = (1.0 - c) / 2.0
    xi = np.empty((P, N), dtype=np.float64)
    xi[0] = np.where(rng.random(N) < 0.5, -1.0, 1.0)
    for mu in range(1, P):
        flip = np.where(rng.random(N) < p_flip, -1.0, 1.0)
        xi[mu] = xi[mu - 1] * flip
    return xi


def gp_circle(N: int, P: int, kappa: float, seed: int) -> np.ndarray:
    K = max(1, round(kappa * P))
    rng = np.random.default_rng(seed)
    theta = 2.0 * np.pi * np.arange(P) / P
    modes = np.arange(1, K + 1)
    A = rng.standard_normal((N, K))
    B = rng.standard_normal((N, K))
    z = A @ np.cos(np.outer(modes, theta)) + B @ np.sin(np.outer(modes, theta))
    xi = np.sign(z.T)
    xi[xi == 0] = 1.0
    return xi.astype(np.float64)


def adjacent_correlations(xi: np.ndarray) -> np.ndarray:
    return np.mean(xi * np.roll(xi, -1, axis=0), axis=1)


def gp_arcsine_prediction(P: int, kappa: float) -> float:
    K = max(1, round(kappa * P))
    rho = np.mean(np.cos(np.arange(1, K + 1) * 2.0 * np.pi / P))
    return float((2.0 / np.pi) * np.arcsin(np.clip(rho, -1.0, 1.0)))

