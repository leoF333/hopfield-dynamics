"""Backend-selectable pattern operators compatible with the historical API.

NumPy float64 is the reference path.  MLX is imported only when
``V5_BACKEND=mlx`` so that a machine without an exposed Metal device can still run
the complete validation suite.
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


def make_patterns(N: int, P: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xi = rng.choice(np.array([-1.0, 1.0]), size=(P, N)).astype(np.float64)
    return xi, np.roll(xi, shift=-1, axis=0)


class Couplings:
    _NP_THRESH = 1000

    def __init__(self, xi: np.ndarray, xi_shift: np.ndarray):
        if xi.shape != xi_shift.shape:
            raise ValueError("xi and xi_shift must have identical shapes")
        self.P, self.N = xi.shape
        self._xi_np = np.asarray(xi, dtype=np.float64)
        self._xis_np = np.asarray(xi_shift, dtype=np.float64)
        self.backend = os.environ.get("V5_BACKEND", "cpu").lower()
        if self.backend not in {"cpu", "mlx"}:
            raise ValueError(f"unknown V5_BACKEND={self.backend!r}")
        self._use_np = self.backend == "cpu"
        self._mx = None
        if self.backend == "mlx":
            import mlx.core as mx  # imported lazily; may require a visible Metal GPU
            self._mx = mx
            self._xi_mx = mx.array(self._xi_np.astype(np.float32))
            self._xis_mx = mx.array(self._xis_np.astype(np.float32))

    def _matvec_np(self, left: np.ndarray, x: np.ndarray) -> np.ndarray:
        return left.T @ (self._xi_np @ np.asarray(x, dtype=np.float64) / self.N)

    def _matvec_mlx(self, left, x: np.ndarray) -> np.ndarray:
        mx = self._mx
        xx = mx.array(np.asarray(x, dtype=np.float32))
        out = left.T @ (self._xi_mx @ xx / self.N)
        mx.eval(out)
        return np.array(out, dtype=np.float64)

    def apply_J(self, x: np.ndarray) -> np.ndarray:
        if self._use_np:
            return self._matvec_np(self._xi_np, x)
        return self._matvec_mlx(self._xi_mx, x)

    def apply_K(self, x: np.ndarray) -> np.ndarray:
        if self._use_np:
            return self._matvec_np(self._xis_np, x)
        return self._matvec_mlx(self._xis_mx, x)

    def apply_C(self, x: np.ndarray, lam: float) -> np.ndarray:
        if self._use_np:
            m = self._xi_np @ np.asarray(x, dtype=np.float64) / self.N
            return (1.0 - lam) * (self._xi_np.T @ m) + lam * (self._xis_np.T @ m)
        mx = self._mx
        xx = mx.array(np.asarray(x, dtype=np.float32))
        m = self._xi_mx @ xx / self.N
        out = (1.0 - lam) * (self._xi_mx.T @ m) + lam * (self._xis_mx.T @ m)
        mx.eval(out)
        return np.array(out, dtype=np.float64)

    def apply_M(self, v: np.ndarray, gain: np.ndarray, lam: float) -> np.ndarray:
        return -v + self.apply_C(gain * v, lam)

    def field_F(self, u: np.ndarray, lam: float, beta: float) -> np.ndarray:
        return -u + self.apply_C(np.tanh(beta * u), lam)

    def dF_dlam(self, u: np.ndarray, beta: float) -> np.ndarray:
        tu = np.tanh(beta * u)
        return self.apply_K(tu) - self.apply_J(tu)

    @staticmethod
    def compute_gain(u: np.ndarray, beta: float) -> np.ndarray:
        return beta * (1.0 - np.tanh(beta * u) ** 2)

    def overlap_raw(self, x: np.ndarray) -> np.ndarray:
        return self._xi_np @ np.asarray(x, dtype=np.float64) / self.N

    def condensed_overlaps(self, u_star: np.ndarray, beta: float,
                           n: int = 5) -> np.ndarray:
        return self.overlap_raw(np.tanh(beta * u_star))[:n]

    def dense_J(self) -> np.ndarray:
        return self._xi_np.T @ self._xi_np / self.N

    def dense_K(self) -> np.ndarray:
        return self._xis_np.T @ self._xi_np / self.N

    def dense_M(self, gain: np.ndarray, lam: float) -> np.ndarray:
        C = (1.0 - lam) * self.dense_J() + lam * self.dense_K()
        return -np.eye(self.N) + C * gain[np.newaxis, :]

    def field_dde(self, u_now: np.ndarray, u_delay: np.ndarray,
                  lam: float, beta: float) -> np.ndarray:
        return (
            -u_now
            + (1.0 - lam) * self.apply_J(np.tanh(beta * u_now))
            + lam * self.apply_K(np.tanh(beta * u_delay))
        )

