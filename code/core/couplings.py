"""
Pattern generation and matrix-free coupling operators.

Array conventions
-----------------
xi       : (P, N) float64, entries ±1  [patterns, 0-indexed: mu=0,...,P-1]
xi_shift : (P, N) float64, xi_shift[mu] = xi[(mu+1)%P]  [forward-shifted]

MLX float32 is used for all O(alpha N^2) matvecs (GPU-accelerated on Apple M-series).
All public methods accept and return NumPy float64.

Operator definitions (matching the spec exactly)
-------------------------------------------------
J x = (1/N) xi^T (xi x)         [symmetric Hebbian]
K x = (1/N) xi_shift^T (xi x)   [asymmetric sequence coupling]
C(lam) x = (1-lam) J x + lam K x

M(lam, gain) v = -v + C(lam)(gain * v)   [linearization at fixed point u*]
  where gain = beta * (1 - tanh^2(beta u*))  [diagonal of D]

Fixed-point residual: F(u, lam) = -u + C(lam) tanh(beta u)
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
import mlx.core as mx


def make_patterns(N: int, P: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate P i.i.d. ±1 patterns of length N.
    Returns (xi, xi_shift), both shape (P, N) float64.
    xi_shift[mu] = xi[(mu+1) % P]  (closed-cycle forward shift).
    """
    rng = np.random.default_rng(seed)
    xi = rng.choice(np.array([-1.0, 1.0]), size=(P, N)).astype(np.float64)
    xi_shift = np.roll(xi, shift=-1, axis=0)
    return xi, xi_shift


class Couplings:
    """
    Matrix-free implementations of J, K, C(lam), M(lam, gain).

    Internally stores xi and xi_shift as MLX float32 for GPU matvecs.
    All public methods accept/return NumPy float64.
    """

    # For N <= _NP_THRESH: NumPy is faster (MLX dispatch ~1ms dominates tiny GPU work)
    _NP_THRESH = 1000

    def __init__(self, xi: np.ndarray, xi_shift: np.ndarray):
        assert xi.shape == xi_shift.shape
        self.N  = xi.shape[1]
        self.P  = xi.shape[0]
        # NumPy copies (float64, used for small-N fast path and overlaps)
        self._xi_np  = xi.astype(np.float64)
        self._xis_np = xi_shift.astype(np.float64)
        # MLX copies for GPU matvecs (large N)
        self._xi_mx  = mx.array(xi.astype(np.float32))
        self._xis_mx = mx.array(xi_shift.astype(np.float32))
        self._use_np = (self.N <= self._NP_THRESH)

    # ------------------------------------------------------------------ core
    def _matvec_np(self, xi_left_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        """Pure NumPy: xi_left^T (xi x / N).  No MLX overhead."""
        m = self._xi_np @ x_np / self.N     # (P,)
        return xi_left_np.T @ m             # (N,)

    def _matvec(self, xi_left_mx, x_np: np.ndarray) -> np.ndarray:
        """
        Compute xi_left^T (xi x / N) matrix-free.
        xi_left has shape (P, N); x has shape (N,).
        Returns shape (N,) float64.
        """
        x_mx = mx.array(x_np.astype(np.float32))
        m    = self._xi_mx @ x_mx / self.N   # (P,): overlaps on xi basis
        out  = xi_left_mx.T @ m              # (N,)
        mx.eval(out)
        return np.array(out, dtype=np.float64)

    def apply_J(self, x: np.ndarray) -> np.ndarray:
        """J x = xi^T (xi x / N).  Shape (N,) -> (N,)."""
        if self._use_np:
            return self._matvec_np(self._xi_np, x)
        return self._matvec(self._xi_mx, x)

    def apply_K(self, x: np.ndarray) -> np.ndarray:
        """K x = xi_shift^T (xi x / N).  Shape (N,) -> (N,)."""
        if self._use_np:
            return self._matvec_np(self._xis_np, x)
        return self._matvec(self._xis_mx, x)

    def apply_C(self, x: np.ndarray, lam: float) -> np.ndarray:
        """C(lam) x = (1-lam) J x + lam K x.  Single pass over xi."""
        if self._use_np:
            m = self._xi_np @ x / self.N
            return (1.0 - lam) * (self._xi_np.T @ m) + lam * (self._xis_np.T @ m)
        x_mx = mx.array(x.astype(np.float32))
        m    = self._xi_mx @ x_mx / self.N
        out  = (1.0 - lam) * (self._xi_mx.T @ m) + lam * (self._xis_mx.T @ m)
        mx.eval(out)
        return np.array(out, dtype=np.float64)

    def apply_M(self, v: np.ndarray, gain: np.ndarray, lam: float) -> np.ndarray:
        """M(lam) v = -v + C(lam)(gain ⊙ v).
        gain : shape (N,), diagonal of D = beta*(1 - tanh^2(beta u*))."""
        return -v + self.apply_C(gain * v, lam)

    # ---------------------------------------------------------------- F, dF
    def field_F(self, u: np.ndarray, lam: float, beta: float) -> np.ndarray:
        """
        Fixed-point residual F(u, lam) = -u + C(lam) tanh(beta u).
        tau-independent.
        """
        return -u + self.apply_C(np.tanh(beta * u), lam)

    def dF_dlam(self, u: np.ndarray, beta: float) -> np.ndarray:
        """
        dF/dlam = (K - J) tanh(beta u).
        Used in continuation tangent and bordered corrector.
        """
        tu = np.tanh(beta * u)
        return self.apply_K(tu) - self.apply_J(tu)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def compute_gain(u: np.ndarray, beta: float) -> np.ndarray:
        """
        Diagonal of D = beta * diag(1 - tanh^2(beta u)).
        Shape (N,) float64.
        """
        return beta * (1.0 - np.tanh(beta * u) ** 2)

    def overlap_raw(self, x: np.ndarray) -> np.ndarray:
        """
        m_mu = xi^mu . x / N,  shape (P,).
        x can be u* or tanh(beta u*) — caller decides.
        """
        return self._xi_np @ x / self.N

    def condensed_overlaps(self, u_star: np.ndarray, beta: float,
                           n: int = 5) -> np.ndarray:
        """
        Physical overlaps m_nu = (1/N) xi^nu . tanh(beta u*),  nu=1,...,n.
        Returns shape (n,) float64.  (0-indexed internally: nu-1)
        """
        tu = np.tanh(beta * u_star)
        return self.overlap_raw(tu)[:n]

    # ---- dense reference (for small-N correctness checks only) ----------
    def dense_J(self) -> np.ndarray:
        """Return the N×N matrix J = xi^T xi / N.  Only call for small N."""
        return (self._xi_np.T @ self._xi_np) / self.N

    def dense_K(self) -> np.ndarray:
        """Return the N×N matrix K = xi_shift^T xi / N.  Only call for small N."""
        return (self._xis_np.T @ self._xi_np) / self.N

    def dense_M(self, gain: np.ndarray, lam: float) -> np.ndarray:
        """
        Return the N×N Jacobian M(lam) = -I + C(lam) D  (diag D = gain).
        Only call for small N.  Used for correctness checks.
        """
        C = (1.0 - lam) * self.dense_J() + lam * self.dense_K()
        return -np.eye(self.N) + C * gain[np.newaxis, :]  # C @ diag(gain)

    # ---- DDE field (used in simulator) -----------------------------------
    def field_dde(self, u_now: np.ndarray, u_delay: np.ndarray,
                  lam: float, beta: float) -> np.ndarray:
        """
        Full DDE right-hand side / t0:
        rhs = -u_now + (1-lam) J tanh(beta u_now) + lam K tanh(beta u_delay)
        Returns shape (N,) float64 (divide by t0 in the integrator).
        """
        tu_now   = np.tanh(beta * u_now)
        tu_delay = np.tanh(beta * u_delay)
        Jt = self.apply_J(tu_now)
        Kt = self.apply_K(tu_delay)
        return -u_now + (1.0 - lam) * Jt + lam * Kt
