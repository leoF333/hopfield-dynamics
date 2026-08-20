"""
Pure-NumPy implementation of the E36 layered Hopfield chain.

Primary model (the user's equations)
------------------------------------
For U[k] in R^N, g(x)=tanh(beta*x), and k=1,...,K-1,

    t0 dU[k]/dt = -U[k] + J g(U[k]) + lam g(U[k-1])
    t0 dU[0]/dt = -U[0] + K_seq g(U[K-1]).

Patterns use the repository convention ``xi.shape == (P, N)``. Therefore

    J x     = xi.T @ (xi @ x / N)
    K_seq x = roll(xi, -1, axis=0).T @ (xi @ x / N).

Equivalently, if ``m = xi @ x / N``, the pattern coefficients of K_seq x are
``roll(m, +1)``. This direction is covered by an independent dense test.

The implementation is intentionally CPU/float64 and self-contained: importing it
does not import MLX. All layerwise Hebbian products are batched as matrix-matrix
products. Dense matrices are available only through ``dense_jacobian`` for tiny
validation cases.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np


Array = np.ndarray


def make_patterns_numpy(N: int, P: int, seed: int = 42) -> tuple[Array, Array]:
    """Generate patterns exactly like ``couplings.make_patterns``, without MLX."""
    if N < 1 or P < 1 or P > N:
        raise ValueError(f"Require 1 <= P <= N, got P={P}, N={N}")
    rng = np.random.default_rng(seed)
    xi = rng.choice(np.array([-1.0, 1.0]), size=(P, N)).astype(np.float64)
    return xi, np.roll(xi, shift=-1, axis=0)


class LayeredChain:
    """E36 chain with vectorized full, projected, and Galerkin fields."""

    _CHECKPOINT_VERSION = 1

    def __init__(
        self,
        xi: Array,
        beta: float,
        lam: float,
        K: int,
        t0: float = 1.0,
        closure: str = "primary",
    ):
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim != 2:
            raise ValueError(f"xi must have shape (P,N), got {xi.shape}")
        self.P, self.N = xi.shape
        if not (1 <= self.P <= self.N):
            raise ValueError("Require 1 <= P <= N")
        if K < 2:
            raise ValueError("E36 requires K >= 2")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lam must be finite and non-negative")
        if not np.isfinite(t0) or t0 <= 0:
            raise ValueError("t0 must be finite and positive")
        if closure not in {"primary", "B"}:
            raise ValueError("closure must be 'primary' or 'B'")

        self.xi = np.ascontiguousarray(xi)
        self.xi_shift = np.ascontiguousarray(np.roll(xi, shift=-1, axis=0))
        self.pattern_sha256 = hashlib.sha256(
            self.xi.view(np.uint8)
        ).hexdigest()
        self.beta = float(beta)
        self.lam = float(lam)
        self.K = int(K)
        self.t0 = float(t0)
        self.closure = closure

        self.gram = (self.xi @ self.xi.T) / self.N
        self.gram_condition = float(np.linalg.cond(self.gram))
        if not np.isfinite(self.gram_condition) or self.gram_condition > 1e12:
            raise ValueError(
                "Pattern Gram matrix is singular or too ill-conditioned: "
                f"cond={self.gram_condition:.3e}"
            )
        self._gram_chol = np.linalg.cholesky(self.gram)

    # ------------------------------------------------------------------
    # Shape and low-rank helpers

    def _state(self, U: Array, *, name: str = "U") -> Array:
        U = np.asarray(U, dtype=np.float64)
        if U.shape != (self.K, self.N):
            raise ValueError(
                f"{name} must have shape {(self.K, self.N)}, got {U.shape}"
            )
        return U

    def _coeff_state(self, A: Array) -> Array:
        A = np.asarray(A, dtype=np.float64)
        if A.shape != (self.K, self.P):
            raise ValueError(
                f"A must have shape {(self.K, self.P)}, got {A.shape}"
            )
        return A

    def _solve_gram(self, rhs: Array) -> Array:
        """Solve Gram*c=rhs for any leading batch shape ending in P."""
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape[-1] != self.P:
            raise ValueError("Gram RHS must end in the pattern dimension P")
        flat = rhs.reshape(-1, self.P).T
        y = np.linalg.solve(self._gram_chol, flat)
        out = np.linalg.solve(self._gram_chol.T, y).T
        return out.reshape(rhs.shape)

    def activation(self, U: Array) -> Array:
        return np.tanh(self.beta * np.asarray(U, dtype=np.float64))

    def activation_gain(self, U: Array) -> Array:
        G = self.activation(U)
        return self.beta * (1.0 - G * G)

    def activation_overlaps(self, U: Array) -> Array:
        """Return xi@g(U)/N, batched over all leading dimensions of U."""
        U = np.asarray(U, dtype=np.float64)
        if U.shape[-1] != self.N:
            raise ValueError("U must end in dimension N")
        return (self.activation(U) @ self.xi.T) / self.N

    def raw_overlaps(self, U: Array) -> Array:
        """Return xi@U/N, batched over all leading dimensions of U."""
        U = np.asarray(U, dtype=np.float64)
        if U.shape[-1] != self.N:
            raise ValueError("U must end in dimension N")
        return (U @ self.xi.T) / self.N

    def project(self, V: Array) -> Array:
        """Orthogonally project vectors ending in N onto span(xi)."""
        V = np.asarray(V, dtype=np.float64)
        if V.shape[-1] != self.N:
            raise ValueError("V must end in dimension N")
        m = (V @ self.xi.T) / self.N
        return self._solve_gram(m) @ self.xi

    def projection_coefficients(self, V: Array) -> Array:
        """Coefficients c such that P_X V = c@xi."""
        V = np.asarray(V, dtype=np.float64)
        if V.shape[-1] != self.N:
            raise ValueError("V must end in dimension N")
        return self._solve_gram((V @ self.xi.T) / self.N)

    def leak_fraction(self, U: Array) -> Array:
        """Relative norm ||(I-P_X)g(U)||/||g(U)|| over the final axis."""
        G = self.activation(U)
        residual = G - self.project(G)
        den = np.linalg.norm(G, axis=-1)
        return np.linalg.norm(residual, axis=-1) / np.maximum(den, 1e-300)

    def closure_drive(self, G_last: Array) -> Array:
        """Apply K_seq to one or a batch of activated last-layer vectors."""
        G_last = np.asarray(G_last, dtype=np.float64)
        if G_last.shape[-1] != self.N:
            raise ValueError("G_last must end in dimension N")
        m = (G_last @ self.xi.T) / self.N
        return np.roll(m, shift=1, axis=-1) @ self.xi

    # ------------------------------------------------------------------
    # Vector fields

    def rhs_full(self, U: Array) -> Array:
        """Vectorized RHS of E36-1/E36-2; exactly two batched Hebbian GEMMs."""
        U = self._state(U)
        G = np.tanh(self.beta * U)
        M = (G @ self.xi.T) / self.N
        JG = M @ self.xi

        out = -U.copy()
        out[1:] += JG[1:] + self.lam * G[:-1]
        close = np.roll(M[-1], shift=1) @ self.xi
        if self.closure == "primary":
            out[0] += close
        else:
            out[0] += JG[0] + self.lam * close
        return out / self.t0

    def rhs_full_loop(self, U: Array) -> Array:
        """Independent per-layer reference; validation only, not production."""
        U = self._state(U)
        G = np.tanh(self.beta * U)
        out = np.empty_like(U)
        for k in range(1, self.K):
            m_k = self.xi @ G[k] / self.N
            out[k] = -U[k] + self.xi.T @ m_k + self.lam * G[k - 1]
        m_last = self.xi @ G[-1] / self.N
        close = self.xi_shift.T @ m_last
        if self.closure == "primary":
            out[0] = -U[0] + close
        else:
            m_zero = self.xi @ G[0] / self.N
            out[0] = (
                -U[0] + self.xi.T @ m_zero + self.lam * close
            )
        return out / self.t0

    def rhs_projected_full(self, U: Array) -> Array:
        """Full-N projected twin: inner injection is lam*P_X*g(previous)."""
        U = self._state(U)
        G = np.tanh(self.beta * U)
        M = (G @ self.xi.T) / self.N
        JG = M @ self.xi
        projected_G = self._solve_gram(M) @ self.xi

        out = -U.copy()
        out[1:] += JG[1:] + self.lam * projected_G[:-1]
        close = np.roll(M[-1], shift=1) @ self.xi
        if self.closure == "primary":
            out[0] += close
        else:
            out[0] += JG[0] + self.lam * close
        return out / self.t0

    # Short alias matching the roadmap language.
    rhs_projected = rhs_projected_full

    def rhs_galerkin(self, A: Array) -> Array:
        """Exact coefficient RHS of the projected twin, with U=A@xi."""
        A = self._coeff_state(A)
        G = np.tanh(self.beta * (A @ self.xi))
        M = (G @ self.xi.T) / self.N
        projected_coeff = self._solve_gram(M)

        out = -A.copy()
        out[1:] += M[1:] + self.lam * projected_coeff[:-1]
        if self.closure == "primary":
            out[0] += np.roll(M[-1], shift=1)
        else:
            out[0] += M[0] + self.lam * np.roll(M[-1], shift=1)
        return out / self.t0

    # ------------------------------------------------------------------
    # Matrix-free variational field

    def jvp(self, U: Array, V: Array, *, projected: bool = False) -> Array:
        """
        Apply the Jacobian at U to one or a batch of perturbations.

        U has shape (K,N). V has shape (...,K,N). All q*K Hebbian products are
        performed in one flattened batch.
        """
        U = self._state(U)
        V = np.asarray(V, dtype=np.float64)
        if V.shape[-2:] != (self.K, self.N):
            raise ValueError(
                f"V must end in {(self.K, self.N)}, got {V.shape}"
            )
        gain = self.activation_gain(U)
        lead = V.shape[:-2]
        gain_b = gain.reshape((1,) * len(lead) + gain.shape)
        DV = V * gain_b

        flat = DV.reshape(-1, self.N)
        DM = (flat @ self.xi.T) / self.N
        JDV = (DM @ self.xi).reshape(V.shape)
        DM_layers = DM.reshape(lead + (self.K, self.P))

        out = -V.copy()
        if projected:
            projected_DV = (
                self._solve_gram(DM_layers) @ self.xi
            )
            out[..., 1:, :] += (
                JDV[..., 1:, :] + self.lam * projected_DV[..., :-1, :]
            )
        else:
            out[..., 1:, :] += (
                JDV[..., 1:, :] + self.lam * DV[..., :-1, :]
            )

        close = np.roll(
            DM_layers[..., -1, :], shift=1, axis=-1
        ) @ self.xi
        if self.closure == "primary":
            out[..., 0, :] += close
        else:
            out[..., 0, :] += JDV[..., 0, :] + self.lam * close
        return out / self.t0

    var_rhs = jvp

    def dense_jacobian(
        self, U: Array, *, projected: bool = False, max_dimension: int = 512
    ) -> Array:
        """Materialize the Jacobian by batched JVP, only for small validation."""
        U = self._state(U)
        dim = self.K * self.N
        if dim > max_dimension:
            raise ValueError(
                f"Refusing dense {dim}x{dim} Jacobian; max_dimension={max_dimension}"
            )
        basis = np.eye(dim, dtype=np.float64).reshape(dim, self.K, self.N)
        images = self.jvp(U, basis, projected=projected).reshape(dim, dim)
        return images.T

    # ------------------------------------------------------------------
    # Schur loop diagnostic (valid only away from local singular blocks)

    def _solve_local_resolvent(
        self, D: Array, B: Array, a: complex
    ) -> tuple[Array, float]:
        """Solve ((a)I-JD)Y=B by Woodbury and return core condition number."""
        dtype = np.result_type(B.dtype, np.asarray(a).dtype)
        B = np.asarray(B, dtype=dtype)
        weighted_xi = self.xi * np.asarray(D, dtype=dtype)[None, :]
        core = np.eye(self.P, dtype=dtype) - (
            weighted_xi @ self.xi.T
        ) / (self.N * a)
        cond = float(np.linalg.cond(core))
        VB = (weighted_xi @ (B / a)) / self.N
        correction = np.linalg.solve(core, VB)
        Y = B / a + (self.xi.T @ correction) / a
        return Y, cond

    def schur_diagnostics(self, U: Array, z: complex = 0.0) -> dict:
        """
        Return the P*P loop matrix Omega(z), without forming an N*N matrix.

        This is a diagnostic only when all local Woodbury cores are well
        conditioned. It does not independently certify a fold.
        """
        U = self._state(U)
        gain = self.activation_gain(U)
        a = complex(self.t0 * z + 1.0)
        if abs(a) < 1e-14:
            raise np.linalg.LinAlgError("Layer-0 block is singular")

        # A0^{-1} K_seq D_last = (xi_shift.T/a) @
        #                              ((xi*D_last)/N).
        W = self.xi_shift.T.astype(np.complex128) / a
        conditions = []
        for k in range(1, self.K):
            rhs = self.lam * gain[k - 1, :, None] * W
            W, cond = self._solve_local_resolvent(gain[k], rhs, a)
            conditions.append(cond)
        V_close = (self.xi * gain[-1][None, :]) / self.N
        omega = V_close @ W
        sigma_min = float(
            np.linalg.svd(
                np.eye(self.P, dtype=np.complex128) - omega,
                compute_uv=False,
            )[-1]
        )
        return {
            "omega": omega,
            "sigma_min_I_minus_omega": sigma_min,
            "local_core_condition": np.asarray(conditions),
            "resolvent_valid": bool(
                np.all(np.isfinite(conditions))
                and np.max(conditions, initial=0.0) < 1e10
            ),
        }

    # ------------------------------------------------------------------
    # Lightweight fixed-step integration and atomic checkpointing

    def _rhs_for_mode(self, mode: str) -> Callable[[Array], Array]:
        table = {
            "full": self.rhs_full,
            "projected": self.rhs_projected_full,
            "galerkin": self.rhs_galerkin,
        }
        try:
            return table[mode]
        except KeyError as exc:
            raise ValueError(
                "mode must be 'full', 'projected', or 'galerkin'"
            ) from exc

    def _write_checkpoint(
        self,
        path: str | os.PathLike[str],
        state: Array,
        step: int,
        dt: float,
        mode: str,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp.npz")
        meta = {
            "version": self._CHECKPOINT_VERSION,
            "K": self.K,
            "N": self.N,
            "P": self.P,
            "beta": self.beta,
            "lam": self.lam,
            "t0": self.t0,
            "closure": self.closure,
            "pattern_sha256": self.pattern_sha256,
        }
        np.savez(
            tmp,
            state=np.asarray(state, dtype=np.float64),
            step=np.int64(step),
            dt=np.float64(dt),
            mode=np.asarray(mode),
            meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
        )
        os.replace(tmp, target)

    def _read_checkpoint(
        self, path: str | os.PathLike[str], dt: float, mode: str
    ) -> tuple[Array, int]:
        with np.load(path, allow_pickle=False) as data:
            stored_dt = float(data["dt"])
            stored_mode = str(data["mode"].item())
            meta = json.loads(str(data["meta_json"].item()))
            if abs(stored_dt - dt) > 1e-15 or stored_mode != mode:
                raise ValueError("Checkpoint dt or mode does not match request")
            expected = {
                "K": self.K,
                "N": self.N,
                "P": self.P,
                "beta": self.beta,
                "lam": self.lam,
                "t0": self.t0,
                "closure": self.closure,
                "pattern_sha256": self.pattern_sha256,
            }
            for key, value in expected.items():
                if meta.get(key) != value:
                    raise ValueError(f"Checkpoint mismatch for {key}")
            return np.array(data["state"], dtype=np.float64), int(data["step"])

    def integrate(
        self,
        U0: Array | None,
        t_final: float,
        dt: float,
        *,
        record_every: int = 1,
        mode: str = "full",
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_every_steps: int | None = None,
        resume: bool = False,
        check_finite_every: int = 1,
        record_callback: Callable[[float, Array], None] | None = None,
        heartbeat_callback: Callable[[dict, Array], None] | None = None,
        heartbeat_every_steps: int | None = None,
        store_records: bool = True,
    ) -> dict:
        """
        Integrate with fixed-step classical RK4.

        ``t_final`` is absolute from t=0. On resume, integration continues from
        the saved step to this absolute final time. Returned records begin at the
        restart point; checkpoints contain only the current state, not trajectories.
        """
        if not np.isfinite(t_final) or t_final < 0:
            raise ValueError("t_final must be finite and non-negative")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if record_every < 1:
            raise ValueError("record_every must be >=1")
        if check_finite_every < 0:
            raise ValueError("check_finite_every must be >=0")
        if heartbeat_every_steps is not None and heartbeat_every_steps < 1:
            raise ValueError("heartbeat_every_steps must be >=1 or None")
        n_steps = int(round(t_final / dt))
        if abs(n_steps * dt - t_final) > 1e-12 * max(1.0, t_final):
            raise ValueError("t_final must be an integer multiple of dt")
        rhs = self._rhs_for_mode(mode)

        if resume:
            if checkpoint_path is None or not Path(checkpoint_path).exists():
                raise ValueError("resume=True requires an existing checkpoint")
            state, start_step = self._read_checkpoint(
                checkpoint_path, dt, mode
            )
        else:
            if U0 is None:
                raise ValueError("U0 is required unless resume=True")
            state = np.asarray(U0, dtype=np.float64).copy()
            if mode == "galerkin":
                self._coeff_state(state)
            else:
                self._state(state)
            start_step = 0
        if start_step > n_steps:
            raise ValueError("Checkpoint is later than requested t_final")

        start_wall = time.perf_counter()
        times = [start_step * dt] if store_records else []
        states = [state.copy()] if store_records else []
        if record_callback is not None:
            record_callback(start_step * dt, state)
        if heartbeat_callback is not None:
            heartbeat_callback(
                {
                    "step": start_step,
                    "total_steps": n_steps,
                    "time": start_step * dt,
                    "elapsed_seconds": 0.0,
                    "steps_per_second": None,
                },
                state,
            )
        for step in range(start_step, n_steps):
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            completed = step + 1
            if (
                check_finite_every
                and completed % check_finite_every == 0
                and not np.all(np.isfinite(state))
            ):
                raise FloatingPointError(
                    f"Non-finite E36 state detected at step {completed}"
                )
            if completed % record_every == 0 or completed == n_steps:
                if store_records:
                    times.append(completed * dt)
                    states.append(state.copy())
                if record_callback is not None:
                    record_callback(completed * dt, state)
            if (
                checkpoint_path is not None
                and checkpoint_every_steps is not None
                and checkpoint_every_steps > 0
                and completed % checkpoint_every_steps == 0
            ):
                self._write_checkpoint(
                    checkpoint_path, state, completed, dt, mode
                )
            if (
                heartbeat_callback is not None
                and (
                    completed == n_steps
                    or (
                        heartbeat_every_steps is not None
                        and completed % heartbeat_every_steps == 0
                    )
                )
            ):
                elapsed = time.perf_counter() - start_wall
                progressed = completed - start_step
                heartbeat_callback(
                    {
                        "step": completed,
                        "total_steps": n_steps,
                        "time": completed * dt,
                        "elapsed_seconds": elapsed,
                        "steps_per_second": (
                            progressed / elapsed if elapsed > 0 else None
                        ),
                        "eta_seconds": (
                            (n_steps-completed) * elapsed / progressed
                            if progressed > 0 else None
                        ),
                    },
                    state,
                )
        if checkpoint_path is not None:
            self._write_checkpoint(
                checkpoint_path, state, n_steps, dt, mode
            )
        return {
            "t": np.asarray(times, dtype=np.float64),
            "state": (
                np.asarray(states, dtype=np.float64)
                if store_records
                else np.empty((0,) + state.shape, dtype=np.float64)
            ),
            "final_state": state,
            "start_step": start_step,
            "final_step": n_steps,
            "mode": mode,
        }


__all__ = ["LayeredChain", "make_patterns_numpy"]
