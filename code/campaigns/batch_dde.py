"""Restart-safe batched float64 exact reduced DDE integrator."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import numpy as np


class ReducedDDEBatch64:
    def __init__(self, xi: np.ndarray, beta: float, lam: float,
                 tau: float, t0: float = 1.0):
        self.xi = np.asarray(xi, dtype=np.float64)
        self.P, self.N = self.xi.shape
        self.beta, self.lam, self.tau, self.t0 = beta, lam, tau, t0

    def m(self, A: np.ndarray) -> np.ndarray:
        return self.xi @ np.tanh(self.beta * (self.xi.T @ A)) / self.N

    def rhs(self, now: np.ndarray, delayed: np.ndarray) -> np.ndarray:
        return (
            -now + (1.0 - self.lam) * self.m(now)
            + self.lam * np.roll(self.m(delayed), 1, axis=0)
        ) / self.t0

    def integrate(self, history: np.ndarray, total: float, dt: float,
                  record_every: int = 1,
                  derivative_history: np.ndarray | None = None) -> dict:
        L = int(round(self.tau / dt))
        if abs(L * dt - self.tau) > 1e-12:
            raise ValueError("dt must divide tau")
        history = np.asarray(history, dtype=np.float64)
        if history.ndim == 2:
            P, K = history.shape
            history = np.repeat(history[None, :, :], L + 1, axis=0)
        elif history.ndim == 3:
            _, P, K = history.shape
        else:
            raise ValueError("history must be (P,K) or (L+1,P,K)")
        if history.shape != (L + 1, self.P, K):
            raise ValueError(
                f"history must have shape {(L + 1, self.P, K)}, got {history.shape}"
            )
        if derivative_history is None:
            derivative_history = np.empty_like(history)
            for g in range(L + 1):
                derivative_history[g] = self.rhs(
                    history[g], history[max(0, g - L)]
                )
        derivative_history = np.asarray(derivative_history, dtype=np.float64)
        if derivative_history.shape != history.shape:
            raise ValueError("derivative history shape mismatch")

        R = L + 2
        buf = np.empty((R, self.P, K))
        dbuf = np.empty_like(buf)
        for g in range(L + 1):
            buf[g % R] = history[g]
            dbuf[g % R] = derivative_history[g]

        def delayed(x: float):
            i0 = int(np.floor(x))
            s = x - i0
            if s < 1e-14:
                return buf[i0 % R]
            return (
                (1 + 2*s) * (1-s)**2 * buf[i0 % R]
                + s * (1-s)**2 * dt * dbuf[i0 % R]
                + s*s * (3-2*s) * buf[(i0 + 1) % R]
                + s*s * (s-1) * dt * dbuf[(i0 + 1) % R]
            )

        n_steps = int(round(total / dt))
        times = [0.0]
        states = [buf[L % R].copy()]
        for n in range(n_steps):
            g = L + n
            a = buf[g % R]
            k1 = self.rhs(a, delayed(g - L))
            k2 = self.rhs(a + 0.5*dt*k1, delayed(g + 0.5 - L))
            k3 = self.rhs(a + 0.5*dt*k2, delayed(g + 0.5 - L))
            k4 = self.rhs(a + dt*k3, delayed(g + 1.0 - L))
            anew = a + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
            buf[(g + 1) % R] = anew
            dbuf[(g + 1) % R] = self.rhs(anew, delayed(g + 1.0 - L))
            if (n + 1) % record_every == 0:
                times.append((n + 1) * dt)
                states.append(anew.copy())
        end = L + n_steps
        indices = range(end - L, end + 1)
        return {
            "t": np.asarray(times),
            "a": np.asarray(states),
            "hist": np.stack([buf[g % R] for g in indices]),
            "dhist": np.stack([dbuf[g % R] for g in indices]),
        }

