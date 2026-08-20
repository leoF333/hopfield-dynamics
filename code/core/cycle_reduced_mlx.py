"""
MLX float32 GPU batched integrator for the reduced delayed DDE (Apple Metal).

Purpose (2026-07-08): the float64 lesson is LOCALIZED to root-finding / fold
detection below the float32 floor (Newton, eig crossings, small Lyapunov exps).
Basin CLASSIFICATION -- a coarse, ensemble-averaged outcome -- tolerates float32,
and the GPU sits idle while E30 uses the CPU, so this runs in PARALLEL with no
contention. USE ONLY AFTER the basin-fraction validation gate (see __main__ and
e32) confirms float32 fractions match float64 within statistical error.

Never use this path for Newton/continuation/Lyapunov (float64 CPU only).
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx


class ReducedDDEMLX:
    def __init__(self, xi_np, beta, lam, tau, t0=1.0):
        self.xi = mx.array(np.asarray(xi_np, np.float32))     # (P,N)
        self.xiT = self.xi.T
        self.P, self.N = self.xi.shape
        self.beta = float(beta); self.lam = float(lam)
        self.tau = float(tau); self.t0 = float(t0)

    def m(self, A):                                            # A (P,K)
        return (self.xi @ mx.tanh(self.beta * (self.xiT @ A))) / self.N

    def rhs(self, A_now, A_del):
        drive = (1.0 - self.lam) * self.m(A_now) \
            + self.lam * mx.roll(self.m(A_del), 1, axis=0)
        return (-A_now + drive) / self.t0

    def integrate(self, A0_np, t_total, dt, record_every=1):
        L = int(round(self.tau / dt)); assert abs(L * dt - self.tau) < 1e-9
        n_steps = int(round(t_total / dt)); R = L + 2
        A0 = mx.array(np.asarray(A0_np, np.float32))           # (P,K)
        d0 = self.rhs(A0, A0); mx.eval(A0, d0)
        ring = [A0] * R; dring = [d0] * R                      # constant history
        # ring[i%R] holds position at index i; indices 0..L seeded with A0/d0
        for i in range(L + 1):
            ring[i % R] = A0; dring[i % R] = d0

        def delayed(idxf):
            i0 = int(np.floor(idxf)); s = idxf - i0
            s0, s1 = i0 % R, (i0 + 1) % R
            if s < 1e-7:
                return ring[s0]
            h00 = (1 + 2 * s) * (1 - s) ** 2; h10 = s * (1 - s) ** 2
            h01 = s * s * (3 - 2 * s); h11 = s * s * (s - 1)
            return (h00 * ring[s0] + h10 * dt * dring[s0]
                    + h01 * ring[s1] + h11 * dt * dring[s1])

        rec = [np.array(A0)]; ts = [0.0]
        for n in range(n_steps):
            i = L + n; a = ring[i % R]
            k1 = self.rhs(a, delayed(i - L))
            k2 = self.rhs(a + 0.5 * dt * k1, delayed(i + 0.5 - L))
            k3 = self.rhs(a + 0.5 * dt * k2, delayed(i + 0.5 - L))
            k4 = self.rhs(a + dt * k3, delayed(i + 1.0 - L))
            nxt = a + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            dn = self.rhs(nxt, delayed(i + 1.0 - L))
            ring[(i + 1) % R] = nxt; dring[(i + 1) % R] = dn
            mx.eval(nxt, dn)                          # bound the lazy graph
            if (n + 1) % record_every == 0:
                rec.append(np.array(nxt)); ts.append((n + 1) * dt)
        return dict(t=np.array(ts), a=np.stack(rec))           # (n_rec,P,K) f32


if __name__ == "__main__":
    from couplings import make_patterns
    from cycle_reduced_batch import ReducedDDEBatch
    import time

    N, P, SEED = 2000, 100, 42
    BETA, TAU, T0, lam = 20.0, 10.0, 1.0, 0.31
    xi, _ = make_patterns(N, P, SEED)
    rng = np.random.default_rng(3)
    K = 8
    A0 = 0.3 * rng.standard_normal((P, K))
    dt = 0.05

    sysG = ReducedDDEMLX(xi, BETA, lam, TAU, T0)
    sysC = ReducedDDEBatch(xi, BETA, lam, TAU, T0)

    # (1) short-time correctness: BEFORE chaotic divergence (t=3), f32 vs f64 close
    tt = 3.0
    aG = sysG.integrate(A0, tt, dt, record_every=1)["a"]
    aC = sysC.integrate(A0, tt, dt, record_every=1)["a"]
    rel = np.max(np.abs(aG - aC)) / (np.max(np.abs(aC)) + 1e-9)
    print(f"[mlx] short-time (t={tt}) max rel diff f32(GPU) vs f64(CPU) = {rel:.2e} "
          f"({'OK ~float32 eps' if rel < 5e-4 else 'CHECK'})")

    # (2) timing at large batch
    Kt = 512
    A0t = 0.3 * rng.standard_normal((P, Kt))
    t0c = time.time(); sysG.integrate(A0t, 400.0, dt, record_every=40); tg = time.time() - t0c
    print(f"[mlx] GPU t=400, K={Kt}: {tg:.1f}s  ({Kt/tg:.0f} traj/s)")
