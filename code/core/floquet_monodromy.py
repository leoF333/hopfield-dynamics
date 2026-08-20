"""
Floquet analysis of the pacemaker cycle via the EXACT P-dim reduction.

Monodromy operator U: history segment da|[-tau,0] (uniform dt grid, dim (L+1)P)
-> segment one period T later, obtained by integrating the exact in-span
variational DDE (CYCLE_worklog §3-(iv)):
    t0 da' = -da + (1-lam) Dm(t) da(t) + lam S Dm(t-tau) da(t-tau),
    Dm(t) v = (1/N) xi ( gain(t) ⊙ (xi^T v) ),   gain(t) = beta(1-tanh^2(beta u_p(t))).
Same RK4 + cubic-Hermite scheme as the certified nonlinear integrator (V1/V4).
Gains at all RK stage times (half-dt grid) are precomputed once in float32
(coefficients only; validated precision) and shared across Arnoldi matvecs.

Leading Floquet multipliers via ARPACK ('LM'). Validation V2: the trivial
multiplier mu_0 = 1 with eigenvector parallel to the phase mode da = a_p'(t).

Stability: cycle orbitally stable iff all nontrivial |mu_k| < 1.
(Transverse spectrum is exactly e^{-T/t0} ~ 0 — worklog §3-(iv) — not computed.)
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence


class Monodromy:
    def __init__(self, xi, cycle, mem_guard_gb=4.0, verbose=True):
        """xi: (P,N) patterns; cycle: dict from pacemaker_cycle.build_cycle."""
        self.xi = np.asarray(xi, float)
        self.P, self.N = self.xi.shape
        self.beta = cycle["beta"]; self.lam = cycle["lam"]
        self.tau = cycle["tau"]; self.t0 = cycle["t0"]; self.dt = cycle["dt"]
        self.L = int(cycle["L"]); self.n_T = int(cycle["n_T"])
        self.T = float(cycle["T"])
        a_g, d_g = cycle["a_grid"], cycle["d_grid"]     # over [-tau, T]

        # ---- a_p on the half-dt grid via cubic Hermite ------------------
        n_half = 2 * (self.L + self.n_T) + 1
        a_half = np.empty((n_half, self.P))
        a_half[0::2] = a_g
        h00, h10, h01, h11 = 0.5, 0.125, 0.5, -0.125    # Hermite at s=1/2
        a_half[1::2] = (h00 * a_g[:-1] + h10 * self.dt * d_g[:-1]
                        + h01 * a_g[1:] + h11 * self.dt * d_g[1:])

        # ---- gain cache (float32) on the half grid ----------------------
        est_gb = n_half * self.N * 4 / 1e9
        if est_gb > mem_guard_gb:
            raise MemoryError(f"gain cache {est_gb:.1f} GB > guard")
        if verbose:
            print(f"[monodromy] precomputing gains: {n_half} stage times x N={self.N} "
                  f"({est_gb:.2f} GB float32) ...", flush=True)
        t0w = time.time()
        self.G = np.empty((n_half, self.N), dtype=np.float32)
        chunk = 4096
        for s0 in range(0, n_half, chunk):
            u = a_half[s0:s0 + chunk] @ self.xi          # (chunk, N)
            self.G[s0:s0 + chunk] = (self.beta *
                                     (1.0 - np.tanh(self.beta * u) ** 2))
        if verbose:
            print(f"[monodromy] gains done ({time.time()-t0w:.0f}s). "
                  f"dim = {(self.L+1)*self.P}, n_T = {self.n_T}", flush=True)
        self.dim = (self.L + 1) * self.P

    # ---- variational RHS at half-grid stage index s ----------------------
    def _rhs(self, d_now, d_del, s):
        xi = self.xi
        inst = xi @ (self.G[s] * (xi.T @ d_now)) / self.N
        dele = xi @ (self.G[s - 2 * self.L] * (xi.T @ d_del)) / self.N
        return (-d_now + (1.0 - self.lam) * inst
                + self.lam * np.roll(dele, 1)) / self.t0

    def matvec(self, v):
        L, P, dt, n_T = self.L, self.P, self.dt, self.n_T
        buf = np.empty((L + n_T + 1, P))
        dbuf = np.empty_like(buf)
        buf[:L + 1] = np.asarray(v, float).reshape(L + 1, P)
        for i in range(L + 1):                     # derivative history (Hermite)
            dbuf[i] = self._rhs(buf[i], buf[max(i - L, 0)], 2 * i)
        def delayed(x):                            # Hermite interp of buf at x
            i0 = int(np.floor(x)); s = x - i0
            if s < 1e-14: return buf[i0]
            return ((1 + 2*s)*(1 - s)**2 * buf[i0] + s*(1 - s)**2 * dt * dbuf[i0]
                    + s*s*(3 - 2*s) * buf[i0+1] + s*s*(s - 1) * dt * dbuf[i0+1])
        for n in range(n_T):
            i = L + n; d = buf[i]; s = 2 * i       # absolute half-grid index of t_n
            k1 = self._rhs(d, delayed(i - L), s)
            k2 = self._rhs(d + 0.5*dt*k1, delayed(i + 0.5 - L), s + 1)
            k3 = self._rhs(d + 0.5*dt*k2, delayed(i + 0.5 - L), s + 1)
            k4 = self._rhs(d + dt*k3, delayed(i + 1.0 - L), s + 2)
            buf[i + 1] = d + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
            dbuf[i + 1] = self._rhs(buf[i + 1], delayed(i + 1.0 - L), s + 2)
        return buf[n_T:].reshape(-1)

    def leading_multipliers(self, k=8, ncv=None, tol=1e-8, maxiter=300):
        op = LinearOperator((self.dim, self.dim), matvec=self.matvec,
                            dtype=np.float64)
        try:
            mu, vec = eigs(op, k=k, which="LM", ncv=ncv or (2*k + 6),
                           tol=tol, maxiter=maxiter, return_eigenvectors=True)
        except ArpackNoConvergence as e:
            mu, vec = e.eigenvalues, e.eigenvectors
        order = np.argsort(-np.abs(mu))
        return mu[order], vec[:, order]

    def phase_mode(self, cycle):
        """Expected mu=1 eigenvector: the history segment of a_p'(t) at t=0."""
        w = cycle["d_grid"][:self.L + 1].reshape(-1)
        return w / np.linalg.norm(w)


if __name__ == "__main__":
    import argparse
    from couplings import make_patterns
    from pacemaker_cycle import build_cycle
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=8)
    a = p.parse_args()

    P = round(a.alpha * a.N)
    xi, _ = make_patterns(a.N, P, a.seed)
    cyc = build_cycle(xi, a.beta, a.lam, a.tau, dt=a.dt)
    if not cyc["ok"]:
        print("cycle FAILED:", cyc["reason"]); sys.exit(1)

    mon = Monodromy(xi, cyc)
    t0w = time.time()
    print("[monodromy] timing one matvec ...", flush=True)
    _ = mon.matvec(np.random.default_rng(0).standard_normal(mon.dim))
    t1 = time.time() - t0w
    print(f"[monodromy] 1 matvec = {t1:.0f}s -> Arnoldi (~{2*a.k+8} matvecs) "
          f"~ {(2*a.k+8)*t1/60:.0f} min", flush=True)

    mu, vec = mon.leading_multipliers(k=a.k)
    w = mon.phase_mode(cyc)
    align = abs(np.vdot(vec[:, 0] / np.linalg.norm(vec[:, 0]), w))
    print("\nLeading Floquet multipliers (|mu| sorted):")
    for m in mu:
        z = np.log(m + 0j) / cyc["T"]
        print(f"  mu = {m.real:+.6f}{m.imag:+.6f}i   |mu|={abs(m):.6f}   "
              f"Re z = {z.real:+.2e}")
    print(f"\nV2  trivial multiplier: |mu_0 - 1| = {abs(mu[0]-1):.2e}   "
          f"phase-mode alignment = {align:.4f}")
    nontriv = np.abs(mu[1:]) if len(mu) > 1 else np.array([0.0])
    print(f"    max nontrivial |mu| = {nontriv.max():.6f}  -> "
          f"{'STABLE cycle' if nontriv.max() < 1 else 'UNSTABLE cycle'}")
