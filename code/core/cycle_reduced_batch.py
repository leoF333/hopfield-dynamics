"""
Batched float64 integrator for the exact P-dim reduced delayed DDE.

Optimization (user request 2026-07-08): integrate K initial conditions at once as
matrix-matrix products (A is P x K), the honest CPU parallelization. NOT GPU:
the accuracy-critical work stays float64 (documented false-fold lesson), and
Apple Metal/MLX has no native float64. Batching turns the two per-step matvecs
(xi^T a, xi (.)) into matmuls over K columns -> big BLAS win, zero accuracy risk.

Memory: a ring buffer of size (L+2, P, K) holds the last positions+derivatives
needed for the delay (delay = L steps; RK stages touch indices [i-L, i+1]), so
memory is O((L+2) P K) not O(n_steps P K). Recorded samples are (n_rec, P, K).

Validated in __main__ against cycle_reduced.ReducedDDE (single trajectory) to
~1e-13 max abs diff over the whole trajectory.
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
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


class ReducedDDEBatch:
    """Batched exact P-dim reduction. xi is (P,N); integrate K histories at once."""

    def __init__(self, xi, beta, lam, tau, t0=1.0):
        self.xi = np.asarray(xi, float)          # (P,N)
        self.P, self.N = self.xi.shape
        self.beta, self.lam, self.tau, self.t0 = beta, lam, tau, t0

    def m(self, A):
        """m(A) = (1/N) xi tanh(beta xi^T A), A is (P,K) -> (P,K)."""
        return self.xi @ np.tanh(self.beta * (self.xi.T @ A)) / self.N

    def rhs(self, A_now, A_del):
        drive = (1.0 - self.lam) * self.m(A_now) \
            + self.lam * np.roll(self.m(A_del), 1, axis=0)   # S: index nu <- nu-1
        return (-A_now + drive) / self.t0

    def integrate(self, A_hist0, t_total, dt, record_every=1):
        """A_hist0: (P,K) constant history, or (L+1,P,K). Returns dict(t, a) with
        a of shape (n_rec, P, K)."""
        L = int(round(self.tau / dt))
        assert abs(L * dt - self.tau) < 1e-12, "dt must divide tau"
        n_steps = int(round(t_total / dt))
        P = self.P
        A_hist0 = np.asarray(A_hist0, float)
        if A_hist0.ndim == 2:
            K = A_hist0.shape[1]
            hist = np.repeat(A_hist0[None, :, :], L + 1, axis=0)   # (L+1,P,K)
        else:
            assert A_hist0.shape[0] == L + 1
            K = A_hist0.shape[2]; hist = A_hist0.copy()
        R = L + 2                                     # ring size
        buf = np.empty((R, P, K)); dbuf = np.empty((R, P, K))
        # seed ring with the history indices 0..L
        for i in range(L + 1):
            buf[i % R] = hist[i]
            j_del = max(i - L, 0)
            dbuf[i % R] = self.rhs(hist[i], hist[j_del])

        def delayed(idx_float):
            i0 = int(np.floor(idx_float)); s = idx_float - i0
            s0, s1 = i0 % R, (i0 + 1) % R
            if s < 1e-14:
                return buf[s0]
            h00 = (1 + 2 * s) * (1 - s) ** 2; h10 = s * (1 - s) ** 2
            h01 = s * s * (3 - 2 * s); h11 = s * s * (s - 1)
            return (h00 * buf[s0] + h10 * dt * dbuf[s0]
                    + h01 * buf[s1] + h11 * dt * dbuf[s1])

        ts = [0.0]; As = [buf[L % R].copy()]
        for n in range(n_steps):
            i = L + n
            a = buf[i % R]
            k1 = self.rhs(a, delayed(i - L))
            k2 = self.rhs(a + 0.5 * dt * k1, delayed(i + 0.5 - L))
            k3 = self.rhs(a + 0.5 * dt * k2, delayed(i + 0.5 - L))
            k4 = self.rhs(a + dt * k3, delayed(i + 1.0 - L))
            nxt = a + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            buf[(i + 1) % R] = nxt
            dbuf[(i + 1) % R] = self.rhs(nxt, delayed(i + 1.0 - L))
            if (n + 1) % record_every == 0:
                ts.append((n + 1) * dt); As.append(nxt.copy())
        return dict(t=np.array(ts), a=np.array(As))     # a: (n_rec,P,K)


if __name__ == "__main__":
    # ---- hard validation vs the single-trajectory reference ------------------
    from couplings import make_patterns
    from cycle_reduced import ReducedDDE

    N, P, SEED = 2000, 100, 42
    BETA, TAU, T0 = 20.0, 10.0, 1.0
    xi, _ = make_patterns(N, P, SEED)
    rng = np.random.default_rng(7)
    K = 5
    lam = 0.31
    dt, t_tot, rec = 0.05, 120.0, 8
    A0 = 0.3 * rng.standard_normal((P, K))

    # batched
    sysB = ReducedDDEBatch(xi, BETA, lam, TAU, T0)
    solB = sysB.integrate(A0, t_tot, dt, record_every=rec)   # (n_rec,P,K)

    # reference: K independent single runs
    sysS = ReducedDDE(xi, BETA, lam, TAU, T0)
    maxdiff = 0.0
    for k in range(K):
        solS = sysS.integrate(A0[:, k], t_tot, dt, record_every=rec)
        d = np.max(np.abs(solB["a"][:, :, k] - solS["a"]))
        maxdiff = max(maxdiff, d)
        print(f"  traj {k}: max|batched - single| = {d:.2e}  "
              f"(final |a|={np.linalg.norm(solS['a'][-1]):.3f})")
    print(f"\n[validate] overall max abs diff (batched vs single) = {maxdiff:.2e}")
    print(f"[validate] {'PASS' if maxdiff < 1e-11 else '*** FAIL ***'} "
          f"(tol 1e-11)")

    # ---- timing: batched vs looped single ------------------------------------
    import time
    Kt = 64
    A0t = 0.3 * rng.standard_normal((P, Kt))
    t0c = time.time(); sysB.integrate(A0t, 400.0, dt, record_every=40)
    tb = time.time() - t0c
    t0c = time.time()
    for k in range(Kt):
        sysS.integrate(A0t[:, k], 400.0, dt, record_every=40)
    tsg = time.time() - t0c
    print(f"\n[timing] K={Kt}, t=400: batched {tb:.2f}s  vs looped-single "
          f"{tsg:.2f}s  -> speedup x{tsg/tb:.1f}")
