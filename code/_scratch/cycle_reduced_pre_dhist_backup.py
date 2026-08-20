"""
Exact P-dimensional reduction of the delayed mixed Hopfield DDE + validations.

Theory (CYCLE_worklog.md §3): the pattern span is EXACTLY invariant and attracting
(transverse component obeys t0 w' = -w exactly). With X = [xi^1 ... xi^P] (N x P)
and u = X a, the full dynamics reduces EXACTLY (finite alpha, sample disorder
included) to
    t0 a' = -a + (1-lam) m(a(t)) + lam S m(a(t-tau)),
    m(a)  = (1/N) X^T tanh(beta X a),        (S m)_nu = m_{nu-1}  (cyclic).
The in-span variational problem around a solution a_p(t):
    t0 da' = -da + (1-lam) Dm(t) da(t) + lam S Dm(t-tau) da(t-tau),
    Dm(t) = (1/N) X^T D(t) X  (= the static G_J along the trajectory).

Index subtlety (worklog §3): the static T_P uses G_K = Dm.S while the variational
uses S.Dm — det-equivalent via AB<->BA. V0 certifies this numerically.

Integrator: method of steps, fixed-dt RK4 with cubic-Hermite interpolation of the
history ring buffer (positions AND stored derivatives), so delayed values at RK
stage times keep high order.

Validations (run as __main__, ~seconds at N=500):
  V0  reduced characteristic matrix roots == static T_P roots (index trap test)
  V1  reduced P-dim trajectory == full-N trajectory (in-span IC), same RK4
  V4  dt-convergence of the integrator (RK4 order check on the DDE)
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


class ReducedDDE:
    """Exact P-dim reduction. Patterns given as xi (P x N), xi_shift = roll(xi,-1,0)."""

    def __init__(self, xi: np.ndarray, beta: float, lam: float, tau: float,
                 t0: float = 1.0):
        self.xi = np.asarray(xi, float)          # (P, N)
        self.P, self.N = self.xi.shape
        self.beta, self.lam, self.tau, self.t0 = beta, lam, tau, t0

    # ---- fields -----------------------------------------------------------
    def m(self, a: np.ndarray) -> np.ndarray:
        """m(a) = (1/N) xi tanh(beta xi^T a)   [X = xi^T]."""
        return self.xi @ np.tanh(self.beta * (self.xi.T @ a)) / self.N

    def rhs(self, a_now: np.ndarray, a_del: np.ndarray) -> np.ndarray:
        """t0 a' = -a + (1-lam) m(a) + lam S m(a_del);  (S m)_nu = m_{nu-1}."""
        drive = (1.0 - self.lam) * self.m(a_now) \
            + self.lam * np.roll(self.m(a_del), 1)      # roll(+1): index nu <- nu-1
        return (-a_now + drive) / self.t0

    def Dm(self, a: np.ndarray) -> np.ndarray:
        """Dm(a) = (1/N) xi diag(gain) xi^T (P x P), gain = beta(1-tanh^2)."""
        g = self.beta * (1.0 - np.tanh(self.beta * (self.xi.T @ a)) ** 2)
        return (self.xi * g[None, :]) @ self.xi.T / self.N

    # ---- integrator: method of steps, RK4 + cubic Hermite on history ------
    def integrate(self, a_hist0, t_total: float, dt: float,
                  record_every: int = 1):
        """
        a_hist0: (L+1, P) history at t = -tau..0 on the dt grid (L = tau/dt), or
                 (P,) constant history. Returns dict(t, a) with recorded samples.
        """
        L = int(round(self.tau / dt))
        assert abs(L * dt - self.tau) < 1e-12, "dt must divide tau"
        n_steps = int(round(t_total / dt))
        P = self.P

        # ring buffers of positions and derivatives (for Hermite interpolation)
        buf = np.empty((L + n_steps + 1, P))
        dbuf = np.empty_like(buf)
        if np.ndim(a_hist0) == 1:
            buf[:L + 1] = np.asarray(a_hist0, float)[None, :]
        else:
            assert a_hist0.shape == (L + 1, P)
            buf[:L + 1] = a_hist0
        # derivative history (needed by Hermite): from the RHS on the history
        for i in range(L + 1):
            j_del = max(i - L, 0)
            dbuf[i] = self.rhs(buf[i], buf[j_del])

        def delayed(idx_float):
            """Cubic-Hermite interpolation of buf at fractional index idx_float."""
            i0 = int(np.floor(idx_float))
            s = idx_float - i0
            if s < 1e-14:
                return buf[i0]
            h00 = (1 + 2 * s) * (1 - s) ** 2
            h10 = s * (1 - s) ** 2
            h01 = s * s * (3 - 2 * s)
            h11 = s * s * (s - 1)
            return (h00 * buf[i0] + h10 * dt * dbuf[i0]
                    + h01 * buf[i0 + 1] + h11 * dt * dbuf[i0 + 1])

        ts, As = [0.0], [buf[L].copy()]
        for n in range(n_steps):
            i = L + n                       # current index (time t_n)
            a = buf[i]
            # RK4 stages; delayed argument at t_n + c*dt - tau -> index i + c - L
            k1 = self.rhs(a, delayed(i - L))
            k2 = self.rhs(a + 0.5 * dt * k1, delayed(i + 0.5 - L))
            k3 = self.rhs(a + 0.5 * dt * k2, delayed(i + 0.5 - L))
            k4 = self.rhs(a + dt * k3, delayed(i + 1.0 - L))
            buf[i + 1] = a + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            dbuf[i + 1] = self.rhs(buf[i + 1], delayed(i + 1.0 - L))
            if (n + 1) % record_every == 0:
                ts.append((n + 1) * dt); As.append(buf[i + 1].copy())
        return dict(t=np.array(ts), a=np.array(As),
                    hist=buf[-(L + 1):].copy())     # final history segment


# ------------------------------------------------------------------ full-N ref
def integrate_fullN(xi, beta, lam, tau, t0, u_hist0, t_total, dt):
    """Same RK4/Hermite scheme on the full N-dim DDE (validation reference)."""
    xiT = xi.T; P, N = xi.shape
    def m(u): return xi @ np.tanh(beta * u) / N
    def rhs(u_now, u_del):
        return (-u_now + xiT @ ((1 - lam) * m(u_now)
                                + lam * np.roll(m(u_del), 1))) / t0
    L = int(round(tau / dt)); n_steps = int(round(t_total / dt))
    buf = np.empty((L + n_steps + 1, N)); dbuf = np.empty_like(buf)
    buf[:L + 1] = u_hist0[None, :] if u_hist0.ndim == 1 else u_hist0
    for i in range(L + 1):
        dbuf[i] = rhs(buf[i], buf[max(i - L, 0)])
    def delayed(x):
        i0 = int(np.floor(x)); s = x - i0
        if s < 1e-14: return buf[i0]
        return ((1 + 2*s)*(1 - s)**2*buf[i0] + s*(1 - s)**2*dt*dbuf[i0]
                + s*s*(3 - 2*s)*buf[i0+1] + s*s*(s - 1)*dt*dbuf[i0+1])
    for n in range(n_steps):
        i = L + n; u = buf[i]
        k1 = rhs(u, delayed(i - L))
        k2 = rhs(u + 0.5*dt*k1, delayed(i + 0.5 - L))
        k3 = rhs(u + 0.5*dt*k2, delayed(i + 0.5 - L))
        k4 = rhs(u + dt*k3, delayed(i + 1.0 - L))
        buf[i + 1] = u + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
        dbuf[i + 1] = rhs(buf[i + 1], delayed(i + 1.0 - L))
    return buf[L:]


# ------------------------------------------------------------------ validations
if __name__ == "__main__":
    from couplings import Couplings, make_patterns
    from robust_branch import trace_branch, anneal_newton
    from reduced_spectrum import GJ_GK, T_P, sigmin_TP, physical_roots

    N, ALPHA, BETA, TAU, T0, SEED = 500, 0.05, 20.0, 10.0, 1.0, 42
    P = round(ALPHA * N)
    xi, xis = make_patterns(N, P, SEED)
    coup = Couplings(xi, xis); coup._use_np = True
    results = []
    def check(name, err, tol):
        ok = err < tol
        results.append((name, err, tol, ok))
        print(f"  {'PASS' if ok else '*** FAIL ***'}  {name:56s} "
              f"err={err:.2e} (tol {tol:.0e})")

    # a memory fixed point below the fold
    cfg = dict(lam_min=0.0, lam_max=0.6, ds=0.01, n_overlaps=5)
    br, fold, _ = trace_branch(coup, BETA, cfg)
    lam = 0.8 * fold
    j = int(np.argmin(np.abs(br["lam"][:np.argmax(br["lam"])+1] - lam)))
    lam = float(br["lam"][j]); u_star = br["u_star"][j]
    print(f"validation at N={N}, P={P}, lam={lam:.4f} (fold={fold:.4f})\n")

    sysP = ReducedDDE(xi, BETA, lam, TAU, T0)

    # ---- V0: reduced characteristic matrix == T_P spectrum ---------------
    # T_red(z) = (t0 z+1)I - (1-lam) Dm - lam e^{-z tau} S.Dm ; must share the
    # roots of T_P (which uses G_K = Dm.S) by the AB<->BA identity.
    a_star = np.linalg.lstsq(xi.T, u_star, rcond=None)[0]
    check("V0a  in-span residual of u* (u* = X a*)",
          np.linalg.norm(xi.T @ a_star - u_star) / np.linalg.norm(u_star), 1e-10)
    Dm = sysP.Dm(a_star)
    GJ, GK = GJ_GK(coup, u_star, BETA, lam)
    check("V0b  Dm == G_J (same P x P object)",
          np.linalg.norm(Dm - GJ) / np.linalg.norm(GJ), 1e-12)
    S = np.roll(np.eye(P), 1, axis=0)          # S e_nu = e_{nu+1}
    check("V0c  G_K == Dm.S (index convention)",
          np.linalg.norm(GK - Dm @ S) / np.linalg.norm(GK), 1e-12)
    # roots of T_P are roots of T_red:
    ph, _, _ = physical_roots(coup, u_star, BETA, lam, TAU, T0, M=24, n_cand=30, k=6)
    def sigmin_Tred(z):
        Tr = (T0 * z + 1) * np.eye(P) - (1 - lam) * Dm \
            - lam * np.exp(-z * TAU) * (S @ Dm)
        return float(np.linalg.svd(Tr, compute_uv=False).min())
    if len(ph):
        check("V0d  T_P roots are T_red roots (AB<->BA identity)",
              max(sigmin_Tred(z) for z in ph[:6]), 1e-8)

    # ---- V1: reduced trajectory == full-N trajectory ---------------------
    rng = np.random.default_rng(1)
    a0 = a_star + 0.05 * rng.standard_normal(P)      # in-span perturbed IC
    dt = 0.01; t_tot = 8.0
    solP = sysP.integrate(a0, t_tot, dt)
    solN = integrate_fullN(xi, BETA, lam, TAU, T0, xi.T @ a0, t_tot, dt)
    uP_final = xi.T @ solP["a"][-1]
    check("V1   P-dim trajectory == full-N trajectory (t=8)",
          np.linalg.norm(uP_final - solN[-1]) / np.linalg.norm(solN[-1]), 1e-10)

    # ---- V4: dt-convergence (RK4 + Hermite on the DDE) --------------------
    sol1 = sysP.integrate(a0, 4.0, 0.02)["a"][-1]
    sol2 = sysP.integrate(a0, 4.0, 0.01)["a"][-1]
    sol3 = sysP.integrate(a0, 4.0, 0.005)["a"][-1]
    e12 = np.linalg.norm(sol1 - sol3); e23 = np.linalg.norm(sol2 - sol3)
    order = np.log2(e12 / max(e23, 1e-300)) if e23 > 0 else np.inf
    print(f"  INFO  V4 dt-convergence: err(0.02)={e12:.2e} err(0.01)={e23:.2e} "
          f"-> apparent order ~{order:.1f}")
    check("V4   dt=0.01 accuracy vs dt=0.005", e23, 1e-6)

    print("\n" + "=" * 74)
    n_ok = sum(1 for *_, ok in results if ok)
    print(f"CYCLE-REDUCTION VALIDATION: {n_ok}/{len(results)} passed")
    for name, err, tol, ok in results:
        if not ok:
            print(f"  FAILED: {name} err={err:.2e}")
    print("=" * 74)
