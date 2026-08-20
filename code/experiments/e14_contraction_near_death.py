"""
E14: total contraction per period A(lambda) = -ln|mu_1| NEAR THE CYCLE DEATH,
measured floor-free by 2-vector orthogonal (Benettin/QR) iteration on the
variational flow, renormalized every ~T1. Vector 1 converges to the phase mode
(check: sum of its log-norms ~ 0 = ln mu_0); vector 2 gives ln|mu_1| exactly,
immune to the ARPACK relative floor (logs accumulate, norms never underflow).

N=2000, seed 42 (the reference realization), lam in {0.40, 0.335, 0.329}.
"""

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
from couplings import make_patterns
from pacemaker_cycle import build_cycle
from floquet_monodromy import Monodromy

N, ALPHA, TAU, BETA, DT, SEED = 2000, 0.05, 10.0, 20.0, 0.01, 42
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)
rng = np.random.default_rng(3)

print("E14: contraction totale par tour A = -ln|mu_1| pres de la mort (Benettin/QR)",
      flush=True)
print(f"{'lam':>7} {'T':>8} {'ln|mu0| (ctrl~0)':>17} {'A=-ln|mu1|':>11} "
      f"{'z1=-A/T':>9} {'A/P':>7}", flush=True)

for lam in [0.40, 0.335, 0.329]:
    t0w = time.time()
    cyc = build_cycle(xi, BETA, lam, TAU, dt=DT, verbose=False,
                      settle_turns=3.0 if lam < 0.36 else 1.5)
    if not cyc.get("ok"):
        print(f"{lam:>7.3f}  cycle non forme: {cyc.get('reason','')[:50]}", flush=True)
        continue
    mon = Monodromy(xi, cyc, verbose=False)
    L, n_T, dt = mon.L, mon.n_T, mon.dt
    k1 = int(round(cyc["T1_mean"] / dt))          # renormalisation ~ tous les T1

    # deux buffers variationnels (positions + derivees pour Hermite)
    dim_win = (L + 1) * P
    V = rng.standard_normal((2, dim_win))
    V[0] = cyc["d_grid"][:L + 1].reshape(-1)      # seed du mode de phase
    Q, _ = np.linalg.qr(V.T); V = Q.T
    bufs = [np.empty((L + n_T + 1, P)) for _ in range(2)]
    dbufs = [np.empty_like(bufs[0]) for _ in range(2)]
    for j in range(2):
        bufs[j][:L + 1] = V[j].reshape(L + 1, P)
        for i in range(L + 1):
            dbufs[j][i] = mon._rhs(bufs[j][i], bufs[j][max(i - L, 0)], 2 * i)
    logs = np.zeros(2)

    def delayed(buf, dbuf, x):
        i0 = int(np.floor(x)); s = x - i0
        if s < 1e-14:
            return buf[i0]
        return ((1 + 2*s)*(1 - s)**2*buf[i0] + s*(1 - s)**2*dt*dbuf[i0]
                + s*s*(3 - 2*s)*buf[i0+1] + s*s*(s - 1)*dt*dbufs_cur[i0+1])

    n = 0
    while n < n_T:
        n_next = min(n + k1, n_T)
        for j in range(2):
            buf, dbuf = bufs[j], dbufs[j]
            global dbufs_cur; dbufs_cur = dbuf
            for m in range(n, n_next):
                i = L + m; d = buf[i]; s2 = 2 * i
                k1_ = mon._rhs(d, delayed(buf, dbuf, i - L), s2)
                k2_ = mon._rhs(d + 0.5*dt*k1_, delayed(buf, dbuf, i + 0.5 - L), s2 + 1)
                k3_ = mon._rhs(d + 0.5*dt*k2_, delayed(buf, dbuf, i + 0.5 - L), s2 + 1)
                k4_ = mon._rhs(d + dt*k3_, delayed(buf, dbuf, i + 1.0 - L), s2 + 2)
                buf[i + 1] = d + dt/6.0*(k1_ + 2*k2_ + 2*k3_ + k4_)
                dbuf[i + 1] = mon._rhs(buf[i + 1], delayed(buf, dbuf, i + 1.0 - L), s2 + 2)
        n = n_next
        # QR sur les fenetres courantes [n, n+L] -> renormalisation + deflation
        W = np.stack([bufs[j][n:n + L + 1].reshape(-1) for j in range(2)])
        Qm, Rm = np.linalg.qr(W.T)
        logs += np.log(np.abs(np.diag(Rm)))
        Wn = Qm.T
        for j in range(2):
            bufs[j][n:n + L + 1] = Wn[j].reshape(L + 1, P)
            for i in range(n, n + L + 1):
                dbufs[j][i] = mon._rhs(bufs[j][i], bufs[j][max(i - L, n)], 2 * i)
    A = -logs[1]
    print(f"{lam:>7.3f} {cyc['T']:>8.1f} {logs[0]:>17.3f} {A:>11.2f} "
          f"{-A/cyc['T']:>9.4f} {A/P:>7.3f}   ({time.time()-t0w:.0f}s)", flush=True)
