"""
Pacemaker cycle construction from the EXACT P-dim reduction (CYCLE_worklog §5.1).

Simulates t0 a' = -a + (1-lam) m(a) + lam S m(a(t-tau)) from a memory IC, detects
the sequential-recall cycle, measures the step period T1 (per-pattern advance) and
the full period T = sum of the P steps, and stores one full period of the cycle
(positions + derivatives on the dt grid, including the [-tau,0] history) for the
Floquet/monodromy stage.

Observables: a_nu(t) directly (reduced amplitudes). The front advance nu(t) is
tracked via argmax_nu a_nu(t). Step times = times where argmax changes.

Usage (demo): python pacemaker_cycle.py [--lam 0.9] [--N 2000] [--alpha 0.05]
              [--tau 10] [--dt 0.01]
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from cycle_reduced import ReducedDDE
from couplings import make_patterns


def build_cycle(xi, beta, lam, tau, t0=1.0, dt=0.01, verbose=True,
                settle_turns=1.0, seed_state=None):
    """
    Simulate to the pacemaker cycle and extract one full period.

    Returns dict:
      T, T1_mean, T1_std : full period, per-step period stats (disorder spread)
      a_grid  : (L + n_T + 1, P) cycle samples on the dt grid over [-tau, T]
      d_grid  : matching derivatives (for Hermite interpolation)
      n_T, L  : steps per period, steps per delay
      step_times : the P front-advance times within the extracted period
      closure : |a(T)-a(0)| / scale  (periodicity residual)
    """
    P = xi.shape[0]
    sysP = ReducedDDE(xi, beta, lam, tau, t0)
    T1_guess = tau + 2.0 * t0
    T_guess = P * T1_guess

    a0 = seed_state if seed_state is not None else None
    if a0 is None:
        a0 = np.zeros(P); a0[0] = 0.99

    # ---- long run: settle + >1 full turn -------------------------------
    t_total = (settle_turns + 1.35) * T_guess
    if verbose:
        print(f"[cycle] lam={lam}: simulating t={t_total:.0f} "
              f"(dt={dt}, ~{int(t_total/dt)} steps) ...", flush=True)
    t0w = time.time()
    sol = sysP.integrate(a0, t_total, dt, record_every=1)
    A = sol["a"]                       # (n+1, P) from t=0
    if verbose:
        print(f"[cycle] simulation done ({time.time()-t0w:.0f}s)", flush=True)

    # ---- front tracking: argmax_nu a_nu(t) ------------------------------
    lead = np.argmax(A, axis=1)
    # step times: changes of the leading pattern (with unwrapping mod P)
    chg = np.nonzero(np.diff(lead) != 0)[0] + 1
    if len(chg) < 2 * P:
        return dict(ok=False, reason=f"front advanced only {len(chg)} steps "
                    f"(< 2P={2*P}) — no sustained pacemaker at lam={lam}")
    # keep the LAST P+1 changes (well past transient) -> one full turn
    idx = chg[-(P + 1):]
    step_times = idx * dt
    T1s = np.diff(step_times)
    T = float(step_times[-1] - step_times[0])
    n_T = int(round(T / dt))
    L = int(round(tau / dt))
    i0 = int(idx[0])                    # cycle phase origin (a front-advance)
    if i0 < L or idx[-1] + 1 > len(A):
        return dict(ok=False, reason="not enough history before the extracted "
                    "period; increase settle_turns")

    # ---- extract [-tau, T] and derivatives ------------------------------
    a_grid = A[i0 - L: i0 + n_T + 1].copy()
    d_grid = np.empty_like(a_grid)
    for k in range(a_grid.shape[0]):
        gidx = i0 - L + k
        d_grid[k] = sysP.rhs(A[gidx], A[max(gidx - L, 0)])
    closure = float(np.linalg.norm(A[i0 + n_T] - A[i0])
                    / max(np.linalg.norm(A[i0]), 1e-30))
    out = dict(ok=True, T=T, T1_mean=float(T1s.mean()), T1_std=float(T1s.std()),
               a_grid=a_grid, d_grid=d_grid, n_T=n_T, L=L, dt=dt,
               step_times=step_times - step_times[0], closure=closure,
               lam=lam, tau=tau, beta=beta, t0=t0)
    if verbose:
        print(f"[cycle] T={T:.2f}  T1={out['T1_mean']:.3f}±{out['T1_std']:.3f} "
              f"(tau+1={tau+1})  closure={closure:.2e}", flush=True)
    return out


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    P = round(a.alpha * a.N)
    xi, _ = make_patterns(a.N, P, a.seed)
    cyc = build_cycle(xi, a.beta, a.lam, a.tau, dt=a.dt)
    if not cyc["ok"]:
        print("FAILED:", cyc["reason"]); sys.exit(1)

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "cycle")
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"cycle_lam{a.lam}_tau{a.tau}_N{a.N}.npz"),
             **{k: v for k, v in cyc.items() if isinstance(v, (int, float, np.ndarray))})

    # waveform figure: a_nu(t) for a few nu + step times
    t_ax = np.arange(cyc["a_grid"].shape[0]) * a.dt - a.tau
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for nu in range(0, 6):
        ax1.plot(t_ax, cyc["a_grid"][:, nu], lw=1.2, label=f"$a_{{{nu+1}}}$")
    ax1.set_xlim(-a.tau, 6 * cyc["T1_mean"])
    ax1.set_xlabel("t"); ax1.set_ylabel(r"$a_\nu(t)$")
    ax1.set_title(f"Pacemaker waveform (first steps), $\\lambda={a.lam}$")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(True, alpha=.3)
    ax2.plot(np.arange(len(cyc["step_times"]) - 1),
             np.diff(cyc["step_times"]), "o-", ms=4)
    ax2.axhline(a.tau + 1, color="k", ls=":", label=r"$\tau+1$")
    ax2.axhline(cyc["T1_mean"], color="crimson", ls="--",
                label=f"mean {cyc['T1_mean']:.2f}")
    ax2.set_xlabel("step index"); ax2.set_ylabel(r"$T_1$ per step")
    ax2.set_title(f"Step periods around the ring "
                  f"(disorder spread ±{cyc['T1_std']:.3f})")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=.3)
    fig.tight_layout()
    fp = os.path.join(OUT, f"waveform_lam{a.lam}_tau{a.tau}_N{a.N}.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"[cycle] figure -> {fp}")
