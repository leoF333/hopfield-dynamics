"""
E19 pilot: small-delay cycle. Establish that the machinery works at small tau
before the sweeps.

Checks:
  1. tau=0 ODE limit: does the integrator run (L=0 buffer) and does a cycle form?
  2. tau=0.25 with dt=0.01 (dt<=tau/20): cycle formation + T1.
  3. dt-convergence of T1 at one small-tau point (dt halving).
  4. Timing of one dim-100 cycle build (to budget the full sweep).
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
from cycle_reduced import ReducedDDE
from pacemaker_cycle import build_cycle

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)
LAM = 0.9


def detect_T1(xi, beta, lam, tau, t0, dt, t_total, seed_state=None,
              n_settle_frac=0.4):
    """Integrate from a memory IC (+ forward bias) and measure per-step period T1
    from argmax-front handoffs, discarding an initial transient."""
    sysP = ReducedDDE(xi, beta, lam, tau, t0)
    if seed_state is None:
        a0 = np.zeros(P)
        a0[0] = 0.99
        a0[1] = 0.05        # small forward bias to launch the front
    else:
        a0 = seed_state
    sol = sysP.integrate(a0, t_total, dt, record_every=1)
    A = sol["a"]
    lead = np.argmax(A, axis=1)
    chg = np.nonzero(np.diff(lead) != 0)[0] + 1
    if len(chg) < 3:
        return None
    step_times = chg * dt
    # discard transient: keep handoffs past n_settle_frac of the run
    keep = step_times > n_settle_frac * t_total
    st = step_times[keep]
    if len(st) < 3:
        st = step_times[len(step_times) // 2:]
    T1s = np.diff(st)
    return dict(n_handoff=len(chg), T1_mean=float(T1s.mean()),
                T1_std=float(T1s.std()), n_used=len(T1s),
                lead_final=int(lead[-1]))


print(f"[E19 pilot] N={N} P={P} beta={BETA} lam={LAM}\n")

# --- 1) tau=0 ODE limit -------------------------------------------------------
print("--- tau=0 (ODE limit) ---", flush=True)
t0w = time.time()
try:
    r0 = detect_T1(xi, BETA, LAM, 0.0, T0, dt=0.01, t_total=60.0)
    if r0 is None:
        print("  tau=0: integrator OK but NO front handoffs -> no cycle "
              "(front pins immediately).")
    else:
        print(f"  tau=0: {r0['n_handoff']} handoffs, "
              f"T1={r0['T1_mean']:.3f}+-{r0['T1_std']:.3f} (n={r0['n_used']})")
except Exception as e:
    print("  tau=0 integrator FAILED:", repr(e))
print(f"  [{time.time()-t0w:.1f}s]\n")

# --- 2) tau=0.25, dt=0.01 -----------------------------------------------------
print("--- tau=0.25, dt=0.01 ---", flush=True)
t0w = time.time()
r = detect_T1(xi, BETA, LAM, 0.25, T0, dt=0.01, t_total=60.0)
if r is None:
    print("  tau=0.25: NO handoffs.")
else:
    print(f"  tau=0.25: {r['n_handoff']} handoffs, "
          f"T1={r['T1_mean']:.4f}+-{r['T1_std']:.4f} (n={r['n_used']})")
print(f"  [{time.time()-t0w:.1f}s]\n")

# --- 3) dt-convergence of T1 at tau=0.5 ---------------------------------------
print("--- dt-convergence of T1 at tau=0.5 (dt=0.02,0.01,0.005) ---", flush=True)
for dt in (0.02, 0.01, 0.005):
    t0w = time.time()
    rr = detect_T1(xi, BETA, LAM, 0.5, T0, dt=dt, t_total=60.0)
    print(f"  dt={dt:.3f}: T1={rr['T1_mean']:.5f}+-{rr['T1_std']:.5f} "
          f"[{time.time()-t0w:.1f}s]", flush=True)

# --- 4) timing of a build_cycle at tau=1 --------------------------------------
print("\n--- build_cycle timing at tau=1.0 ---", flush=True)
t0w = time.time()
cyc = build_cycle(xi, BETA, LAM, 1.0, dt=0.01, verbose=False, settle_turns=1.0)
print(f"  build_cycle tau=1: ok={cyc.get('ok')} "
      f"T1={cyc.get('T1_mean', float('nan')):.4f} "
      f"closure={cyc.get('closure', float('nan')):.1e} "
      f"[{time.time()-t0w:.1f}s]")
print("\n[E19 pilot] done.")
