"""
E19b - Death of the small-delay cycle under decreasing lambda (seeded hysteresis).

For tau in {0.25, 1, 2}: start from a converged travelling state at high lambda,
step lambda DOWN by 0.005, at each step re-using the final history segment as the
new IC ('seeded hysteresis'). Detect the last surviving lambda (front still
advancing) and classify the post-death state by overlap signature:
  - 'pinned front'  : |a| has two dominant consecutive components, ||a'||~0
  - 'chaos'         : ||a'|| stays finite, many components active, no lock
  - 'memory'        : a single dominant component (~1), ||a'||~0
Refine the death lambda to +-0.001 by bisection on cycle survival.
Compare lambda*(tau) with the tau=10 value 0.327-0.328.

Outputs: results/cycle/E19b_death.npz + printed tables.
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

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)
DT = 0.01
TAUS = [0.25, 1.0, 2.0]
LAM_HI = 0.45          # start deep in the travelling phase (>> lambda*~0.33)
LAM_STEP = 0.005
LAM_FLOOR = 0.20       # below static lambda_c=0.2822 -> stop
T_SEED = 200.0         # settle time per lambda step (long enough to lock/die)


def make_hist(sysP, a_last, L):
    """Constant history from a single state vector (length L+1)."""
    return np.repeat(a_last[None, :], L + 1, axis=0)


def is_advancing(A, dt, tail_frac=0.5):
    """Front still advancing? Count argmax handoffs in the last part of the run."""
    lead = np.argmax(A, axis=1)
    i0 = int((1 - tail_frac) * len(A))
    chg = np.nonzero(np.diff(lead[i0:]) != 0)[0]
    # advancing if >=3 handoffs over the tail window
    return len(chg) >= 3, len(chg)


def classify_state(A, dt, sysP, tau):
    """Classify the final (dead) state from the tail of the trajectory."""
    a = A[-1]
    # derivative magnitude (use constant-history approximation for a fixed point)
    L = int(round(tau / dt))
    adot = sysP.rhs(a, a)
    dnorm = float(np.linalg.norm(adot))
    order = np.argsort(np.abs(a))[::-1]
    top = order[:4]
    vals = np.abs(a[top])
    consecutive = (abs(int(top[0]) - int(top[1])) % (P - 1)) == 1 or \
                  abs(int(top[0]) - int(top[1])) == 1
    # also measure temporal variability over the tail (chaos test)
    tail = A[int(0.7 * len(A)):]
    var = float(np.mean(np.std(tail, axis=0)))
    if dnorm < 1e-3 and var < 1e-3:
        if vals[0] > 0.9 and vals[1] < 0.15:
            label = "memory"
        elif vals[1] > 0.15 and consecutive:
            label = "pinned front"
        else:
            label = "fixed (other)"
    else:
        label = "chaos/drift"
    return dict(label=label, dnorm=dnorm, var=var,
                top=[int(k) for k in top], vals=[float(v) for v in vals])


def seeded_descent(tau, verbose=True):
    sysP0 = ReducedDDE(xi, BETA, LAM_HI, tau, T0)
    dt = dt_for(tau)
    # build an initial travelling state at LAM_HI from memory IC + bias
    a0 = np.zeros(P); a0[0] = 0.99; a0[1] = 0.05
    sol = sysP0.integrate(a0, max(200.0, 60 * (tau + 1)), dt)
    hist = sol["hist"]
    rows = []
    lam = LAM_HI
    last_alive_lam = None
    last_alive_hist = None
    dead_state = None
    while lam >= LAM_FLOOR - 1e-9:
        sysP = ReducedDDE(xi, BETA, lam, tau, T0)
        sol = sysP.integrate(hist, T_SEED, dt)
        A = sol["a"]
        alive, nchg = is_advancing(A, dt)
        st = classify_state(A, dt, sysP, tau)
        rows.append(dict(lam=lam, alive=alive, nchg=nchg, **st))
        if verbose:
            print(f"  tau={tau} lam={lam:.3f}: alive={alive} (nchg={nchg:>3})  "
                  f"{st['label']:14s} dnorm={st['dnorm']:.1e} var={st['var']:.1e} "
                  f"top={st['top'][:2]} vals={[round(v,3) for v in st['vals'][:2]]}",
                  flush=True)
        if alive:
            last_alive_lam = lam
            last_alive_hist = sol["hist"].copy()
            hist = sol["hist"]           # seed next (lower) lambda
        else:
            dead_state = st
            break
        lam -= LAM_STEP
    return dict(tau=tau, rows=rows, last_alive_lam=last_alive_lam,
                last_alive_hist=last_alive_hist, dead_state=dead_state,
                first_dead_lam=lam)


def dt_for(tau):
    # resolve the delay: dt <= tau/20; keep 0.01 for tau>=0.2, finer below
    if tau <= 0.0:
        return 0.01
    return min(0.01, tau / 25.0)


def refine_death(tau, lam_alive, lam_dead, hist_alive, tol=0.001):
    """Bisection on cycle survival between lam_dead (< ) and lam_alive (>)."""
    dt = dt_for(tau)
    lo, hi = lam_dead, lam_alive       # lo dead, hi alive
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        sysP = ReducedDDE(xi, BETA, mid, tau, T0)
        sol = sysP.integrate(hist_alive, T_SEED, dt)
        alive, nchg = is_advancing(sol["a"], dt)
        if alive:
            hi = mid
            hist_alive = sol["hist"]
        else:
            lo = mid
    return 0.5 * (lo + hi), hi, lo


print(f"[E19b] N={N} P={P} beta={BETA}  seeded descent, step={LAM_STEP}\n")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
summary = {}
for tau in TAUS:
    print(f"=== tau={tau} (dt={dt_for(tau)}) ===", flush=True)
    t0w = time.time()
    res = seeded_descent(tau)
    lam_star = np.nan
    if res["last_alive_lam"] is not None and res["dead_state"] is not None:
        lam_star, hi, lo = refine_death(tau, res["last_alive_lam"],
                                        res["first_dead_lam"],
                                        res["last_alive_hist"])
        print(f"  -> refined lambda*({tau}) = {lam_star:.4f} "
              f"(bracket [{lo:.4f},{hi:.4f}])", flush=True)
    print(f"  last-alive lam={res['last_alive_lam']}, "
          f"post-death: {res['dead_state']['label'] if res['dead_state'] else 'n/a'} "
          f"[{time.time()-t0w:.0f}s]\n", flush=True)
    summary[tau] = dict(lam_star=lam_star,
                        last_alive=res["last_alive_lam"],
                        dead_label=res["dead_state"]["label"] if res["dead_state"] else "none",
                        dead_top=res["dead_state"]["top"] if res["dead_state"] else [],
                        dead_vals=res["dead_state"]["vals"] if res["dead_state"] else [],
                        rows=res["rows"])

print(f"\n{'tau':>6} {'lambda*':>9} {'post-death':>16} {'top2':>12} {'vals2':>16}")
for tau in TAUS:
    s = summary[tau]
    print(f"{tau:>6} {s['lam_star']:>9.4f} {s['dead_label']:>16} "
          f"{str(s['dead_top'][:2]):>12} "
          f"{str([round(v,3) for v in s['dead_vals'][:2]]):>16}")
print(f"\n  tau=10 reference: lambda* = 0.327-0.328, post-death = pinned front (91,92)")

# save (rows are dicts -> store as object arrays of the key scalars)
np.savez(os.path.join(OUT, "E19b_death.npz"),
         taus=np.array(TAUS),
         lam_star=np.array([summary[t]["lam_star"] for t in TAUS]),
         last_alive=np.array([summary[t]["last_alive"] if summary[t]["last_alive"]
                              else np.nan for t in TAUS]),
         dead_labels=np.array([summary[t]["dead_label"] for t in TAUS]),
         **{f"lamgrid_tau{t}": np.array([r["lam"] for r in summary[t]["rows"]])
            for t in TAUS},
         **{f"nchg_tau{t}": np.array([r["nchg"] for r in summary[t]["rows"]])
            for t in TAUS},
         **{f"labels_tau{t}": np.array([r["label"] for r in summary[t]["rows"]])
            for t in TAUS})
print("[E19b] npz saved.")
