"""
E20 PILOT: validate the classifier on KNOWN initial conditions (front / memory /
cycle) and time one lam with K=50 random ICs end-to-end. Prints a confusion check
and per-IC timing so the full sweep can be sized under ~40 min.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tqdm import tqdm

from cycle_reduced import ReducedDDE
import e20_common as C

T_TOTAL = 300.0
REC = 40                       # record every REC*dt = 2.0 time units

def known_ic_check():
    print("=" * 70)
    print("CLASSIFIER VALIDATION on KNOWN initial states")
    print("=" * 70)
    # (a) start ON the pinned front, deep memory phase (should stay 'front')
    for lam in (0.20, 0.30):
        a_front = C.reach_front(lam); rf = C._front_shape_ratio(lam)
        sysP = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)
        lab, info, _ = C.integrate_classify(sysP, a_front.copy(), a_front, lam,
                                             T_TOTAL, C.DT, REC, r_front=rf)
        print(f"  ON-FRONT   lam={lam:.3f} -> {lab:6s}  d_front={info['d_front']:.2e} "
              f"bal={info['balance']:.3f} (rf={rf:.3f}) s0={info['s0']:.3f}")
    # (b) start ON pure memory xi^1 (should stay 'memory' below lam_c=0.2822)
    for lam in (0.15, 0.25):
        a_front = C.reach_front(lam); rf = C._front_shape_ratio(lam)
        a_mem = C.reach_memory(lam, 0)
        sysP = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)
        lab, info, _ = C.integrate_classify(sysP, a_mem.copy(), a_front, lam,
                                             T_TOTAL, C.DT, REC, r_front=rf)
        print(f"  ON-MEMORY  lam={lam:.3f} -> {lab:6s}  d_front={info['d_front']:.2e} "
              f"bal={info['balance']:.3f} (rf={rf:.3f}) s0={info['s0']:.3f}")
    # (c) start ON memory at high lam (cycle regime) -> should form the cycle
    for lam in (0.40, 0.60):
        a_front = C.reach_front(lam) if lam <= 0.328 else np.zeros(C.P)
        rf = C._front_shape_ratio(lam) if lam <= 0.328 else 0.5
        a_mem = 0.99 * np.eye(C.P)[0]      # IC-memory constant history
        sysP = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)
        lab, info, _ = C.integrate_classify(sysP, a_mem.copy(), a_front, lam,
                                             T_TOTAL, C.DT, REC, r_front=rf)
        print(f"  ON-MEMORY(hi) lam={lam:.3f} -> {lab:6s}  n_adv={info.get('n_adv')} "
              f"mono={info.get('mono',0):.2f} drift={info['drift']:.2e}")
    # (d) chaotic sector: random mixture at lam=0.31 -> expect chaos/other
    lam = 0.31
    a_front = C.reach_front(lam); rf = C._front_shape_ratio(lam)
    sysP = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)
    rng = np.random.default_rng(0)
    labs = {}
    for _ in range(6):
        a0 = C.sample_ic(rng, "random_small")
        lab, info, _ = C.integrate_classify(sysP, a0, a_front, lam, T_TOTAL, C.DT, REC, r_front=rf)
        labs[lab] = labs.get(lab, 0) + 1
    print(f"  RANDOM-SMALL lam={lam:.3f} (6 ICs) -> {labs}")


def timing_pilot(lam=0.30, K=50):
    print("\n" + "=" * 70)
    print(f"TIMING PILOT: lam={lam}, K={K} random ICs, t={T_TOTAL}, dt={C.DT}")
    print("=" * 70)
    a_front = C.reach_front(lam); rf = C._front_shape_ratio(lam)
    sysP = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)
    rng = np.random.default_rng(20)
    a0s, kinds = C.sample_batch(rng, K)
    counts = {}
    t0 = time.time()
    for a0 in tqdm(a0s, desc=f"pilot lam={lam}"):
        lab, info, _ = C.integrate_classify(sysP, a0, a_front, lam, T_TOTAL, C.DT, REC, r_front=rf)
        counts[lab] = counts.get(lab, 0) + 1
    dt_run = time.time() - t0
    print(f"  counts: {counts}")
    print(f"  time {dt_run:.1f}s for K={K}  ->  {dt_run/K*1000:.0f} ms / IC")
    print(f"  ESTIMATE for K=400 x 14 lam = {dt_run/K*400*14/60:.1f} min "
          f"(+ front continuation overhead per lam)")


if __name__ == "__main__":
    t_start = time.time()
    known_ic_check()
    timing_pilot(lam=0.30, K=50)
    print(f"\n[E20 pilot] total {time.time()-t_start:.0f}s")
