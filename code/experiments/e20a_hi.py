"""
E20a (completion) -- run the HIGH-lambda points that the main sweep missed after
the a_front=None crash near depinning. Saves to a separate npz for merging.
Run:  conda run -n mcmc_env python src/e20a_hi.py [K] [nproc]
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
import sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from cycle_reduced import ReducedDDE
import e20_common as C

LAM_GRID = [0.328, 0.33, 0.34, 0.40, 0.60, 0.90]
T_TOTAL = 300.0
REC = 40
LABELS = ["memory", "front", "cycle", "chaos", "other"]
NPZ = os.path.join(C.OUT, "E20a_basin_fractions_hi.npz")

_CTX = {}

def _init_worker(lam, a_front, r_front):
    os.environ["OMP_NUM_THREADS"] = "2"
    _CTX.update(lam=lam, a_front=a_front, r_front=r_front,
                sysP=ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0))

def _run_one(a0):
    lab, info, _ = C.integrate_classify(
        _CTX["sysP"], a0, _CTX["a_front"], _CTX["lam"],
        T_TOTAL, C.DT, REC, r_front=_CTX["r_front"])
    return lab

def sweep(K=250, nproc=6, seed=20260705):
    frac_rows, all_counts = [], {}
    rng = np.random.default_rng(seed)
    for lam in LAM_GRID:
        # front unreachable above depinning -> None (handled in classify)
        a_front = C.reach_front(lam) if lam <= 0.3285 else None
        r_front = C._front_shape_ratio(lam) if lam <= 0.3285 else 0.5
        a0s, _ = C.sample_batch(rng, K)
        counts = {L: 0 for L in LABELS}
        t0 = time.time()
        with Pool(processes=nproc, initializer=_init_worker,
                  initargs=(lam, a_front, r_front)) as pool:
            for lab in tqdm(pool.imap_unordered(_run_one, a0s, chunksize=4),
                            total=K, desc=f"lam={lam:.3f}", leave=False):
                counts[lab] = counts.get(lab, 0) + 1
        fr = {L: counts[L] / K for L in LABELS}
        frac_rows.append([lam] + [fr[L] for L in LABELS])
        all_counts[f"{lam:.4f}"] = counts
        print(f"[E20a-hi] lam={lam:.3f}  mem={fr['memory']:.3f} front={fr['front']:.3f} "
              f"cyc={fr['cycle']:.3f} chaos={fr['chaos']:.3f} other={fr['other']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        np.savez(NPZ, lam_grid=np.array([r[0] for r in frac_rows]),
                 fractions=np.array(frac_rows)[:, 1:], labels=np.array(LABELS),
                 K=K, t_total=T_TOTAL, dt=C.DT, counts=json.dumps(all_counts))
    print(f"[E20a-hi] saved {NPZ}")

if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    sweep(K, nproc)
