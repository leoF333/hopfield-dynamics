"""
E20a -- Monte-Carlo basin fractions vs lambda (headline).

For each lam in a grid over [0.15, 0.90] (denser near the depinning window
[0.28,0.34]), draw K random constant-history initial overlap vectors a0 from three
documented families (near_pattern / mixture / random_small, see e20_common), then
integrate the reduced DDE (tau=10) to t=T_TOTAL and classify the final state as
memory / front / cycle / chaos / other. Reports basin FRACTIONS vs lam.

Parallelised over ICs with multiprocessing (each worker OMP_NUM_THREADS=2).
Results appended INCREMENTALLY to results/cycle/E20_RESULTS.md and saved to npz.

Run:  conda run -n mcmc_env python src/e20a_basin_fractions.py [K] [nproc]
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

LAM_GRID = [0.15, 0.20, 0.25, 0.28, 0.30, 0.31, 0.32, 0.325, 0.328,
            0.33, 0.34, 0.40, 0.60, 0.90]
T_TOTAL = 300.0
REC = 40
LABELS = ["memory", "front", "cycle", "chaos", "other"]

RES_MD = os.path.join(C.OUT, "E20_RESULTS.md")
NPZ = os.path.join(C.OUT, "E20a_basin_fractions.npz")

# ---- per-lam worker context (set once per pool via initializer) ----------
_CTX = {}

def _init_worker(lam, a_front, r_front):
    os.environ["OMP_NUM_THREADS"] = "2"
    _CTX["lam"] = lam
    _CTX["a_front"] = a_front
    _CTX["r_front"] = r_front
    _CTX["sysP"] = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)

def _run_one(a0):
    lab, info, _ = C.integrate_classify(
        _CTX["sysP"], a0, _CTX["a_front"], _CTX["lam"],
        T_TOTAL, C.DT, REC, r_front=_CTX["r_front"])
    return lab


def append_md(header=False, line=""):
    with open(RES_MD, "a") as f:
        if header:
            f.write(line)
        else:
            f.write(line + "\n")


def sweep(K=300, nproc=5, seed=20260705):
    if not os.path.exists(RES_MD):
        append_md(header=True, line=(
            "# E20 -- basin competition of the delayed mixed Hopfield network\n\n"
            f"Reduced DDE tau={C.TAU}, N={C.N}, alpha={C.ALPHA} (P={C.P}), beta={C.BETA}, "
            f"seed={C.SEED}, t_total={T_TOTAL}, dt={C.DT}.\n"
            "Sampling: 1/3 near_pattern (alpha*e_mu+noise), 1/3 mixture "
            "(2-4 consecutive patterns), 1/3 random_small (broad small-norm overlap).\n"
            "Classes: memory (recall on a strong bond) / front (weak bond 91-92, "
            "metastable) / cycle (ring advance) / chaos (non-stationary) / other.\n\n"
            "## E20a -- Monte-Carlo basin fractions vs lam\n\n"
            f"K={K} random ICs per lam.\n\n"
            "| lam | memory | front | cycle | chaos | other |\n"
            "|-----|--------|-------|-------|-------|-------|\n"))
    all_counts = {}
    frac_rows = []
    rng = np.random.default_rng(seed)
    t_start = time.time()
    for lam in LAM_GRID:
        a_front = C.reach_front(lam) if lam <= 0.3285 else np.zeros(C.P)
        r_front = C._front_shape_ratio(lam) if lam <= 0.3285 else 0.5
        a0s, kinds = C.sample_batch(rng, K)
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
        dt_lam = time.time() - t0
        line = (f"| {lam:.3f} | {fr['memory']:.3f} | {fr['front']:.3f} | "
                f"{fr['cycle']:.3f} | {fr['chaos']:.3f} | {fr['other']:.3f} |")
        append_md(line=line)
        print(f"[E20a] lam={lam:.3f}  mem={fr['memory']:.3f} front={fr['front']:.3f} "
              f"cyc={fr['cycle']:.3f} chaos={fr['chaos']:.3f} other={fr['other']:.3f} "
              f"({dt_lam:.0f}s)", flush=True)
        # incremental npz save
        np.savez(NPZ, lam_grid=np.array([r[0] for r in frac_rows]),
                 fractions=np.array(frac_rows)[:, 1:], labels=np.array(LABELS),
                 K=K, t_total=T_TOTAL, dt=C.DT, counts=json.dumps(all_counts))
    append_md(line=f"\n(E20a total {time.time()-t_start:.0f}s, K={K}, nproc={nproc})\n")
    print(f"\n[E20a] TOTAL {time.time()-t_start:.0f}s  ->  {NPZ}")
    print(f"[E20a] results appended to {RES_MD}")
    return frac_rows


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sweep(K=K, nproc=nproc)
