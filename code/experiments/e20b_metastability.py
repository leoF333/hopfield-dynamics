"""
E20b -- metastability quantification (front vs memory).

For lam in [0.15, 0.32] where BOTH the memory xi^1 and the weak-bond pinned front
are stable fixed points (E18c), quantify how metastable the front is relative to
the memory:

(i)   relative basin-volume ratio memory:front from the E20a Monte-Carlo counts.
(ii)  SEPARATRIX location along the straight segment in overlap space from the
      front a_front to the nearest memory alpha*xi^mu:  a(s) = (1-s) a_front + s a_mem.
      Bisect s in (0,1) to the basin boundary s*: small s* (boundary close to the
      front) => the front basin is THIN in the memory direction => front metastable.
      Also bisect toward the pure-mixture / symmetric direction as a control.
(iii) relaxation-depth proxy: |eig_max(M)| of the instantaneous Jacobian at the
      front vs at the memory (deeper well = larger |eig|).

Assembles a metastability index (front_basin_fraction and s*) vs lam and shows the
front basin -> 0 as lam -> lam* while the memory basin persists.

Run:  conda run -n mcmc_env python src/e20b_metastability.py
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

from cycle_reduced import ReducedDDE
import e20_common as C

LAM_GRID = [0.15, 0.20, 0.25, 0.28, 0.30, 0.31, 0.32]
T_TOTAL = 300.0
REC = 40
NPZ = os.path.join(C.OUT, "E20b_metastability.npz")
RES_MD = os.path.join(C.OUT, "E20_RESULTS.md")
E20A_NPZ = os.path.join(C.OUT, "E20a_basin_fractions.npz")


def settle_label(sysP, a0, a_front, lam, r_front):
    """Integrate and return the coarse label (front/memory/cycle/chaos/other)."""
    lab, info, _ = C.integrate_classify(sysP, a0, a_front, lam, T_TOTAL, C.DT, REC,
                                        r_front=r_front)
    return lab


def bisect_separatrix(sysP, a_front, a_target, lam, r_front, iters=14):
    """Bisect s in [0,1] along a(s) = (1-s) a_front + s a_target to the basin
    boundary: s=0 stays on the front, s=1 (if it reaches a_target's basin) leaves.
    Returns s* (fraction of the way from front to target at which the trajectory
    STOPS returning to the front). If the whole segment stays in the front basin,
    returns 1.0 (front basin spans the segment)."""
    def in_front(s):
        a0 = (1 - s) * a_front + s * a_target
        return settle_label(sysP, a0, a_front, lam, r_front) == "front"
    if in_front(1.0):
        return 1.0, "front_spans"
    if not in_front(0.05):
        return 0.0, "front_thin"
    lo, hi = 0.05, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if in_front(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "bisected"


def main():
    print(f"[E20b] metastability of the weak-bond front vs memory "
          f"(reduced DDE tau={C.TAU}, t={T_TOTAL})\n")
    # basin fractions from E20a (if available)
    fr_map = {}
    if os.path.exists(E20A_NPZ):
        d = np.load(E20A_NPZ, allow_pickle=True)
        lg = d["lam_grid"]; fr = d["fractions"]; labs = list(d["labels"])
        im, ifr = labs.index("memory"), labs.index("front")
        for i, lam in enumerate(lg):
            fr_map[round(float(lam), 4)] = (float(fr[i, im]), float(fr[i, ifr]))

    rows = []
    t_start = time.time()
    for lam in LAM_GRID:
        a_front = C.reach_front(lam)
        r_front = C._front_shape_ratio(lam)
        a_mem = C.reach_memory(lam, 0)          # nearest memory (pattern xi^1)
        sysP = ReducedDDE(C.xi, C.BETA, lam, C.TAU, C.T0)

        # (iii) depth proxy
        eig_front = C.eigmax_at_amp(a_front, lam)
        eig_mem = C.eigmax_at_amp(a_mem, lam) if a_mem is not None else np.nan

        # (ii) separatrix toward the memory
        s_mem, tag_mem = bisect_separatrix(sysP, a_front, a_mem, lam, r_front)
        # control: toward the symmetric equal-mixture of the weak-bond pair
        b0, b1 = C.classify(
            {"a": a_front[None, :].repeat(2, 0)}, sysP, lam, a_front, r_front)[1]["bond"]
        a_sym = np.zeros(C.P)
        amp_sym = 0.5 * (abs(a_front[b0]) + abs(a_front[b1]))
        a_sym[b0] = amp_sym; a_sym[b1] = amp_sym   # equal 50/50 mixture direction
        s_sym, tag_sym = bisect_separatrix(sysP, a_front, a_sym, lam, r_front)

        mem_fr, front_fr = fr_map.get(round(lam, 4), (np.nan, np.nan))
        ratio = (mem_fr / front_fr) if (front_fr and front_fr > 0) else np.inf
        rows.append([lam, mem_fr, front_fr, ratio, s_mem, s_sym,
                     eig_front, eig_mem])
        print(f"[E20b] lam={lam:.3f}  s*(->mem)={s_mem:.3f} ({tag_mem})  "
              f"s*(->sym)={s_sym:.3f}  |eig|front={abs(eig_front):.3f} "
              f"|eig|mem={abs(eig_mem):.3f}  mem:front={ratio:.2f}", flush=True)
        np.savez(NPZ, rows=np.array(rows, float),
                 cols=np.array(["lam", "frac_mem", "frac_front", "ratio_mem_front",
                                "s_sep_mem", "s_sep_sym", "eig_front", "eig_mem"]),
                 t_total=T_TOTAL, dt=C.DT)

    # append to RESULTS.md
    with open(RES_MD, "a") as f:
        f.write("\n## E20b -- metastability of the weak-bond front vs memory\n\n")
        f.write("s* = fraction along the segment front->memory at which the front "
                "basin ends (small = thin front basin). |eig| = instantaneous "
                "eig_max(M) depth. mem:front = basin-volume ratio from E20a.\n\n")
        f.write("| lam | frac_mem | frac_front | mem:front | s*(->mem) | s*(->sym) "
                "| |eig|_front | |eig|_mem |\n")
        f.write("|-----|----------|-----------|-----------|-----------|-----------"
                "|-----------|----------|\n")
        for r in rows:
            f.write(f"| {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.2f} | "
                    f"{r[4]:.3f} | {r[5]:.3f} | {abs(r[6]):.3f} | {abs(r[7]):.3f} |\n")
    print(f"\n[E20b] TOTAL {time.time()-t_start:.0f}s  ->  {NPZ}")


if __name__ == "__main__":
    main()
