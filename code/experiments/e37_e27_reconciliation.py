"""Why E37's baseline forward fraction differs from E27's, at the same c.

E27's dynamic control reads the leader from the reduced COEFFICIENTS a, on a
400 t.u. window, from a nudged initial condition. E35 froze the rule that for
correlated patterns the cycle events must be read from the PHYSICAL overlap
m = xi tanh(beta u)/N, and E37 uses a window of several full periods.

This runs E27's protocol verbatim and then re-reads the SAME trajectory with the
physical observable, so the gap is attributed to the observable rather than left
as an unexplained disagreement between two experiments.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import json
import os
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ACCELERATE_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from cycle_reduced import ReducedDDE
from e37_modern_k_markov import make_markov
from mhn_reduced import MhnContext

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/6_motifs_correles/data"

N, P, SEED, BETA, TAU, T0V, DT = 2000, 100, 42, 20.0, 10.0, 1.0, 0.01
CS = [0.0, 0.2, 0.4, 0.6, 0.8]


def handoffs(lead: np.ndarray) -> tuple[int, int, float]:
    chg = np.nonzero(np.diff(lead) != 0)[0]
    adv = int(np.sum((lead[chg + 1] - lead[chg]) % P == 1))
    return len(chg), adv, adv / max(len(chg), 1)


def main() -> int:
    rows = []
    for c in CS:
        xi = make_markov(N, P, c, SEED)
        ctx = MhnContext.from_patterns(xi, BETA)
        sysP = ReducedDDE(xi, BETA, 0.9, TAU, T0V)
        a0 = np.zeros(P)
        a0[0], a0[1] = 0.99, 0.05                    # E27's nudged start
        sol = sysP.integrate(a0, 600.0, DT, record_every=20)
        A = sol["a"][int(200 / (DT * 20)):]          # E27's 400 t.u. window
        n_a, f_a, frac_a = handoffs(np.argmax(A, axis=1))
        M = ctx.physical_overlaps_batch(A)
        n_m, f_m, frac_m = handoffs(np.argmax(M, axis=1))
        rows.append(dict(c=c, changes_a=n_a, forward_a=f_a, frac_a=frac_a,
                         changes_m=n_m, forward_m=f_m, frac_m=frac_m,
                         cover_a=len(np.unique(np.argmax(A, axis=1))) / P,
                         cover_m=len(np.unique(np.argmax(M, axis=1))) / P))
        print(f"c={c:.1f}  leader from a: {n_a:3d} changes, forward "
              f"{frac_a:.2f}, coverage {rows[-1]['cover_a']:.2f}   |   "
              f"leader from m: {n_m:3d} changes, forward {frac_m:.2f}, "
              f"coverage {rows[-1]['cover_m']:.2f}", flush=True)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    p = OUT / f"E37_e27_reconciliation_{stamp}.json"
    p.write_text(json.dumps(dict(
        note="E27 protocol verbatim; same trajectory read with both observables",
        n=N, p=P, seed=SEED, lam=0.9, t_total=600.0, window=400.0,
        generated=datetime.now(timezone.utc).isoformat(), rows=rows),
        indent=1, default=float))
    print(f"wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
