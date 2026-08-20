"""Is the usual system's recall a transient or an attractor? Resolve it in time.

The E27 dynamic control reports forward fraction 0.65 at c=0.4 on a 400 t.u.
window opened after 200 t.u.; E37 reports 0.19 on 4800 t.u. after 3200. The
reconciliation script showed the observable (coefficients vs physical overlaps)
explains none of the gap. The remaining candidate is the window itself: 400 t.u.
is 0.37 of a recall period, so E27 may be describing a transient sweep that later
arrests.

This integrates the SAME nudged initial condition far longer and reports the
forward fraction and the leader in consecutive windows, so an arrest -- if that is
what happens -- is visible as a time series rather than inferred.
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
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ACCELERATE_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from e35_dynamics import MhnReducedDDE, integrate_chunked
from e37_modern_k_markov import make_markov
from mhn_reduced import MhnContext, factorial_architectures

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/6_motifs_correles/data"

N, P, SEED, BETA, TAU, T0V, DT = 2000, 100, 42, 20.0, 10.0, 1.0, 0.01
WINDOW = 400.0                       # exactly E27's analysis window


def main() -> int:
    rows = []
    for c in (0.4, 0.6):
        for arm in ("JH_KH", "JP_KP"):
            xi = make_markov(N, P, c, SEED)
            ctx = MhnContext.from_patterns(xi, BETA)
            arch = factorial_architectures(ctx)[arm]
            sysP = MhnReducedDDE(arch, 0.9, TAU, T0V)
            a0 = np.zeros(P)
            a0[0], a0[1] = 0.99, 0.05          # E27's nudged start, verbatim
            o = integrate_chunked(sysP, a0, da_hist0=np.zeros(P), t_total=8000.0,
                                  dt=DT, record_every=20, chunk_time=500.0)
            t, a = o["t"], o["a"]
            m = ctx.physical_overlaps_batch(a)
            lead = np.argmax(m, axis=1)
            wins = []
            edges = np.arange(0.0, t[-1] + 1e-9, WINDOW)
            for lo, hi in zip(edges[:-1], edges[1:]):
                w = (t >= lo) & (t < hi)
                lw = lead[w]
                ch = np.flatnonzero(np.diff(lw) != 0)
                fwd = (float(np.mean((lw[ch + 1] - lw[ch]) % P == 1))
                       if ch.size else float("nan"))
                wins.append(dict(t_lo=float(lo), t_hi=float(hi),
                                 changes=int(ch.size), forward=fwd,
                                 n_distinct=int(len(np.unique(lw)))))
            rows.append(dict(c=c, arm=arm, windows=wins))
            head = ", ".join(f"{w['changes']}" for w in wins[:20])
            print(f"c={c} {arm}: lead changes per {WINDOW:g} t.u. window -> {head}",
                  flush=True)
            first_dead = next((w["t_lo"] for w in wins if w["changes"] == 0), None)
            print(f"    first window with NO lead change: "
                  f"{first_dead if first_dead is not None else 'none in 8000 t.u.'}",
                  flush=True)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    p = OUT / f"E37_arrest_in_time_{stamp}.json"
    p.write_text(json.dumps(dict(n=N, p=P, seed=SEED, lam=0.9, window=WINDOW,
                                 t_total=8000.0, rows=rows), indent=1,
                            default=float))
    print(f"wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
