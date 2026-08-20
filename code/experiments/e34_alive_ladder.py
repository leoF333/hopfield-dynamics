"""Measured survival threshold of every bead, by WARM-START continuation.

Why this exists. The E24 threshold table is early for at least 33/100 beads
(E34_threshold_table_audit): a stable equilibrium carrying identity mu still
exists above the tabulated lam_c. The audit could not say why, because two
readings fit -- the table is early, or the branch is S-shaped and the table
records the FIRST fold. Distinguishing them by full pseudo-arclength on 100
beads costs ~4 h. This does it in minutes and, unlike the audit, it also
answers the "why":

walk lambda upward on a fine grid from lam_lo (where every bead is a stable
memory) with the previous solution as the Newton seed, and record the FULL
stability pattern of the branch -- the list of lambda intervals on which a
stable equilibrium with identity mu exists. Then

  * one interval ending at lam_survive > lam_c(table)  -> table EARLY
  * two or more intervals (stable, unstable, stable again)
                                                       -> S-SHAPED branch
  * one interval ending at lam_c(table)                -> table consistent

Warm start is what makes this trustworthy where Newton-from-pattern fails: the
seed is the previous point on the branch, not the raw pattern, so the solver
tracks the sheet instead of jumping to it. Past the fold, Newton either fails or
lands on an unstable object -- both are detected by requiring eigmax_M < 0, so
the last stable grid point brackets the fold to one step.

Also emits, for any requested lambda, the CORRECTED alive set -- the census
`solve_alive_nodes` cannot produce because it reads the table.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from e34_lib import (
    NumpyLowRankCouplings, gram_matrix, make_iid_patterns,
    memory_branch_identity, solve_memory_node,
)
from robust_branch import eigmax_M, woodbury_newton

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"
THR = ROOT / "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_here(coup, u, mu, lam, beta, Q, res_tol=1e-9):
    """Is u a stable equilibrium carrying identity mu at lam?"""
    res = float(np.linalg.norm(coup.field_F(u, lam, beta)) / np.sqrt(coup.N))
    if not np.isfinite(res) or res > res_tol:
        return False, res, np.nan, False
    ident = bool(memory_branch_identity(coup, u, mu, Q=Q))
    em = float(eigmax_M(coup, u, lam, beta))
    return bool(ident and em < 0.0), res, em, ident


def ladder(coup, Q, mu, beta, lam_lo, lam_hi, step, refine, res_tol=1e-9):
    """Walk lambda up with warm starts; return the stability pattern."""
    u, ok, res = solve_memory_node(coup, mu, lam_lo, beta)
    if not ok or res > res_tol:
        return dict(mu=int(mu), status="seed_failed", seed_residual=float(res))
    intervals, cur = [], None
    lam, last_stable = lam_lo, None
    pts = []
    while lam <= lam_hi + 1e-12:
        u_new, _ = woodbury_newton(coup, u, lam, beta, tol=1e-12, max_iter=80)
        good, res, em, ident = _stable_here(coup, u_new, mu, lam, beta, Q, res_tol)
        if good:
            u = u_new                      # advance the warm start on the branch
            last_stable = lam
            if cur is None:
                cur = [lam, lam]
            else:
                cur[1] = lam
            pts.append((float(lam), float(em)))
        else:
            # second route before declaring the window closed: seed from the
            # PATTERN. A stable sheet lying above the primary fold is often
            # unreachable from a stale warm start (its basin is elsewhere) yet
            # reachable from the pattern -- bead 3 at lam=0.2988 is the measured
            # example. Missing it would under-count the fragmentation.
            u_alt, ok_alt, res_alt = solve_memory_node(coup, mu, lam, beta)
            good_alt = False
            if ok_alt and res_alt <= res_tol:
                good_alt, res, em, ident = _stable_here(
                    coup, u_alt, mu, lam, beta, Q, res_tol)
            if good_alt:
                u, last_stable = u_alt, lam
                if cur is None:
                    cur = [lam, lam]
                else:
                    cur[1] = lam
                pts.append((float(lam), float(em)))
                lam += step
                continue
            if cur is not None:
                intervals.append(cur)
                cur = None
            # do NOT advance the warm start through a non-stable point: keep the
            # last stable solution as the seed so the walk can re-enter a second
            # stable sheet without having drifted onto the unstable one.
        lam += step
    if cur is not None:
        intervals.append(cur)

    # refine the top edge of the last stable interval to `refine`
    lam_survive = float(intervals[-1][1]) if intervals else np.nan
    if intervals and refine < step:
        u_ref, lam_ref = None, lam_survive
        # re-seed at the last stable grid point
        u_ref, _ = woodbury_newton(coup, u, lam_ref, beta, tol=1e-12, max_iter=80)
        lam_try = lam_ref + refine
        while lam_try <= lam_ref + step + 1e-12:
            u_new, _ = woodbury_newton(coup, u_ref, lam_try, beta, tol=1e-12,
                                       max_iter=80)
            good, _, _, _ = _stable_here(coup, u_new, mu, lam_try, beta, Q,
                                         res_tol)
            if not good:
                break
            u_ref, lam_survive = u_new, lam_try
            lam_try += refine
    return dict(mu=int(mu), status="ok",
                intervals=[[float(a), float(b)] for a, b in intervals],
                n_intervals=len(intervals), lam_survive=float(lam_survive),
                eig_trace=pts[-4:])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lam-lo", dest="lam_lo", type=float, default=0.150)
    p.add_argument("--lam-hi", dest="lam_hi", type=float, default=0.360)
    p.add_argument("--step", type=float, default=1e-3)
    p.add_argument("--refine", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--beads", default="all")
    p.add_argument("--census-at", dest="census_at", default="",
                   help="comma-separated lambdas: emit the corrected alive set")
    args = p.parse_args(argv)

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
        N, P, seed = int(d["N"]), int(d["P"]), int(d["seed"])
    xi, xis = make_iid_patterns(N, P, seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    beads = (list(range(P)) if args.beads == "all"
             else [int(x) for x in args.beads.split(",")])

    print(f"{_now()} ladder lam {args.lam_lo}->{args.lam_hi} step {args.step} "
          f"refine {args.refine}; {len(beads)} beads", flush=True)
    rows, t0w = [], time.perf_counter()
    for k, mu in enumerate(beads):
        r = ladder(coup, Q, mu, args.beta, args.lam_lo, args.lam_hi,
                   args.step, args.refine)
        r["lam_c_table"] = float(lam_c[mu])
        if r["status"] == "ok":
            r["offset"] = r["lam_survive"] - float(lam_c[mu])
            r["reading"] = ("S_shaped_branch" if r["n_intervals"] > 1
                            else ("table_early" if r["offset"] > 2 * args.step
                                  else ("table_late" if r["offset"] < -2 * args.step
                                        else "table_consistent")))
        rows.append(r)
        if k % 10 == 0 or r.get("reading") == "S_shaped_branch":
            print(f"{_now()}  mu={mu:3d} table={lam_c[mu]:.6f} "
                  f"survive={r.get('lam_survive', float('nan')):.6f} "
                  f"offset={r.get('offset', float('nan')):+.2e} "
                  f"n_int={r.get('n_intervals')} {r.get('reading','')} "
                  f"[{time.perf_counter()-t0w:.0f}s]", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    from collections import Counter
    tally = Counter(r["reading"] for r in ok)
    print(f"\n{_now()} READINGS over {len(ok)} beads: {dict(tally)}", flush=True)
    off = np.array([r["offset"] for r in ok])
    print(f"{_now()} offset (survive - table): median {np.median(off):+.2e}, "
          f"max {off.max():+.2e} (mu={ok[int(np.argmax(off))]['mu']}), "
          f"min {off.min():+.2e}", flush=True)
    sshape = [r["mu"] for r in ok if r["reading"] == "S_shaped_branch"]
    print(f"{_now()} S-shaped beads ({len(sshape)}): {sshape}", flush=True)

    census = {}
    for s in [x for x in args.census_at.split(",") if x.strip()]:
        lam = float(s)
        alive = sorted(r["mu"] for r in ok if any(a <= lam <= b
                                                 for a, b in r["intervals"]))
        alive_table = sorted(int(m) for m in range(P)
                             if np.isfinite(lam_c[m]) and lam_c[m] > lam + 3e-3)
        census[s] = dict(alive=alive, n_alive=len(alive),
                         alive_from_table=alive_table,
                         n_from_table=len(alive_table),
                         missed_by_table=sorted(set(alive) - set(alive_table)),
                         false_in_table=sorted(set(alive_table) - set(alive)))
        print(f"{_now()} lam={lam:.6f}: corrected alive={len(alive)} "
              f"(table said {len(alive_table)}); table MISSED {census[s]['missed_by_table']}; "
              f"table WRONGLY kept {census[s]['false_in_table']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = OUT / f"E34_alive_ladder_{stamp}.json"
    path.write_text(json.dumps(
        dict(N=N, P=P, seed=seed, beta=args.beta, lam_lo=args.lam_lo,
             lam_hi=args.lam_hi, step=args.step, refine=args.refine,
             readings=dict(tally), s_shaped=sshape, census=census, rows=rows),
        indent=1, default=float))
    print(f"{_now()} wrote {path.name} ({time.perf_counter()-t0w:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
