"""Bifurcation diagram of ONE bead: node, twin and their index-1 partners vs lambda.

The twin experiments (E34_twin_fold) showed that a bead carries a LADDER of
equilibria -- node, a saddle, a twin, another saddle, sometimes a third stable
state -- and that the ring connection is handed on at the end of that ladder.
Drawing it as a bifurcation diagram needs the unstable branches too.

Deflated Newton was tried first and does not deliver them: it was already
measured to diverge from stale seeds (residual 1e-1) in the ET1' diagnostics.
So each object is instead obtained ONCE, by the pipeline that is known to work
(arclength for the saddles, W^u shooting for the twin), and then CONTINUED in
lambda by warm-start Newton -- which is indifferent to stability, so saddles
continue exactly as nodes do. A branch simply ends where Newton stops
converging, and that endpoint is the fold.
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from cycle_reduced import ReducedDDE
from e34_lib import (
    NumpyLowRankCouplings, approach_memory_fold, continue_node_fold_saddle,
    exponential_history, extract_null_mode, gram_distance, gram_matrix,
    make_iid_patterns, memory_branch_identity, reduced_field_coefficients,
)
from e34_chain_closure import load_ladder, node_by_continuation
from reduced_spectrum import GJ_GK, rightmost_real_root
from robust_branch import eigmax_M, woodbury_newton

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _m_mu(coup, u, mu, beta):
    return float(coup.overlap_raw(np.tanh(beta * u))[mu])


def continue_branch(coup, Q, mu, u0, lam0, beta, step, lam_lo, lam_hi,
                    res_tol=1e-9):
    """Warm-start continuation of one object, both ways in lambda."""
    pts = []
    for direction in (+1, -1):
        u, lam = np.array(u0, float), lam0
        while lam_lo - 1e-12 <= lam <= lam_hi + 1e-12:
            u_new, _ = woodbury_newton(coup, u, lam, beta, tol=1e-12,
                                       max_iter=120)
            r = float(np.linalg.norm(coup.field_F(u_new, lam, beta))
                      / np.sqrt(coup.N))
            if not np.isfinite(r) or r > res_tol \
                    or not memory_branch_identity(coup, u_new, mu, Q=Q):
                break
            u = u_new
            pts.append(dict(lam=float(lam), m_mu=_m_mu(coup, u, mu, beta),
                            eigmax=float(eigmax_M(coup, u, lam, beta)),
                            residual=r,
                            a=reduced_field_coefficients(coup, u, Q)))
            lam += direction * step
    pts.sort(key=lambda d: d["lam"])
    return pts


def objects_at(coup, Q, mu, lam, beta, tau, t0, fold_hint, log):
    """node, saddle_1, twin, saddle_2 at one lambda (the validated pipeline)."""
    out = {}
    node_u, nres, route = node_by_continuation(coup, mu, lam, beta, Q, tau=tau,
                                               t0=t0)
    if node_u is None:
        return out
    out["node"] = node_u
    a_node = reduced_field_coefficients(coup, node_u, Q)
    P = coup.P
    S = np.roll(np.eye(P), 1, axis=0)
    sysP = ReducedDDE(coup._xi_np, beta, lam, tau, t0)

    def saddle_from(seed_u, lam_limit):
        near, near_lam, _ = approach_memory_fold(coup, mu, beta, seed_u, lam,
                                                 lam_limit=lam_limit,
                                                 eig_stop=1e9)
        pair = continue_node_fold_saddle(coup, mu, beta, seed_u, lam, near,
                                         near_lam)
        return pair.saddle

    lim = (float(fold_hint - 4e-3) if np.isfinite(fold_hint)
           and fold_hint - 4e-3 > lam else float(lam + 4e-3))
    u_s1 = saddle_from(node_u, lim)
    if u_s1 is None:
        return out
    out["saddle1"] = u_s1
    a_s1 = reduced_field_coefficients(coup, u_s1, Q)
    GJ, GK = GJ_GK(coup, u_s1, beta, lam)
    z1 = float(rightmost_real_root(GJ, GK, t0, tau, lam, x_hi=2.0, x_lo=-0.5,
                                   n=400).real)
    v1 = np.asarray(extract_null_mode(
        lambda z: (t0 * z + 1.0) * np.eye(P) - (1.0 - lam) * GJ
        - lam * np.exp(-z * tau) * (S @ GJ), z1, Q=Q, expect_real=True).vector,
        float)
    if v1[(mu + 1) % P] < 0:
        v1 = -v1
    _, hist, dhist = exponential_history(a_s1, z1, v1, 1e-3, tau, 0.01, Q=Q)
    # the budget must exceed a full ring tour (~1990 t.u. measured at bead 91),
    # otherwise a not-yet-converged return to the node passes for a twin and the
    # continuation then simply retraces the node branch.
    el, a_last = 0.0, None
    while el < 6000.0:
        o = sysP.integrate(hist, 250.0, 0.01, record_every=100, da_hist0=dhist)
        hist, dhist = o["hist"], o["dhist"]
        el += 250.0
        a_last = o["a"][-1]
        tail = o["a"][-20:]
        if max(float(gram_distance(tail[k], tail[-1], Q))
               for k in range(len(tail))) < 1e-10:
            break
    u_tw = coup._xi_np.T @ a_last
    if memory_branch_identity(coup, u_tw, mu, Q=Q) \
            and gram_distance(a_last, a_node, Q) > 1e-4 \
            and eigmax_M(coup, u_tw, lam, beta) < 0:
        out["twin"] = u_tw
        u_s2 = saddle_from(u_tw, float(lam + 1e-2))
        if u_s2 is not None:
            out["saddle2"] = u_s2
    log(f"[mu={mu} lam={lam:.6f}] objects: {sorted(out)}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=int, required=True)
    p.add_argument("--lam-ref", dest="lam_ref", type=float, required=True)
    p.add_argument("--lam-lo", dest="lam_lo", type=float, default=0.26)
    p.add_argument("--lam-hi", dest="lam_hi", type=float, default=0.32)
    p.add_argument("--step", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    args = p.parse_args(argv)

    fold, _, ladder = load_ladder(None)
    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    def log(msg):
        print(f"{_now()} {msg}", flush=True)

    t0w = time.perf_counter()
    objs = objects_at(coup, Q, args.mu, args.lam_ref, args.beta, args.tau,
                      args.t0, fold.get(args.mu, np.nan), log)
    if "node" not in objs:
        raise SystemExit("no node at the reference lambda")

    branches = {}
    for name, u0 in objs.items():
        pts = continue_branch(coup, Q, args.mu, u0, args.lam_ref, args.beta,
                              args.step, args.lam_lo, args.lam_hi)
        if not pts:
            continue
        lam_v = [q["lam"] for q in pts]
        branches[name] = dict(
            lam=lam_v, m_mu=[q["m_mu"] for q in pts],
            eigmax=[q["eigmax"] for q in pts],
            residual=[q["residual"] for q in pts],
            lam_min=min(lam_v), lam_max=max(lam_v),
            stable_fraction=float(np.mean([q["eigmax"] < 0 for q in pts])))
        log(f"branch {name:8s}: lam in [{min(lam_v):.5f}, {max(lam_v):.5f}], "
            f"{len(pts)} points, stable fraction "
            f"{branches[name]['stable_fraction']:.2f}, "
            f"m_mu {pts[0]['m_mu']:.4f} -> {pts[-1]['m_mu']:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = OUT / f"E34_branch_diagram_mu{args.mu}_{stamp}.json"
    path.write_text(json.dumps(dict(mu=args.mu, lam_ref=args.lam_ref,
                                    step=args.step, ladder=ladder,
                                    branches=branches), indent=1, default=float))
    log(f"wrote {path.name} [{time.perf_counter()-t0w:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
