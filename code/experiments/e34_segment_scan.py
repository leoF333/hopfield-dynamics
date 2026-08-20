"""Are the basins of two beads ADJACENT, when the fold partner says otherwise?

Measured in E34_circle_traj: for a bead whose brick "breaks", the node, the
fold-partner saddle and the twin are nearly COLLINEAR in the Gram metric
(d(node,saddle) + d(saddle,twin) exceeds d(node,twin) by only 3-7%), with the
saddle strictly between the other two, and the twin carries the same overlap
pattern as the node with uniformly larger cross-talk. So that saddle is the
separatrix between the memory and its OWN twin -- a local bistability on the
memory branch -- not the separatrix toward the next bead.

If so, a broken brick does not prove the necklace is interrupted: it proves the
arclength returned a different saddle than the one the necklace needs. This
tests the premise directly and without any saddle: sample the straight segment
node_mu -> node_nu in the reduced space, integrate each sample to its omega-
limit, and see whether both basins are met. If they are, the two basins are
adjacent and some index-1 saddle mediates the connection -- just not the fold
partner.
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
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from cycle_reduced import ReducedDDE
from e34_lib import (
    NumpyLowRankCouplings, gram_distance, gram_matrix, make_iid_patterns,
    memory_branch_identity, reduced_field_coefficients,
)
from e34_chain_closure import node_by_continuation
from robust_branch import eigmax_M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def settle(sysP, a0, Q, dt, tau, t_max, chunk=250.0):
    hist, elapsed = a0, 0.0
    a_last = a0
    while elapsed < t_max - 1e-9:
        step = min(chunk, t_max - elapsed)
        o = sysP.integrate(hist, step, dt, record_every=int(round(1.0 / dt)))
        hist = o["hist"]
        elapsed += step
        a_last = o["a"][-1]
        tail = o["a"][-int(round(2.0 * tau)):]
        if max(float(gram_distance(tail[k], tail[-1], Q))
               for k in range(len(tail))) < 1e-10:
            return a_last, elapsed, True
    return a_last, elapsed, False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--mu", type=int, required=True)
    p.add_argument("--nu", type=int, required=True)
    p.add_argument("--n-samples", dest="n", type=int, default=15)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--t-max", dest="t_max", type=float, default=2500.0)
    args = p.parse_args(argv)

    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    sysP = ReducedDDE(xi, args.beta, args.lam, args.tau, args.t0)

    nodes = {}
    for k in (args.mu, args.nu):
        u, r, route = node_by_continuation(coup, k, args.lam, args.beta, Q,
                                           tau=args.tau, t0=args.t0)
        if u is None:
            raise SystemExit(f"node_{k} not found at lam={args.lam}")
        nodes[k] = reduced_field_coefficients(coup, u, Q)
        print(f"{_now()} node_{k}: res={r:.1e} route={route}", flush=True)
    a_mu, a_nu = nodes[args.mu], nodes[args.nu]
    print(f"{_now()} d_G(node_{args.mu}, node_{args.nu}) = "
          f"{gram_distance(a_mu, a_nu, Q):.4f}", flush=True)

    rows = []
    for s in np.linspace(0.0, 1.0, args.n):
        a0 = (1.0 - s) * a_mu + s * a_nu
        a_f, el, stat = settle(sysP, a0, Q, args.dt, args.tau, args.t_max)
        u_f = coup._xi_np.T @ a_f
        m = np.asarray(coup.overlap_raw(np.tanh(args.beta * u_f)), float)
        lead = int(np.argmax(np.abs(m)))
        res = float(np.linalg.norm(coup.field_F(u_f, args.lam, args.beta))
                    / np.sqrt(coup.N))
        d_mu = float(gram_distance(a_f, a_mu, Q))
        d_nu = float(gram_distance(a_f, a_nu, Q))
        kind = ("node_mu" if d_mu < 1e-6 else
                "node_nu" if d_nu < 1e-6 else
                ("twin_of_%d" % lead if res < 1e-8 and
                 memory_branch_identity(coup, u_f, lead, Q=Q) else
                 ("other_equilibrium" if res < 1e-8 else "not_stationary")))
        rec = dict(s=float(s), lead=lead, lead_m=float(m[lead]), residual=res,
                   eigmax=float(eigmax_M(coup, u_f, args.lam, args.beta))
                   if res < 1e-8 else float("nan"),
                   d_mu=d_mu, d_nu=d_nu, stationary=bool(stat),
                   elapsed=float(el), kind=kind)
        rows.append(rec)
        print(f"{_now()}  s={s:.4f} -> {kind:18s} lead={lead:3d} "
              f"m={m[lead]:+.4f} d({args.mu})={d_mu:.4f} d({args.nu})={d_nu:.4f} "
              f"res={res:.1e} t={el:.0f}", flush=True)

    kinds = [r["kind"] for r in rows]
    reach_nu = kinds.count("node_nu")
    verdict = ("basins ADJACENT: the segment meets both nodes"
               if reach_nu and "node_mu" in kinds else
               f"node_{args.nu} NOT reached from this segment "
               f"(kinds seen: {sorted(set(kinds))})")
    print(f"{_now()} VERDICT {verdict}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = OUT / f"E34_segment_lam{args.lam:.6f}_mu{args.mu}_nu{args.nu}_{stamp}.json"
    path.write_text(json.dumps(dict(lam=args.lam, mu=args.mu, nu=args.nu,
                                    verdict=verdict, rows=rows), indent=1,
                               default=float))
    print(f"{_now()} wrote {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
