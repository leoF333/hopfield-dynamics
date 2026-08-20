"""E34-ET1' diagnostic: WHAT lies on the far side of the node_mu basin boundary?

The ET1' smoke cell (mu=0, delta=0.020, N=2000) bisected cleanly (s* stable in dt
and across segments) but the polished boundary object came out as the STABLE NODE
(index 0), not an index-1 saddle. Before patching the solver (deflated Newton),
this script tests the prior question: the bisection discriminant is only
"settles onto node_mu" vs "does not". If the not-A side falls into the chaotic
sea rather than onto node_{mu+1}, the tracked object is the edge of chaos, not
the necklace saddle -- and no index-1 equilibrium is expected there.

Samples the straight segment node_mu -> node_far, classifies each omega-limit,
and refines the classification around s*. Output: npz + printed table.
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

from cycle_reduced import ReducedDDE
from e34_lib import (
    NumpyLowRankCouplings,
    gram_distance,
    gram_matrix,
    make_iid_patterns,
    reduced_field_coefficients,
    solve_memory_node,
)
from e34b_edge_tracking import (
    classify_omega,
    integrate_to_settle,
    solve_alive_nodes,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / \
    "results/2_cycle_rappel_snic/data/E34"
DEFAULT_THR = Path(__file__).resolve().parents[1] / \
    "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chaos_diagnostics(a_traj, t, Q, tail_time=60.0):
    """Cheap non-stationarity descriptors on the trajectory tail."""
    mask = t >= (t[-1] - tail_time)
    tail = a_traj[mask]
    if len(tail) < 3:
        return dict(tail_spread=np.nan, n_macro=np.nan, winner_changes=np.nan)
    mean = tail.mean(axis=0)
    spread = max(float(gram_distance(x, mean, Q)) for x in tail)
    last = np.abs(tail[-1])
    n_macro = int(np.count_nonzero(last > 0.15))
    winners = np.argmax(np.abs(tail), axis=1)
    changes = int(np.count_nonzero(np.diff(winners) != 0))
    return dict(tail_spread=spread, n_macro=n_macro, winner_changes=changes)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.020)
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--P", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--s-star", type=float, default=0.229880)
    p.add_argument("--t-max", type=float, default=800.0)
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    with np.load(args.thresholds) as d:
        lam_c = np.asarray(d["lam_c"], float)
    lam = float(lam_c[args.mu] - args.delta)

    xi, xis = make_iid_patterns(args.N, args.P, args.seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    sysP = ReducedDDE(xi, args.beta, lam, args.tau, args.t0)

    print(f"{_now()} E34-ET1' boundary diagnostic  mu={args.mu} "
          f"lam={lam:.6f} (lam_c={lam_c[args.mu]:.6f}, delta={args.delta})",
          flush=True)

    node_u, ok, res = solve_memory_node(coup, args.mu, lam, args.beta)
    if not ok:
        raise RuntimeError("node solve failed")
    a_node = reduced_field_coefficients(coup, node_u, Q)
    a_nodes = solve_alive_nodes(coup, lam, args.beta, lam_c)
    a_nodes[int(args.mu)] = a_node
    succ = (args.mu + 1) % args.P
    if succ not in a_nodes:
        raise RuntimeError(f"successor node {succ} not alive at lam={lam}")
    a_far = a_nodes[succ]
    print(f"{_now()} node residual {res:.2e}; alive nodes: {len(a_nodes)}; "
          f"far = node_{succ}", flush=True)

    # sample: coarse scan + fine bracket around s*
    s_vals = sorted(set(
        [round(x, 6) for x in np.linspace(0.0, 1.0, 11)]
        + [round(args.s_star + off, 6) for off in
           (-2e-3, -5e-4, 5e-4, 2e-3, 1e-2, 3e-2, 8e-2)]))
    s_vals = [s for s in s_vals if 0.0 <= s <= 1.0]

    rows = []
    for s in s_vals:
        ic = (1.0 - s) * a_node + s * a_far
        t_w = time.perf_counter()
        out = integrate_to_settle(
            sysP, ic, Q, dt=0.01, tau=args.tau, t_max=args.t_max,
            record_every=25)
        label, d0, d1 = classify_omega(out["a"][-1], a_nodes, Q)
        diag = chaos_diagnostics(out["a"], out["t"], Q)
        rows.append(dict(
            s=float(s), label=str(label), stationary=bool(out["stationary"]),
            d_nearest=float(d0), d_second=float(d1),
            elapsed=float(out["elapsed"]),
            d_to_node_mu=float(gram_distance(out["a"][-1], a_node, Q)),
            d_to_node_far=float(gram_distance(out["a"][-1], a_far, Q)),
            wall=time.perf_counter() - t_w, **diag))
        r = rows[-1]
        print(f"{_now()}   s={s:.6f}  label={r['label']:>6}  "
              f"stat={r['stationary']:d}  d_mu={r['d_to_node_mu']:.4f}  "
              f"d_far={r['d_to_node_far']:.4f}  spread={r['tail_spread']:.4f}  "
              f"n_macro={r['n_macro']}  wchg={r['winner_changes']}  "
              f"({r['wall']:.1f}s)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    npz = args.output_dir / f"E34_ET1prime_boundary_diag_{run_id}.npz"
    np.savez_compressed(
        npz,
        mu=args.mu, lam=lam, lam_c=lam_c[args.mu], delta=args.delta,
        s=np.array([r["s"] for r in rows]),
        labels=np.array([r["label"] for r in rows]),
        stationary=np.array([r["stationary"] for r in rows]),
        d_to_node_mu=np.array([r["d_to_node_mu"] for r in rows]),
        d_to_node_far=np.array([r["d_to_node_far"] for r in rows]),
        tail_spread=np.array([r["tail_spread"] for r in rows]),
        n_macro=np.array([r["n_macro"] for r in rows]),
        winner_changes=np.array([r["winner_changes"] for r in rows]),
        alive_nodes=np.array(sorted(int(k) for k in a_nodes)),
        a_node=a_node, a_far=a_far, s_star=args.s_star)

    labels = [r["label"] for r in rows]
    far_side = [r for r in rows if r["s"] > args.s_star]
    n_far_node = sum(1 for r in far_side if r["label"] == str(succ))
    n_far_other = sum(1 for r in far_side if r["label"] == "other")
    summary = dict(
        mu=args.mu, lam=lam, s_star=args.s_star,
        n_samples=len(rows),
        labels_seen=sorted(set(labels)),
        far_side_node_far=n_far_node, far_side_other=n_far_other,
        verdict=("boundary node_mu | node_far" if n_far_other == 0 else
                 ("boundary node_mu | non-node (chaotic sea) "
                  if n_far_node == 0 else "mixed far side")),
        npz=str(npz))
    (args.output_dir / f"E34_ET1prime_boundary_diag_{run_id}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"{_now()} VERDICT: {summary['verdict']}  "
          f"(far side: {n_far_node} -> node_{succ}, {n_far_other} -> other)",
          flush=True)
    print(f"{_now()} npz={npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
