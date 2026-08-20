"""Speed/distance profile of an edge-tracking A-side trajectory (ET1' seed bug).

The refined seed came out at d_G=0.021 from node_mu while the true (arclength)
saddle sits at d_G=0.134: the global minimum-speed rule is captured by the slow
final settling onto the node instead of the saddle dwell. This prints the full
profile so the dwell criterion can be fixed on evidence: expected structure is
decrease (approach along W^s) -> local min AT THE SADDLE -> local max (peel-off
along W^u) -> monotone decrease onto the node.
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
import os
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from pathlib import Path

from cycle_reduced import ReducedDDE
from e34_lib import (
    NumpyLowRankCouplings, approach_memory_fold, continue_node_fold_saddle,
    gram_distance, gram_matrix, make_iid_patterns, reduced_field_coefficients,
    solve_memory_node,
)
from e34b_edge_tracking import (
    _speed_gram, edge_track_segment, integrate_to_settle, solve_alive_nodes,
)

THR = Path(__file__).resolve().parents[1] / \
    "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.020)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--t-max", type=float, default=400.0)
    args = p.parse_args(argv)

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
    lam = float(lam_c[args.mu] - args.delta)
    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    sysP = ReducedDDE(xi, args.beta, lam, args.tau, args.t0)

    node_u, _, _ = solve_memory_node(coup, args.mu, lam, args.beta)
    a_node = reduced_field_coefficients(coup, node_u, Q)
    a_nodes = solve_alive_nodes(coup, lam, args.beta, lam_c)
    a_nodes[int(args.mu)] = a_node
    a_far = a_nodes[(args.mu + 1) % 100]

    near_node, near_lam, _ = approach_memory_fold(
        coup, args.mu, args.beta, node_u, lam,
        lam_limit=float(lam_c[args.mu] - 4e-3), eig_stop=1e9)
    pair = continue_node_fold_saddle(coup, args.mu, args.beta, node_u, lam,
                                     near_node, near_lam)
    a_arc = reduced_field_coefficients(coup, pair.saddle, Q)
    print(f"reference: d_G(arclength saddle, node) = "
          f"{gram_distance(a_arc, a_node, Q):.4f}", flush=True)

    et = edge_track_segment(sysP, a_node, a_far, Q, a_nodes, dt=0.01,
                            tau=args.tau, t_max=800.0)
    s_star = et["s_star"]
    print(f"edge s*={s_star:.8e}", flush=True)

    # A-side trajectory just inside the basin, recorded densely
    s_A = et["s_lo"]
    ic = (1.0 - s_A) * a_node + s_A * a_far
    rec_every = 10                       # 0.1 t.u. at dt=0.01
    out = integrate_to_settle(sysP, ic, Q, dt=0.01, tau=args.tau,
                              t_max=args.t_max, record_every=rec_every)
    a, t = out["a"], out["t"]
    sp = _speed_gram(a, 0.1, Q)
    d_node = np.array([float(gram_distance(x, a_node, Q)) for x in a])
    d_arc = np.array([float(gram_distance(x, a_arc, Q)) for x in a])

    print(f"\n{'t':>8} {'speed':>11} {'d_G(node)':>10} {'d_G(saddle)':>11}",
          flush=True)
    step = max(1, len(t) // 60)
    for i in range(0, len(t), step):
        print(f"{t[i]:8.1f} {sp[i]:11.3e} {d_node[i]:10.4f} {d_arc[i]:11.4f}",
              flush=True)

    i_close = int(np.argmin(d_arc))
    print(f"\nclosest approach to the arclength saddle: d_G={d_arc[i_close]:.4f} "
          f"at t={t[i_close]:.1f} (speed {sp[i_close]:.3e}, "
          f"d_G(node)={d_node[i_close]:.4f})", flush=True)
    i_gmin = int(np.argmin(sp[3:-3])) + 3
    print(f"global min-speed point:  t={t[i_gmin]:.1f} speed={sp[i_gmin]:.3e} "
          f"d_G(node)={d_node[i_gmin]:.4f} d_G(saddle)={d_arc[i_gmin]:.4f}",
          flush=True)
    # local maxima of speed after the initial transient
    lo = int(np.searchsorted(t, 2.0 * args.tau))
    loc_max = [i for i in range(lo + 1, len(sp) - 1)
               if sp[i] >= sp[i - 1] and sp[i] > sp[i + 1]]
    print(f"local speed maxima after 2tau: "
          f"{[(round(float(t[i]),1), float(f'{sp[i]:.2e}')) for i in loc_max[:8]]}",
          flush=True)
    if loc_max:
        jmax = loc_max[-1]
        win = range(lo, jmax)
        jmin = min(win, key=lambda i: sp[i])
        print(f"LAST-local-max rule -> dwell at t={t[jmin]:.1f} "
              f"speed={sp[jmin]:.3e} d_G(node)={d_node[jmin]:.4f} "
              f"d_G(saddle)={d_arc[jmin]:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
