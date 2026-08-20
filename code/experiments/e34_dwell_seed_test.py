"""Which equilibrium sits on the node_mu | node_mu+1 boundary at N=2000?

The A-side profile shows a genuine slow plateau (speed ~1e-3 for t in [50,100])
at d_G(node)~0.14, but its closest approach to the arclength fold-partner saddle
is only 0.071 -- while the global minimum-speed rule used so far is captured by
the final settling onto the node (speed 1e-13), which is why the seed was bad.

This seeds Newton (plain and node-deflated) from several points ACROSS the
plateau, certifies each limit, and measures the distance to the independent
arclength saddle. Outcome decides between:
  (i) the boundary object IS the fold partner  -> necklace brick holds at N=2000;
  (ii) it is a DIFFERENT certified index-1 saddle -> the basin boundary is not
       the fold partner at this size (a substantive finding, not a solver bug).
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
    NumpyLowRankCouplings, approach_memory_fold, continue_node_fold_saddle,
    gram_distance, gram_matrix, make_iid_patterns, memory_branch_identity,
    reduced_field_coefficients, solve_memory_node,
)
from e34b_edge_tracking import (
    _speed_gram, certify_equilibrium, deflated_newton, edge_track_segment,
    integrate_to_settle, solve_alive_nodes,
)
from robust_branch import woodbury_newton

OUT = Path(__file__).resolve().parents[1] / "results/2_cycle_rappel_snic/data/E34"
THR = Path(__file__).resolve().parents[1] / \
    "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.020)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    args = p.parse_args(argv)

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
    lam = float(lam_c[args.mu] - args.delta)
    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    sysP = ReducedDDE(xi, args.beta, lam, args.tau, args.t0)
    res_of = lambda u: float(
        np.linalg.norm(coup.field_F(u, lam, args.beta)) / np.sqrt(coup.N))

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
    cert_arc, _, _ = certify_equilibrium(coup, pair.saddle, args.beta, lam,
                                         args.tau, args.t0, Q)
    print(f"{_now()} arclength saddle: d_G(node)={gram_distance(a_arc,a_node,Q):.4f} "
          f"n_unst={cert_arc['n_unstable']} res={res_of(pair.saddle):.2e}",
          flush=True)

    et = edge_track_segment(sysP, a_node, a_far, Q, a_nodes, dt=0.01,
                            tau=args.tau, t_max=800.0)
    print(f"{_now()} edge s*={et['s_star']:.8e}", flush=True)
    ic = (1.0 - et["s_lo"]) * a_node + et["s_lo"] * a_far
    out = integrate_to_settle(sysP, ic, Q, dt=0.01, tau=args.tau, t_max=400.0,
                              record_every=10)
    a, t = out["a"], out["t"]
    sp = _speed_gram(a, 0.1, Q)
    d_node = np.array([float(gram_distance(x, a_node, Q)) for x in a])

    # plateau window: far from the node (excludes the slow terminal settling)
    lo = int(np.searchsorted(t, 2.0 * args.tau))
    d_plateau = float(d_node[lo:].max())
    win = [i for i in range(lo, len(t)) if d_node[i] > 0.5 * d_plateau]
    i_best = min(win, key=lambda i: sp[i])
    cand_t = [float(t[i_best])] + [50.0, 60.0, 70.0, 80.0, 90.0]
    seen, cands = set(), []
    for tc in cand_t:
        i = int(np.argmin(np.abs(t - tc)))
        if i not in seen and i in win:
            seen.add(i)
            cands.append(i)
    print(f"{_now()} plateau d_max={d_plateau:.4f}, min-speed seed at "
          f"t={t[i_best]:.1f} (speed {sp[i_best]:.2e}, d_G(node)="
          f"{d_node[i_best]:.4f})", flush=True)

    rows = []
    for i in cands:
        seed_a = a[i]
        u_seed = coup._xi_np.T @ seed_a
        rec = dict(t=float(t[i]), speed=float(sp[i]),
                   d_node_seed=float(d_node[i]),
                   d_arc_seed=float(gram_distance(seed_a, a_arc, Q)))
        u_p, _ = woodbury_newton(coup, u_seed, lam, args.beta, tol=1e-12,
                                 max_iter=160)
        c_p, _, _ = certify_equilibrium(coup, u_p, args.beta, lam, args.tau,
                                        args.t0, Q)
        a_p = reduced_field_coefficients(coup, u_p, Q)
        rec.update(plain_res=res_of(u_p), plain_nunst=int(c_p["n_unstable"]),
                   plain_d_node=float(gram_distance(a_p, a_node, Q)),
                   plain_d_arc=float(gram_distance(a_p, a_arc, Q)))
        u_d, _ = deflated_newton(coup, u_seed, lam, args.beta, [node_u],
                                 tol=1e-12)
        c_d, _, _ = certify_equilibrium(coup, u_d, args.beta, lam, args.tau,
                                        args.t0, Q)
        a_d = reduced_field_coefficients(coup, u_d, Q)
        rec.update(defl_res=res_of(u_d), defl_nunst=int(c_d["n_unstable"]),
                   defl_eigmax=float(c_d["eigmax"]), defl_zu=float(c_d["z_u"]),
                   defl_d_node=float(gram_distance(a_d, a_node, Q)),
                   defl_d_arc=float(gram_distance(a_d, a_arc, Q)),
                   defl_identity=bool(memory_branch_identity(
                       coup, u_d, args.mu, Q=Q)))
        rows.append(rec)
        print(f"{_now()}  t={rec['t']:6.1f} | plain: res={rec['plain_res']:.1e} "
              f"n={rec['plain_nunst']:2d} d_node={rec['plain_d_node']:.4f} "
              f"d_arc={rec['plain_d_arc']:.4f} | defl: res={rec['defl_res']:.1e} "
              f"n={rec['defl_nunst']:2d} d_node={rec['defl_d_node']:.4f} "
              f"d_arc={rec['defl_d_arc']:.4f}", flush=True)

    hits_arc = [r for r in rows if r["defl_res"] < 1e-10
                and r["defl_nunst"] == 1 and r["defl_d_arc"] < 1e-6]
    hits_other = [r for r in rows if r["defl_res"] < 1e-10
                  and r["defl_nunst"] == 1 and r["defl_d_arc"] >= 1e-6]
    verdict = ("boundary object IS the fold partner" if hits_arc else
               ("boundary object is a DIFFERENT certified index-1 saddle"
                if hits_other else "no certified index-1 object recovered"))
    print(f"{_now()} VERDICT: {verdict} "
          f"(arc-hits={len(hits_arc)}, other-hits={len(hits_other)}, "
          f"n_seeds={len(rows)})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    (OUT / f"E34_dwell_seed_test_{stamp}.json").write_text(json.dumps(
        dict(mu=args.mu, lam=lam, delta=args.delta, verdict=verdict,
             d_arc_node=float(gram_distance(a_arc, a_node, Q)),
             s_star=float(et["s_star"]), rows=rows), indent=1))
    print(f"{_now()} wrote E34_dwell_seed_test_{stamp}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
