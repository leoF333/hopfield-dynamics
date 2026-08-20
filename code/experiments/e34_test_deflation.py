"""Focused test of the deflated-Newton fix for the ET1' boundary object.

The ET1' smoke cell bisected cleanly but the polished boundary object came out
as the STABLE NODE (index 0). The boundary diagnostic showed the s* crossing is
a genuine node_mu | node_{mu+1} boundary (no chaotic sea), so an index-1 saddle
must exist there and the failure is solver-side. This script runs ONE edge-track
segment, then compares: plain Newton polish vs deflated Newton, certifying both,
and measures the distance to the INDEPENDENT arclength-through-fold saddle
(cross-check only, never used as a seed).
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
    approach_memory_fold,
    continue_node_fold_saddle,
    gram_distance,
    gram_matrix,
    make_iid_patterns,
    memory_branch_identity,
    reduced_field_coefficients,
    solve_memory_node,
)
from e34b_edge_tracking import (
    certify_equilibrium,
    deflated_newton,
    edge_track_segment,
    refine_saddle_seed,
    solve_alive_nodes,
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
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--P", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--t-settle", type=float, default=800.0)
    p.add_argument("--skip-arclength", action="store_true")
    args = p.parse_args(argv)

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
    lam = float(lam_c[args.mu] - args.delta)
    xi, xis = make_iid_patterns(args.N, args.P, args.seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    sysP = ReducedDDE(xi, args.beta, lam, args.tau, args.t0)
    res_of = lambda u: float(
        np.linalg.norm(coup.field_F(u, lam, args.beta)) / np.sqrt(coup.N))

    print(f"{_now()} deflation test  mu={args.mu} lam={lam:.6f} "
          f"(lam_c={lam_c[args.mu]:.6f})", flush=True)

    node_u, ok, nres = solve_memory_node(coup, args.mu, lam, args.beta)
    a_node = reduced_field_coefficients(coup, node_u, Q)
    a_nodes = solve_alive_nodes(coup, lam, args.beta, lam_c)
    a_nodes[int(args.mu)] = a_node
    succ = (args.mu + 1) % args.P
    a_far = a_nodes[succ]
    print(f"{_now()} node res={nres:.2e}; alive={len(a_nodes)}; far=node_{succ}",
          flush=True)

    t_edge = time.perf_counter()
    et = edge_track_segment(sysP, a_node, a_far, Q, a_nodes,
                            dt=0.01, tau=args.tau, t_max=args.t_settle)
    print(f"{_now()} edge: ok={et['ok']} s*={et['s_star']:.6e} "
          f"({time.perf_counter()-t_edge:.1f}s)", flush=True)
    if not et["ok"] or et["dwell"] is None:
        raise RuntimeError("edge tracking failed")

    seed = refine_saddle_seed(sysP, et["dwell"], a_node, Q, dt=0.01,
                              tau=args.tau, t_max=args.t_settle, rounds=3)
    u_seed = coup._xi_np.T @ seed
    print(f"{_now()} refined seed: d_G(seed,node)={gram_distance(seed,a_node,Q):.4f}",
          flush=True)

    rows = {}
    # (A) plain Newton
    u_a, ok_a = woodbury_newton(coup, u_seed, lam, args.beta, tol=1e-12,
                                max_iter=160)
    cert_a, _, _ = certify_equilibrium(coup, u_a, args.beta, lam, args.tau,
                                       args.t0, Q)
    a_a = reduced_field_coefficients(coup, u_a, Q)
    rows["plain"] = dict(res=res_of(u_a), n_unstable=int(cert_a["n_unstable"]),
                         eigmax=float(cert_a["eigmax"]),
                         d_to_node=float(gram_distance(a_a, a_node, Q)))
    print(f"{_now()} PLAIN   res={rows['plain']['res']:.2e} "
          f"n_unst={rows['plain']['n_unstable']} "
          f"eigmax={rows['plain']['eigmax']:+.4f} "
          f"d_G(node)={rows['plain']['d_to_node']:.4f}", flush=True)

    # (B) deflated Newton, deflating what plain Newton found (+ the node)
    defl = [u_a.copy()]
    if float(gram_distance(a_a, a_node, Q)) > 1e-9:
        defl.append(node_u.copy())
    t_d = time.perf_counter()
    u_b, ok_b = deflated_newton(coup, u_seed, lam, args.beta, defl, tol=1e-12)
    cert_b, _, _ = certify_equilibrium(coup, u_b, args.beta, lam, args.tau,
                                       args.t0, Q)
    a_b = reduced_field_coefficients(coup, u_b, Q)
    ident_b = memory_branch_identity(coup, u_b, args.mu, Q=Q)
    rows["deflated"] = dict(
        res=res_of(u_b), n_unstable=int(cert_b["n_unstable"]),
        eigmax=float(cert_b["eigmax"]), z_u=float(cert_b["z_u"]),
        z_u_imag=float(cert_b["z_u_imag"]),
        crosscheck=bool(cert_b["z_u_crosscheck_ok"]),
        contour_stable=bool(cert_b["contour_stable"]),
        identity=bool(ident_b),
        d_to_node=float(gram_distance(a_b, a_node, Q)),
        wall=time.perf_counter() - t_d)
    r = rows["deflated"]
    print(f"{_now()} DEFLATED res={r['res']:.2e} n_unst={r['n_unstable']} "
          f"eigmax={r['eigmax']:+.4f} z_u={r['z_u']:.5f} "
          f"crosscheck={r['crosscheck']} identity={r['identity']} "
          f"d_G(node)={r['d_to_node']:.4f} ({r['wall']:.1f}s)", flush=True)

    # (C) independent cross-check: arclength through the fold
    if not args.skip_arclength:
        t_arc = time.perf_counter()
        near_node, near_lam, _ = approach_memory_fold(
            coup, args.mu, args.beta, node_u, lam,
            lam_limit=float(lam_c[args.mu] - 4e-3), eig_stop=1e9)
        pair = continue_node_fold_saddle(coup, args.mu, args.beta, node_u, lam,
                                         near_node, near_lam)
        if pair.converged and pair.saddle is not None:
            a_arc = reduced_field_coefficients(coup, pair.saddle, Q)
            rows["arclength"] = dict(
                d_to_deflated=float(gram_distance(a_b, a_arc, Q)),
                d_to_plain=float(gram_distance(a_a, a_arc, Q)),
                d_to_node=float(gram_distance(a_arc, a_node, Q)),
                fold=float(pair.fold_parameter),
                wall=time.perf_counter() - t_arc)
            q = rows["arclength"]
            print(f"{_now()} ARCLENGTH saddle: d_G(deflated,arc)="
                  f"{q['d_to_deflated']:.3e}  d_G(plain,arc)={q['d_to_plain']:.4f}"
                  f"  d_G(arc,node)={q['d_to_node']:.4f}  fold={q['fold']:.6f}"
                  f"  ({q['wall']:.1f}s)", flush=True)
        else:
            rows["arclength"] = dict(converged=False)
            print(f"{_now()} ARCLENGTH did not converge", flush=True)

    verdict = ("FIX WORKS: deflation yields a certified index-1 saddle"
               if rows["deflated"]["n_unstable"] == 1 and
               rows["deflated"]["res"] < 1e-10 else
               "FIX INSUFFICIENT")
    print(f"{_now()} VERDICT: {verdict}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    (OUT / f"E34_deflation_test_{stamp}.json").write_text(json.dumps(
        dict(mu=args.mu, lam=lam, delta=args.delta, s_star=float(et["s_star"]),
             verdict=verdict, **rows), indent=1))
    print(f"{_now()} wrote {OUT / f'E34_deflation_test_{stamp}.json'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
