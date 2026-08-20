"""Does the necklace close THROUGH the twins?

Measured in E34_circle_traj: when a brick "breaks", node_mu, its fold-partner
saddle and the twin are nearly collinear, the saddle strictly between the other
two. So that saddle is the node<->twin separatrix, and the ring connection --
which E34_segment_scan shows DOES exist, the basins being adjacent -- must be
carried by some other saddle.

The natural candidate: the twin is itself a stable equilibrium, so it has its
own fold partner. If the forward branch of THAT saddle reaches the next bead,
the necklace is not broken at all -- it simply has more beads than patterns:

    node_mu -> saddle_1 -> twin_mu -> saddle_2 -> node_next

This takes the twin (obtained dynamically, as the forward landing of the
node<->twin saddle), continues it by arclength to its own fold, certifies the
partner's index, and shoots both branches.
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
    NumpyLowRankCouplings, approach_memory_fold, continue_node_fold_saddle,
    exponential_history, extract_null_mode, gram_distance, gram_matrix,
    make_iid_patterns, memory_branch_identity, reduced_field_coefficients,
)
from e34_chain_closure import load_ladder, node_by_continuation
from e34b_edge_tracking import certify_equilibrium
from reduced_spectrum import GJ_GK, rightmost_real_root
from robust_branch import eigmax_M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _m(coup, a, beta):
    return np.asarray(coup.overlap_raw(np.tanh(beta * (coup._xi_np.T @ a))),
                      float)


def saddle_of(coup, Q, mu, lam, seed_u, lam_limit, beta):
    near_node, near_lam, _ = approach_memory_fold(
        coup, mu, beta, seed_u, lam, lam_limit=lam_limit, eig_stop=1e9)
    pair = continue_node_fold_saddle(coup, mu, beta, seed_u, lam, near_node,
                                     near_lam)
    if pair.saddle is None:
        return None, pair
    return pair.saddle, pair


def shoot(sysP, coup, Q, a_s, z_u, v, sign, eps, beta, tau, dt, t_max):
    _, hist, dhist = exponential_history(a_s, z_u, v, sign * eps, tau, dt, Q=Q)
    elapsed, a_last, stat = 0.0, None, False
    while elapsed < t_max - 1e-9:
        step = min(500.0, t_max - elapsed)
        o = sysP.integrate(hist, step, dt, record_every=int(round(1.0 / dt)),
                           da_hist0=dhist)
        hist, dhist = o["hist"], o["dhist"]
        elapsed += step
        a_last = o["a"][-1]
        tail = o["a"][-int(round(2.0 * tau)):]
        if max(float(gram_distance(tail[k], tail[-1], Q))
               for k in range(len(tail))) < 1e-10:
            stat = True
            break
    return a_last, elapsed, stat


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--mu", type=int, required=True)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--t-wu", dest="t_wu", type=float, default=4000.0)
    p.add_argument("--probe", type=float, default=1e-2,
                   help="upward lambda probe for the twin's own fold")
    args = p.parse_args(argv)

    fold, intervals, ladder_name = load_ladder(None)
    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    P = coup.P
    lam, mu = args.lam, args.mu
    sysP = ReducedDDE(xi, args.beta, lam, args.tau, args.t0)
    rec = dict(lam=lam, mu=mu, ladder=ladder_name)

    # -- 1. node, its fold partner, and the twin it flows to ---------------
    node_u, nres, route = node_by_continuation(coup, mu, lam, args.beta, Q,
                                               tau=args.tau, t0=args.t0)
    if node_u is None:
        raise SystemExit(f"node_{mu} not found at lam={lam}")
    a_node = reduced_field_coefficients(coup, node_u, Q)
    fm = fold.get(mu, np.nan)
    lim1 = (float(fm - 4e-3) if np.isfinite(fm) and fm - 4e-3 > lam
            else float(lam + 4e-3))
    u_s1, pair1 = saddle_of(coup, Q, mu, lam, node_u, lim1, args.beta)
    if u_s1 is None:
        raise SystemExit(f"arclength from the node failed: {pair1.reason}")
    a_s1 = reduced_field_coefficients(coup, u_s1, Q)
    GJ, GK = GJ_GK(coup, u_s1, args.beta, lam)
    z1 = float(rightmost_real_root(GJ, GK, args.t0, args.tau, lam, x_hi=2.0,
                                   x_lo=-0.5, n=400).real)
    S = np.roll(np.eye(P), 1, axis=0)
    v1 = np.asarray(extract_null_mode(
        lambda z: (args.t0 * z + 1.0) * np.eye(P) - (1.0 - lam) * GJ
        - lam * np.exp(-z * args.tau) * (S @ GJ), z1, Q=Q,
        expect_real=True).vector, float)
    if v1[(mu + 1) % P] < 0:
        v1 = -v1
    a_twin, el, stat = shoot(sysP, coup, Q, a_s1, z1, v1, +1.0, args.eps,
                             args.beta, args.tau, args.dt, args.t_wu)
    u_twin = coup._xi_np.T @ a_twin
    d_tw = float(gram_distance(a_twin, a_node, Q))
    m_tw = _m(coup, a_twin, args.beta)
    lead_tw = int(np.argmax(np.abs(m_tw)))
    rec.update(node_route=route, saddle1_d_G=float(gram_distance(a_s1, a_node, Q)),
               z_u1=z1, twin_d_G_to_node=d_tw, twin_lead=lead_tw,
               twin_overlap=float(m_tw[lead_tw]),
               twin_eigmax=float(eigmax_M(coup, u_twin, lam, args.beta)),
               twin_stationary=bool(stat))
    print(f"{_now()} node_{mu} route={route}; saddle1 d_G={rec['saddle1_d_G']:.4f} "
          f"z_u={z1:.5f}", flush=True)
    print(f"{_now()} forward landing: lead={lead_tw} m={m_tw[lead_tw]:+.4f} "
          f"d_G(node)={d_tw:.4f} eig={rec['twin_eigmax']:+.4f} stat={stat}",
          flush=True)
    if lead_tw != mu or d_tw < 1e-6:
        rec["status"] = "no_twin_here"
        print(f"{_now()} VERDICT no twin at this cell -- nothing to continue",
              flush=True)
    else:
        # -- 2. the TWIN's own fold partner ----------------------------------
        t_a = time.perf_counter()
        u_s2, pair2 = saddle_of(coup, Q, mu, lam, u_twin,
                                float(lam + args.probe), args.beta)
        rec["twin_arc_wall"] = time.perf_counter() - t_a
        if u_s2 is None:
            rec["status"] = "twin_arclength_failed"
            rec["reason"] = str(pair2.reason)
            print(f"{_now()} VERDICT twin arclength FAILED: {pair2.reason}",
                  flush=True)
        else:
            a_s2 = reduced_field_coefficients(coup, u_s2, Q)
            res2 = float(np.linalg.norm(coup.field_F(u_s2, lam, args.beta))
                         / np.sqrt(coup.N))
            d_s2_twin = float(gram_distance(a_s2, a_twin, Q))
            d_s2_node = float(gram_distance(a_s2, a_node, Q))
            d_s2_s1 = float(gram_distance(a_s2, a_s1, Q))
            cert, GJ2, GK2 = certify_equilibrium(coup, u_s2, args.beta, lam,
                                                 args.tau, args.t0, Q,
                                                 samples_per_edge=32,
                                                 max_refinements=9)
            z2 = (float(cert["z_u"]) if np.isfinite(cert["z_u"])
                  else float(cert["real_root"]))
            rec.update(saddle2_residual=res2, saddle2_d_G_twin=d_s2_twin,
                       saddle2_d_G_node=d_s2_node, saddle2_d_G_saddle1=d_s2_s1,
                       saddle2_n_unstable=int(cert["n_unstable"]),
                       saddle2_eigmax=float(cert["eigmax"]),
                       saddle2_z_u=z2, saddle2_fold=float(pair2.fold_parameter),
                       saddle2_contour_stable=bool(cert["contour_stable"]))
            print(f"{_now()} twin's fold partner: res={res2:.1e} "
                  f"n_unst={cert['n_unstable']} eigmax={cert['eigmax']:+.4f} "
                  f"z_u={z2:.5f} fold={pair2.fold_parameter:.6f} "
                  f"d_G(twin)={d_s2_twin:.4f} d_G(node)={d_s2_node:.4f} "
                  f"d_G(saddle1)={d_s2_s1:.4f} "
                  f"[{rec['twin_arc_wall']:.0f}s]", flush=True)
            if cert["n_unstable"] != 1 or not np.isfinite(z2) or cert["eigmax"] <= 0:
                rec["status"] = "saddle2_not_index_one"
            else:
                v2 = (np.asarray(cert["mode"], float) if cert["mode"] is not None
                      else np.asarray(extract_null_mode(
                          lambda z: (args.t0 * z + 1.0) * np.eye(P)
                          - (1.0 - lam) * GJ2
                          - lam * np.exp(-z * args.tau) * (S @ GJ2), z2, Q=Q,
                          expect_real=True).vector, float))
                if v2[(mu + 1) % P] < 0:
                    v2 = -v2
                lands = {}
                for sign, name in ((-1.0, "minus"), (+1.0, "plus")):
                    a_f, el2, st2 = shoot(sysP, coup, Q, a_s2, z2, v2, sign,
                                          args.eps, args.beta, args.tau,
                                          args.dt, args.t_wu)
                    u_f = coup._xi_np.T @ a_f
                    m_f = _m(coup, a_f, args.beta)
                    nu = int(np.argmax(np.abs(m_f)))
                    d_node_nu = np.nan
                    is_node_nu = None
                    if nu != mu:
                        u_n, r_n, rt = node_by_continuation(
                            coup, nu, lam, args.beta, Q, tau=args.tau,
                            t0=args.t0)
                        if u_n is not None:
                            a_n = reduced_field_coefficients(coup, u_n, Q)
                            d_node_nu = float(gram_distance(a_f, a_n, Q))
                            is_node_nu = bool(d_node_nu < 1e-6)
                    lands[name] = dict(
                        lead=nu, overlap=float(m_f[nu]),
                        residual=float(np.linalg.norm(
                            coup.field_F(u_f, lam, args.beta)) / np.sqrt(coup.N)),
                        eigmax=float(eigmax_M(coup, u_f, lam, args.beta)),
                        d_G_to_node_mu=float(gram_distance(a_f, a_node, Q)),
                        d_G_to_twin=float(gram_distance(a_f, a_twin, Q)),
                        d_G_to_node_lead=d_node_nu, is_node_lead=is_node_nu,
                        stationary=bool(st2), elapsed=float(el2))
                    print(f"{_now()}   W^u {name} -> lead={nu} "
                          f"m={m_f[nu]:+.4f} d_G(node_{mu})="
                          f"{lands[name]['d_G_to_node_mu']:.4f} "
                          f"d_G(twin)={lands[name]['d_G_to_twin']:.4f} "
                          f"d_G(node_{nu})={d_node_nu:.4f} "
                          f"is_node={is_node_nu} stat={st2}", flush=True)
                rec["landings"] = lands
                fwd = lands["plus"]
                rec["status"] = (
                    "necklace_closes_through_twin"
                    if (fwd["lead"] != mu and fwd["is_node_lead"]) else
                    "twin_saddle_does_not_reach_next_bead")
                print(f"{_now()} VERDICT {rec['status']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = OUT / f"E34_twin_fold_lam{lam:.6f}_mu{mu}_{stamp}.json"
    path.write_text(json.dumps(rec, indent=1, default=float))
    print(f"{_now()} wrote {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
