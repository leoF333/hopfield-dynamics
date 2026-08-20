"""Record the invariant-circle trajectories (and the twins) for plotting.

E34_snic_closure and E34_chain_closure established WHAT the branches do but kept
only summaries; the figure needs the trajectories themselves. This recomputes
the fold-partner saddle of each requested bead, shoots both W^u branches, and
stores the full reduced trajectory a(t) together with the physical overlaps
m(t), so the circle can be drawn in the (m_mu, m_nu) plane and as a kymograph.

The index-1 certification is NOT redone here (it was done, with contour
escalation, in E34_chain_closure / E34_ET1pp); the unstable rate is taken from
the bracketed rightmost REAL root, which those runs showed to agree with the
localised root, and the mode is the variational null mode at that root.

It also stores the LANDING states, which is what makes the twins analysable:
for a broken brick the forward landing is a second stable equilibrium carrying
the same dominant pattern, and its overlap profile is the thing to look at.
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
from reduced_spectrum import GJ_GK, rightmost_real_root
from robust_branch import eigmax_M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def overlaps(coup, a, beta):
    u = coup._xi_np.T @ np.asarray(a, float)
    return np.asarray(coup.overlap_raw(np.tanh(beta * u)), float)


def shoot_record(sysP, coup, a_s, z_u, v, sign, eps, Q, beta, tau, dt, t_max):
    _, hist, dhist = exponential_history(a_s, z_u, v, sign * eps, tau, dt, Q=Q)
    ts, As, elapsed, stat = [], [], 0.0, False
    while elapsed < t_max - 1e-9:
        step = min(250.0, t_max - elapsed)
        o = sysP.integrate(hist, step, dt, record_every=int(round(1.0 / dt)),
                           da_hist0=dhist)
        hist, dhist = o["hist"], o["dhist"]
        ts.append(o["t"] + elapsed)
        As.append(o["a"])
        elapsed += step
        tail = o["a"][-int(round(2.0 * tau)):]
        if max(float(gram_distance(tail[k], tail[-1], Q))
               for k in range(len(tail))) < 1e-10:
            stat = True
            break
    t = np.concatenate(ts)
    A = np.concatenate(As)
    M = np.array([overlaps(coup, x, beta) for x in A])
    return dict(t=t, a=A, m=M, stationary=stat, elapsed=elapsed)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True,
                   help="lam:beads pairs, e.g. '0.322629:91;0.307629:43,91'")
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--t-max", dest="t_max", type=float, default=4000.0)
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    fold, intervals, ladder_name = load_ladder(None)
    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    P = coup.P

    store, summary = {}, []
    for cell in args.cells.split(";"):
        lam_s, beads_s = cell.split(":")
        lam = float(lam_s)
        beads = [int(x) for x in beads_s.split(",")]
        sysP = ReducedDDE(xi, args.beta, lam, args.tau, args.t0)
        for mu in beads:
            t_b = time.perf_counter()
            node_u, nres, route = node_by_continuation(coup, mu, lam, args.beta,
                                                       Q, tau=args.tau,
                                                       t0=args.t0)
            if node_u is None:
                print(f"{_now()} [lam={lam} mu={mu}] node FAILED", flush=True)
                continue
            a_node = reduced_field_coefficients(coup, node_u, Q)
            fm = fold.get(mu, np.nan)
            lam_limit = (float(fm - 4e-3) if np.isfinite(fm) and fm - 4e-3 > lam
                         else float(lam + 4e-3))
            near_node, near_lam, _ = approach_memory_fold(
                coup, mu, args.beta, node_u, lam, lam_limit=lam_limit,
                eig_stop=1e9)
            pair = continue_node_fold_saddle(coup, mu, args.beta, node_u, lam,
                                             near_node, near_lam)
            if pair.saddle is None:
                print(f"{_now()} [lam={lam} mu={mu}] arclength FAILED",
                      flush=True)
                continue
            u_s = pair.saddle
            a_s = reduced_field_coefficients(coup, u_s, Q)
            res_s = float(np.linalg.norm(coup.field_F(u_s, lam, args.beta))
                          / np.sqrt(coup.N))
            GJ, GK = GJ_GK(coup, u_s, args.beta, lam)
            rr = rightmost_real_root(GJ, GK, args.t0, args.tau, lam,
                                     x_hi=2.0, x_lo=-0.5, n=400)
            z_u = float(rr.real)
            S = np.roll(np.eye(P), 1, axis=0)
            mode = extract_null_mode(
                lambda z: (args.t0 * z + 1.0) * np.eye(P)
                - (1.0 - lam) * GJ - lam * np.exp(-z * args.tau) * (S @ GJ),
                z_u, Q=Q, expect_real=True)
            v = np.asarray(mode.vector, float)
            if v[(mu + 1) % P] < 0:
                v = -v
            print(f"{_now()} [lam={lam:.6f} mu={mu}] saddle res={res_s:.1e} "
                  f"z_u={z_u:.5f} d_G(node)={gram_distance(a_s, a_node, Q):.4f} "
                  f"node_route={route} mode_res={mode.residual:.1e}", flush=True)

            key = f"lam{lam:.6f}_mu{mu}"
            store[f"{key}_a_node"] = a_node
            store[f"{key}_a_saddle"] = a_s
            store[f"{key}_m_node"] = overlaps(coup, a_node, args.beta)
            store[f"{key}_m_saddle"] = overlaps(coup, a_s, args.beta)
            for sign, name in ((-1.0, "minus"), (+1.0, "plus")):
                sh = shoot_record(sysP, coup, a_s, z_u, v, sign, args.eps, Q,
                                  args.beta, args.tau, args.dt, args.t_max)
                store[f"{key}_{name}_t"] = sh["t"]
                store[f"{key}_{name}_m"] = sh["m"]
                store[f"{key}_{name}_a_final"] = sh["a"][-1]
                a_f = sh["a"][-1]
                m_f = overlaps(coup, a_f, args.beta)
                nu = int(np.argmax(np.abs(m_f)))
                u_f = coup._xi_np.T @ a_f
                rec = dict(lam=lam, mu=int(mu), branch=name,
                           stationary=bool(sh["stationary"]),
                           elapsed=float(sh["elapsed"]),
                           lead=nu, lead_overlap=float(m_f[nu]),
                           residual=float(np.linalg.norm(
                               coup.field_F(u_f, lam, args.beta))
                               / np.sqrt(coup.N)),
                           eigmax=float(eigmax_M(coup, u_f, lam, args.beta)),
                           identity=bool(memory_branch_identity(
                               coup, u_f, nu, Q=Q)),
                           d_G_to_node_mu=float(gram_distance(a_f, a_node, Q)),
                           top5=[[int(k), float(m_f[k])]
                                 for k in np.argsort(-np.abs(m_f))[:5]],
                           saddle_residual=res_s, z_u=z_u,
                           d_saddle_node=float(gram_distance(a_s, a_node, Q)))
                # if the landing carries mu itself but is NOT node_mu, it is a
                # twin: record where it sits relative to the node and the saddle
                if nu == int(mu) and rec["d_G_to_node_mu"] > 1e-6:
                    rec["is_twin"] = True
                    rec["d_G_twin_to_saddle"] = float(
                        gram_distance(a_f, a_s, Q))
                    store[f"{key}_twin_m"] = m_f
                    store[f"{key}_twin_a"] = a_f
                else:
                    rec["is_twin"] = False
                summary.append(rec)
                print(f"{_now()}   {name}: lead={nu} m={m_f[nu]:+.4f} "
                      f"res={rec['residual']:.1e} eig={rec['eigmax']:+.4f} "
                      f"d_G(node_{mu})={rec['d_G_to_node_mu']:.4f} "
                      f"twin={rec['is_twin']} t={sh['elapsed']:.0f} "
                      f"stat={sh['stationary']}", flush=True)
            print(f"{_now()} [lam={lam:.6f} mu={mu}] done in "
                  f"{time.perf_counter()-t_b:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    np.savez_compressed(OUT / f"E34_circle_traj{tag}_{stamp}.npz", **store)
    (OUT / f"E34_circle_traj{tag}_{stamp}.json").write_text(
        json.dumps(dict(ladder=ladder_name, eps=args.eps, dt=args.dt,
                        t_max=args.t_max, rows=summary), indent=1, default=float))
    print(f"{_now()} wrote E34_circle_traj{tag}_{stamp}.npz/.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
