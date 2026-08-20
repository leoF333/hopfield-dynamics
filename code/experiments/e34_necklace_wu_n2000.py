"""ET1'' -- the necklace brick tested DIRECTLY at N=2000 via W^u of the fold partner.

Why this replaces edge-tracking as the primary localiser at N=2000
------------------------------------------------------------------
ET1' tried to localise the boundary saddle by node-node edge tracking, then
polish it with Newton. Three measured facts (see E34_ET1prime_boundary_diag,
E34_deflation_test, E34_dwell_seed_test) changed the picture:
  1. the s* crossing IS a clean node_mu | node_{mu+1} boundary (13/13 far-side
     samples settle on node_{mu+1}; no chaotic sea in between);
  2. the A-side trajectory does dwell (speed ~1e-3 over ~50 t.u.) but its closest
     approach to the fold-partner saddle is only d_G ~ 0.071, while that saddle
     sits d_G = 0.134 from the node;
  3. Newton -- plain AND node-deflated -- STALLS at residual ~1e-3 from every
     plateau seed: no equilibrium is recovered at the dwell.
So the edge state at N=2000 is (apparently) not an equilibrium, and edge tracking
cannot serve as the saddle localiser here. The fold partner itself, however, is
obtained cleanly by arclength continuation (residual ~1e-15, certified index 1).

The necklace hypothesis is a statement about that saddle's unstable manifold:
    W^u(saddle_mu) -> node_mu  (one branch)  and  node_{mu+1}  (the other).
This script tests exactly that, per (mu, delta) cell: certified fold-partner
saddle + both W^u branches + omega-limit classification, robust in eps and dt.
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
    gram_distance, gram_matrix, make_iid_patterns, memory_branch_identity,
    reduced_field_coefficients, solve_memory_node,
)
from e34b_edge_tracking import (
    certify_equilibrium, linear_gate, shoot_branch, solve_alive_nodes,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"
THR = ROOT / "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cell(coup, Q, lam_c, mu, delta, beta, tau, t0, log, epss=(1e-4, 1e-3),
             dts=(0.01, 0.005), t_wu=1200.0):
    lam = float(lam_c[mu] - delta)
    rec = dict(mu=int(mu), delta=float(delta), lam=lam,
               lam_c=float(lam_c[mu]), status="running")
    sysP = ReducedDDE(coup._xi_np, beta, lam, tau, t0)

    node_u, ok, nres = solve_memory_node(coup, mu, lam, beta)
    if not ok:
        rec["status"] = "node_failed"
        return rec
    a_node = reduced_field_coefficients(coup, node_u, Q)
    a_nodes = solve_alive_nodes(coup, lam, beta, lam_c)
    a_nodes[int(mu)] = a_node
    succ, pred = (mu + 1) % coup.P, (mu - 1) % coup.P
    # Beads die one by one, so mu+1 may already be dead at lambda_c(mu)-delta:
    # the necklace target is the FIRST ALIVE node forward on the ring (roadmap
    # E34 sec.5) -- recorded explicitly so a non-adjacent landing is not scored
    # as a failure when the adjacent bead simply no longer exists.
    target = next((int((mu + k) % coup.P) for k in range(1, coup.P)
                   if int((mu + k) % coup.P) in a_nodes), None)
    rec.update(node_residual=float(nres), n_alive=len(a_nodes),
               succ_alive=bool(succ in a_nodes), pred_alive=bool(pred in a_nodes),
               forward_target=target,
               target_is_succ=bool(target == succ))

    # fold-partner saddle by arclength through the fold (independent of dynamics)
    t_arc = time.perf_counter()
    near_node, near_lam, _ = approach_memory_fold(
        coup, mu, beta, node_u, lam, lam_limit=float(lam_c[mu] - 4e-3),
        eig_stop=1e9)
    pair = continue_node_fold_saddle(coup, mu, beta, node_u, lam, near_node,
                                     near_lam)
    if not (pair.converged and pair.saddle is not None):
        rec["status"] = "arclength_failed"
        return rec
    u_s = pair.saddle
    a_s = reduced_field_coefficients(coup, u_s, Q)
    res_s = float(np.linalg.norm(coup.field_F(u_s, lam, beta)) / np.sqrt(coup.N))
    # escalate the contour resolution until the winding count stabilises: P=100
    # needs a finer contour than the P=20 pilot (measured, E34_certif_tune)
    for spe, mr in ((16, 7), (32, 9), (48, 11)):
        cert, GJ, GK = certify_equilibrium(
            coup, u_s, beta, lam, tau, t0, Q,
            samples_per_edge=spe, max_refinements=mr)
        rec["certif_setting"] = [spe, mr]
        if cert["contour_stable"]:
            break
    index_one = bool(cert["contour_stable"] and cert["n_unstable"] == 1
                     and cert["localize_ok"] and abs(cert["z_u_imag"]) < 1e-8
                     and cert["eigmax"] > 0 and cert["z_u_crosscheck_ok"])
    rec.update(saddle_residual=res_s, n_unstable=int(cert["n_unstable"]),
               z_u=float(cert["z_u"]), eigmax=float(cert["eigmax"]),
               index_one=index_one, real_root=float(cert["real_root"]),
               contour_stable=bool(cert["contour_stable"]),
               base_reason=str(cert["base_reason"]),
               shifted_reason=str(cert["shifted_reason"]),
               localize_ok=bool(cert["localize_ok"]), rho=float(cert["rho"]),
               saddle_identity=bool(memory_branch_identity(coup, u_s, mu, Q=Q)),
               d_saddle_node=float(gram_distance(a_s, a_node, Q)),
               fold_parameter=float(pair.fold_parameter),
               arc_wall=time.perf_counter() - t_arc,
               a_saddle=a_s, m_saddle=coup.overlap_raw(np.tanh(beta * u_s)),
               a_node=a_node)
    log(f"[mu={mu} d={delta}] saddle res={res_s:.1e} n_unst={cert['n_unstable']} "
        f"z_u={cert['z_u']:.4f} eigmax={cert['eigmax']:+.4f} "
        f"real_root={cert['real_root']:.4f} d_G(node)={rec['d_saddle_node']:.4f} "
        f"fold={pair.fold_parameter:.6f} contour_stable={cert['contour_stable']} "
        f"base={cert['base_reason']} shifted={cert['shifted_reason']} "
        f"({rec['arc_wall']:.1f}s)")
    if not index_one:
        rec["status"] = "index_certification_failed"
        return rec

    # flow eigenvector: variational null mode (S.Dm), already built by the
    # certification -- never the T_P mode (worklog V0 subtlety, ~3% rate error)
    v = np.asarray(cert["mode"], float)
    rec["mode_residual"] = float(cert["mode_residual"])
    if v[(mu + 1) % coup.P] < 0:
        v = -v
    gate = linear_gate(sysP, a_s, cert["z_u"], v, Q, tau)
    rec["linear_gate_rel_err"] = float(gate)

    lands = {}
    for sign, name in ((-1.0, "minus"), (+1.0, "plus")):
        for eps in epss:
            for dt in dts:
                t_s = time.perf_counter()
                sh = shoot_branch(sysP, a_s, cert["z_u"], v, sign, eps, Q,
                                  a_nodes, tau=tau, dt=dt, t_max=t_wu)
                key = f"{name}_eps{eps}_dt{dt}"
                lands[key] = dict(omega=int(sh["omega"]),
                                  stationary=bool(sh["stationary"]),
                                  elapsed=float(sh["elapsed"]),
                                  wall=time.perf_counter() - t_s)
                log(f"[mu={mu} d={delta}] W^u {key} -> omega={sh['omega']} "
                    f"({lands[key]['wall']:.1f}s)")
    rec["landings"] = lands
    om_minus = {v_["omega"] for k, v_ in lands.items() if k.startswith("minus")}
    om_plus = {v_["omega"] for k, v_ in lands.items() if k.startswith("plus")}
    rec["omega_minus"] = sorted(om_minus)
    rec["omega_plus"] = sorted(om_plus)
    robust = len(om_minus) == 1 and len(om_plus) == 1
    rec["wu_robust"] = bool(robust)
    if robust:
        m_, p_ = om_minus.pop(), om_plus.pop()
        rec["landing_minus"], rec["landing_plus"] = int(m_), int(p_)
        rec["adjacent_pair"] = bool(m_ == int(mu) and target is not None
                                    and p_ == int(target))
        rec["status"] = ("necklace_brick_positive" if rec["adjacent_pair"]
                         else "non_adjacent_landing")
    else:
        rec["adjacent_pair"] = False
        rec["status"] = "wu_not_robust"
    log(f"[mu={mu} d={delta}] VERDICT {rec['status']} "
        f"(minus={rec['omega_minus']}, plus={rec['omega_plus']})")
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--links", default="28,1,91")
    p.add_argument("--deltas", default="0.005,0.010,0.020")
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--t-wu", type=float, default=1200.0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
        N, P, seed = int(d["N"]), int(d["P"]), int(d["seed"])
    links = [int(x) for x in args.links.split(",") if x.strip()]
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    if args.smoke:
        links, deltas = links[:1], deltas[-1:]

    xi, xis = make_iid_patterns(N, P, seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    OUT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = OUT / f"E34_ET1pp_wu_log_{run_id}.txt"
    fh = log_path.open("w", encoding="utf-8")

    def log(msg):
        line = f"{_now()} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    log(f"E34 ET1'' (W^u of the fold partner) N={N} P={P} seed={seed} "
        f"beta={args.beta} tau={args.tau}")
    log(f"links={links} deltas={deltas}  lam_c=" +
        ", ".join(f"{m}:{lam_c[m]:.6f}" for m in links))
    t_all = time.perf_counter()
    rows = []
    for mu in links:
        for dl in deltas:
            t_c = time.perf_counter()
            rec = run_cell(coup, Q, lam_c, mu, dl, args.beta, args.tau, args.t0,
                           log, t_wu=args.t_wu)
            rec["wall"] = time.perf_counter() - t_c
            rows.append(rec)
    total = time.perf_counter() - t_all
    n_pos = sum(1 for r in rows if r["status"] == "necklace_brick_positive")
    log(f"DONE {len(rows)} cells in {total:.1f}s; "
        f"necklace_brick_positive={n_pos}/{len(rows)}")

    npz = OUT / f"E34_ET1pp_wu_results_{run_id}.npz"
    np.savez_compressed(
        npz,
        mu=np.array([r["mu"] for r in rows]),
        delta=np.array([r["delta"] for r in rows]),
        lam=np.array([r["lam"] for r in rows]),
        status=np.array([r["status"] for r in rows]),
        saddle_residual=np.array([r.get("saddle_residual", np.nan) for r in rows]),
        n_unstable=np.array([r.get("n_unstable", -1) for r in rows]),
        z_u=np.array([r.get("z_u", np.nan) for r in rows]),
        d_saddle_node=np.array([r.get("d_saddle_node", np.nan) for r in rows]),
        fold_parameter=np.array([r.get("fold_parameter", np.nan) for r in rows]),
        linear_gate=np.array([r.get("linear_gate_rel_err", np.nan) for r in rows]),
        adjacent=np.array([bool(r.get("adjacent_pair", False)) for r in rows]),
        wu_robust=np.array([bool(r.get("wu_robust", False)) for r in rows]),
        a_saddle=np.array([r.get("a_saddle", np.full(P, np.nan)) for r in rows]),
        m_saddle=np.array([r.get("m_saddle", np.full(P, np.nan)) for r in rows]),
        a_node=np.array([r.get("a_node", np.full(P, np.nan)) for r in rows]),
    )
    js = {k: v for k, v in dict(
        run_id=run_id, N=N, P=P, seed=seed, links=links, deltas=deltas,
        total_wall=total, n_positive=n_pos,
        rows=[{k2: (v2 if not isinstance(v2, np.ndarray) else None)
               for k2, v2 in r.items()} for r in rows]).items()}
    (OUT / f"E34_ET1pp_wu_summary_{run_id}.json").write_text(json.dumps(js, indent=1))
    log(f"npz={npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
