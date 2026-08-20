"""Do ALL the alive saddle-node pairs sit on ONE invariant circle, at a FIXED lambda?

ET1'' verified the necklace BRICK one bead at a time, each at its own
lambda = lam_c(mu) - delta. That is not the user's question. The question is
whether, at a SINGLE lambda, the alive beads chain into one closed circle:

    node_{m1} -> saddle -> node_{m2} -> saddle -> ... -> node_{mn} -> saddle -> node_{m1}

and E34_snic_closure has just shown the n=1 case closes exactly (mu=91: the
forward branch tours the whole ring and lands back on node_91, d_G=0).

For each alive bead at the requested lambda this script:
  1. gets node_mu (warm-start continuation, not pattern Newton -- the pattern
     seed demonstrably fails near folds);
  2. continues to the FOLD PARTNER by pseudo-arclength, with the approach bound
     taken from the MEASURED fold (e34_alive_ladder), not from the E24 table --
     the table is early for a third of the beads and that alone made mu=28 fail
     in ET1'';
  3. certifies index 1 (argument principle, contour escalated as at P=100);
  4. shoots both W^u branches with t_max long enough for a full ring tour
     (the mu=91 tour took ~2000 t.u., so 1200 was too short in ET1'');
  5. classifies each omega-limit BY IDENTITY, never by census membership: the
     limit is accepted as bead nu if it is a stable equilibrium whose dominant
     overlap is nu. This removes the census -- and hence the flawed table --
     from the verdict entirely.

Then it assembles the directed map mu -> forward landing and reports whether it
is a single cycle covering every alive bead (one circle), several disjoint
cycles, or a broken chain.

Shard with --beads to run several processes; merge with --merge.
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
    exponential_history, gram_distance, gram_matrix, make_iid_patterns,
    memory_branch_identity, reduced_field_coefficients, solve_memory_node,
)
from e34b_edge_tracking import certify_equilibrium, linear_gate
from robust_branch import eigmax_M, woodbury_newton

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"
THR = ROOT / "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ladder(path: Path | None):
    p = path or sorted(OUT.glob("E34_alive_ladder_*.json"))[-1]
    d = json.loads(p.read_text())
    fold = {}
    intervals = {}
    for r in d["rows"]:
        if r.get("status") == "ok":
            fold[int(r["mu"])] = float(r["lam_survive"])
            intervals[int(r["mu"])] = [tuple(x) for x in r["intervals"]]
    return fold, intervals, p.name


def node_by_continuation(coup, mu, lam, beta, Q, lam_lo=0.15, step=2e-3,
                         tau=10.0, t0=1.0):
    """Track bead mu's branch up to lam; fall back to the pattern seed."""
    u, ok, res = solve_memory_node(coup, mu, lam_lo, beta)
    if ok and res < 1e-9:
        cur = lam_lo
        while cur < lam - 1e-12:
            cur = min(cur + step, lam)
            u_new, _ = woodbury_newton(coup, u, cur, beta, tol=1e-12,
                                       max_iter=80)
            r = float(np.linalg.norm(coup.field_F(u_new, cur, beta))
                      / np.sqrt(coup.N))
            if r < 1e-9:
                u = u_new
            else:
                break
        r = float(np.linalg.norm(coup.field_F(u, lam, beta)) / np.sqrt(coup.N))
        if r < 1e-9 and eigmax_M(coup, u, lam, beta) < 0 \
                and memory_branch_identity(coup, u, mu, Q=Q):
            return u, r, "continuation"
    u2, ok2, res2 = solve_memory_node(coup, mu, lam, beta)
    if ok2 and res2 < 1e-9:
        return u2, float(res2), "pattern"
    # route 3: DYNAMICAL relaxation. Measured fact (bead 3 at lam=0.298819):
    # a stable equilibrium with identity mu can exist on a sheet that neither
    # continuation from below nor the pattern seed reaches -- the flow finds it,
    # and the forward W^u branch of bead 1 lands exactly on it. A bead is alive
    # at lam if such an equilibrium exists, whatever route reveals it; so relax
    # from the last point of the branch and polish what the flow settles on.
    if u is not None:
        sysP = ReducedDDE(coup._xi_np, beta, lam, tau, t0)
        a0 = reduced_field_coefficients(coup, u, Q)
        hist, elapsed = a0, 0.0
        while elapsed < 600.0:
            o = sysP.integrate(hist, 100.0, 0.01, record_every=100)
            hist = o["hist"]
            elapsed += 100.0
            tail = o["a"][-20:]
            if max(float(gram_distance(tail[k], tail[-1], Q))
                   for k in range(len(tail))) < 1e-10:
                break
        u3 = coup._xi_np.T @ o["a"][-1]
        u3, _ = woodbury_newton(coup, u3, lam, beta, tol=1e-12, max_iter=120)
        r3 = float(np.linalg.norm(coup.field_F(u3, lam, beta)) / np.sqrt(coup.N))
        if r3 < 1e-9 and eigmax_M(coup, u3, lam, beta) < 0 \
                and memory_branch_identity(coup, u3, mu, Q=Q):
            return u3, r3, "relaxation"
    return None, float(res2), "failed"


def classify_by_identity(coup, a_final, lam, beta, Q, res_tol=1e-8,
                         a_ref=None):
    """Census-free classification of an omega-limit."""
    u_f = coup._xi_np.T @ a_final
    res = float(np.linalg.norm(coup.field_F(u_f, lam, beta)) / np.sqrt(coup.N))
    m = np.asarray(coup.overlap_raw(np.tanh(beta * u_f)), float)
    nu = int(np.argmax(np.abs(m)))
    rec = dict(residual=res, lead=nu, lead_overlap=float(m[nu]))
    if res > res_tol:
        rec.update(kind="not_equilibrium", bead=None)
        return rec
    em = float(eigmax_M(coup, u_f, lam, beta))
    ident = bool(memory_branch_identity(coup, u_f, nu, Q=Q))
    rec.update(eigmax=em, identity=ident)
    if a_ref is not None:
        rec["d_G_to_ref_node"] = float(gram_distance(a_final, a_ref, Q))
    if em < 0 and ident and abs(m[nu]) > 0.8:
        # identity alone is NOT enough: E17 found memory TWINS -- two distinct
        # stable equilibria carrying the same dominant pattern, ~1e-2 apart near
        # the fold. Landing on a twin is not landing back on node_mu, so record
        # the distance and let the caller tell them apart.
        rec.update(kind="stable_bead", bead=nu)
        if a_ref is not None:
            rec["is_ref_node"] = bool(rec["d_G_to_ref_node"] < 1e-6)
    elif em < 0:
        rec.update(kind="stable_other", bead=None)
    else:
        rec.update(kind="unstable_equilibrium", bead=None)
    return rec


def run_bead(coup, Q, mu, lam, fold_mu, beta, tau, t0, log, t_wu, epss, dts):
    rec = dict(mu=int(mu), lam=float(lam), fold_measured=float(fold_mu))
    node_u, nres, how = node_by_continuation(coup, mu, lam, beta, Q,
                                            tau=tau, t0=t0)
    if node_u is None:
        rec.update(status="node_failed", node_residual=nres)
        log(f"[mu={mu}] node FAILED (res={nres:.1e})")
        return rec
    a_node = reduced_field_coefficients(coup, node_u, Q)
    rec.update(node_residual=nres, node_route=how,
               node_eigmax=float(eigmax_M(coup, node_u, lam, beta)))
    sysP = ReducedDDE(coup._xi_np, beta, lam, tau, t0)

    t_arc = time.perf_counter()
    lam_limit = (float(fold_mu - 4e-3) if np.isfinite(fold_mu)
                 and fold_mu - 4e-3 > lam else float(lam + 4e-3))
    rec["lam_limit"] = lam_limit
    near_node, near_lam, _ = approach_memory_fold(
        coup, mu, beta, node_u, lam, lam_limit=lam_limit, eig_stop=1e9)
    pair = continue_node_fold_saddle(coup, mu, beta, node_u, lam, near_node,
                                     near_lam)
    if pair.saddle is None:
        rec.update(status="arclength_failed", reason=str(pair.reason))
        log(f"[mu={mu}] arclength FAILED: {pair.reason}")
        return rec
    u_s = pair.saddle
    a_s = reduced_field_coefficients(coup, u_s, Q)
    res_s = float(np.linalg.norm(coup.field_F(u_s, lam, beta)) / np.sqrt(coup.N))
    rec.update(saddle_residual=res_s, fold_reached=float(pair.fold_parameter),
               d_saddle_node=float(gram_distance(a_s, a_node, Q)),
               arc_wall=time.perf_counter() - t_arc,
               arc_converged=bool(pair.converged))
    if res_s > 1e-9:
        rec["status"] = "saddle_not_exact"
        log(f"[mu={mu}] saddle residual {res_s:.1e} -- rejected")
        return rec

    for spe, mr in ((16, 7), (32, 9), (48, 11)):
        cert, GJ, GK = certify_equilibrium(coup, u_s, beta, lam, tau, t0, Q,
                                           samples_per_edge=spe,
                                           max_refinements=mr)
        rec["certif_setting"] = [spe, mr]
        if cert["contour_stable"]:
            break
    z_u = float(cert["z_u"]) if np.isfinite(cert["z_u"]) else float(cert["real_root"])
    rec.update(n_unstable=int(cert["n_unstable"]), z_u=z_u,
               eigmax=float(cert["eigmax"]), real_root=float(cert["real_root"]),
               contour_stable=bool(cert["contour_stable"]),
               localize_ok=bool(cert["localize_ok"]))
    index_one = bool(cert["contour_stable"] and cert["n_unstable"] == 1
                     and cert["eigmax"] > 0 and np.isfinite(z_u))
    rec["index_one"] = index_one
    log(f"[mu={mu}] saddle res={res_s:.1e} n_unst={cert['n_unstable']} "
        f"z_u={z_u:.5f} eigmax={cert['eigmax']:+.4f} "
        f"d_G={rec['d_saddle_node']:.4f} fold={pair.fold_parameter:.6f} "
        f"contour={cert['contour_stable']} ({rec['arc_wall']:.0f}s arc)")
    if not index_one:
        rec["status"] = "index_certification_failed"
        return rec

    if cert["mode"] is not None:
        v = np.asarray(cert["mode"], float)
    else:
        from e34_lib import extract_null_mode
        P = coup.P
        S = np.roll(np.eye(P), 1, axis=0)
        v = np.asarray(extract_null_mode(
            lambda z: (t0 * z + 1.0) * np.eye(P) - (1.0 - lam) * GJ
            - lam * np.exp(-z * tau) * (S @ GJ), z_u, Q=Q,
            expect_real=True).vector, float)
    if v[(mu + 1) % coup.P] < 0:
        v = -v
    rec["linear_gate_rel_err"] = float(linear_gate(sysP, a_s, z_u, v, Q, tau))

    lands = {}
    for sign, name in ((-1.0, "minus"), (+1.0, "plus")):
        for eps in epss:
            for dt in dts:
                _, hist, dhist = exponential_history(a_s, z_u, v, sign * eps,
                                                     tau, dt, Q=Q)
                elapsed, a_last, stat = 0.0, None, False
                while elapsed < t_wu - 1e-9:
                    stepT = min(500.0, t_wu - elapsed)
                    out = sysP.integrate(hist, stepT, dt,
                                         record_every=int(round(1.0 / dt)),
                                         da_hist0=dhist)
                    hist, dhist = out["hist"], out["dhist"]
                    elapsed += stepT
                    a_last = out["a"][-1]
                    w = int(round(2.0 * tau))
                    tail = out["a"][-w:]
                    if max(float(gram_distance(tail[k], tail[-1], Q))
                           for k in range(len(tail))) < 1e-9:
                        stat = True
                        break
                cl = classify_by_identity(coup, a_last, lam, beta, Q,
                                          a_ref=a_node)
                cl.update(stationary=stat, t_used=float(elapsed))
                cl["_a"] = np.asarray(a_last, float).tolist()
                lands[f"{name}_eps{eps}_dt{dt}"] = cl
                log(f"[mu={mu}]  W^u {name} eps={eps} dt={dt} -> "
                    f"{cl['kind']}"
                    + (f" bead {cl['bead']}" if cl.get("bead") is not None else "")
                    + f" (lead {cl['lead']}, m={cl['lead_overlap']:+.4f}, "
                      f"d_G(node)={cl.get('d_G_to_ref_node', float('nan')):.4f}, "
                      f"is_node={cl.get('is_ref_node')}, "
                      f"t={elapsed:.0f}, stat={stat})")
    rec["landings"] = lands
    # A circle must connect NODE to NODE. A cross-bead landing carries the
    # target's pattern, but E17 twins mean that is not enough: resolve, for each
    # distinct target nu, whether the landing IS node_nu or a twin of it.
    targets = sorted({v_["bead"] for v_ in lands.values()
                      if v_.get("bead") is not None and v_["bead"] != int(mu)})
    tgt_nodes = {}
    for nu in targets:
        u_n, r_n, route = node_by_continuation(coup, nu, lam, beta, Q,
                                              tau=tau, t0=t0)
        if u_n is not None:
            tgt_nodes[nu] = reduced_field_coefficients(coup, u_n, Q)
        rec.setdefault("target_node_route", {})[str(nu)] = route
    for k, v_ in lands.items():
        nu = v_.get("bead")
        if nu is None or nu == int(mu) or nu not in tgt_nodes:
            continue
        a_l = np.asarray(v_.pop("_a", None)) if v_.get("_a") is not None else None
        if a_l is None:
            continue
        v_["d_G_to_target_node"] = float(gram_distance(a_l, tgt_nodes[nu], Q))
        v_["is_target_node"] = bool(v_["d_G_to_target_node"] < 1e-6)
        log(f"[mu={mu}]  landing {k}: bead {nu}, d_G(node_{nu})="
            f"{v_['d_G_to_target_node']:.4f}, is_node={v_['is_target_node']}")
    def _tag(v_):
        """bead label, but a twin of mu is not mu -- it is a distinct object."""
        if v_.get("bead") is None:
            return None
        if v_["bead"] == int(mu) and v_.get("is_ref_node") is False:
            return f"twin{mu}"
        if v_["bead"] != int(mu) and v_.get("is_target_node") is False:
            return f"twin{v_['bead']}"
        return int(v_["bead"])

    bm = {_tag(v_) for k, v_ in lands.items() if k.startswith("minus")}
    bp = {_tag(v_) for k, v_ in lands.items() if k.startswith("plus")}
    rec["back_bead"] = (bm.pop() if len(bm) == 1 and None not in bm else None)
    rec["forward_bead"] = (bp.pop() if len(bp) == 1 and None not in bp
                           else None)
    rec["robust"] = bool(rec["back_bead"] is not None
                         and rec["forward_bead"] is not None)
    rec["brick_ok"] = bool(rec["back_bead"] == int(mu)
                           and isinstance(rec["forward_bead"], int)
                           and rec["forward_bead"] != int(mu))
    rec["status"] = "brick_positive" if rec["brick_ok"] else "brick_broken"
    log(f"[mu={mu}] BRICK {rec['status']} (back={rec['back_bead']}, "
        f"forward={rec['forward_bead']})")
    return rec


def assemble(rows, alive, log):
    """Is mu -> forward_bead a single cycle covering every alive bead?"""
    fwd = {int(r["mu"]): r.get("forward_bead") for r in rows
           if r.get("status") == "brick_positive"}
    missing = [m for m in alive if m not in fwd]
    verdict = dict(n_alive=len(alive), n_bricks=len(fwd),
                   beads_without_brick=missing, map=fwd)
    if missing:
        verdict["closure"] = "incomplete: some alive bead has no certified brick"
        log(f"CHAIN incomplete -- no brick for {missing}")
        return verdict
    # walk the map
    start = alive[0]
    seen, cur = [], start
    while cur not in seen:
        seen.append(cur)
        cur = fwd.get(cur)
        if cur is None:
            break
    verdict["walk"] = seen
    if cur == start and len(seen) == len(alive):
        verdict["closure"] = "SINGLE CLOSED CIRCLE over all alive beads"
    elif cur == start:
        verdict["closure"] = (f"closed cycle over {len(seen)}/{len(alive)} beads "
                              "-- several disjoint circles")
    else:
        verdict["closure"] = "chain does not close"
    log(f"CHAIN {verdict['closure']}; walk={seen}")
    return verdict


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--beads", default="alive",
                   help="'alive' (from the ladder) or a comma-separated list")
    p.add_argument("--ladder", type=Path, default=None)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--t-wu", dest="t_wu", type=float, default=4000.0)
    p.add_argument("--eps", default="1e-3")
    p.add_argument("--dts", default="0.01")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    fold, intervals, ladder_name = load_ladder(args.ladder)
    with np.load(THR) as d:
        N, P, seed = int(d["N"]), int(d["P"]), int(d["seed"])
    xi, xis = make_iid_patterns(N, P, seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    alive = sorted(m for m, iv in intervals.items()
                   if any(a <= args.lam <= b for a, b in iv))
    beads = alive if args.beads == "alive" else [int(x) for x in
                                                args.beads.split(",")]
    epss = [float(x) for x in args.eps.split(",")]
    dts = [float(x) for x in args.dts.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    log_path = OUT / f"E34_chain_lam{args.lam:.6f}{tag}_{stamp}.log"
    fh = log_path.open("w", encoding="utf-8")

    def log(msg):
        line = f"{_now()} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    log(f"chain closure at lam={args.lam:.6f}; corrected alive set "
        f"({len(alive)}) = {alive}  [ladder {ladder_name}]")
    log(f"beads to run ({len(beads)}): {beads}; t_wu={args.t_wu}")
    rows, t0w = [], time.perf_counter()
    for mu in beads:
        rows.append(run_bead(coup, Q, mu, args.lam, fold.get(mu, np.nan),
                             args.beta, args.tau, args.t0, log, args.t_wu,
                             epss, dts))
    verdict = (assemble(rows, alive, log) if args.beads == "alive"
               else dict(closure="partial shard -- merge required",
                         map={int(r["mu"]): r.get("forward_bead") for r in rows}))
    log(f"DONE {len(rows)} beads in {time.perf_counter()-t0w:.0f}s")

    path = OUT / f"E34_chain_lam{args.lam:.6f}{tag}_{stamp}.json"
    path.write_text(json.dumps(dict(lam=args.lam, alive=alive, beads=beads,
                                    t_wu=args.t_wu, ladder=ladder_name,
                                    verdict=verdict, rows=rows),
                               indent=1, default=float))
    log(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
