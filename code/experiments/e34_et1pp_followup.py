"""ET1'' follow-ups: identify the three non-positive cells of the W^u grid.

The 9-cell grid (E34_ET1pp_wu_summary_20260727T154356.json) left three distinct
open questions, each of which changes how the cell must be SCORED:

  (a) mu=1, delta=0.005 -- the forward W^u branch settles on a fixed point that
      is not in the alive-node census (omega=-4). The census excludes any node
      with lam_c[k] <= lam + margin (margin 3e-3), so a perfectly good memory
      bead sitting just above lambda can be missing from the census. If the
      landing IS such a bead, the cell is a necklace brick, not a failure.
      -> mode "land": re-shoot the plus branch and identify the limit by overlap.

  (b) mu=91, delta=0.020 -- count is stable at 1 and eigmax_M > 0, but the
      rectangle localisation of the root failed (localize_ok False, z_u NaN), so
      no mode and no shooting. With a stable count of 1 and a bracketed REAL
      root from rightmost_real_root, the single unstable root must be that real
      root: build the variational mode there and shoot.
      -> mode "loc": documented fallback localisation + shooting.

  (c) mu=28 (all three deltas) -- arclength continuation to the fold partner
      returned not-converged in ~1.5 s, i.e. it failed immediately, before any
      spectral work. mu=28 is the EARLIEST bead to die (lam_c=0.2195, far below
      the bulk), so the fold is reached at a lambda where all 100 beads are
      still alive.  -> mode "arc": instrument approach_memory_fold /
      continue_node_fold_saddle to see which stage fails and why.

Each mode is independently runnable and single-threaded.
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
    certify_equilibrium, linear_gate, shoot_branch, solve_alive_nodes,
)
from e34_lib import extract_null_mode
from reduced_spectrum import GJ_GK, rightmost_real_root
from robust_branch import eigmax_M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"
THR = ROOT / "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup(mu, delta, lam_c, beta):
    lam = float(lam_c[mu] - delta)
    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    return coup, Q, lam


def _load_saddle(coup, mu, delta):
    npz = sorted(OUT.glob("E34_ET1pp_wu_results_*.npz"))[-1]
    d = np.load(npz, allow_pickle=True)
    idx = [i for i in range(len(d["mu"]))
           if int(d["mu"][i]) == mu and abs(float(d["delta"][i]) - delta) < 1e-9]
    if not idx:
        raise SystemExit(f"cell mu={mu} delta={delta} not in {npz.name}")
    a_s = np.asarray(d["a_saddle"][idx[0]], float)
    if not np.all(np.isfinite(a_s)):
        raise SystemExit(f"cell mu={mu} delta={delta} has no saddle stored")
    print(f"{_now()} loaded saddle from {npz.name}", flush=True)
    return coup._xi_np.T @ a_s, a_s


def _identify(coup, a_final, lam, beta, Q, lam_c, margin=3e-3):
    """Which object is this? overlaps -> best bead -> node solve -> distance."""
    u_f = coup._xi_np.T @ a_final
    m = np.asarray(coup.overlap_raw(np.tanh(beta * u_f)), float)
    order = np.argsort(-np.abs(m))[:5]
    rows = [dict(nu=int(k), m=float(m[k]), lam_c=float(lam_c[k]),
                 in_census=bool(lam_c[k] > lam + margin)) for k in order]
    nu = int(order[0])
    out = dict(top5=rows, best=nu, res_final=float(
        np.linalg.norm(coup.field_F(u_f, lam, beta)) / np.sqrt(coup.N)))
    # identity and stability of the LANDING ITSELF -- never conditioned on a
    # naive node solve succeeding: when the tabulated threshold under-estimates
    # the fold, Newton-from-pattern fails for exactly the bead we just landed on.
    out["identity_of_final"] = bool(memory_branch_identity(coup, u_f, nu, Q=Q))
    out["eigmax_of_final"] = float(eigmax_M(coup, u_f, lam, beta))
    u_n, ok, res = solve_memory_node(coup, nu, lam, beta)
    out["node_solve_ok"] = bool(ok)
    out["node_residual"] = float(res)
    if ok and res < 1e-9:
        a_n = reduced_field_coefficients(coup, u_n, Q)
        out["d_G_to_node"] = float(gram_distance(a_final, a_n, Q))
    return out


def mode_land(args, lam_c, beta):
    """(a) re-shoot the forward branch and identify its omega-limit."""
    coup, Q, lam = _setup(args.mu, args.delta, lam_c, beta)
    u_s, a_s = _load_saddle(coup, args.mu, args.delta)
    sysP = ReducedDDE(coup._xi_np, beta, lam, args.tau, args.t0)
    a_nodes = solve_alive_nodes(coup, lam, beta, lam_c)
    print(f"{_now()} lam={lam:.6f} census n_alive={len(a_nodes)}", flush=True)
    cert, _, _ = certify_equilibrium(coup, u_s, beta, lam, args.tau, args.t0, Q,
                                     samples_per_edge=32, max_refinements=9)
    v = np.asarray(cert["mode"], float)
    if v[(args.mu + 1) % coup.P] < 0:
        v = -v
    print(f"{_now()} saddle n_unst={cert['n_unstable']} z_u={cert['z_u']:.6f} "
          f"mode_res={cert['mode_residual']:.1e}", flush=True)
    sh = shoot_branch(sysP, a_s, cert["z_u"], v, +1.0, 1e-3, Q, a_nodes,
                      tau=args.tau, dt=0.01, t_max=1200.0)
    ident = _identify(coup, sh["final"], lam, beta, Q, lam_c)
    print(f"{_now()} plus branch omega={sh['omega']} stationary={sh['stationary']} "
          f"elapsed={sh['elapsed']}", flush=True)
    print(f"{_now()} landing residual={ident['res_final']:.2e} best bead="
          f"{ident['best']} identity={ident['identity_of_final']} "
          f"eigmax={ident['eigmax_of_final']:+.4f} "
          f"node_solve_ok={ident['node_solve_ok']} "
          f"d_G(node)={ident.get('d_G_to_node', float('nan')):.2e}", flush=True)
    for r in ident["top5"]:
        print(f"    nu={r['nu']:3d} m={r['m']:+.4f} lam_c={r['lam_c']:.6f} "
              f"in_census={r['in_census']}", flush=True)
    is_bead = bool(ident["res_final"] < 1e-9 and ident["identity_of_final"]
                   and ident["eigmax_of_final"] < 0)
    verdict = (
        "landing IS the stable memory bead nu={} but the table under-estimates "
        "its fold (lam_c={:.6f} < lambda={:.6f}), so the census dropped it"
        .format(ident["best"], ident["top5"][0]["lam_c"], lam)
        if (is_bead and not ident["top5"][0]["in_census"])
        else ("landing IS a stable memory bead in the census (classifier bug)"
              if is_bead
              else "landing is NOT a memory bead (genuine non-census state)"))
    print(f"{_now()} VERDICT {verdict}", flush=True)
    return dict(mode="land", mu=args.mu, delta=args.delta, lam=lam,
                omega=int(sh["omega"]), stationary=bool(sh["stationary"]),
                z_u=float(cert["z_u"]), identify=ident, verdict=verdict)


def mode_loc(args, lam_c, beta):
    """(b) fallback localisation when the rectangle refinement fails."""
    coup, Q, lam = _setup(args.mu, args.delta, lam_c, beta)
    u_s, a_s = _load_saddle(coup, args.mu, args.delta)
    sysP = ReducedDDE(coup._xi_np, beta, lam, args.tau, args.t0)
    a_nodes = solve_alive_nodes(coup, lam, beta, lam_c)
    GJ, GK = GJ_GK(coup, u_s, beta, lam)
    em = float(eigmax_M(coup, u_s, lam, beta))
    cert, _, _ = certify_equilibrium(coup, u_s, beta, lam, args.tau, args.t0, Q,
                                     samples_per_edge=32, max_refinements=9)
    rr = rightmost_real_root(GJ, GK, args.t0, args.tau, lam,
                             x_hi=cert["rho"] / args.t0, x_lo=-0.5, n=400)
    print(f"{_now()} lam={lam:.6f} n_unst={cert['n_unstable']} "
          f"contour_stable={cert['contour_stable']} localize_ok={cert['localize_ok']} "
          f"eigmax={em:+.4f} real_root={rr.real:.6f} n_alive={len(a_nodes)}",
          flush=True)
    if not (cert["contour_stable"] and cert["n_unstable"] == 1 and em > 0
            and rr is not None):
        print(f"{_now()} VERDICT fallback not licensed", flush=True)
        return dict(mode="loc", mu=args.mu, delta=args.delta,
                    verdict="fallback not licensed")
    # count==1 and a bracketed REAL root exists => the unique unstable root is
    # real and equals rr; build the variational mode there (same char_red as
    # certify_equilibrium) and verify with the linear gate.
    z_u = float(rr.real)
    P = GJ.shape[0]
    S = np.roll(np.eye(P), 1, axis=0)

    def char_red(z):
        return (args.t0 * z + 1.0) * np.eye(P) - (1.0 - lam) * GJ \
            - lam * np.exp(-z * args.tau) * (S @ GJ)

    mode = extract_null_mode(char_red, z_u, Q=Q, expect_real=True)
    v = np.asarray(mode.vector, float)
    if v[(args.mu + 1) % coup.P] < 0:
        v = -v
    gate = float(linear_gate(sysP, a_s, z_u, v, Q, args.tau))
    print(f"{_now()} fallback mode residual={mode.residual:.1e} "
          f"realif={mode.realification_error:.1e} linear_gate_rel_err={gate:.2e}",
          flush=True)
    lands = {}
    for sign, name in ((-1.0, "minus"), (+1.0, "plus")):
        for eps in (1e-4, 1e-3):
            sh = shoot_branch(sysP, a_s, z_u, v, sign, eps, Q, a_nodes,
                              tau=args.tau, dt=0.01, t_max=1200.0)
            lands[f"{name}_eps{eps}"] = int(sh["omega"])
            print(f"{_now()} W^u {name} eps={eps} -> omega={sh['omega']} "
                  f"stationary={sh['stationary']}", flush=True)
    om_m = {v_ for k, v_ in lands.items() if k.startswith("minus")}
    om_p = {v_ for k, v_ in lands.items() if k.startswith("plus")}
    target = next((int((args.mu + k) % coup.P) for k in range(1, coup.P)
                   if int((args.mu + k) % coup.P) in a_nodes), None)
    ok = (len(om_m) == 1 and len(om_p) == 1 and om_m.pop() == args.mu
          and target is not None and om_p.pop() == target)
    print(f"{_now()} forward_target={target} VERDICT "
          f"{'necklace_brick_positive' if ok else 'non_adjacent_landing'} "
          f"(gate {gate:.2e})", flush=True)
    return dict(mode="loc", mu=args.mu, delta=args.delta, lam=lam, z_u=z_u,
                mode_residual=float(mode.residual), linear_gate=gate,
                landings=lands, forward_target=target,
                status="necklace_brick_positive" if ok else "non_adjacent_landing")


def mode_arc(args, lam_c, beta):
    """(c) why does the arclength continuation fail for this bead?"""
    coup, Q, lam = _setup(args.mu, args.delta, lam_c, beta)
    node_u, ok, nres = solve_memory_node(coup, args.mu, lam, beta)
    print(f"{_now()} mu={args.mu} lam={lam:.6f} lam_c={lam_c[args.mu]:.6f} "
          f"node ok={ok} res={nres:.2e} eigmax={eigmax_M(coup, node_u, lam, beta):+.5f}",
          flush=True)
    out = dict(mode="arc", mu=args.mu, delta=args.delta, lam=lam,
               node_ok=bool(ok), node_residual=float(nres))
    for lim_off in (4e-3, 2e-3, 1e-3, 5e-4):
        lam_limit = float(lam_c[args.mu] - lim_off)
        try:
            near_node, near_lam, info = approach_memory_fold(
                coup, args.mu, beta, node_u, lam, lam_limit=lam_limit,
                eig_stop=1e9)
            em = float(eigmax_M(coup, near_node, near_lam, beta))
            print(f"{_now()}  approach(limit=lam_c-{lim_off:g}): near_lam="
                  f"{near_lam:.8f} eigmax={em:+.6f} gap_to_fold="
                  f"{lam_c[args.mu]-near_lam:.2e}", flush=True)
            pair = continue_node_fold_saddle(coup, args.mu, beta, node_u, lam,
                                             near_node, near_lam)
            print(f"{_now()}  continue: converged={pair.converged} "
                  f"saddle={'yes' if pair.saddle is not None else 'no'} "
                  f"fold={getattr(pair, 'fold_parameter', float('nan')):.6f} "
                  f"(table lam_c={lam_c[args.mu]:.6f}, "
                  f"offset={getattr(pair, 'fold_parameter', np.nan)-lam_c[args.mu]:+.2e}) "
                  f"node_eigmax={pair.node_eigmax:+.5f} "
                  f"saddle_eigmax={pair.saddle_eigmax if pair.saddle_eigmax is None else f'{pair.saddle_eigmax:+.5f}'} "
                  f"reason={pair.reason!r}", flush=True)
            rec = dict(lim_off=lim_off, near_lam=float(near_lam), eigmax=em,
                       converged=bool(pair.converged),
                       has_saddle=pair.saddle is not None,
                       reason=str(pair.reason),
                       node_eigmax=float(pair.node_eigmax),
                       saddle_eigmax=(None if pair.saddle_eigmax is None
                                      else float(pair.saddle_eigmax)),
                       fold=float(getattr(pair, "fold_parameter", np.nan)))
            if pair.saddle is not None:
                res_any = float(np.linalg.norm(
                    coup.field_F(pair.saddle, lam, beta)) / np.sqrt(coup.N))
                a_s = reduced_field_coefficients(coup, pair.saddle, Q)
                a_n = reduced_field_coefficients(coup, node_u, Q)
                rec.update(saddle_res_any=res_any,
                           d_G_any=float(gram_distance(a_s, a_n, Q)),
                           identity_any=bool(memory_branch_identity(
                               coup, pair.saddle, args.mu, Q=Q)))
                print(f"{_now()}   saddle-as-returned: res={res_any:.2e} "
                      f"d_G(node)={rec['d_G_any']:.4f} "
                      f"identity_mu={rec['identity_any']}", flush=True)
            if pair.converged and pair.saddle is not None:
                res = float(np.linalg.norm(coup.field_F(pair.saddle, lam, beta))
                            / np.sqrt(coup.N))
                a_s = reduced_field_coefficients(coup, pair.saddle, Q)
                a_n = reduced_field_coefficients(coup, node_u, Q)
                rec.update(saddle_res=res,
                           d_G=float(gram_distance(a_s, a_n, Q)),
                           identity=bool(memory_branch_identity(
                               coup, pair.saddle, args.mu, Q=Q)))
                print(f"{_now()}  SADDLE FOUND res={res:.2e} "
                      f"d_G(node)={rec['d_G']:.4f} identity={rec['identity']}",
                      flush=True)
                out.setdefault("attempts", []).append(rec)
                out["fixed_by"] = lim_off
                break
            out.setdefault("attempts", []).append(rec)
        except Exception as exc:                       # noqa: BLE001
            print(f"{_now()}  approach/continue raised: {type(exc).__name__}: "
                  f"{exc}", flush=True)
            out.setdefault("attempts", []).append(
                dict(lim_off=lim_off, error=f"{type(exc).__name__}: {exc}"))
    return out


def mode_cycle(args, lam_c, beta):
    """(d) is the non-stationary forward landing the RECALL CYCLE?

    At mu=91 (lam_c = 0.32763, i.e. essentially lambda*) the forward W^u branch
    is still moving at t=1200 and far from every census node (omega=-1).  The
    SNIC picture predicts it should be the recall limit cycle: a periodic orbit
    whose dominant overlap advances one bead per link time around the ring.
    Test: record the argmax-overlap sequence, check it advances forward, and
    measure the period by returning to a Poincare section.
    """
    coup, Q, lam = _setup(args.mu, args.delta, lam_c, beta)
    u_s, a_s = _load_saddle(coup, args.mu, args.delta)
    sysP = ReducedDDE(coup._xi_np, beta, lam, args.tau, args.t0)
    cert, _, _ = certify_equilibrium(coup, u_s, beta, lam, args.tau, args.t0, Q,
                                     samples_per_edge=32, max_refinements=9)
    v = np.asarray(cert["mode"], float)
    if v[(args.mu + 1) % coup.P] < 0:
        v = -v
    print(f"{_now()} lam={lam:.6f} z_u={cert['z_u']:.6f}", flush=True)
    from e34_lib import exponential_history
    from e34b_edge_tracking import integrate_to_settle
    _, hist, dhist = exponential_history(a_s, cert["z_u"], v, 1e-3, args.tau,
                                         0.01, Q=Q)
    out = integrate_to_settle(sysP, hist, Q, dt=0.01, tau=args.tau,
                              t_chunk=50.0, t_max=args.t_wu, record_every=100,
                              da_hist0=dhist)
    a, t = out["a"], out["t"]
    m = np.array([coup.overlap_raw(np.tanh(beta * (coup._xi_np.T @ x)))
                  for x in a])
    lead = np.argmax(np.abs(m), axis=1)
    amp = np.array([abs(m[i, lead[i]]) for i in range(len(lead))])
    # transitions of the leading bead in the second half (post-transient)
    half = len(t) // 2
    trans = [(float(t[i]), int(lead[i - 1]), int(lead[i]))
             for i in range(half, len(t)) if lead[i] != lead[i - 1]]
    steps = [((b - a_) % coup.P) for _, a_, b in trans]
    fwd = [s for s in steps if s <= coup.P // 2]
    print(f"{_now()} stationary={out['stationary']} elapsed={out['elapsed']} "
          f"lead-amp mean={amp[half:].mean():.4f} min={amp[half:].min():.4f}",
          flush=True)
    print(f"{_now()} {len(trans)} leader changes after t={t[half]:.0f}; "
          f"ring steps={sorted(set(steps))}; first 12 transitions="
          f"{[(round(x[0],1), x[1], x[2]) for x in trans[:12]]}", flush=True)
    period = np.nan
    if len(trans) >= 3:
        dts = np.diff([x[0] for x in trans])
        period = float(np.median(dts))
        print(f"{_now()} median link duration={period:.3f} "
              f"cv={float(np.std(dts)/max(np.mean(dts),1e-30)):.2e} "
              f"=> full-ring period ~ {period*coup.P:.1f}", flush=True)
    all_fwd = bool(steps) and all(s == steps[0] for s in steps) and steps[0] >= 1
    verdict = ("forward landing is a RECALL-TYPE travelling orbit "
               f"(uniform ring step {steps[0] if steps else 'n/a'})"
               if all_fwd else
               "forward landing is non-stationary but NOT a uniform ring orbit")
    print(f"{_now()} VERDICT {verdict}", flush=True)
    return dict(mode="cycle", mu=args.mu, delta=args.delta, lam=lam,
                stationary=bool(out["stationary"]), elapsed=float(out["elapsed"]),
                lead_amp_mean=float(amp[half:].mean()),
                n_transitions=len(trans), ring_steps=sorted({int(s) for s in steps}),
                link_duration=period, n_forward=len(fwd), verdict=verdict,
                transitions=[(float(x[0]), int(x[1]), int(x[2]))
                             for x in trans[:40]])


def mode_recell(args, lam_c, beta):
    """(e) re-run a full ET1'' cell with the fold taken from the CONTINUATION.

    For mu=28 the tabulated threshold is 1.27e-2 below the fold the arclength
    actually turns at, so lambda = lam_c_table - delta is nowhere near the fold
    and the cell fails at the continuation stage. Re-parameterise that bead with
    the measured fold and run the identical cell logic.
    """
    from e34_necklace_wu_n2000 import run_cell
    coup, Q, _ = _setup(args.mu, args.delta, lam_c, beta)
    lam_c_fix = lam_c.copy()
    lam_c_fix[args.mu] = float(args.fold)
    print(f"{_now()} re-running cell mu={args.mu} delta={args.delta} with "
          f"lam_c := {args.fold:.6f} (table {lam_c[args.mu]:.6f})", flush=True)
    rec = run_cell(coup, Q, lam_c_fix, args.mu, args.delta, beta, args.tau,
                   args.t0, lambda msg: print(f"{_now()} {msg}", flush=True),
                   t_wu=args.t_wu)
    for k in ("a_saddle", "m_saddle", "a_node"):
        rec.pop(k, None)
    rec["mode"] = "recell"
    rec["lam_c_table"] = float(lam_c[args.mu])
    rec["lam_c_used"] = float(args.fold)
    print(f"{_now()} VERDICT {rec.get('status')}", flush=True)
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("land", "loc", "arc", "cycle", "recell"),
                   required=True)
    p.add_argument("--fold", type=float, default=None,
                   help="recell: fold parameter to use in place of lam_c[mu]")
    p.add_argument("--t-wu", dest="t_wu", type=float, default=1200.0)
    p.add_argument("--mu", type=int, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    args = p.parse_args(argv)

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
    if args.mode == "recell" and args.fold is None:
        raise SystemExit("--fold is required for mode recell")
    fn = dict(land=mode_land, loc=mode_loc, arc=mode_arc, cycle=mode_cycle,
              recell=mode_recell)[args.mode]
    out = fn(args, lam_c, args.beta)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = OUT / f"E34_ET1pp_followup_{args.mode}_mu{args.mu}_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"{_now()} wrote {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
