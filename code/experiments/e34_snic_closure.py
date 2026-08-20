"""Does the last bead's invariant circle CLOSE? -- SNIC vs bistability.

ET1'' left the decisive question open for a purely technical reason. At mu=91
(lam_c = 0.32763 ~ lambda*) the forward W^u branch was integrated to t_max=1200
while the estimated full-ring period is ~1900: the orbit had advanced from bead
~21 to bead 51 when integration stopped, i.e. ~40 links (~760 t.u.) short of
returning to node_91. So "non-stationary at t_max" is compatible with BOTH

  (a) an in-transit orbit that closes back onto node_91
      -> node + saddle + two connections = an INVARIANT CIRCLE, the SNIC picture;
  (b) a genuine attracting limit cycle coexisting with the stable node_91
      -> BISTABILITY, which is NOT a SNIC.

This integrates the same branch far past one tour and classifies:
  * settles on node_mu            -> circle closed (SNIC)
  * settles on another node       -> chain leaves the ring elsewhere
  * still travelling after >=2 full tours with a stable tour time
                                  -> recurrent orbit, coexisting cycle

The saddle and its unstable rate are RELOADED from the ET1'' npz (no contour
work); only the variational null mode is rebuilt, which costs one SVD.
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
    NumpyLowRankCouplings, exponential_history, extract_null_mode,
    gram_distance, gram_matrix, make_iid_patterns, reduced_field_coefficients,
    solve_memory_node,
)
from e34b_edge_tracking import solve_alive_nodes
from reduced_spectrum import GJ_GK

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"
THR = ROOT / "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cell(mu: int, delta: float, z_u_override: float | None = None):
    npz = sorted(OUT.glob("E34_ET1pp_wu_results_*.npz"))[-1]
    d = np.load(npz, allow_pickle=True)
    idx = [i for i in range(len(d["mu"]))
           if int(d["mu"][i]) == mu and abs(float(d["delta"][i]) - delta) < 1e-9]
    if not idx:
        raise SystemExit(f"cell mu={mu} delta={delta} absent from {npz.name}")
    i = idx[0]
    a_s = np.asarray(d["a_saddle"][i], float)
    z_u = float(d["z_u"][i])
    if z_u_override is not None:
        # the delta=0.020 cell has z_u = NaN: its rectangle localisation failed
        # while the count was stable at 1 with a bracketed REAL root, so the
        # unique unstable root IS that real root (ET1'' sec. 4.3 fallback).
        z_u = float(z_u_override)
    if not np.all(np.isfinite(a_s)) or not np.isfinite(z_u):
        raise SystemExit(f"cell mu={mu} delta={delta} has no usable saddle/z_u")
    return a_s, z_u, npz.name


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=int, default=91)
    p.add_argument("--delta", type=float, default=0.005)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--t-max", dest="t_max", type=float, default=10000.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--chunk", type=float, default=250.0)
    p.add_argument("--z-u", dest="z_u_override", type=float, default=None,
                   help="override the reloaded z_u (delta=0.020 fallback)")
    args = p.parse_args(argv)

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
    lam = float(lam_c[args.mu] - args.delta)
    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    P = coup.P
    sysP = ReducedDDE(xi, args.beta, lam, args.tau, args.t0)

    a_s, z_u, src = load_cell(args.mu, args.delta, args.z_u_override)
    print(f"{_now()} mu={args.mu} delta={args.delta} lam={lam:.6f} "
          f"z_u={z_u:.6f} (saddle from {src})", flush=True)

    node_u, ok, nres = solve_memory_node(coup, args.mu, lam, args.beta)
    a_node = reduced_field_coefficients(coup, node_u, Q)
    a_nodes = solve_alive_nodes(coup, lam, args.beta, lam_c)
    a_nodes[int(args.mu)] = a_node
    print(f"{_now()} node_{args.mu} res={nres:.1e}; census alive={sorted(a_nodes)}",
          flush=True)

    # variational null mode at the reloaded root (one SVD, no contour work)
    u_s = coup._xi_np.T @ a_s
    GJ, GK = GJ_GK(coup, u_s, args.beta, lam)
    S = np.roll(np.eye(P), 1, axis=0)

    def char_red(z):
        return (args.t0 * z + 1.0) * np.eye(P) - (1.0 - lam) * GJ \
            - lam * np.exp(-z * args.tau) * (S @ GJ)

    mode = extract_null_mode(char_red, z_u, Q=Q, expect_real=True)
    v = np.asarray(mode.vector, float)
    if v[(args.mu + 1) % P] < 0:
        v = -v
    print(f"{_now()} mode residual={mode.residual:.1e} "
          f"realif={mode.realification_error:.1e}", flush=True)

    # ---- long integration of the FORWARD branch, chunk by chunk -------------
    _, hist, dhist = exponential_history(a_s, z_u, v, args.eps, args.tau,
                                         args.dt, Q=Q)
    rec_every = int(round(1.0 / args.dt))          # one sample per time unit
    elapsed, t_all, lead_all, d_node_all = 0.0, [], [], []
    stationary = False
    t_wall = time.perf_counter()
    while elapsed < args.t_max - 1e-9:
        step = min(args.chunk, args.t_max - elapsed)
        out = sysP.integrate(hist, step, args.dt, record_every=rec_every,
                             da_hist0=dhist)
        # out["t"] already starts at record_every*dt, so nothing to drop here:
        # dropping a sample per chunk would fake a leader jump at the seam.
        a_chunk = out["a"]
        t_chunk = out["t"] + elapsed
        hist, dhist = out["hist"], out["dhist"]
        elapsed += step
        m = np.array([coup.overlap_raw(np.tanh(args.beta * (coup._xi_np.T @ x)))
                      for x in a_chunk])
        lead = np.argmax(np.abs(m), axis=1)
        d_node = np.array([float(gram_distance(x, a_node, Q)) for x in a_chunk])
        t_all.append(t_chunk); lead_all.append(lead); d_node_all.append(d_node)
        # stationarity over the last 2 tau
        w = int(round(2.0 * args.tau))
        tail = a_chunk[-w:] if len(a_chunk) >= w else a_chunk
        spread = max(float(gram_distance(tail[k], tail[-1], Q))
                     for k in range(len(tail)))
        print(f"{_now()}   t={elapsed:7.0f} lead={int(lead[-1]):3d} "
              f"d_G(node_{args.mu})={d_node[-1]:.4f} min_d={d_node.min():.4f} "
              f"spread(2tau)={spread:.2e} [{time.perf_counter()-t_wall:.0f}s]",
              flush=True)
        if spread < 1e-9:
            stationary = True
            break
    t = np.concatenate(t_all); lead = np.concatenate(lead_all)
    d_node = np.concatenate(d_node_all)

    # ---- classification ----------------------------------------------------
    a_final = None
    out_rec = dict(mu=args.mu, delta=args.delta, lam=lam, z_u=z_u,
                   t_max=args.t_max, t_reached=float(elapsed),
                   stationary=bool(stationary),
                   d_node_final=float(d_node[-1]),
                   d_node_min=float(d_node.min()),
                   t_of_min=float(t[int(np.argmin(d_node))]),
                   lead_final=int(lead[-1]))
    # ring advance and tour times: cumulative forward steps of the leader
    changes = [(float(t[i]), int(lead[i - 1]), int(lead[i]))
               for i in range(1, len(t)) if lead[i] != lead[i - 1]]
    steps = [((b - a_) % P) for _, a_, b in changes]
    advance = np.cumsum([s for s in steps]) if steps else np.array([])
    n_tours = float(advance[-1] / P) if advance.size else 0.0
    tour_times = []
    if advance.size:
        for k in range(1, int(advance[-1] // P) + 1):
            j = int(np.searchsorted(advance, k * P))
            if j < len(changes):
                tour_times.append(float(changes[j][0]))
    tour_dt = list(np.diff(tour_times)) if len(tour_times) >= 2 else []
    out_rec.update(n_leader_changes=len(changes),
                   ring_steps=sorted({int(s) for s in steps}),
                   n_tours=n_tours, tour_times=tour_times,
                   tour_dt=[float(x) for x in tour_dt],
                   tour_dt_cv=(float(np.std(tour_dt) / np.mean(tour_dt))
                               if len(tour_dt) >= 2 else float("nan")))

    if stationary:
        # which object did it settle on?
        d_all = {k: float(gram_distance(a_chunk[-1], a_nodes[k], Q))
                 for k in a_nodes}
        best = min(d_all, key=d_all.get)
        out_rec.update(settled_on=int(best), settled_dist=d_all[best])
        if best == args.mu and d_all[best] < 1e-3:
            verdict = ("CIRCLE CLOSED: the forward branch returns to "
                       f"node_{args.mu} -- invariant circle, SNIC picture")
        else:
            verdict = (f"settled on node_{best} (d_G={d_all[best]:.2e}), "
                       f"not back on node_{args.mu}")
    elif n_tours >= 2.0 and len(tour_dt) >= 2 and out_rec["tour_dt_cv"] < 0.05:
        verdict = (f"RECURRENT ORBIT: {n_tours:.2f} tours, tour time "
                   f"{np.mean(tour_dt):.1f} (cv {out_rec['tour_dt_cv']:.1e}) "
                   f"-- coexisting limit cycle, i.e. BISTABILITY, not a SNIC")
    else:
        verdict = (f"INCONCLUSIVE: {n_tours:.2f} tours reached, not stationary, "
                   "tour time not yet repeatable -- integrate further")
    out_rec["verdict"] = verdict
    print(f"{_now()} tours={n_tours:.2f} tour_times={[round(x) for x in tour_times]} "
          f"cv={out_rec['tour_dt_cv']:.2e}", flush=True)
    print(f"{_now()} VERDICT {verdict}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = OUT / f"E34_snic_closure_mu{args.mu}_d{args.delta}_{stamp}.json"
    path.write_text(json.dumps(out_rec, indent=1, default=float))
    np.savez_compressed(
        OUT / f"E34_snic_closure_mu{args.mu}_d{args.delta}_{stamp}.npz",
        t=t, lead=lead, d_node=d_node, a_node=a_node, a_saddle=a_s,
        lam=lam, z_u=z_u, mu=args.mu, delta=args.delta)
    print(f"{_now()} wrote {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
