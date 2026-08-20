"""Above lambda*: is there ONE limit cycle, and is it the same for every pattern?

This is the direct form of the question. Everything measured so far concerns
lambda BELOW lambda*, where the invariant circle still carries equilibria. Above
lambda* the pairs are gone and the circle should be a genuine periodic orbit --
but "the recall cycle" has always been observed from a handful of initial
conditions, never from all P patterns, and never with the orbits compared to
each other.

So: start from EVERY pattern (a = e_mu, i.e. u = xi^mu), integrate past the
transient, and identify the attractor reached:

  * is it periodic, and with what period,
  * does the leading bead advance by +1 and sweep the WHOLE ring,
  * and -- the decisive part -- is the orbit the SAME one for every starting
    pattern? Two trajectories on the same cycle differ only by a time shift, so
    they must share the period AND the state at a canonical section. The section
    used here is the instant the leading bead switches to bead 0; the state
    m at that instant is a translation-invariant fingerprint of the orbit.

Distinct fingerprints would mean several coexisting cycles (different patterns
belonging to different attractors); identical fingerprints mean one global cycle
threading all P patterns.
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

# every BLAS backend must be pinned, not just OMP: on macOS the big overlap
# matmuls otherwise spawn threads per process, and 7 shards oversubscribe the
# machine (measured: 8.3 min/bead instead of 1.5).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ACCELERATE_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from cycle_reduced import ReducedDDE
from e34_lib import NumpyLowRankCouplings, gram_matrix, make_iid_patterns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def overlaps_batch(coup, A, beta, block=4000):
    """m(t) for a whole trajectory, in BLAS blocks.

    The per-sample Python loop was the reason sampling had to stay coarse at
    1 t.u.; at that resolution the parabolic peak fit carries ~0.2 t.u. of
    error, which is the entire fingerprint discrepancy seen in the pilot.
    Batched, 0.1 t.u. sampling costs almost nothing and buys ~100x on the peak
    times (the error of a parabolic fit scales with the square of the step).
    """
    xi = coup._xi_np
    out = np.empty((A.shape[0], xi.shape[0]), dtype=float)
    for i in range(0, A.shape[0], block):
        U = A[i:i + block] @ xi                    # (b, N)
        np.multiply(U, beta, out=U)
        np.tanh(U, out=U)
        out[i:i + block] = U @ xi.T / xi.shape[1]
    return out


def peak_times(t, M, P):
    """Sub-sample peak time and height of each m_nu, on ONE ring pass.

    Two pitfalls the pilot exposed, both fixed here:
      * the state at a section instant is not a usable fingerprint -- the
        section is found on a 1 t.u. grid while a link lasts ~16 t.u., so a
        half-sample offset moves it by ~0.2 in norm;
      * taking each bead's GLOBAL maximum over a multi-period window picks the
        peak from an arbitrary pass (they are equal up to round-off), which
        scrambles the peak ORDER. The window must hold exactly one pass.
    Peak times are then sub-sample accurate (parabolic), and the intervals
    between consecutive peaks around the ring are translation invariant.
    """
    out = {}
    for nu in range(P):
        y = np.abs(M[:, nu])
        i = int(np.argmax(y))
        if i <= 0 or i >= len(y) - 1:
            continue
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = (y0 - 2.0 * y1 + y2)
        shift = 0.5 * (y0 - y2) / den if abs(den) > 1e-300 else 0.0
        shift = float(np.clip(shift, -1.0, 1.0))
        out[nu] = (float(t[i] + shift * (t[1] - t[0])), float(y1))
    return out


def phase_waveform(t, M, P, T, anchor=0, n_phase=256):
    """The orbit's waveform, resampled on a phase grid anchored on bead `anchor`.

    This is the fingerprint that actually decides the question. Peak-derived
    dwell times inherit the sampling error of 100 separate parabolic fits; here
    a SINGLE peak (bead `anchor`) sets the phase origin, and the whole waveform
    m_nu(phase) is then linearly interpolated on the recorded grid. Two
    trajectories on the same periodic orbit differ only by a time shift, so
    after this alignment their waveforms must coincide to interpolation error.
    """
    y = np.abs(M[:, anchor])
    h = float(t[1] - t[0])

    def refine(i):
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2.0 * y1 + y2
        sh = 0.5 * (y0 - y2) / den if abs(den) > 1e-300 else 0.0
        return float(t[i] + float(np.clip(sh, -1.0, 1.0)) * h)

    # local maxima of the anchor bead, refined to sub-sample accuracy
    loc = [i for i in range(1, len(y) - 1)
           if y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] > 0.5]
    if not loc:
        return None, np.nan, np.nan
    # a shoulder on the anchor bead's overlap registers as a SECOND local maximum
    # a few samples from the true peak; "the last two maxima" then returns a few
    # t.u. instead of ~T, and the phase grid built from it is meaningless (this is
    # exactly what beads 25/47/78/86 hit at lambda=0.35). Keep only the highest
    # maximum inside each half-period cluster before reading any period off them.
    if np.isfinite(T) and T > 0 and len(loc) > 1:
        keep = []
        for i in sorted(loc, key=lambda j: -y[j]):
            if all(abs(t[i] - t[j]) > 0.5 * T for j in keep):
                keep.append(i)
        loc = sorted(keep)
    t0 = refine(loc[-1])
    # the PERIOD from consecutive anchor peaks: the section-based estimate is
    # quantised by the sampling step, and a 1e-4 relative period error already
    # drifts the phase by ~0.1 t.u. over a full turn, which is the whole residual
    # discrepancy seen in the previous pilot. Only differences compatible with the
    # section estimate count as consecutive passes.
    if len(loc) >= 2:
        dT = np.diff([refine(i) for i in loc])
        if np.isfinite(T) and T > 0:
            dT = dT[np.abs(dT - T) < 0.2 * T]
        if dT.size:
            T = float(np.median(dT))
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    tq = t0 - T + phase * T                      # one full period ending at t0
    if tq[0] < t[0]:
        tq = t0 + phase * T
        if tq[-1] > t[-1]:
            return None, t0, T
    W = np.empty((n_phase, P))
    for nu in range(P):
        W[:, nu] = np.interp(tq, t, M[:, nu])
    return W, t0, T


def analyse(coup, t, M, P, section_bead=0):
    """Period, ring sweep and a translation-invariant fingerprint."""
    lead = np.argmax(np.abs(M), axis=1)
    ch = [(float(t[i]), int(lead[i - 1]), int(lead[i]))
          for i in range(1, len(t)) if lead[i] != lead[i - 1]]
    steps = [((b - a) % P) for _, a, b in ch]
    rec = dict(n_transitions=len(ch), ring_steps=sorted({int(s) for s in steps}),
               n_beads_visited=int(len(set(lead.tolist()))))
    # full-ring times: instants the leader switches TO the section bead
    hits = [(tt, i) for i, (tt, a, b) in enumerate(ch) if b == section_bead]
    rec["n_section_hits"] = len(hits)
    if len(hits) >= 2:
        per = np.diff([h[0] for h in hits])
        rec["period_mean"] = float(np.mean(per))
        rec["period_std"] = float(np.std(per))
        rec["period_cv"] = float(np.std(per) / max(np.mean(per), 1e-30))
        rec["periods"] = [float(x) for x in per]
    else:
        rec.update(period_mean=float("nan"), period_std=float("nan"),
                   period_cv=float("nan"), periods=[])
    # canonical, translation-invariant fingerprint: the dwell time on each bead,
    # read as the interval between consecutive peak times around the ring, plus
    # the peak height per bead. Both are properties of the ORBIT, not of when we
    # happened to look at it. The ring period is estimated from the link
    # durations (no section crossing required), and the peaks are searched in a
    # window holding exactly ONE pass.
    link = np.diff([c[0] for c in ch]) if len(ch) >= 2 else np.array([])
    rec["link_median"] = float(np.median(link)) if link.size else float("nan")
    # the link durations are quantised by the 1 t.u. sampling, so P*median is
    # off by ~2% (1600 vs 1633 measured); the section period is exact, use it.
    T_est = (rec["period_mean"] if np.isfinite(rec["period_mean"])
             else (float(P * np.median(link)) if link.size else float("nan")))
    rec["T_used"] = T_est
    fp = None
    pk = {}
    if np.isfinite(T_est) and T_est > 0 and t[-1] - t[0] > 1.15 * T_est:
        # a window slightly WIDER than one pass: a bead whose peak sits at the
        # edge would otherwise be dropped. Two peaks of the same bead differ by
        # exactly T, and the dwell is taken mod T, so the extra width is safe.
        w = t >= t[-1] - 1.15 * T_est
        pk = peak_times(t[w], M[w], P)
        if len(pk) == P:
            tp = np.array([pk[nu][0] for nu in range(P)])
            hp = np.array([pk[nu][1] for nu in range(P)])
            d = np.array([(tp[(nu + 1) % P] - tp[nu]) % T_est
                          for nu in range(P)])
            fp = np.concatenate([d, hp])
            rec["dwell_sum"] = float(d.sum())
            rec["dwell_min"] = float(d.min())
            rec["dwell_max"] = float(d.max())
            rec["dwell_argmax"] = int(np.argmax(d))
            rec["dwell_argmin"] = int(np.argmin(d))
            rec["peak_min"] = float(hp.min())
            rec["peak_mean"] = float(hp.mean())
    rec["n_peaks"] = len(pk)
    W, t_anchor, T_ref = (phase_waveform(t, M, P, T_est)
                          if np.isfinite(T_est) else (None, np.nan, np.nan))
    rec["anchor_time"] = float(t_anchor) if np.isfinite(t_anchor) else float("nan")
    rec["T_refined"] = float(T_ref) if np.isfinite(T_ref) else float("nan")
    rec["waveform_ok"] = W is not None
    return rec, fp, lead, W


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--beads", default="all")
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sample-dt", dest="sample_dt", type=float, default=0.1,
                   help="recording step; peak-time accuracy scales with it")
    p.add_argument("--t-total", dest="t_total", type=float, default=6000.0)
    p.add_argument("--transient", type=float, default=0.5,
                   help="fraction of t_total discarded before analysis")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    P = coup.P
    sysP = ReducedDDE(xi, args.beta, args.lam, args.tau, args.t0)
    beads = (list(range(P)) if args.beads == "all"
             else [int(x) for x in args.beads.split(",")])

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    print(f"{_now()} lam={args.lam} beads={len(beads)} t_total={args.t_total} "
          f"dt={args.dt}", flush=True)

    rows, fps = [], {}
    t_all = time.perf_counter()
    for mu in beads:
        t_b = time.perf_counter()
        a0 = np.zeros(P)
        a0[mu] = 1.0                       # u = xi^mu : start ON pattern mu
        hist, elapsed, ts, Ms = a0, 0.0, [], []
        while elapsed < args.t_total - 1e-9:
            step = min(500.0, args.t_total - elapsed)
            o = sysP.integrate(
                hist, step, args.dt,
                record_every=max(1, int(round(args.sample_dt / args.dt))))
            hist = o["hist"]
            ts.append(o["t"] + elapsed)
            Ms.append(overlaps_batch(coup, o["a"], args.beta))
            elapsed += step
        t = np.concatenate(ts)
        M = np.concatenate(Ms)
        cut = int(args.transient * len(t))
        rec, fp, lead, W = analyse(coup, t[cut:], M[cut:], P)
        rec.update(mu=int(mu), lam=args.lam, wall=time.perf_counter() - t_b)
        # is it stationary instead of periodic?
        tail = M[-int(round(2.0 * args.tau / args.sample_dt)):]
        rec["stationary"] = bool(np.max(np.linalg.norm(tail - tail[-1], axis=1))
                                 < 1e-9)
        rows.append(rec)
        if W is not None:
            fps[int(mu)] = W
        print(f"{_now()} mu={mu:3d} period={rec['period_mean']:.2f} "
              f"T={rec.get('T_refined', float('nan')):.4f} "
              f"dwell_sum={rec.get('dwell_sum', float('nan')):.2f} "
              f"slowest={rec.get('dwell_argmax', -1)} "
              f"steps={rec['ring_steps']} beads={rec['n_beads_visited']} "
              f"peaks={rec['n_peaks']} peak_min={rec.get('peak_min', float('nan')):.4f} "
              f"stat={rec['stationary']} [{rec['wall']:.0f}s]", flush=True)

    # ---- comparison across starting patterns ----------------------------
    keys = sorted(fps)
    summary = dict(lam=args.lam, n_ics=len(rows), n_with_fingerprint=len(keys))
    if len(keys) >= 2:
        ref = fps[keys[0]]
        d = {k: float(np.max(np.linalg.norm(fps[k] - ref, axis=1)))
             for k in keys}
        summary["fingerprint_ref_bead"] = keys[0]
        summary["fingerprint_dist_max"] = max(d.values())
        summary["fingerprint_metric"] = ("max over phase of ||m_ic(phase) "
                                         "- m_ref(phase)||, phase-aligned")
        summary["fingerprint_dist_median"] = float(np.median(list(d.values())))
        summary["fingerprint_dists"] = d
        per = [r["period_mean"] for r in rows if np.isfinite(r["period_mean"])]
        summary["period_mean_over_ics"] = float(np.mean(per)) if per else float("nan")
        summary["period_spread_over_ics"] = (float(np.max(per) - np.min(per))
                                             if per else float("nan"))
        same = summary["fingerprint_dist_max"] < 1e-2
        summary["verdict"] = (
            "ONE cycle: every starting pattern reaches the same orbit "
            f"(max fingerprint distance {summary['fingerprint_dist_max']:.2e})"
            if same else
            "SEVERAL attractors: fingerprints differ "
            f"(max {summary['fingerprint_dist_max']:.2e}, "
            f"median {summary['fingerprint_dist_median']:.2e})")
        print(f"\n{_now()} period over ICs: mean "
              f"{summary['period_mean_over_ics']:.3f}, spread "
              f"{summary['period_spread_over_ics']:.2e}", flush=True)
        print(f"{_now()} VERDICT {summary['verdict']}", flush=True)

    (OUT / f"E34_cycle_uniqueness_lam{args.lam:.6f}{tag}_{stamp}.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=1, default=float))
    if fps:
        np.savez_compressed(
            OUT / f"E34_cycle_uniqueness_lam{args.lam:.6f}{tag}_{stamp}.npz",
            **{f"fp_{k}": v for k, v in fps.items()})
    print(f"{_now()} wrote E34_cycle_uniqueness_lam{args.lam:.6f}{tag}_{stamp} "
          f"({time.perf_counter()-t_all:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
