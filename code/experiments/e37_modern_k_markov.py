"""E37 -- what the Modern-Hopfield component buys on CORRELATED sequences.

E27 established, on the Markov-flip "video" (xi^{mu+1}_i = xi^mu_i s_i, iid flips,
consecutive-frame correlation c), that the ordinary Hebbian ring loses ordered
recall once c is large: the individual memories melt into delocalised packets and
the leader wanders back and forth. E35-V1 established, on ONE advected-texture
video, that the J x K factorial separates sharply and that K = PINV is the
condition of ordered recall. Neither measured the EFFECT SIZE against a tunable
correlation.

This runs the full factorial {J in hebb,pinv} x {K in hebb,pinv} against the E27
generator at several c, so the gain attributable to modernising K (and to
modernising J) can be read as a dose-response curve.

Everything is measured with ONE architecture-blind readout: the physical overlap
m(t) = xi tanh(beta u)/N. Using G-dagger-decoded coordinates as the observable for
the pinv arms only would build the answer into the measurement. The decoded
leader is recorded as a SECONDARY diagnostic, for all four arms alike.

Metrics per (architecture, c):
  coverage_fraction, forward_fraction, valid_cycle   -- E35 classify_cycle
  period_mean, period_cv
  selectivity  = m_leader - max_{nu != leader} m_nu, at the leader's peak
  packet_width = #{nu : m_nu > 0.5 m_leader}, at the leader's peak
The last two separate "the ring turns" from "the ring turns while resolving one
frame at a time", which is the failure mode E27 actually saw at high c.
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ACCELERATE_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from e35_dynamics import MhnReducedDDE, classify_cycle, integrate_chunked
from mhn_reduced import MhnContext, factorial_architectures

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/6_motifs_correles/data"

ARMS = ("JH_KH", "JP_KH", "JH_KP", "JP_KP")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_markov(n: int, p: int, c: float, seed: int) -> np.ndarray:
    """E27 generator, reproduced verbatim so the two experiments are comparable.

    xi^{mu+1}_i = xi^mu_i s_i with P(s_i = -1) = (1-c)/2, so the consecutive-frame
    correlation is c. The chain is OPEN: the seam (P-1 -> 0) carries q ~ c^{P-1},
    i.e. it is the one uncorrelated bond of the ring.
    """
    rng = np.random.default_rng(seed)
    flip = (1.0 - c) / 2.0
    xi = np.empty((p, n))
    xi[0] = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    for mu in range(1, p):
        s = np.where(rng.random(n) < flip, -1.0, 1.0)
        xi[mu] = xi[mu - 1] * s
    return xi


def peak_selectivity(m: np.ndarray, lead: np.ndarray) -> dict:
    """How cleanly one frame is resolved, read at each leader's own peak.

    For every bead that ever leads, take the sample where its overlap is largest
    among the samples it leads; report the margin over the runner-up and how many
    frames stay within half of it. A delocalised packet has a small margin and a
    wide support even when the leader label still advances.
    """
    p = m.shape[1]
    margin, width, height = [], [], []
    for nu in range(p):
        w = np.flatnonzero(lead == nu)
        if w.size == 0:
            continue
        i = w[np.argmax(m[w, nu])]
        row = m[i]
        top = float(row[nu])
        rest = np.delete(row, nu)
        margin.append(top - float(rest.max()))
        width.append(int(np.count_nonzero(row > 0.5 * top)) if top > 0 else p)
        height.append(top)
    if not margin:
        return dict(n_leaders=0, selectivity_median=float("nan"),
                    selectivity_min=float("nan"), packet_width_median=float("nan"),
                    peak_height_median=float("nan"))
    return dict(n_leaders=len(margin),
                selectivity_median=float(np.median(margin)),
                selectivity_min=float(np.min(margin)),
                packet_width_median=float(np.median(width)),
                peak_height_median=float(np.median(height)))


def run_one(xi: np.ndarray, arm: str, lam: float, beta: float, tau: float,
            t0: float, dt: float, t_total: float, sample_dt: float,
            transient: float) -> tuple[dict, np.ndarray, np.ndarray]:
    ctx = MhnContext.from_patterns(xi, beta)
    arch = factorial_architectures(ctx)[arm]
    sysP = MhnReducedDDE(arch, lam, tau, t0)
    p = ctx.p
    a0 = np.zeros(p)
    a0[0] = 1.0                                   # start ON frame 0
    out = integrate_chunked(sysP, a0, da_hist0=np.zeros(p), t_total=t_total,
                            dt=dt, record_every=max(1, int(round(sample_dt / dt))),
                            chunk_time=500.0)
    t, a = out["t"], out["a"]
    m = ctx.physical_overlaps_batch(a)
    diag = classify_cycle(t, m, transient_fraction=transient)
    rec = {k: v for k, v in diag.__dict__.items()}
    cut = int(np.floor(transient * len(t)))
    m_tail = m[cut:]
    rec.update(peak_selectivity(m_tail, np.argmax(m_tail, axis=1)))
    # secondary, architecture-aware readout: the decoded coefficient. Reported for
    # ALL arms so it never becomes the pinv arms' private yardstick.
    dec = ctx.apply_pinv(m_tail.T).T
    lead_dec = np.argmax(dec, axis=1)
    lead_phy = np.argmax(m_tail, axis=1)
    rec["decoded_leader_agreement"] = float(np.mean(lead_dec == lead_phy))
    ch = np.flatnonzero(np.diff(lead_dec) != 0) + 1
    if ch.size:
        ev = lead_dec[np.concatenate(([0], ch))]
        rec["decoded_forward_fraction"] = float(
            np.mean((ev[1:] - ev[:-1]) % p == 1))
        rec["decoded_coverage_fraction"] = float(len(np.unique(lead_dec)) / p)
    else:
        rec["decoded_forward_fraction"] = 0.0
        rec["decoded_coverage_fraction"] = float(len(np.unique(lead_dec)) / p)
    rec["gram"] = ctx.diagnostics.to_dict()
    return rec, t, m


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--c", type=float, required=True,
                   help="consecutive-frame correlation of the Markov video")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--p", dest="n_patterns", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sample-dt", dest="sample_dt", type=float, default=0.1)
    p.add_argument("--t-total", dest="t_total", type=float, default=6000.0)
    p.add_argument("--transient", type=float, default=0.5)
    p.add_argument("--save-traces", dest="save_traces", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)

    xi = make_markov(args.n, args.n_patterns, args.c, args.seed)
    q = float(np.mean(xi[:-1] * xi[1:]))          # measured bulk correlation
    print(f"{_now()} c={args.c} N={args.n} P={args.n_patterns} seed={args.seed} "
          f"lam={args.lam} measured bulk q={q:.4f}", flush=True)

    rows, traces = [], {}
    t_all = time.perf_counter()
    for arm in args.arms.split(","):
        t_a = time.perf_counter()
        rec, t, m = run_one(xi, arm, args.lam, args.beta, args.tau, args.t0,
                            args.dt, args.t_total, args.sample_dt, args.transient)
        rec.update(arm=arm, c=args.c, q_measured=q, lam=args.lam, seed=args.seed,
                   n=args.n, p=args.n_patterns, wall=time.perf_counter() - t_a)
        rows.append(rec)
        if args.save_traces:
            traces[f"t_{arm}"] = t.astype(np.float32)
            traces[f"m_{arm}"] = m.astype(np.float32)
        print(f"{_now()} {arm}: coverage={rec['coverage_fraction']:.3f} "
              f"forward={rec['forward_fraction']:.3f} "
              f"valid={rec['valid_cycle']} T={rec['period_mean']:.1f} "
              f"cv={rec['period_cv']:.1e} "
              f"select={rec['selectivity_median']:.4f} "
              f"width={rec['packet_width_median']:.0f} "
              f"[{rec['wall']:.0f}s] {rec['reason']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    # NB: the stem contains dots ("c0.60_..."), so Path.with_suffix would treat
    # ".60_seed42_..." as the extension and collapse every c onto one filename.
    base = OUT / f"E37_modernK_c{args.c:.2f}_seed{args.seed}{tag}_{stamp}"
    Path(str(base) + ".json").write_text(json.dumps(
        dict(c=args.c, q_measured=q, lam=args.lam, beta=args.beta, tau=args.tau,
             n=args.n, p=args.n_patterns, seed=args.seed, dt=args.dt,
             t_total=args.t_total, transient=args.transient, rows=rows),
        indent=1, default=float))
    if traces:
        np.savez_compressed(str(base) + ".npz", **traces)
    print(f"{_now()} wrote {base.name} [{time.perf_counter()-t_all:.0f}s]",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
