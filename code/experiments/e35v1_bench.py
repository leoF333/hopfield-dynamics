"""E35-V1 step B: benchmark + smoke test BEFORE any heavy run.

Measures integrator throughput (steps/s) and RSS for the frozen dimensioning,
certifies exact periodicity of the new generator, and projects the total wall
cost of the whole V1 (design 6 seeds x 4 architectures + eval 10 seeds x 1
architecture).  No hidden-frame metric and no design decision is taken here.

Run: /opt/homebrew/Caskroom/miniforge/base/envs/mcmc_env/bin/python src/e35v1_bench.py
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e35_video_tools import (  # noqa: E402
    FramePreprocessor,
    loop_displacement_statistics,
    make_nonlinear_advection_loop,
    seam_diagnostics,
    split_keyframes,
)
from mhn_reduced import MhnContext, factorial_architectures  # noqa: E402
from e35_dynamics import (  # noqa: E402
    MhnReducedDDE,
    classify_cycle,
    integrate_chunked,
)

HEIGHT = WIDTH = 64
K = 4
BETA = 20.0
LAM = 0.9
TAU = 10.0
DT = 0.05
T0 = 1.0


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024.0 * 1024.0)  # darwin reports bytes


def build(p: int, seed: int, winding=(2, 1)):
    frames = make_nonlinear_advection_loop(
        p * K, height=HEIGHT, width=WIDTH, winding=winding, seed=seed
    )
    split = split_keyframes(frames, K)
    pre = FramePreprocessor.fit_from_keyframes(split.keyframes, binary=False)
    patterns = pre.encode(split.keyframes).reshape(p, -1)
    context = MhnContext.from_patterns(patterns, BETA, tsvd_rtol=None)
    return frames, split, pre, patterns, context


def bench_steps(p: int, seed: int, t_bench: float) -> tuple[float, dict]:
    _, _, _, _, context = build(p, seed)
    arch = factorial_architectures(context)["JP_KP"]
    system = MhnReducedDDE(arch, LAM, TAU, T0)
    a0 = np.zeros(p)
    a0[0] = 0.95
    a0[1] = 0.05
    t_start = time.time()
    sol = integrate_chunked(
        system, a0, da_hist0=np.zeros_like(a0), t_total=t_bench, dt=DT,
        record_every=1, chunk_time=t_bench,
    )
    elapsed = time.time() - t_start
    steps = int(round(t_bench / DT))
    return steps / elapsed, {
        "p": p, "steps": steps, "elapsed": elapsed,
        "cond": context.diagnostics.condition_retained,
        "rank": context.diagnostics.rank,
        "sol_samples": len(sol["t"]),
    }


def main() -> int:
    print("=== E35-V1 bench (step B) ===")
    print(f"grid {HEIGHT}x{WIDTH} N={HEIGHT*WIDTH}, k={K}, dt={DT}, tau={TAU}, "
          f"lam={LAM}, beta={BETA}")

    # ---- periodicity certification of the generator -------------------------
    closed = make_nonlinear_advection_loop(
        64, height=HEIGHT, width=WIDTH, winding=(2, 1), seed=1,
        include_endpoint=True,
    )
    print(f"periodic closure max|frame[-1]-frame[0]| = "
          f"{np.max(np.abs(closed[-1] - closed[0])):.3e}")
    loop = make_nonlinear_advection_loop(
        256, height=HEIGHT, width=WIDTH, winding=(2, 1), seed=1
    )
    seam = seam_diagnostics(loop)
    print("seam diagnostics:", {kk: round(vv, 5) for kk, vv in seam.items()})
    disp = loop_displacement_statistics(
        256, K, height=HEIGHT, width=WIDTH, winding=(2, 1),
        curve_amplitude=(6.0, 4.0), speed_modulation=0.35,
    )
    print("keyframe displacement (px):", {kk: round(vv, 3) for kk, vv in disp.items()})
    print(f"frames range [{loop.min():.3f},{loop.max():.3f}] dtype {loop.dtype}")

    # ---- throughput at both candidate dimensionings --------------------------
    results = {}
    for p in (64, 100):
        rate, info = bench_steps(p, seed=1, t_bench=20.0)
        results[p] = rate
        print(f"P={p:3d}: {rate:8.1f} steps/s  ({info['elapsed']:.2f} s for "
              f"{info['steps']} steps)  Gram rank={info['rank']} "
              f"cond={info['cond']:.3e}")
    print(f"peak RSS so far: {rss_mb():.1f} MB")

    # ---- measure the recall period to size t_total ---------------------------
    for p in (64,):
        _, _, _, _, context = build(p, seed=1)
        arch = factorial_architectures(context)["JP_KP"]
        system = MhnReducedDDE(arch, LAM, TAU, T0)
        a0 = np.zeros(p)
        a0[0] = 0.95
        a0[1] = 0.05
        t_probe = 4000.0
        t_start = time.time()
        sol = integrate_chunked(
            system, a0, da_hist0=np.zeros_like(a0), t_total=t_probe, dt=DT,
            record_every=1, chunk_time=500.0,
        )
        m = context.physical_overlaps_batch(sol["a"])
        cyc = classify_cycle(sol["t"], m, transient_fraction=0.5)
        print(f"P={p} probe t_total={t_probe} took {time.time()-t_start:.1f} s: "
              f"valid={cyc.valid_cycle} coverage={cyc.coverage_fraction:.3f} "
              f"fwd={cyc.forward_fraction:.3f} tours={cyc.tours_estimate:.2f} "
              f"period_mean={cyc.period_mean} cv={cyc.period_cv} "
              f"reason={cyc.reason}")
        link = cyc.median_link_duration
        print(f"median link duration = {link}")

    # ---- projection ----------------------------------------------------------
    print("\n--- cost projection (see report) ---")
    for p in (64, 100):
        rate = results[p]
        for t_total in (2000.0, 3000.0, 4000.0, 6000.0, 9000.0):
            steps = t_total / DT
            per_run = steps / rate
            design = 6 * 4 * per_run
            evalph = 10 * per_run
            print(f"P={p:3d} t_total={t_total:6.0f}: run {per_run/60:6.2f} min | "
                  f"design {design/60:6.1f} min | eval {evalph/60:5.1f} min | "
                  f"TOTAL {(design+evalph)/3600:5.2f} h")
    print(f"peak RSS: {rss_mb():.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
