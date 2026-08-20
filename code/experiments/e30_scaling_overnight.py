"""
E30 -- overnight scaling campaign for the per-pattern threshold law lambda_c(mu).

Goal (user task 1): with many more (N, seed) points, pin down
  (a) the fixed-P series sigma(N) at P=100 (isolates the per-threshold
      fluctuation -> is it CLT, sigma ~ N^{-1/2}?);
  (b) the fixed-alpha series at alpha=0.05 (capacity-fixed, physically relevant),
      distribution shape, and the gap = max-min -> 0 as N -> infinity;
  (c) reach points at N=14000, 20000 to see the large-N trend of the gap.

Each (N,P,seed) is delegated to e24_thresholds.py, which saves ONE npz -> the
campaign is naturally checkpointed and resumable (existing outputs are skipped).
Outputs land in results/cycle/E30_scaling/.

Run:  python e30_scaling_overnight.py            (full ladder, background)
      python e30_scaling_overnight.py --dry      (print the plan + time estimate)
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, sys, time, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TRACER = os.path.join(HERE, "e24_thresholds.py")
OUTDIR = os.path.join(os.path.dirname(HERE), "results", "cycle", "E30_scaling")
os.makedirs(OUTDIR, exist_ok=True)
NPROC = 8

# ---- the ladder -------------------------------------------------------------
# fixed P=100 (alpha shrinks): isolates the fluctuation of a single threshold
FIXED_P = [
    (1000, 100, 24), (1414, 100, 24), (2000, 100, 24), (2828, 100, 24),
    (4000, 100, 24), (5657, 100, 20), (8000, 100, 20), (11314, 100, 16),
    (16000, 100, 16), (22627, 100, 10), (32000, 100, 8),
]
# fixed alpha=0.05 (capacity fixed): the physical series. eig ON up to 10000.
FIXED_A = [
    (1000, 50, 24), (1414, 71, 24), (2000, 100, 20), (2828, 141, 16),
    (4000, 200, 12), (5657, 283, 8), (8000, 400, 6), (10000, 500, 2),
]
# N>=10000 fixed-alpha: use --no-eig (eig cross-check non-essential, bisection lam_c
# is the primary estimator, already validated at N<=8000; the O(P^2 N) eig calls at
# P=500,N=10000 were pathologically slow ~48 min/seed). Reordered after edit 2026-07-08.
# reach points: bigger N, bisection-only (--no-eig), for the gap trend. REVISED
# 2026-07-09 00:25: even no-eig, Woodbury O(P^2 N) per solve makes cost ~N^{3-4};
# measured N=10000/P500 no-eig ~35 min/seed => N=14000 ~100 min, N=20000 ~6 h/seed
# (my earlier "N=20000 overnight" estimate was far too optimistic). Realistic
# overnight ceiling at full P, alpha=0.05 is ~N=14000. Dropped N=20000; N=14000 x1.
REACH = [
    (14000, 700, 1, True),
]
SEED0 = 1000   # fresh seed range, does not collide with the seed-42 runs


def out_path(N, P, seed):
    return os.path.join(OUTDIR, f"E30_N{N}_P{P}_s{seed}.npz")


def jobs():
    for N, P, ns in FIXED_P:
        for k in range(ns):
            yield (N, P, SEED0 + k, False)
    for N, P, ns in FIXED_A:
        for k in range(ns):
            yield (N, P, SEED0 + k, N >= 10000)      # no-eig for the slow large-N
    for N, P, ns, noeig in REACH:
        for k in range(ns):
            yield (N, P, SEED0 + k, noeig)


def run_one(N, P, seed, no_eig):
    op = out_path(N, P, seed)
    if os.path.exists(op):
        return "skip"
    cmd = [PY, TRACER, "--N", str(N), "--P", str(P), "--seed", str(seed),
           "--nproc", str(NPROC), "--out", op]
    if no_eig:
        cmd.append("--no-eig")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-3:] if r.stdout else []
    print(f"[E30] N={N} P={P} s{seed}{' [no-eig]' if no_eig else ''}: "
          f"{'OK' if r.returncode == 0 else 'FAIL'}")
    for ln in tail:
        print("     " + ln)
    if r.returncode != 0 and r.stderr:
        print("     ERR " + r.stderr.strip().splitlines()[-1])
    sys.stdout.flush()
    return "done" if r.returncode == 0 else "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    allj = list(jobs())
    print(f"[E30] {len(allj)} runs planned -> {OUTDIR}")
    if args.dry:
        for N, P, seed, ne in allj:
            print(f"   N={N} P={P} s{seed}{' no-eig' if ne else ''}"
                  f"{'  (exists)' if os.path.exists(out_path(N,P,seed)) else ''}")
        return
    t0 = time.time()
    done = skip = fail = 0
    for N, P, seed, ne in allj:
        st = run_one(N, P, seed, ne)
        done += st == "done"; skip += st == "skip"; fail += st == "fail"
    print(f"\n[E30] finished: {done} done, {skip} skipped, {fail} failed, "
          f"{(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
