"""
Multi-seed static-fold study via the robust Woodbury tracer (float64, exact).

For a given (N, alpha) it traces the xi^1-connected memory branch for several seeds
and reports the saddle-node fold lambda_c = fold (where eig_max(M) -> 0, tau-indep).
Near the Hopfield capacity (alpha~0.10) the fold is strongly sample- and N-dependent,
so a mean +/- std over seeds is the meaningful quantity.

Parallelism note: the tracer needs float64 EXACT Woodbury solves (GPU float32 is what
corrupted the branch tracing in the first place), so each trace runs on CPU. We
parallelize by running independent SEEDS concurrently across CPU cores.

Usage:
    python multiseed_fold.py --N 10000 --alpha 0.10 --seeds 42 43 44 45 46 --outdir <dir>
Output: <outdir>/fold_N{N}_a{alpha}.png  + .npz  + printed table.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from couplings import Couplings, make_patterns
from robust_branch import trace_branch

BETA = 20.0


def _one_seed(N, alpha, seed, lam_max, ds):
    cfg = dict(lam_min=0.0, lam_max=lam_max, ds=ds, n_overlaps=5,
               N=N, alpha=alpha, beta=BETA)
    P = round(alpha * N)
    xi, xis = make_patterns(N, P, seed)
    coup = Couplings(xi, xis); coup._use_np = True     # float64 exact path
    t0 = time.time()
    br, fold, status = trace_branch(coup, BETA, cfg)
    return dict(seed=seed, fold=fold, status=status,
                lam=br["lam"], eig=br["eig_max"], m1=br["m_nu"][:, 0],
                t=time.time() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--lam-max", type=float, default=0.6, dest="lam_max")
    p.add_argument("--ds", type=float, default=0.01)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = min(#seeds, cpu-2)")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    nw = args.workers or min(len(args.seeds), max(1, (os.cpu_count() or 2) - 2))
    print(f"multiseed fold: N={args.N} alpha={args.alpha} seeds={args.seeds} "
          f"workers={nw}", flush=True)

    t_all = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(_one_seed, args.N, args.alpha, s, args.lam_max, args.ds)
                for s in args.seeds]
        for f in futs:
            r = f.result(); res.append(r)
            print(f"  seed {r['seed']}: fold={r['fold']:.4f} ({r['status']}) "
                  f"{r['t']:.0f}s", flush=True)
    res.sort(key=lambda r: r["seed"])
    folds = np.array([r["fold"] for r in res])
    print(f"\nN={args.N} alpha={args.alpha}:  lambda_c = {folds.mean():.4f} "
          f"+/- {folds.std():.4f}  (over {len(folds)} seeds)   "
          f"[{time.time()-t_all:.0f}s]", flush=True)

    # ---- figure: eig_max(M) vs lambda per seed (crossing 0 = fold) ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(res)))
    for r, c in zip(res, cols):
        ax1.plot(r["lam"], r["eig"], color=c, lw=1.2, label=f"seed {r['seed']}")
        ax1.plot(r["fold"], 0.0, "o", color=c, ms=7, mec="k", mew=.5)
        ax2.plot(r["lam"], r["m1"], color=c, lw=1.2, label=f"seed {r['seed']}")
    ax1.axhline(0, color="k", lw=.8, ls=":")
    ax1.set_xlabel(r"$\lambda$"); ax1.set_ylabel(r"$\max\,\mathrm{Re}\,\mathrm{eig}(M)$")
    ax1.set_title("Static stability: eig_max(M) (fold where it hits 0)")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=.3)
    ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel(r"$m_1$")
    ax2.set_title("Memory overlap along the branch")
    ax2.grid(True, alpha=.3)
    fig.suptitle(f"$N={args.N},\\ \\alpha={args.alpha},\\ \\beta={BETA}$:  "
                 f"$\\lambda_c={folds.mean():.3f}\\pm{folds.std():.3f}$ "
                 f"({len(folds)} seeds)", fontsize=13)
    fig.tight_layout()
    tag = f"N{args.N}_a{args.alpha}"
    path = os.path.join(args.outdir, f"fold_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    np.savez(os.path.join(args.outdir, f"fold_{tag}.npz"),
             seeds=[r["seed"] for r in res], folds=folds,
             status=[r["status"] for r in res],
             lam_c_mean=folds.mean(), lam_c_std=folds.std())
    print(f"[multiseed_fold] -> {path}", flush=True)


if __name__ == "__main__":
    main()
