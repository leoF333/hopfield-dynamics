"""
E31 (large-alpha task, Exp C) -- per-pattern memory SURVIVAL census.

Robust, tracer-independent probe of the static memory -> (no memory) breakdown as
alpha grows toward capacity. For each pattern mu, beta-anneal Newton (5->20) from
u = 0.99 xi^mu at the target (alpha, lambda) and ask: does a macroscopic, stable
memory fixed point still exist on that pattern?

(Method choice: Newton from DIFFUSE random ICs almost never converges -- verified
in the smoke test, 4/40. Newton from the pattern itself with beta-annealing is the
robust method used across this codebase, 40/40 at alpha=0.05. The order parameter
is the FRACTION of patterns that retain a memory, f_mem(alpha,lambda); the
proliferation of spurious/mixture attractors is measured dynamically in Exp A.)

Per (alpha, lambda, seed) over ALL P patterns, record: converged?, max overlap,
n_cond = #{|a_mu|>0.15}, eigmax_M (stable?). Deliverables:
  - f_mem(alpha, lambda) = fraction of patterns with a macroscopic (maxov>0.7)
    stable memory  -> the memory-phase order parameter;
  - overlap / n_cond distributions of survivors;
  - pairwise |q| between distinct survivor states (do they stay distinct patterns
    or collapse together = spin-glass condensation?).

CLI: e31_fixedpoint_census.py --N 2000 --alphas 0.05,0.07,0.09,0.11,0.13 \
     --lams 0.0,0.05,0.15,0.25 --seeds 42,43 --nproc 8 \
     --out results/cycle/E31_survival_N2000.npz
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool

from couplings import Couplings, make_patterns
from robust_branch import woodbury_newton, eigmax_M

BETA = 20.0
ANNEAL = (5.0, 8.0, 12.0, 16.0, BETA)
_CTX = {}


def _init(N, P, seed):
    os.environ["OMP_NUM_THREADS"] = "2"
    xi, xis = make_patterns(N, P, seed)
    coup = Couplings(xi, xis); coup._use_np = True
    _CTX.update(N=N, P=P, xi=xi, coup=coup, gram=(xi @ xi.T) / N)


def _amp(u):
    return np.linalg.solve(_CTX["gram"], _CTX["coup"].overlap_raw(u))


def _worker(args):
    mu, lam = args
    xi, N, coup = _CTX["xi"], _CTX["N"], _CTX["coup"]
    u = 0.99 * xi[mu].copy()
    ok = False
    for b in ANNEAL:
        u, ok = woodbury_newton(coup, u, lam, b, tol=1e-12)
        if not ok:
            return dict(mu=mu, conv=False)
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    if res > 1e-9:
        return dict(mu=mu, conv=False)
    a = _amp(u); aa = np.abs(a)
    return dict(mu=mu, conv=True, a=a, maxov=float(aa.max()),
                lead=int(np.argmax(aa)), ncond=int((aa > 0.15).sum()),
                eigmax=float(eigmax_M(coup, u, lam, BETA)))


def run_cell(alpha, lam, N, seed, nproc):
    P = round(alpha * N)
    t0 = time.time()
    with Pool(nproc, initializer=_init, initargs=(N, P, seed)) as pool:
        res = pool.map(_worker, [(mu, lam) for mu in range(P)])
    surv = [r for r in res if r["conv"] and r["maxov"] > 0.7 and r["eigmax"] < 0]
    n_surv = len(surv)
    f_mem = n_surv / P
    maxovs = np.array([r["maxov"] for r in surv]) if surv else np.array([np.nan])
    nconds = np.array([r["ncond"] for r in surv]) if surv else np.array([np.nan])
    # pairwise |q| between distinct survivor amplitude vectors (leading-normalized)
    qs = []
    V = [np.abs(r["a"]) / (np.linalg.norm(r["a"]) + 1e-12) for r in surv]
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            qs.append(abs(float(V[i] @ V[j])))
    qs = np.array(qs) if qs else np.array([np.nan])
    dt = time.time() - t0
    print(f"[E31] a={alpha:.3f} P={P} lam={lam:.3f} s{seed}: "
          f"f_mem={f_mem:.3f} ({n_surv}/{P}), "
          f"<maxov>={np.nanmedian(maxovs):.3f}, <ncond>={np.nanmedian(nconds):.1f}, "
          f"P(q) med={np.nanmedian(qs):.3f}  ({dt:.0f}s)")
    sys.stdout.flush()
    return dict(alpha=alpha, lam=lam, P=P, seed=seed, n_surv=n_surv, f_mem=f_mem,
                maxovs=maxovs, nconds=nconds, qs=qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--alphas", default="0.05,0.07,0.09,0.11,0.13")
    ap.add_argument("--lams", default="0.0,0.05,0.15,0.25")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "cycle", f"E31_survival_N{args.N}.npz")

    print(f"[E31] survival census: N={args.N}, alphas={alphas}, lams={lams}, "
          f"seeds={seeds}")
    save = dict(N=args.N, alphas=np.array(alphas), lams=np.array(lams),
                seeds=np.array(seeds))
    fmem = np.full((len(alphas), len(lams), len(seeds)), np.nan)
    for ia, a in enumerate(alphas):
        for il, lam in enumerate(lams):
            for isd, sd in enumerate(seeds):
                r = run_cell(a, lam, args.N, sd, args.nproc)
                fmem[ia, il, isd] = r["f_mem"]
                tag = f"a{a:.3f}_l{lam:.3f}_s{sd}".replace(".", "p")
                for k in ("maxovs", "nconds", "qs"):
                    save[f"{k}_{tag}"] = r[k]
    save["f_mem"] = fmem
    np.savez(out, **save)
    print(f"\n[E31] f_mem(alpha,lam) [seed-averaged]:")
    for ia, a in enumerate(alphas):
        row = "  ".join(f"{np.nanmean(fmem[ia, il]):.2f}" for il in range(len(lams)))
        print(f"  a={a:.3f}: [lam {lams}] {row}")
    print(f"[E31] -> {out}")


if __name__ == "__main__":
    main()
