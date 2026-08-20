"""
E27 -- correlated patterns I: MARKOV FLIP CHAIN (user task 2, option 1).

Patterns form a "coherent video": xi^{mu+1}_i = xi^mu_i * s_i with iid flips
P(s_i=-1) = p, so consecutive-frame correlation c = 1-2p (c=0 <=> iid control).
The ring does NOT close: the seam bond (P-1 -> 0) has q ~ c^{P-1} ~ 0 while all
bulk bonds have q ~ c. E25 predicts (corr(q, lam_c) < 0) that the seam becomes
the STRONGEST bond (highest threshold) -- the last dike, where the cycle dies.

For each c in {0, 0.2, 0.4, 0.6, 0.8} (N=2000, P=100, seed 42):
  - trace all P memory->front branches (same operational lambda_c as E24);
  - stats: distribution of lambda_c(mu), seam vs bulk, corr(q_mu, lam_c) in bulk;
  - fold shape r_c(mu) vs lambda_c (does the E25 one-parameter family persist?).
Then a DYNAMICS check at c=0.6: does the recall cycle still run at lam=0.9, and
does the seeded descent die by pinning ON THE SEAM?

Outputs: figX_markov_video.png, E27_markov.npz, printed tables.
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
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sstats

from couplings import Couplings
import e24_thresholds as T

# CLI: e27_markov_thresholds.py [N] [P] [subset]   (defaults: 2000 100 0=all)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
P = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SUB = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SEED, BETA = 42, 20.0
NO_EIG = P > 200            # skip O(P^2 N) spectral calls at large P
SUFF = f"_N{N}" if N != 2000 else ""
CS = [0.0, 0.2, 0.4, 0.6, 0.8]
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")


def make_markov(N, P, c, seed):
    """Markov flip chain: consecutive-frame correlation c (open chain, seam P-1->0)."""
    rng = np.random.default_rng(seed)
    p = (1.0 - c) / 2.0
    xi = np.empty((P, N))
    xi[0] = np.where(rng.random(N) < 0.5, -1.0, 1.0)
    for mu in range(1, P):
        s = np.where(rng.random(N) < p, -1.0, 1.0)
        xi[mu] = xi[mu - 1] * s
    return xi


def _init_corr(c):
    os.environ["OMP_NUM_THREADS"] = "2"
    xi = make_markov(N, P, c, SEED)
    xis = np.roll(xi, -1, axis=0)
    coup = Couplings(xi, xis); coup._use_np = True
    T._CTX.update(N=N, P=P, xi=xi, coup=coup, gram=(xi @ xi.T) / N,
                  relaxed=True)   # dressed memories: identity = leading index only


def run_thresholds(c):
    lam_c = np.full(P, np.nan)
    shape = np.full((P, 2), np.nan)     # (a_mu, a_mu1) at the fold
    if SUB and SUB < P:
        rng = np.random.default_rng(1234)
        mus = sorted(set(rng.choice(P, SUB, replace=False).tolist()) | {P - 1})
    else:                               # seam branch always included
        mus = list(range(P))
    with Pool(6, initializer=_init_corr, initargs=(c,)) as pool:
        for r in tqdm(pool.imap_unordered(T._worker,
                                          [(mu, True, NO_EIG) for mu in mus]),
                      total=len(mus), desc=f"c={c}", leave=False):
            lam_c[r["mu"]] = r["lam_c"]
            if r["rows"] is not None and len(r["rows"]):
                shape[r["mu"]] = np.abs(r["rows"][-1][1:3])
    return lam_c, shape


results = {}

# NOTE: everything below is wrapped by fix_guard.py into _experiment_main(),
# called only under __name__ == "__main__" -- macOS spawn re-imports this module
# in every Pool worker, and unguarded top-level execution deadlocks (the first
# launch hung silently for 11 hours: daemonic children cannot create Pools).
def _experiment_main():
    print(f"[E27] Markov flip video, N={N}, P={P}, seed={SEED}")
    for c in CS:
        t0 = time.time()
        xi = make_markov(N, P, c, SEED)
        q = np.array([(xi[mu] @ xi[(mu + 1) % P]) / N for mu in range(P)])
        lam_c, shape = run_thresholds(c)
        ok = np.isfinite(lam_c)
        bulk = np.arange(P - 1)             # branches 0..P-2 tilt within the chain
        seam = P - 1                        # branch P-1 tilts toward xi^0 across the seam
        r_c = shape[:, 1] / np.maximum(shape[:, 0], 1e-12)
        okb = ok[bulk]
        corr_q = (sstats.pearsonr(q[bulk][okb], lam_c[bulk][okb])[0]
                  if okb.sum() > 3 else np.nan)
        results[c] = dict(lam_c=lam_c, q=q, shape=shape, r_c=r_c)
        print(f"  c={c:.1f}: q_bulk={q[bulk].mean():+.3f}  q_seam={q[seam]:+.3f}  "
              f"traced {ok.sum()}/{P}  "
              f"lam_c bulk: med={np.nanmedian(lam_c[bulk]):.4f} "
              f"[{np.nanmin(lam_c[bulk]):.4f}, {np.nanmax(lam_c[bulk]):.4f}] "
              f"sig={np.nanstd(lam_c[bulk]):.4f}  "
              f"SEAM lam_c={lam_c[seam]:.4f} "
              f"({'MAX of all' if ok[seam] and lam_c[seam] >= np.nanmax(lam_c) - 1e-9 else 'not max'})  "
              f"corr(q,lam_c)_bulk={corr_q:+.2f}  [{time.time()-t0:.0f}s]", flush=True)

    # --------------------------------------------------------- dynamics checks ----
    from cycle_reduced import ReducedDDE
    TAU, T0V, DT = 10.0, 1.0, 0.05

    def handoffs(A):
        lead = np.argmax(A, axis=1)
        chg = np.nonzero(np.diff(lead) != 0)[0]
        adv = np.sum((lead[chg + 1] - lead[chg]) % P == 1)
        return len(chg), adv, lead

    # (i) cycle QUALITY at lam=0.9 for every correlation level
    print("\n[E27dyn] recall-cycle quality at lam=0.90 vs frame correlation:")
    qual = {}
    for c in CS:
        xi = make_markov(N, P, c, SEED)
        sysP = ReducedDDE(xi, BETA, 0.9, TAU, T0V)
        a0 = np.zeros(P); a0[0] = 0.99; a0[1] = 0.05
        sol = sysP.integrate(a0, 600.0, DT, record_every=20)
        nchg, nadv, lead = handoffs(sol["a"][int(200 / (DT * 20)):])
        fwd = nadv / max(nchg, 1)
        qual[c] = (nchg, nadv, fwd, sol["a"][-1].copy())
        print(f"  c={c:.1f}: {nchg:3d} lead changes, {nadv:3d} forward "
              f"(fwd fraction {fwd:.2f}) -> "
              f"{'CYCLE RUNS' if nadv >= 10 and fwd > 0.8 else 'DEGRADED/NO cycle'}",
              flush=True)

    # (ii) seeded descent at c_dyn, from HIGH lambda with fine steps
    c_dyn = 0.6
    print(f"\n[E27dyn] seeded descent at c={c_dyn} (start 0.85, step 0.01):")
    xi = make_markov(N, P, c_dyn, SEED)
    a_hist = qual[c_dyn][3]
    lam = 0.85
    death_lam, death_pair = np.nan, None
    while lam > 0.20:
        sysP = ReducedDDE(xi, BETA, lam, TAU, T0V)
        sol = sysP.integrate(a_hist, 300.0, DT, record_every=20)
        A = sol["a"]
        nchg, nadv, lead = handoffs(A[-int(150 / (DT * 20)):])
        a_hist = A[-1]
        if nchg == 0:                                     # pinned: stationary
            top = np.argsort(np.abs(A[-1]))[::-1][:2]
            death_lam, death_pair = lam, sorted(int(k) for k in top)
            print(f"  PINNED at lam={lam:.3f} on pair {death_pair} "
                  f"(seam pair = {[P-1, 0]})")
            break
        lam = round(lam - 0.01, 4)
    if death_pair is None:
        print(f"  no pinning found down to lam={lam:.3f}")

    # ---------------------------------------------------------------- figure ------
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))
    cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(CS)))

    ax = axes[0, 0]
    for c, col in zip(CS, cols):
        lc = results[c]["lam_c"]
        ax.hist(lc[:P - 1][np.isfinite(lc[:P - 1])], bins=22, histtype="step", lw=1.8,
                color=col, density=True, label=rf"$c={c}$ (bulk)")
        if np.isfinite(lc[P - 1]):
            ax.axvline(lc[P - 1], color=col, ls="--", lw=1.4)
    ax.set_xlabel(r"$\lambda_c(\mu)$"); ax.set_ylabel("density")
    ax.set_title("Bulk threshold distributions vs frame correlation $c$\n"
                 "(dashed: the SEAM branch -- pushed to the far right)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[0, 1]
    for stat, marker, lab in [(np.nanmedian, "o", "bulk median"),
                              (np.nanmin, "v", "bulk min"),
                              (np.nanmax, "^", "bulk max")]:
        ax.plot(CS, [stat(results[c]["lam_c"][:P - 1]) for c in CS], marker + "-",
                label=lab)
    ax.plot(CS, [results[c]["lam_c"][P - 1] for c in CS], "s--", color="crimson",
            lw=2, label="SEAM branch (q$\\approx$0)")
    ax.set_xlabel("frame correlation $c$"); ax.set_ylabel(r"$\lambda_c$")
    ax.set_title("Correlation shifts the bulk thresholds DOWN;\n"
                 "the uncorrelated seam stays put and becomes the last dike")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[1, 0]
    for c, col in zip(CS, cols):
        lc, rc = results[c]["lam_c"], results[c]["r_c"]
        m = np.isfinite(lc[:P - 1]) & np.isfinite(rc[:P - 1])
        ax.scatter(lc[:P - 1][m], rc[:P - 1][m], s=22, color=col, alpha=0.7,
                   label=rf"$c={c}$")
    ax.set_xlabel(r"$\lambda_c(\mu)$"); ax.set_ylabel(r"fold shape $r_c$")
    ax.set_title("Death-shape family $r_c(\\lambda_c)$ per correlation level\n"
                 "(is the E25 one-parameter law preserved / shifted?)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[1, 1]
    for c, col in zip(CS, cols):
        lc, q = results[c]["lam_c"], results[c]["q"]
        m = np.isfinite(lc[:P - 1])
        ax.scatter(q[:P - 1][m], lc[:P - 1][m], s=22, color=col, alpha=0.7,
                   label=rf"$c={c}$")
        ax.scatter([q[P - 1]], [lc[P - 1]], marker="s", s=90, color=col,
                   edgecolor="k", zorder=5)
    ax.set_xlabel(r"bond overlap $q_\mu$"); ax.set_ylabel(r"$\lambda_c(\mu)$")
    ax.set_title("Threshold vs quenched bond overlap, all correlation levels\n"
                 "(squares: the seam bonds)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    fig.suptitle("E27 -- Markov-flip 'video' patterns: per-pattern thresholds vs frame "
                 rf"correlation ($N={N}$, $P={P}$, seed {SEED})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(BASE, "figures", f"figX_markov_video{SUFF}.png")
    fig.savefig(fp, dpi=175, bbox_inches="tight")
    print(f"\n[E27] figure -> {fp}")

    np.savez(os.path.join(BASE, f"E27_markov{SUFF}.npz"),
             cs=np.array(CS),
             **{f"lam_c_{c}": results[c]["lam_c"] for c in CS},
             **{f"q_{c}": results[c]["q"] for c in CS},
             **{f"shape_{c}": results[c]["shape"] for c in CS},
             death_lam=death_lam,
             death_pair=np.array(death_pair if death_pair else [-1, -1]))
    print("[E27] npz saved")


if __name__ == "__main__":
    _experiment_main()
