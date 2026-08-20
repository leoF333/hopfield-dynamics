"""
E28 -- correlated patterns II: CIRCULAR GAUSSIAN LATENT PROCESS (user task 2, option 2).

Each pixel carries an independent stationary periodic Gaussian process on the circle,
built from K Fourier modes:  z_i(theta) = sum_{k=1..K} a_ik cos(k theta) + b_ik sin(k theta),
a,b ~ N(0,1) iid. Frames are xi^mu_i = sign(z_i(theta_mu)), theta_mu = 2 pi mu / P.
The "video" loops EXACTLY (no seam); the frame-to-frame binary correlation is
q1 = (2/pi) arcsin(rho(2pi/P)) with rho(d) = (1/K) sum_k cos(k d) -- tuned by K
(small K = smooth slow video = high correlation; large K = fast video = low corr).

For each K in {80, 40, 20, 10, 5} (N=2000, P=100, seed 42): trace all P branches,
distribution of lambda_c(mu), fold shapes, corr(q_mu, lam_c). All bonds are now
statistically EQUIVALENT (no seam): the weakest/strongest links are pure sample
fluctuations, so the extreme-value picture should survive with shifted parameters.
Then a DYNAMICS check at K=10: cycle at lam=0.9 and seeded-descent death.

Outputs: figY_gpcircle_video.png, E28_gpcircle.npz, printed tables.
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

# CLI: e28_gpcircle_thresholds.py [N] [P] [subset]   (defaults: 2000 100 0=all)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
P = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SUB = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SEED, BETA = 42, 20.0
NO_EIG = P > 200            # skip O(P^2 N) spectral calls at large P
SUFF = f"_N{N}" if N != 2000 else ""
# q1 depends on K/P (frame spacing 2*pi/P): scale the mode counts with P so the
# correlation levels match across sizes (at P=100: KS=[80,40,20,10,5])
KS = [max(1, round(f * P)) for f in (0.8, 0.4, 0.2, 0.1, 0.05)]
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")


def make_gpcircle(N, P, K, seed):
    """Periodic latent GP per pixel, K Fourier modes, flat spectrum."""
    rng = np.random.default_rng(seed)
    theta = 2 * np.pi * np.arange(P) / P
    k = np.arange(1, K + 1)
    A = rng.standard_normal((N, K)); B = rng.standard_normal((N, K))
    z = A @ np.cos(np.outer(k, theta)) + B @ np.sin(np.outer(k, theta))   # (N,P)
    xi = np.sign(z.T)
    xi[xi == 0] = 1.0
    return xi


def _init_corr(K):
    os.environ["OMP_NUM_THREADS"] = "2"
    xi = make_gpcircle(N, P, K, SEED)
    xis = np.roll(xi, -1, axis=0)
    coup = Couplings(xi, xis); coup._use_np = True
    T._CTX.update(N=N, P=P, xi=xi, coup=coup, gram=(xi @ xi.T) / N,
                  relaxed=True)   # dressed memories: identity = leading index only


def run_thresholds(K):
    lam_c = np.full(P, np.nan)
    shape = np.full((P, 2), np.nan)
    if SUB and SUB < P:
        mus = sorted(np.random.default_rng(1234).choice(P, SUB,
                                                        replace=False).tolist())
    else:
        mus = list(range(P))
    with Pool(6, initializer=_init_corr, initargs=(K,)) as pool:
        for r in tqdm(pool.imap_unordered(T._worker,
                                          [(mu, True, NO_EIG) for mu in mus]),
                      total=len(mus), desc=f"K={K}", leave=False):
            lam_c[r["mu"]] = r["lam_c"]
            if r["rows"] is not None and len(r["rows"]):
                shape[r["mu"]] = np.abs(r["rows"][-1][1:3])
    return lam_c, shape


results = {}
def _experiment_main():
    print(f"[E28] circular-GP video, N={N}, P={P}, seed={SEED}")
    for K in KS:
        t0 = time.time()
        xi = make_gpcircle(N, P, K, SEED)
        q = np.array([(xi[mu] @ xi[(mu + 1) % P]) / N for mu in range(P)])
        rho1 = np.mean(np.cos(np.arange(1, K + 1) * 2 * np.pi / P))
        q_pred = (2 / np.pi) * np.arcsin(rho1)
        lam_c, shape = run_thresholds(K)
        ok = np.isfinite(lam_c)
        r_c = shape[:, 1] / np.maximum(shape[:, 0], 1e-12)
        corr_q = (sstats.pearsonr(q[ok], lam_c[ok])[0] if ok.sum() > 3 else np.nan)
        results[K] = dict(lam_c=lam_c, q=q, shape=shape, r_c=r_c, q_pred=q_pred)
        print(f"  K={K:3d}: q1 measured={q.mean():+.3f} (pred {q_pred:+.3f}, sd {q.std():.3f})  "
              f"traced {ok.sum()}/{P}  "
              f"lam_c: med={np.nanmedian(lam_c):.4f} "
              f"[{np.nanmin(lam_c):.4f}, {np.nanmax(lam_c):.4f}] sig={np.nanstd(lam_c):.4f}  "
              f"corr(q,lam_c)={corr_q:+.2f}  [{time.time()-t0:.0f}s]", flush=True)

    np.savez(os.path.join(BASE, f"E28_gpcircle{SUFF}.npz"),
             Ks=np.array(KS),
             **{f"lam_c_{K}": results[K]["lam_c"] for K in KS},
             **{f"q_{K}": results[K]["q"] for K in KS},
             **{f"shape_{K}": results[K]["shape"] for K in KS})
    print("[E28] thresholds npz saved (pre-dynamics checkpoint)", flush=True)

    # --------------------------------------------------------- dynamics checks ----
    from cycle_reduced import ReducedDDE
    TAU, T0V, DT = 10.0, 1.0, 0.05

    def handoffs(A):
        lead = np.argmax(A, axis=1)
        chg = np.nonzero(np.diff(lead) != 0)[0]
        adv = np.sum((lead[chg + 1] - lead[chg]) % P == 1)
        return len(chg), adv, lead

    # (i) cycle QUALITY at lam=0.9 for every smoothness level
    print("\n[E28dyn] recall-cycle quality at lam=0.90 vs video smoothness:")
    qual = {}
    for K in KS:
        xi = make_gpcircle(N, P, K, SEED)
        sysP = ReducedDDE(xi, BETA, 0.9, TAU, T0V)
        a0 = np.zeros(P); a0[0] = 0.99; a0[1] = 0.05
        sol = sysP.integrate(a0, 600.0, DT, record_every=20)
        nchg, nadv, lead = handoffs(sol["a"][int(200 / (DT * 20)):])
        fwd = nadv / max(nchg, 1)
        qual[K] = (nchg, nadv, fwd, sol["a"][-1].copy())
        print(f"  K={K:3d} (q1={results[K]['q'].mean():+.2f}): {nchg:3d} lead changes, "
              f"{nadv:3d} forward (fwd fraction {fwd:.2f}) -> "
              f"{'CYCLE RUNS' if nadv >= 10 and fwd > 0.8 else 'DEGRADED/NO cycle'}",
              flush=True)

    # (ii) seeded descent at K_dyn, from HIGH lambda with fine steps
    K_dyn = max(1, round(0.1 * P))   # =10 at P=100; scales with P
    print(f"\n[E28dyn] seeded descent at K={K_dyn} (start 0.85, step 0.01):")
    xi = make_gpcircle(N, P, K_dyn, SEED)
    a_hist = qual[K_dyn][3]
    lam = 0.85
    death_lam, death_pair = np.nan, None
    lam_c_dyn = results[K_dyn]["lam_c"]
    while lam > 0.20:
        sysP = ReducedDDE(xi, BETA, lam, TAU, T0V)
        sol = sysP.integrate(a_hist, 300.0, DT, record_every=20)
        A = sol["a"]
        nchg, nadv, lead = handoffs(A[-int(150 / (DT * 20)):])
        a_hist = A[-1]
        if nchg == 0:
            top = np.argsort(np.abs(A[-1]))[::-1][:2]
            death_lam, death_pair = lam, sorted(int(k) for k in top)
            imax = int(np.nanargmax(lam_c_dyn)) if np.isfinite(lam_c_dyn).any() else -1
            print(f"  PINNED at lam={lam:.3f} on pair {death_pair} "
                  f"(max-threshold branch mu={imax})")
            break
        lam = round(lam - 0.01, 4)
    if death_pair is None:
        print(f"  no pinning found down to lam={lam:.3f}")

    # ---------------------------------------------------------------- figure ------
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(KS)))

    ax = axes[0, 0]
    for K, col in zip(KS, cols):
        lc = results[K]["lam_c"]
        ax.hist(lc[np.isfinite(lc)], bins=22, histtype="step", lw=1.8, color=col,
                density=True, label=rf"$K={K}$ ($q_1$={results[K]['q'].mean():+.2f})")
    ax.set_xlabel(r"$\lambda_c(\mu)$"); ax.set_ylabel("density")
    ax.set_title("Threshold distributions vs video smoothness (no seam)\n"
                 "(K latent Fourier modes; small K = slow coherent video)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[0, 1]
    q1s = [results[K]["q"].mean() for K in KS]
    ax.errorbar(q1s, [np.nanmedian(results[K]["lam_c"]) for K in KS],
                yerr=[np.nanstd(results[K]["lam_c"]) for K in KS], fmt="o-",
                color="#2b6cb0", label="median ± σ")
    ax.plot(q1s, [np.nanmax(results[K]["lam_c"]) for K in KS], "^-",
            color="#2f9e44", label=r"max = $\lambda^*$")
    ax.plot(q1s, [np.nanmin(results[K]["lam_c"]) for K in KS], "v-",
            color="#b03a5b", label="min (first death)")
    ax.set_xlabel(r"frame correlation $q_1$"); ax.set_ylabel(r"$\lambda_c$")
    ax.set_title("Threshold statistics vs frame correlation\n"
                 "(compare E27 bulk: same trend without any seam)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[1, 0]
    for K, col in zip(KS, cols):
        lc, rc = results[K]["lam_c"], results[K]["r_c"]
        m = np.isfinite(lc) & np.isfinite(rc)
        ax.scatter(lc[m], rc[m], s=22, color=col, alpha=0.7, label=rf"$K={K}$")
    ax.set_xlabel(r"$\lambda_c(\mu)$"); ax.set_ylabel(r"fold shape $r_c$")
    ax.set_title("Death-shape family across correlation levels")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[1, 1]
    for K, col in zip(KS, cols):
        lc, q = results[K]["lam_c"], results[K]["q"]
        m = np.isfinite(lc)
        ax.scatter(q[m], lc[m], s=22, color=col, alpha=0.7, label=rf"$K={K}$")
    ax.set_xlabel(r"bond overlap $q_\mu$"); ax.set_ylabel(r"$\lambda_c(\mu)$")
    ax.set_title("Threshold vs quenched bond overlap\n"
                 "(within- and across-level dependence)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    fig.suptitle("E28 -- circular-GP 'video' patterns (exactly periodic, seamless) "
                 rf"($N={N}$, $P={P}$, seed {SEED})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(BASE, "figures", f"figY_gpcircle_video{SUFF}.png")
    fig.savefig(fp, dpi=175, bbox_inches="tight")
    print(f"\n[E28] figure -> {fp}")

    np.savez(os.path.join(BASE, f"E28_gpcircle{SUFF}.npz"),
             Ks=np.array(KS),
             **{f"lam_c_{K}": results[K]["lam_c"] for K in KS},
             **{f"q_{K}": results[K]["q"] for K in KS},
             **{f"shape_{K}": results[K]["shape"] for K in KS},
             death_lam=death_lam,
             death_pair=np.array(death_pair if death_pair else [-1, -1]))
    print("[E28] npz saved")


if __name__ == "__main__":
    _experiment_main()
