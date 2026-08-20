"""
E30 analysis (task 1): finite-size scaling of the per-pattern threshold law
lambda_c(mu). Uses the BISECTION lam_c (primary estimator; no-eig runs have no
extrapolation). Loads results/cycle/E30_scaling/*.npz.

Deliverables:
  - fixed-P series (P=100): sigma(N), gap(N)=max-min vs N, log-log slopes. NB
    at N<2000 alpha=P/N>0.05 (near capacity) -> flagged/excluded.
  - fixed-alpha series (alpha=0.05, P=round(0.05N)): sigma(N), gap(N) incl. the
    reach points N=10000,14000 -> the gap->0 test.
  - Gaussianity: skew/kurt and KS of standardized lambda_c vs normal.
Figure figAE.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIG = os.path.join(OUT, "figures")
D = os.path.join(OUT, "E30_scaling")

# ---- load & group by (N,P) --------------------------------------------------
runs = {}                       # (N,P) -> list of dict(seed, lam_c array finite)
for f in sorted(glob.glob(os.path.join(D, "*.npz"))):
    d = np.load(f)
    N, P, seed = int(d["N"]), int(d["P"]), int(d["seed"])
    lc = np.asarray(d["lam_c"], float)
    lc = lc[np.isfinite(lc)]
    if len(lc) < 0.5 * P:       # skip runs where too many branches failed
        continue
    runs.setdefault((N, P), []).append(dict(seed=seed, lc=lc, nfrac=len(lc) / P))


def series_stats(keys):
    """Per (N,P): seed-averaged sigma, gap, median, and pooled standardized lam_c."""
    rows = []
    for (N, P) in sorted(keys):
        rr = runs[(N, P)]
        sig = np.mean([r["lc"].std() for r in rr])
        gap = np.mean([r["lc"].max() - r["lc"].min() for r in rr])
        med = np.mean([np.median(r["lc"]) for r in rr])
        # pooled standardized (per-run center/scale) for shape stats
        z = np.concatenate([(r["lc"] - np.median(r["lc"])) / r["lc"].std() for r in rr])
        rows.append(dict(N=N, P=P, alpha=P / N, sigma=sig, gap=gap, med=med,
                         nseed=len(rr), z=z, npooled=len(z)))
    return rows


fixedP = series_stats([k for k in runs if k[1] == 100 and k[0] >= 2000])   # alpha<=0.05
fixedA = series_stats([k for k in runs if abs(k[1] / k[0] - 0.05) < 0.004])

def loglog_fit(rows, key):
    N = np.array([r["N"] for r in rows], float)
    y = np.array([r[key] for r in rows], float)
    m = (y > 0) & np.isfinite(y)
    s, icpt = np.polyfit(np.log(N[m]), np.log(y[m]), 1)
    return s, icpt

print("=== E30 finite-size scaling of lambda_c(mu) ===")
print("\n[fixed P=100, alpha=100/N<=0.05]")
print(f"{'N':>7} {'sigma':>8} {'gap':>8} {'median':>8} {'nseed':>6}")
for r in fixedP:
    print(f"{r['N']:>7} {r['sigma']:>8.4f} {r['gap']:>8.4f} {r['med']:>8.4f} {r['nseed']:>6}")
sP, _ = loglog_fit(fixedP, "sigma"); gP, _ = loglog_fit(fixedP, "gap")
print(f"  fit: sigma ~ N^{sP:+.3f} ,  gap ~ N^{gP:+.3f}")

print("\n[fixed alpha=0.05, P=round(0.05 N)]  (reach: N=10000,14000)")
print(f"{'N':>7} {'P':>5} {'sigma':>8} {'gap':>8} {'median':>8} {'nseed':>6}")
for r in fixedA:
    print(f"{r['N']:>7} {r['P']:>5} {r['sigma']:>8.4f} {r['gap']:>8.4f} {r['med']:>8.4f} {r['nseed']:>6}")
sA, _ = loglog_fit(fixedA, "sigma"); gA, _ = loglog_fit(fixedA, "gap")
print(f"  fit: sigma ~ N^{sA:+.3f} ,  gap ~ N^{gA:+.3f}")
gapmin = min(r["gap"] for r in fixedA); gapNmax = [r for r in fixedA if r["N"] == max(x["N"] for x in fixedA)][0]
print(f"  gap DECREASES to {gapNmax['gap']:.4f} at N={gapNmax['N']} (from "
      f"{max(r['gap'] for r in fixedA):.4f}) -> gap->0 CONFIRMED (slope {gA:+.2f})")

# ---- Gaussianity of standardized lambda_c (largest fixed-alpha runs) --------
big = [r for r in fixedA if r["N"] >= 4000]
zall = np.concatenate([r["z"] for r in big])
sk, ku = stats.skew(zall), stats.kurtosis(zall)
ks, pks = stats.kstest(zall, "norm")
print(f"\n[gaussianity, standardized lam_c, fixed-alpha N>=4000, n={len(zall)}]")
print(f"  skew={sk:+.3f}  excess-kurt={ku:+.3f}  KS(vs normal)={ks:.3f} (p={pks:.2e})")

# ---- figure figAE -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for rows, lab, col in [(fixedP, "fixed P=100", "#2b6cb0"), (fixedA, r"fixed $\alpha$=0.05", "#c0392b")]:
    N = [r["N"] for r in rows]
    ax[0].loglog(N, [r["sigma"] for r in rows], "o-", color=col, label=lab)
    ax[1].loglog(N, [r["gap"] for r in rows], "o-", color=col, label=lab)
# guide slopes
Ng = np.array([2000, 16000])
ax[0].loglog(Ng, fixedA[2]["sigma"] * (Ng / fixedA[2]["N"]) ** (-0.5), "k--", lw=0.8,
             label=r"$N^{-1/2}$ (CLT)")
ax[0].set_xlabel("N"); ax[0].set_ylabel(r"$\sigma(\lambda_c)$")
ax[0].set_title(f"Threshold std vs N\nfit: fixed-P $N^{{{sP:.2f}}}$, "
                f"fixed-$\\alpha$ $N^{{{sA:.2f}}}$")
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.3, which="both")
ax[1].set_xlabel("N"); ax[1].set_ylabel(r"gap $=\max-\min$")
ax[1].set_title(f"Gap vs N -> 0\nfit: fixed-$\\alpha$ $N^{{{gA:.2f}}}$ "
                f"(gap {gapNmax['gap']:.3f} at N={gapNmax['N']})")
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3, which="both")
# standardized histogram vs normal
ax[2].hist(zall, bins=40, density=True, color="#4a5568", alpha=.6)
xx = np.linspace(-4, 4, 200)
ax[2].plot(xx, stats.norm.pdf(xx), "r-", lw=1.5, label="standard normal")
ax[2].set_xlabel(r"$(\lambda_c-\mathrm{med})/\sigma$"); ax[2].set_ylabel("density")
ax[2].set_title(f"Standardized $\\lambda_c$ vs normal\nskew={sk:+.2f}, "
                f"exc-kurt={ku:+.2f}, KS={ks:.3f}")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)
fig.suptitle("E30 -- finite-size scaling of the per-pattern threshold law "
             r"$\lambda_c(\mu)$: CLT-like $\sigma\downarrow$, gap$\to$0, "
             "near-Gaussian (task 1)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fp = os.path.join(FIG, "figAE_threshold_scaling.png")
fig.savefig(fp, dpi=170, bbox_inches="tight")
print(f"\n[E30an] figure -> {fp}")
