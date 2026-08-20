"""
E24b/c/d -- finite-size scaling and distribution of the per-pattern thresholds
lambda_c(mu), from the npz produced by e24_thresholds.py.

(b) Does the gap = max_mu - min_mu shrink with N (fixed alpha=0.05)?
(c) Distribution of lambda_c(mu): shape, sigma(N) scaling (fixed-alpha AND fixed-P
    series), and the extreme-value prediction of the gap:
    for ~Gaussian thresholds, E[max of P] ~ mean + sigma*a_P,
    a_P = sqrt(2 ln P) - (ln ln P + ln 4 pi)/(2 sqrt(2 ln P)).

Outputs: figV_threshold_distribution.png + printed tables.
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")

def a_P(P):
    L = np.sqrt(2 * np.log(P))
    return L - (np.log(np.log(P)) + np.log(4 * np.pi)) / (2 * L)

def load_series():
    runs = {}
    for f in glob.glob(os.path.join(BASE, "E24_thr_N*_P*_s*.npz")):
        d = np.load(f)
        key = (int(d["N"]), int(d["P"]), int(d["seed"]))
        lc = np.asarray(d["lam_c"], float)
        runs[key] = lc[np.isfinite(lc)]
    return runs

runs = load_series()
print(f"[E24] loaded {len(runs)} runs")

FIXED_A = [(500, 25), (1000, 50), (2000, 100), (4000, 200), (8000, 400)]
FIXED_P = [(2000, 100), (4000, 100), (8000, 100), (16000, 100)]

def series_stats(pairs):
    out = []
    for N, P in pairs:
        lcs = [v for (n, p, s), v in runs.items() if n == N and p == P]
        if not lcs:
            continue
        mins = [v.min() for v in lcs]; maxs = [v.max() for v in lcs]
        meds = [np.median(v) for v in lcs]; sigs = [v.std(ddof=1) for v in lcs]
        gaps = [v.max() - v.min() for v in lcs]
        pooled = np.concatenate(lcs)
        pred_max = [np.mean(v) + np.std(v, ddof=1) * a_P(P) for v in lcs]
        out.append(dict(N=N, P=P, nseed=len(lcs),
                        med=np.mean(meds), med_e=np.std(meds),
                        sig=np.mean(sigs), sig_e=np.std(sigs),
                        gap=np.mean(gaps), gap_e=np.std(gaps),
                        mx=np.mean(maxs), mx_e=np.std(maxs),
                        mn=np.mean(mins), mn_e=np.std(mins),
                        pred_mx=np.mean(pred_max), pooled=pooled,
                        skew=stats.skew(pooled), kurt=stats.kurtosis(pooled)))
    return out

SA = series_stats(FIXED_A)
SP = series_stats(FIXED_P)

print("\n[E24b] FIXED ALPHA = 0.05 (P = N/20):")
print(f"{'N':>6} {'P':>4} {'sd':>3} {'median':>8} {'sigma':>8} {'gap':>8}±{'':<6} "
      f"{'max(=lam*)':>10} {'EVT pred max':>12} {'skew':>6} {'kurt':>6}")
for r in SA:
    print(f"{r['N']:>6} {r['P']:>4} {r['nseed']:>3} {r['med']:8.4f} "
          f"{r['sig']:8.4f} {r['gap']:8.4f}±{r['gap_e']:<6.4f} "
          f"{r['mx']:10.4f} {r['pred_mx']:12.4f} {r['skew']:6.2f} {r['kurt']:6.2f}")

print("\n[E24c] FIXED P = 100 (alpha = 100/N, isolates per-threshold fluctuation):")
print(f"{'N':>6} {'alpha':>7} {'sd':>3} {'median':>8} {'sigma':>8} {'gap':>8}")
for r in SP:
    print(f"{r['N']:>6} {100/r['N']:>7.4f} {r['nseed']:>3} {r['med']:8.4f} "
          f"{r['sig']:8.4f} {r['gap']:8.4f}")

# scaling exponents
def loglog_slope(xs, ys):
    A = np.vstack([np.log(xs), np.ones(len(xs))]).T
    return np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0]

if len(SA) >= 3:
    print(f"\n[E24] sigma(N) fixed-alpha slope: "
          f"{loglog_slope([r['N'] for r in SA], [r['sig'] for r in SA]):+.3f} "
          f"(pure per-threshold CLT would give -0.5)")
    print(f"[E24] gap(N) fixed-alpha slope:   "
          f"{loglog_slope([r['N'] for r in SA], [r['gap'] for r in SA]):+.3f} "
          f"(sigma*sqrt(2 ln alpha N) ~ N^-1/2 * sqrt(ln N): slightly slower than -0.5)")
if len(SP) >= 3:
    print(f"[E24] sigma(N) fixed-P slope:     "
          f"{loglog_slope([r['N'] for r in SP], [r['sig'] for r in SP]):+.3f}")

# ------------------------------------------------------------------ figure ----
fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))

ax = axes[0, 0]
cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(SA)))
for r, c in zip(SA, cols):
    ax.hist(r["pooled"], bins=30, density=True, histtype="step", lw=1.8, color=c,
            label=rf"$N={r['N']}$ ($P={r['P']}$, {r['nseed']} seeds)")
ax.set_xlabel(r"$\lambda_c(\mu)$"); ax.set_ylabel("density")
ax.set_title("Distribution of the per-pattern thresholds (fixed $\\alpha=0.05$)\n"
             "narrows and drifts with $N$")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[0, 1]
for r, c in zip(SA, cols):
    z = (r["pooled"] - r["pooled"].mean()) / r["pooled"].std(ddof=1)
    xs = np.sort(z); ys = (np.arange(len(xs)) + 0.5) / len(xs)
    ax.plot(xs, ys, lw=1.6, color=c, label=rf"$N={r['N']}$")
zz = np.linspace(-4, 4, 200)
ax.plot(zz, stats.norm.cdf(zz), "k--", lw=1.4, label="standard normal")
ax.set_xlabel(r"$(\lambda_c-\bar\lambda_c)/\sigma$"); ax.set_ylabel("CDF")
ax.set_title("Standardized CDFs vs Gaussian\n(skewness/kurtosis printed in the table)")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[1, 0]
Ns_a = [r["N"] for r in SA]
ax.errorbar(Ns_a, [r["sig"] for r in SA], yerr=[r["sig_e"] for r in SA], fmt="o-",
            color="#2b6cb0", label=r"$\sigma(N)$, fixed $\alpha$ ($P=\alpha N$)")
if SP:
    ax.errorbar([r["N"] for r in SP], [r["sig"] for r in SP],
                yerr=[r["sig_e"] for r in SP], fmt="s-", color="#e07b00",
                label=r"$\sigma(N)$, fixed $P=100$")
ref = SA[0]["sig"] * (np.array(Ns_a, float) / SA[0]["N"]) ** -0.5
ax.plot(Ns_a, ref, "k:", lw=1.2, label=r"$N^{-1/2}$ guide")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$\sigma$ of $\lambda_c(\mu)$")
ax.set_title("Threshold dispersion vs size")
ax.legend(fontsize=8.5); ax.grid(alpha=.3, which="both")

ax = axes[1, 1]
ax.errorbar(Ns_a, [r["gap"] for r in SA], yerr=[r["gap_e"] for r in SA], fmt="o-",
            color="#b03a5b", lw=2, label="observed gap = max$-$min")
evt_gap = [2 * r["sig"] * a_P(r["P"]) for r in SA]   # max-min ~ 2*sigma*a_P (symmetric)
ax.plot(Ns_a, evt_gap, "k--", lw=1.4, label=r"EVT: $2\sigma(N)\,a_P$ (Gaussian iid)")
ax.errorbar(Ns_a, [r["mx"] - r["med"] for r in SA],
            yerr=[r["mx_e"] for r in SA], fmt="^-", color="#2f9e44",
            label=r"$\lambda^*-\mathrm{median}$ (cycle-birth gap)")
ax.plot(Ns_a, [r["sig"] * a_P(r["P"]) for r in SA], "g:", lw=1.3,
        label=r"EVT: $\sigma(N)\,a_P$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$\lambda$ gap")
ax.set_title("The recall-death / cycle-birth gap shrinks with $N$\n"
             "and is quantitatively the Gaussian extreme-value gap")
ax.legend(fontsize=8.5); ax.grid(alpha=.3, which="both")

fig.suptitle("E24b/c/d -- finite-size statistics of the per-pattern saddle-node "
             r"thresholds $\lambda_c(\mu)$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fp = os.path.join(BASE, "figures", "figV_threshold_distribution.png")
fig.savefig(fp, dpi=175, bbox_inches="tight")
print(f"\n[E24] figure -> {fp}")
