"""
E26 -- is the threshold law a FIXED distribution independent of P, from which the
P thresholds are drawn (user task 1)? Analysis of the alpha-grid at N=10^4.

Runs: E26_thr_N10000_P{100,200}_s{42,43} (full coverage) and
      E26_thr_N10000_P{500(s42,43),800(s42),1000(s42)} (random subsets, --no-eig).
Subsets estimate the MARGINAL law (median/sigma/shape) but not the P-extremes;
extreme-value statements use the full-coverage runs + the E24 fixed-alpha series.

Output: figZ_alpha_law.png + printed tables.
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

runs = {}
for f in sorted(glob.glob(os.path.join(BASE, "E26_thr_N10000_P*_s*.npz"))):
    d = np.load(f)
    P, s = int(d["P"]), int(d["seed"])
    lc = np.asarray(d["lam_c"], float)
    n_att = len(lc); lc = lc[np.isfinite(lc)]
    if P >= 800:
        # near-capacity breakdown: most branches lose their 2-pattern identity
        # before any fold (third-overlap condensation) -> no clean lambda_c
        print(f"[E26] P={P} s{s} (alpha={P/10000:.2f}): TRACER BREAKDOWN -- "
              f"{len(lc)}/{n_att} sampled branches traced, and the recorded values "
              f"(median {np.median(lc):.3f}) mark branch-identity loss, NOT folds. "
              f"Consistent with the near-capacity breakdown (REPORT statique §5.4). "
              f"EXCLUDED from the law analysis.")
        continue
    runs.setdefault(P, {})[s] = lc
print(f"[E26] loaded (clean, alpha<=0.05): "
      + ", ".join(f"P={P} ({len(v)} seeds)" for P, v in runs.items()))

# reference: the memory lambda_c(alpha) from the static study (xi^1 probe, N=10^4)
REF_ALPHA = {0.01: 0.379, 0.03: 0.314, 0.05: 0.263}

print(f"\n[E26] N=10000 marginal law vs alpha:")
print(f"{'alpha':>7} {'P':>5} {'n_thr':>6} {'median':>8} {'sigma':>8} "
      f"{'skew':>6} {'kurt':>6} {'frac_traced':>11}")
TAB = []
for P in sorted(runs):
    pooled = np.concatenate(list(runs[P].values()))
    alpha = P / 10000
    TAB.append(dict(alpha=alpha, P=P, pooled=pooled,
                    med=np.median(pooled), sig=pooled.std(ddof=1),
                    skew=stats.skew(pooled), kurt=stats.kurtosis(pooled)))
    print(f"{alpha:7.3f} {P:5d} {len(pooled):6d} {np.median(pooled):8.4f} "
          f"{pooled.std(ddof=1):8.4f} {stats.skew(pooled):6.2f} "
          f"{stats.kurtosis(pooled):6.2f}")

# pairwise KS tests on standardized samples (shape universality)
print(f"\n[E26] KS tests between STANDARDIZED distributions (shape comparison):")
for i in range(len(TAB)):
    for j in range(i + 1, len(TAB)):
        zi = (TAB[i]["pooled"] - TAB[i]["med"]) / TAB[i]["sig"]
        zj = (TAB[j]["pooled"] - TAB[j]["med"]) / TAB[j]["sig"]
        ks, p = stats.ks_2samp(zi, zj)
        print(f"  alpha {TAB[i]['alpha']:.3f} vs {TAB[j]['alpha']:.3f}: "
              f"KS={ks:.3f} p={p:.3f} {'(compatible)' if p > 0.05 else '(DIFFER)'}")

# ---------------------------------------------------------------- figure ------
fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))
cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(TAB)))

ax = axes[0, 0]
for t, c in zip(TAB, cols):
    ax.hist(t["pooled"], bins=24, density=True, histtype="step", lw=1.9, color=c,
            label=rf"$\alpha={t['alpha']:.2f}$ ($P={t['P']}$)")
ax.set_xlabel(r"$\lambda_c(\mu)$"); ax.set_ylabel("density")
ax.set_title("The law is NOT fixed: its center tracks the interference "
             r"$\bar\lambda_c(\alpha)$" + "\n(N=10$^4$; subsets for P$\\geq$500)")
ax.legend(fontsize=8.5); ax.grid(alpha=.3)

ax = axes[0, 1]
als = [t["alpha"] for t in TAB]
ax.errorbar(als, [t["med"] for t in TAB],
            yerr=[t["sig"] / np.sqrt(len(t["pooled"])) for t in TAB], fmt="o-",
            color="#2b6cb0", lw=2, label=r"median of $\lambda_c(\mu)$ (E26)")
ax.plot(list(REF_ALPHA.keys()), list(REF_ALPHA.values()), "k^--", ms=8,
        label=r"static $\lambda_c(\alpha)$ ($\xi^1$ probe, §5.3)")
ax2 = ax.twinx()
ax2.plot(als, [t["sig"] for t in TAB], "s-", color="#e07b00", label=r"$\sigma(\alpha)$")
ax2.set_ylabel(r"$\sigma$", color="#e07b00"); ax2.set_ylim(0, 0.02)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\lambda_c$")
ax.set_title("Center moves strongly with $\\alpha$;\nwidth $\\sigma$ barely moves "
             "(right axis)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right"); ax.grid(alpha=.3)

ax = axes[1, 0]
for t, c in zip(TAB, cols):
    z = np.sort((t["pooled"] - t["med"]) / t["sig"])
    ax.plot(z, (np.arange(len(z)) + 0.5) / len(z), lw=1.7, color=c,
            label=rf"$\alpha={t['alpha']:.2f}$")
zz = np.linspace(-4, 4, 200)
ax.plot(zz, stats.norm.cdf(zz), "k--", lw=1.4, label="standard normal")
ax.set_xlabel(r"$(\lambda_c-\mathrm{med})/\sigma$"); ax.set_ylabel("CDF")
ax.set_title("Standardized CDFs collapse across $\\alpha$:\n"
             "ONE universal shape, location(-scale) family")
ax.legend(fontsize=8.5); ax.grid(alpha=.3)

ax = axes[1, 1]
# sigma vs alpha at N=1e4 (E26) + sigma vs N at alpha=0.05 (E24) in one map
ax.plot(als, [t["sig"] for t in TAB], "o-", color="#e07b00", lw=2,
        label=r"$\sigma(\alpha)$ at $N=10^4$ (E26)")
# E24 fixed-alpha series for context
e24 = {}
for f in glob.glob(os.path.join(BASE, "E24_thr_N*_P*_s*.npz")):
    d = np.load(f); Nn, Pp = int(d["N"]), int(d["P"])
    if Pp * 20 == Nn:
        lc = np.asarray(d["lam_c"], float); lc = lc[np.isfinite(lc)]
        e24.setdefault(Nn, []).append(lc.std(ddof=1))
Ns = sorted(e24)
ax.plot([0.05] * len(Ns), [np.mean(e24[n]) for n in Ns], "v", ms=8, color="#2b6cb0")
for n in Ns:
    ax.annotate(f"N={n}", (0.0505, np.mean(e24[n])), fontsize=7.5, color="#2b6cb0")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\sigma$ of $\lambda_c(\mu)$")
ax.set_title("Width: weak in $\\alpha$ (at fixed N), strong in $N$ (blue: E24)\n"
             r"$\Rightarrow$ $\lambda_c(\mu) \approx m(\alpha) + \sigma(N)\,X$, X universal")
ax.legend(fontsize=8.5); ax.grid(alpha=.3)

fig.suptitle("E26 -- the per-pattern threshold law vs load $\\alpha$ at $N=10^4$",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fp = os.path.join(BASE, "figures", "figZ_alpha_law.png")
fig.savefig(fp, dpi=175, bbox_inches="tight")
print(f"\n[E26] figure -> {fp}")
