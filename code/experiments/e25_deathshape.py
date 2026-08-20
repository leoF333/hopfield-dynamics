"""
E25 -- is the death shape (a_mu : a_mu+1 at the fold) universal, or does it depend
on mu (user task 3)?

Background: E13 (4 seeds) reported a quasi-universal frozen-front shape ~[0.68, 0.32]
-- but it only ever measured the WEAKEST bond of each seed (the pattern with the
maximal threshold). figU-d (E24a) suggests the fold shape varies across branches.

Here, from the stored full curves of all 100 branches (N=2000, seed 42):
  1. extract the fold-point shape (a_mu^c, a_mu1^c, ratio r_c) for every branch;
  2. test r_c vs lambda_c(mu)  (is the shape a universal function of the death
     POSITION lambda, sampled at different lambda_c(mu)?);
  3. test the microscopic predictor: the quenched bond overlap
     q_mu = (1/N) xi^mu . xi^{mu+1}  vs both lambda_c(mu) and r_c;
  4. independence check (task 1 side-result): ring autocorrelation of the
     lambda_c(mu) sequence, and corr(lambda_c, q).
Output: figW_deathshape.png + printed tables + E25_deathshape.npz.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

from couplings import make_patterns

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")
# CLI: e25_deathshape.py [curves_npz] [N] [P]   (defaults: N=2000 full-P run)
FNAME = sys.argv[1] if len(sys.argv) > 1 else "E24_curves_N2000_s42.npz"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
P = int(sys.argv[3]) if len(sys.argv) > 3 else 100
SEED = 42
SUFF = f"_N{N}" if N != 2000 else ""
d = np.load(os.path.join(BASE, FNAME))
# subset-aware: lam_c[i] corresponds to mus[i] (mus saved by the tracer)
mus = (np.asarray(d["mus"], int) if "mus" in d.files else np.arange(P))
lam_c_full = np.full(P, np.nan); lam_c_full[mus] = np.asarray(d["lam_c"], float)
xi, _ = make_patterns(N, P, SEED)
q_full = np.array([(xi[mu] @ xi[(mu + 1) % P]) / N for mu in range(P)])

cur_mus = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("curve_"))
a_cf, a1_cf = np.full(P, np.nan), np.full(P, np.nan)
for mu in cur_mus:
    C = d[f"curve_{mu}"]
    a_cf[mu] = abs(C[-1, 1]); a1_cf[mu] = abs(C[-1, 2])
sel = np.array(cur_mus)
sel = sel[np.isfinite(lam_c_full[sel]) & np.isfinite(a_cf[sel])]
lam_c, q = lam_c_full[sel], q_full[sel]
a_c, a1_c = a_cf[sel], a1_cf[sel]
r_c = a1_c / np.maximum(a_c, 1e-12)
P_eff = len(sel)
print(f"[E25] using {P_eff} branches from {FNAME}")
P = P_eff   # downstream stats/figures operate on the selected branches

print(f"[E25] fold shape across the 100 branches (N=2000, seed 42):")
print(f"  a_mu^c:  min {a_c.min():.3f}  median {np.median(a_c):.3f}  max {a_c.max():.3f}")
print(f"  ratio r_c = a_mu1/a_mu: min {r_c.min():.3f} median {np.median(r_c):.3f} "
      f"max {r_c.max():.3f}  -> NOT a single universal number")

def corr(x, y, name):
    r, p = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    print(f"  {name}: Pearson r={r:+.3f} (p={p:.1e})  Spearman {rs:+.3f} (p={ps:.1e})")
    return r

print("\n[E25] correlations:")
r_rl = corr(lam_c, r_c, "r_c  vs lambda_c(mu)")
r_ql = corr(q, lam_c, "q_mu vs lambda_c(mu)")
r_qr = corr(q, r_c, "q_mu vs r_c")
# partial: r_c vs q at fixed lambda_c (residuals)
res_r = r_c - np.poly1d(np.polyfit(lam_c, r_c, 1))(lam_c)
res_q = q - np.poly1d(np.polyfit(lam_c, q, 1))(lam_c)
rp, pp = stats.pearsonr(res_q, res_r)
print(f"  partial corr(q, r_c | lambda_c): {rp:+.3f} (p={pp:.1e})")

# also the E13 reconciliation: the shape at the MAX threshold branch
imax, imin = int(np.argmax(lam_c)), int(np.argmin(lam_c))
print(f"\n[E25] extremes: weakest bond mu={sel[imax]} (lam_c={lam_c[imax]:.4f}): "
      f"shape ({a_c[imax]:.3f}, {a1_c[imax]:.3f}) r={r_c[imax]:.3f}  <- E13's '68/32'")
print(f"          first death mu={sel[imin]} (lam_c={lam_c[imin]:.4f}): "
      f"shape ({a_c[imin]:.3f}, {a1_c[imin]:.3f}) r={r_c[imin]:.3f}")

# independence check (task 1 side-result): ring autocorrelation of lambda_c(mu)
lc0 = lam_c - lam_c.mean()
ac1 = float(np.corrcoef(lc0, np.roll(lc0, 1))[0, 1])
ac2 = float(np.corrcoef(lc0, np.roll(lc0, 2))[0, 1])
print(f"\n[E25] ring autocorrelation of lambda_c(mu): lag1 {ac1:+.3f}, lag2 {ac2:+.3f} "
      f"(iid -> ~0 +- {1/np.sqrt(P):.2f})")

# ---------------------------------------------------------------- figure ------
fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
ax = axes[0, 0]
sc = ax.scatter(lam_c, r_c, c=q, cmap="coolwarm", s=45, edgecolor="k", lw=0.4)
fig.colorbar(sc, ax=ax, label=r"bond overlap $q_\mu=\frac{1}{N}\xi^\mu\!\cdot\!\xi^{\mu+1}$")
ax.plot(*zip(*sorted(zip(lam_c, np.poly1d(np.polyfit(lam_c, r_c, 1))(lam_c)))),
        "k--", lw=1.2, label=f"linear fit (r={r_rl:+.2f})")
ax.scatter([lam_c[imax]], [r_c[imax]], marker="*", s=260, color="gold",
           edgecolor="k", zorder=5, label="weakest bond (E13's probe)")
ax.set_xlabel(r"$\lambda_c(\mu)$"); ax.set_ylabel(r"fold shape $r_c=a_{\mu+1}/a_\mu$")
ax.set_title("The death shape is NOT one number: it grows with the death position\n"
             "(E13's 68/32 was conditioned on the extreme bond)")
ax.legend(fontsize=8.5); ax.grid(alpha=.3)

ax = axes[0, 1]
ax.scatter(q, lam_c, c=r_c, cmap="viridis", s=45, edgecolor="k", lw=0.4)
ax.set_xlabel(r"$q_\mu$"); ax.set_ylabel(r"$\lambda_c(\mu)$")
ax.set_title(f"Microscopic predictor test: bond overlap vs threshold\n"
             f"(Pearson {r_ql:+.2f})")
ax.grid(alpha=.3)
cb = fig.colorbar(ax.collections[0], ax=ax); cb.set_label(r"$r_c$")

ax = axes[1, 0]
ax.scatter(a_c, a1_c, c=lam_c, cmap="plasma", s=45, edgecolor="k", lw=0.4)
cb = fig.colorbar(ax.collections[0], ax=ax); cb.set_label(r"$\lambda_c(\mu)$")
ax.plot([0.6, 0.85], [0.32 / 0.68 * 0.6, 0.32 / 0.68 * 0.85], "k:", lw=1.1,
        label="68/32 direction")
ax.set_xlabel(r"$a_\mu$ at the fold"); ax.set_ylabel(r"$a_{\mu+1}$ at the fold")
ax.set_title("Fold states in the $(a_\\mu, a_{\\mu+1})$ plane\n"
             "one-parameter family ordered by $\\lambda_c(\\mu)$")
ax.legend(fontsize=9); ax.grid(alpha=.3)

ax = axes[1, 1]
# multi-run lag-1 autocorrelation (all fixed-alpha runs with full P coverage)
import glob
pts = []
for f in sorted(glob.glob(os.path.join(BASE, "E24_thr_N*_P*_s*.npz"))):
    dd = np.load(f); lc = dd["lam_c"]; lc = lc[np.isfinite(lc)]
    Np, Pp = int(dd["N"]), int(dd["P"])
    if (Np, Pp) not in [(2000, 100), (4000, 200), (8000, 400)] or len(lc) < Pp:
        continue
    x = lc - lc.mean()
    pts.append((Pp, float(np.corrcoef(x, np.roll(x, 1))[0, 1]), int(dd["seed"])))
Ps = sorted(set(p for p, _, _ in pts))
for Pp in Ps:
    vals = [v for p, v, s in pts if p == Pp]
    ax.scatter([Pp] * len(vals), vals, s=45, edgecolor="k", lw=0.4, zorder=3,
               color="#2b6cb0")
    ax.plot([Pp * 0.9, Pp * 1.1], [2 / np.sqrt(Pp)] * 2, "k--", lw=1)
    ax.plot([Pp * 0.9, Pp * 1.1], [-2 / np.sqrt(Pp)] * 2, "k--", lw=1,
            label="iid band" if Pp == Ps[0] else None)
mean_l1 = np.mean([v for _, v, _ in pts])
ax.axhline(mean_l1, color="crimson", lw=1.3, label=f"mean lag-1 = {mean_l1:+.3f}")
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("P (12 runs, 5 seeds)")
ax.set_ylabel(r"lag-1 ring autocorr of $\lambda_c(\mu)$")
ax.set_title("Independence check (multi-seed): neighbor thresholds are\n"
             "near-iid -- weak negative tendency, shrinking with N")
ax.legend(fontsize=9); ax.grid(alpha=.3)

fig.suptitle("E25 -- what determines each pattern's death: shape vs position vs "
             "quenched bond overlap ($N=2000$, seed 42)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fp = os.path.join(BASE, "figures", f"figW_deathshape{SUFF}.png")
fig.savefig(fp, dpi=175, bbox_inches="tight")
print(f"\n[E25] figure -> {fp}")

np.savez(os.path.join(BASE, f"E25_deathshape{SUFF}.npz"), lam_c=lam_c, q=q, a_c=a_c,
         a1_c=a1_c, r_c=r_c, lag1=ac1, lag2=ac2, sel=sel)
print("[E25] npz saved")
