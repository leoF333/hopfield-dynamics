"""
E21e -- figS rev. 2: extend the left panel of figS_front_birth.png to the full
branch [0, lambda*] (user request). The E21a/a2 arclength was deliberately seeded
at lam=0.15 to probe the birth region below E18a's spurious fold (0.1322); the
upper limb [0.132, 0.3276] already exists in E18_front_branch.npz. Here:
  (1) merge the two limbs for the a91/a92 curves;
  (2) recompute the INDEPENDENT pure-xi^91-memory comparison points on the whole
      range (beta-annealed Newton at lam=0.03, then warm-start continuation up),
      extending E21a2's merge proof to lambda*;
  (3) regenerate figS (right panel unchanged). Saves E21_mem_marks.npz.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from couplings import Couplings, make_patterns
from robust_branch import woodbury_newton, eigmax_M

N, ALPHA, BETA, SEED = 2000, 0.05, 20.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIG = os.path.join(OUT, "figures")
MU_A, MU_B = 91, 92
GRAM = (xi @ xi.T) / N
LAM_STAR, LAM_C = 0.328, 0.2822


def amp(u):
    return np.linalg.solve(GRAM, coup.overlap_raw(u))


def residual(u, lam):
    return np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)


def main():
    t0 = time.time()
    # ---- (2) independent pure-xi^91 memory branch, lam = 0.03 -> lam* --------
    marks = [0.03, 0.07, 0.11, 0.15, 0.19, 0.23, 0.27, 0.30, 0.32, 0.327]
    u = 0.99 * xi[MU_A].copy()
    for b in [6.0, 10.0, 14.0, BETA]:                 # beta-anneal at lam=0.03
        u, ok = woodbury_newton(coup, u, 0.03, b, tol=1e-12)
    rows = []
    lam = 0.03
    for lam_t in marks:
        step = 0.004
        while abs(lam - lam_t) > 1e-9:
            ln = lam + np.sign(lam_t - lam) * min(step, abs(lam_t - lam))
            un, ok = woodbury_newton(coup, u, ln, BETA, tol=1e-12)
            a = amp(un)
            if ok and int(np.argmax(np.abs(a))) == MU_A and residual(un, ln) < 1e-10:
                u, lam = un, ln
                step = min(step * 1.3, 0.004)
            else:
                step *= 0.5
                assert step > 1e-5, f"memory continuation stalled at lam={lam:.4f}"
        a = amp(u)
        rows.append([lam, a[MU_A], a[MU_B], eigmax_M(coup, u, lam, BETA),
                     residual(u, lam)])
        print(f"[E21e] mem xi^91 @ lam={lam:.3f}: a91={a[MU_A]:+.4f} "
              f"a92={a[MU_B]:+.4f} eigM={rows[-1][3]:+.4f} res={rows[-1][4]:.1e}")
    mem = np.array(rows)
    np.savez(os.path.join(OUT, "E21_mem_marks.npz"), mem=mem,
             cols=np.array(["lam", "a91", "a92", "eig_max", "res"]))

    # ---- (1) merge the two limbs of the front branch -------------------------
    dd = np.load(os.path.join(OUT, "E21_dissolution.npz"))
    lo = dd["branch"]                                  # lam,a91,a92,eig_max,res
    e18 = np.load(os.path.join(OUT, "E18_front_branch.npz"))["rows"]
    hi = e18[:, [0, 2, 3, 4]]                          # lam,a91,a92,eigM
    lo_keep = lo[lo[:, 0] < hi[:, 0].min()]
    br = np.concatenate([lo_keep[:, :4], hi], axis=0)
    br = br[np.argsort(br[:, 0])]
    print(f"[E21e] merged branch: {len(br)} pts, lam in "
          f"[{br[:,0].min():.4f}, {br[:,0].max():.4f}]")
    # merge agreement in the overlap zone (lo top vs hi bottom)
    j = int(np.argmin(np.abs(lo[:, 0] - hi[:, 0].min())))
    print(f"[E21e] seam check near lam={hi[:,0].min():.4f}: "
          f"lo a91={lo[j,1]:+.4f} vs hi a91={hi[0,1]:+.4f}")

    # ---- (3) regenerate figS --------------------------------------------------
    d = np.load(os.path.join(OUT, "E21_front_birth.npz"))
    arows = d["rows"]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    a = ax[0]
    a.plot(br[:, 0], br[:, 1], "-", color="#1f77b4", lw=2.2, label=r"front $a_{91}$")
    a.plot(br[:, 0], br[:, 2], "-", color="#d62728", lw=2.2, label=r"front $a_{92}$")
    a.plot(mem[:, 0], mem[:, 1], "o", color="#1f77b4", ms=6, mfc="none", mew=1.6,
           label=r"pure $\xi^{91}$ memory $a_{91}$ (independent Newton)")
    a.plot(mem[:, 0], mem[:, 2], "s", color="#d62728", ms=6, mfc="none", mew=1.6,
           label=r"pure $\xi^{91}$ memory $a_{92}$")
    a.axhline(0.0, color="k", lw=0.7, ls=":")
    a.axvline(LAM_C, color="gray", lw=1.0, ls="--")
    a.axvline(LAM_STAR, color="k", lw=1.0, ls="--")
    a.text(LAM_C, 0.92, r"$\lambda_c$", color="gray", ha="center", fontsize=10)
    a.text(LAM_STAR, 0.92, r"$\lambda^*$", color="k", ha="center", fontsize=10)
    a.annotate(r"$a_{92}\to 0$: dissolves into bare memory",
               xy=(0.02, 0.01), xytext=(0.05, 0.45), fontsize=9.5,
               arrowprops=dict(arrowstyle="->", color="#d62728"))
    a.annotate("dies at the depinning\nsaddle-node (SNIC)",
               xy=(LAM_STAR, 0.664), xytext=(0.19, 0.62), fontsize=9.5,
               arrowprops=dict(arrowstyle="->", color="k"))
    a.set_xlabel(r"$\lambda$"); a.set_ylabel("reduced amplitude")
    a.set_title("Front branch = single-pattern memory branch, followed on its "
                "FULL range\n(continuous from $\\lambda=0$ to the depinning fold "
                "$\\lambda^*$; memory and front coincide everywhere)")
    a.legend(fontsize=8.5, loc="center left"); a.grid(alpha=0.25)
    a.set_xlim(-0.02, 0.345)

    b = ax[1]
    lam_all = np.concatenate([arows[:, 1], br[:, 0]])
    em_all = np.concatenate([arows[:, 3], br[:, 3]])
    o = np.argsort(lam_all)
    b.plot(lam_all[o], em_all[o], "-", color="#2ca02c", lw=2.2,
           label=r"$\mathrm{eig}_{\max}(M)$ (full branch)")
    b.plot(arows[:, 1], arows[:, 4], "--", color="#9467bd", lw=2.0,
           label=r"$M$-eig closest to 0 (birth region)")
    b.axhline(0.0, color="k", lw=1.0)
    b.axvline(0.1322, color="orange", lw=1.3, ls=":")
    b.text(0.1322, -0.55, "E18a spurious\n'fold' 0.1322", color="orange",
           ha="center", fontsize=8.5)
    b.axvline(LAM_STAR, color="k", lw=1.0, ls="--")
    b.text(LAM_STAR, -0.15, r"true fold $\lambda^*$" + "\n(depinning SN)", color="k",
           ha="right", fontsize=8.5)
    b.set_xlabel(r"$\lambda$"); b.set_ylabel(r"$\mathrm{Re}\,\mathrm{eig}(M)$")
    b.set_title("No real eigenvalue of $M$ crosses 0 below $\\lambda^*$\n"
                "$\\Rightarrow$ no saddle-node birth")
    b.legend(fontsize=9, loc="lower right"); b.grid(alpha=0.25)
    b.set_xlim(-0.02, 0.345); b.set_ylim(-1.05, 0.15)

    fig.suptitle("E21a/b (rev. 2) - Birth of the pinned front: a continuous limb "
                 "of the $\\xi^{91}$ memory branch, not a fold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG, "figS_front_birth.png")
    fig.savefig(p, dpi=180)
    print(f"[E21e] wrote {p}  ({time.time()-t0:.0f}s)")

    # front-vs-memory distance table over the FULL range
    print("\n[E21e] |front - memory| on the two dominant overlaps:")
    for r in mem:
        j = int(np.argmin(np.abs(br[:, 0] - r[0])))
        dsum = abs(br[j, 1] - r[1]) + abs(br[j, 2] - r[2])
        print(f"  lam={r[0]:.3f}: |d(a91)|+|d(a92)| = {dsum:.2e} "
              f"(branch pt at lam={br[j,0]:.4f})")


if __name__ == "__main__":
    main()
