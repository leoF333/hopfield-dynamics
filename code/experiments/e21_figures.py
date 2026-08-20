"""
E21 figures (presentation quality, English).

figS_front_birth.png       - the front branch through the (non-)turning region + the
                             M-spectrum mechanism: eig_max(M) and eig-closest-to-0
                             stay pinned at ~-1 (no fold); a91->1, a92->0 (dissolution
                             into the pure xi^91 memory); front == memory overlay.
figT_front_stability_vs_tau.png - the front's rightmost T_P(z) root vs lam for
                             tau in {5,10,20,50,100}; real (fold) vs complex (Hopf)
                             marked; the stability boundary and its type per tau.

Data: results/cycle/E21_front_birth.npz, E21_dissolution.npz, E21_tau_nature.npz.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
LAM_STAR = 0.328
LAM_C = 0.2822


def fig_birth():
    d = np.load(os.path.join(OUT, "E21_front_birth.npz"))
    rows = d["rows"]                      # s,lam,res,eig_max,re_closest,im,a1,a2,a3,i1,i2,dlam
    dd = np.load(os.path.join(OUT, "E21_dissolution.npz"))
    br = dd["branch"]                     # lam,a91,a92,eig_max,res
    cmp = dd["cmp"]                       # lam,front_a91,front_a92,mem_a91,mem_a92,dist,...

    order = np.argsort(br[:, 0])
    br = br[order]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # -- left: amplitudes a91, a92 vs lam (front == memory) --
    a = ax[0]
    a.plot(br[:, 0], br[:, 1], "-", color="#1f77b4", lw=2.2, label=r"front $a_{91}$")
    a.plot(br[:, 0], br[:, 2], "-", color="#d62728", lw=2.2, label=r"front $a_{92}$")
    a.plot(cmp[:, 0], cmp[:, 3], "o", color="#1f77b4", ms=6, mfc="none", mew=1.6,
           label=r"pure $\xi^{91}$ memory $a_{91}$")
    a.plot(cmp[:, 0], cmp[:, 4], "s", color="#d62728", ms=6, mfc="none", mew=1.6,
           label=r"pure $\xi^{91}$ memory $a_{92}$")
    a.axhline(0.0, color="k", lw=0.7, ls=":")
    a.axvline(LAM_C, color="gray", lw=1.0, ls="--")
    a.axvline(LAM_STAR, color="k", lw=1.0, ls="--")
    a.text(LAM_C, 0.92, r"$\lambda_c$", color="gray", ha="center", fontsize=10)
    a.text(LAM_STAR, 0.92, r"$\lambda^*$", color="k", ha="center", fontsize=10)
    a.annotate(r"$a_{92}\to 0$: dissolves into bare memory",
               xy=(0.02, 0.01), xytext=(0.06, 0.45), fontsize=9.5,
               arrowprops=dict(arrowstyle="->", color="#d62728"))
    a.set_xlabel(r"$\lambda$"); a.set_ylabel("reduced amplitude")
    a.set_title("Front branch = single-pattern memory branch\n(no lower fold; continuous to $\\lambda=0$)")
    a.legend(fontsize=8.5, loc="center left"); a.grid(alpha=0.25)
    a.set_xlim(-0.02, 0.34)

    # -- right: M-spectrum mechanism: eig_max & eig-closest-to-0 stay near -1 --
    b = ax[1]
    # full branch eig_max: E21 lower limb (rows) + E18a upper limb (up to lam*)
    lam_lo = rows[:, 1]; em_lo = rows[:, 3]
    try:
        e18 = np.load(os.path.join(OUT, "E18_front_branch.npz"))["rows"]  # lam,res,a91,a92,eigM,...
        lam_hi = e18[:, 0]; em_hi = e18[:, 4]
        lam_all = np.concatenate([lam_lo, lam_hi]); em_all = np.concatenate([em_lo, em_hi])
        o = np.argsort(lam_all); lam_all, em_all = lam_all[o], em_all[o]
        b.plot(lam_all, em_all, "-", color="#2ca02c", lw=2.2,
               label=r"$\mathrm{eig}_{\max}(M)$ (full branch)")
    except Exception:
        b.plot(lam_lo, em_lo, "-", color="#2ca02c", lw=2.2,
               label=r"$\mathrm{eig}_{\max}(M)$")
    b.plot(rows[:, 1], rows[:, 4], "--", color="#9467bd", lw=2.0,
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
    b.set_xlim(-0.02, 0.34); b.set_ylim(-1.05, 0.15)

    fig.suptitle("E21a/b - Birth of the pinned front: a continuous limb of the "
                 "$\\xi^{91}$ memory branch, not a fold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG, "figS_front_birth.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f"[E21fig] wrote {p}")


def fig_stability():
    d = np.load(os.path.join(OUT, "E21_tau_nature.npz"))
    taus = d["taus"]
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(taus)))

    # -- left: rightmost Re z vs lam, one curve per tau; markers = type --
    a = ax[0]
    for tau, c in zip(taus, colors):
        r = d[f"tau_{tau:.0f}"]          # lam,eigM,re,im,is_cplx,n_unstable
        lam = r[:, 0]; re = r[:, 2]; isc = r[:, 4] > 0
        a.plot(lam, re, "-", color=c, lw=1.8, label=fr"$\tau={tau:.0f}$")
        a.plot(lam[isc], re[isc], "o", color=c, ms=6)          # complex (Hopf)
        a.plot(lam[~isc], re[~isc], "^", color=c, ms=6, mfc="none")  # real (fold)
    a.axhline(0.0, color="k", lw=1.1)
    a.axvline(LAM_C, color="gray", lw=1.0, ls="--")
    a.axvline(LAM_STAR, color="k", lw=1.0, ls="--")
    a.text(LAM_C, a.get_ylim()[1]*0.9, r"$\lambda_c$", color="gray", ha="center", fontsize=9)
    a.text(LAM_STAR, a.get_ylim()[1]*0.9, r"$\lambda^*$", color="k", ha="center", fontsize=9)
    a.set_xlabel(r"$\lambda$"); a.set_ylabel(r"rightmost $\mathrm{Re}\,z$ of $T_P(z)$")
    a.set_title("Front delayed stability vs $\\lambda$\n"
                r"$\circ$ = complex (Hopf mode), $\triangle$ = real (fold mode)")
    a.legend(fontsize=8.5, ncol=2); a.grid(alpha=0.25)

    # -- right: stability boundary lam_b(tau) and its type --
    b = ax[1]
    lam_b = []; typ_b = []
    for tau in taus:
        r = d[f"tau_{tau:.0f}"]
        lam = r[:, 0]; re = r[:, 2]; isc = r[:, 4] > 0
        # first lam (ascending) where re crosses 0
        unstable = np.where(re > 0)[0]
        if len(unstable) == 0:
            lam_b.append(np.nan); typ_b.append(0)
        else:
            i = unstable[0]
            if i > 0:
                # linear interp of the crossing
                l0, l1 = lam[i-1], lam[i]; r0, r1 = re[i-1], re[i]
                lb = l0 + (0 - r0) / (r1 - r0) * (l1 - l0)
            else:
                lb = lam[i]
            lam_b.append(lb); typ_b.append(1 if isc[i] else 2)
    lam_b = np.array(lam_b); typ_b = np.array(typ_b)
    for tau, lb, tp, c in zip(taus, lam_b, typ_b, colors):
        if np.isnan(lb):
            b.plot(tau, LAM_STAR, "*", color=c, ms=15,
                   label=f"tau={tau:.0f}: stable to lam*" if tau == taus[0] else None)
        else:
            mk = "o" if tp == 1 else "^"
            b.plot(tau, lb, mk, color=c, ms=11)
    b.axhline(LAM_STAR, color="k", lw=1.0, ls="--", label=r"$\lambda^*$ (fold, $\tau$-indep.)")
    b.axhline(LAM_C, color="gray", lw=1.0, ls="--", label=r"$\lambda_c$")
    b.set_xscale("log")
    b.set_xlabel(r"$\tau$"); b.set_ylabel(r"stability boundary $\lambda_b(\tau)$")
    b.set_title("Onset of front instability in $\\lambda$ vs $\\tau$\n"
                r"($\circ$ Hopf onset, $\triangle$ real onset)")
    b.legend(fontsize=8.5, loc="best"); b.grid(alpha=0.25, which="both")

    fig.suptitle("E21c - Front stability boundary and its NATURE vs delay $\\tau$",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG, "figT_front_stability_vs_tau.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f"[E21fig] wrote {p}")


if __name__ == "__main__":
    fig_birth()
    if os.path.exists(os.path.join(OUT, "E21_tau_nature.npz")):
        fig_stability()
    else:
        print("[E21fig] E21_tau_nature.npz not ready yet; skipping figT")
