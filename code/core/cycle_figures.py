"""
Presentation-quality figures for the cycle study (E1-E10 measured results).

Data provenance: values measured in experiments E1 (Floquet scan), E3/E6a/E8c
(T1 near lambda*), E4 (tau-collapse), E7 (attractor census npz), E8a (drift +
Lyapunov), E9 (per-bond passage times, npz for lam=0.328), E10 (pinning
positions). See CYCLE_worklog.md sec.11 for protocols.

Outputs -> results/cycle/figures/*.png
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "cycle")
OUT = os.path.join(R, "figures"); os.makedirs(OUT, exist_ok=True)
LAM_C, LAM_STAR = 0.2822, 0.3270

# ---------------- measured data (E1, E3, E6a, E8c) --------------------------
scan = np.array([  # (lam, T1) tau=10, N=2000, seed 42
    (0.95, 10.774), (0.90, 10.832), (0.85, 10.901), (0.80, 10.983),
    (0.75, 11.081), (0.70, 11.203), (0.65, 11.357), (0.60, 11.558),
    (0.55, 11.831), (0.50, 12.222), (0.45, 12.831), (0.40, 13.905),
    (0.35, 16.335), (0.34, 17.279), (0.335, 17.92), (0.332, 18.33),
    (0.331, 18.48), (0.33, 18.72), (0.329, 18.85), (0.328, 19.31)])
tau5  = np.array([(0.90, 5.831), (0.60, 6.555), (0.45, 7.824)])
tau20 = np.array([(0.90, 20.832), (0.60, 21.558), (0.45, 22.831), (0.35, 26.335)])
tmax = np.array([(0.332, 35.62), (0.330, 41.45), (0.329, 46.91), (0.328, 72.87)])
drift = np.array([(100, 3.268e-2), (300, 2.932e-2), (1000, 2.019e-2),
                  (3000, 2.055e-2), (6000, 2.100e-2), (9999, 1.168e-2)])

# ================= Fig A: pacemaker law & tau-collapse (E1+E4) ==============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
a1.plot(scan[:13, 0], scan[:13, 1], "o-", color="teal", ms=6, label=r"$\tau=10$")
a1.plot(tau5[:, 0], tau5[:, 1], "s-", color="darkorange", ms=6, label=r"$\tau=5$")
a1.plot(tau20[:, 0], tau20[:, 1], "^-", color="purple", ms=6, label=r"$\tau=20$")
for tv, c in ((5, "darkorange"), (10, "teal"), (20, "purple")):
    a1.axhline(tv + 1, color=c, ls=":", lw=.8)
a1.set_xlabel(r"$\lambda$"); a1.set_ylabel(r"$T_1$ (step period)")
a1.set_title(r"Raw step period $T_1(\lambda,\tau)$  (dotted: $\tau+1$)")
a1.legend(); a1.grid(alpha=.3); a1.invert_xaxis()
a2.plot(scan[:13, 0], scan[:13, 1] - 10, "o-", color="teal", ms=7, label=r"$\tau=10$")
a2.plot(tau5[:, 0], tau5[:, 1] - 5, "s", color="darkorange", ms=9, mfc="none",
        mew=2, label=r"$\tau=5$")
a2.plot(tau20[:, 0], tau20[:, 1] - 20, "^", color="purple", ms=9, mfc="none",
        mew=2, label=r"$\tau=20$")
a2.set_xlabel(r"$\lambda$"); a2.set_ylabel(r"$t_{\rm esc}=T_1-\tau$")
a2.set_title(r"Collapse: $T_1 = \tau + t_{\rm esc}(\lambda)$ "
             r"($\tau$-independent to $10^{-3}$)")
a2.legend(); a2.grid(alpha=.3); a2.invert_xaxis()
fig.suptitle(r"Pacemaker law — the delay is a pure additive floor "
             r"($N=2000,\ \alpha=0.05,\ \beta=20$)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "figA_pacemaker_law_collapse.png"), dpi=170,
            bbox_inches="tight")

# ================= Fig B: depinning transition (E1+E8c+E9) ==================
fig, (b1, b2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
b1.plot(scan[:, 0], scan[:, 1], "o-", color="navy", ms=5)
b1.axvline(LAM_STAR, color="crimson", ls="--", lw=1.6,
           label=r"$\lambda^*\!\approx0.327$ (cycle death: depinning)")
b1.axvline(LAM_C, color="gray", ls="--", lw=1.6,
           label=r"$\lambda_c=0.2822$ (static fold, $\xi^1$)")
b1.axvspan(LAM_C, LAM_STAR, color="orange", alpha=.15)
b1.set_xlim(0.26, 0.97); b1.invert_xaxis()
b1.set_xlabel(r"$\lambda$"); b1.set_ylabel(r"$T_1$ (mean step period)")
b1.set_title(r"Recall cycle slows toward depinning")
b1.legend(fontsize=9); b1.grid(alpha=.3)
lam_fit = np.linspace(0.3276, 0.3325, 200)
C_fit = np.mean(tmax[:, 1] * np.sqrt(tmax[:, 0] - LAM_STAR))
b2.plot(tmax[:, 0], tmax[:, 1], "o", color="crimson", ms=9,
        label=r"$t_{\rm MAX}$ (weakest bond = #91, measured)")
b2.plot(lam_fit, C_fit / np.sqrt(lam_fit - LAM_STAR), "-", color="k", lw=1.4,
        label=(r"SNIC law $C/\sqrt{\lambda-\lambda^*}$, "
               rf"$C={C_fit:.2f}$, $\lambda^*={LAM_STAR}$"))
b2.plot(tmax[:, 0], [17.8, 18.0, 18.1, 18.3], "s--", color="steelblue", ms=6,
        label="median bond time (barely moves)")
b2.set_xlabel(r"$\lambda$"); b2.set_ylabel("bond passage time")
b2.set_title("Local saddle-node at the weakest bond\n"
             "(log/homoclinic law rejected: local slope explodes)")
b2.legend(fontsize=8.5); b2.grid(alpha=.3); b2.invert_xaxis()
# inset: linearization 1/t^2 vs lambda -> intercept = lambda*
bi = b2.inset_axes([0.55, 0.45, 0.42, 0.35])
x, y = tmax[:, 0], 1.0 / tmax[:, 1] ** 2
p = np.polyfit(x, y, 1)
xx = np.linspace(0.3265, 0.333, 50)
bi.plot(x, 1e4 * y, "o", color="crimson", ms=5)
bi.plot(xx, 1e4 * np.polyval(p, xx), "k-", lw=1)
bi.axhline(0, color="gray", lw=.6)
bi.axvline(-p[1] / p[0], color="crimson", ls=":", lw=1)
bi.set_title(rf"$1/t_{{\rm MAX}}^2$: $\lambda^*={-p[1]/p[0]:.4f}$", fontsize=8)
bi.tick_params(labelsize=7); bi.grid(alpha=.3)
fig.suptitle(r"Cycle termination = SNIC localized at the weakest bond "
             r"($\tau=10,\ N=2000$)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(OUT, "figB_depinning_transition.png"), dpi=170,
            bbox_inches="tight")

# ============ Fig C: the intermediate mixed phase (E7+E8a+E9) ===============
fig, (c1, c2, c3) = plt.subplots(1, 3, figsize=(15.5, 4.6))
try:
    d7 = np.load(os.path.join(R, "E7_attractor_census_lam0.31.npz"))
    c1.hist(d7["Q"], bins=20, range=(0, 1), color="slateblue", alpha=.85,
            edgecolor="k", lw=.4)
except FileNotFoundError:
    c1.text(.5, .5, "E7 npz missing", ha="center")
c1.set_xlabel(r"$q$ (cross-overlap of final states)")
c1.set_ylabel("pairs"); c1.grid(alpha=.3)
c1.set_title(r"Broad $P(q)$ — SG-like landscape"
             "\n(40 random ICs, $\\lambda=0.31$)")
c2.loglog(drift[:, 0], drift[:, 1], "o-", color="darkgreen", ms=7)
c2.set_ylim(5e-3, 6e-2)
c2.set_xlabel(r"$t$"); c2.set_ylabel(r"$\|\dot a\|$")
c2.set_title("No freezing: velocity plateau over 2 decades\n"
             r"$\lambda_L\approx+0.034>0$ $\Rightarrow$ chaotic itinerancy")
c2.grid(alpha=.3, which="both")
try:
    d9 = np.load(os.path.join(R, "E9_bond_times.npz"))
    c3.hist(d9["dts"], bins=30, color="peru", edgecolor="k", lw=.4)
    c3.axvline(72.87, color="crimson", lw=2)
    c3.annotate("bond #91\n(the weakest link)", xy=(72.87, 3), xytext=(45, 20),
                fontsize=9, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson"))
except FileNotFoundError:
    c3.text(.5, .5, "E9 npz missing", ha="center")
c3.set_yscale("log")
c3.set_xlabel("bond passage time $t_\\mu$  ($\\lambda=0.328$)")
c3.set_ylabel("count"); c3.grid(alpha=.3)
c3.set_title("Quenched threshold dispersion\n(interface-glass fingerprint)")
fig.suptitle(r"The intermediate window $(\lambda_c,\lambda^*)$: "
             r"chaotic sea + embedded frozen states", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(os.path.join(OUT, "figC_intermediate_phase.png"), dpi=170,
            bbox_inches="tight")

# ================= Fig D: 1-D phase map along lambda ========================
fig, ax = plt.subplots(figsize=(12.5, 3.4))
ax.axvspan(0.20, LAM_C, color="steelblue", alpha=.30)
ax.axvspan(LAM_C, LAM_STAR, color="orange", alpha=.35)
ax.axvspan(LAM_STAR, 1.0, color="seagreen", alpha=.28)
ax.axvline(LAM_C, color="k", lw=1.4); ax.axvline(LAM_STAR, color="k", lw=1.4)
ax.text(0.24, .78, "MEMORY\n(static retrieval)\n+ pinned fronts\n(bistability)",
        ha="center", fontsize=10, transform=ax.get_xaxis_transform())
ax.text((LAM_C + LAM_STAR) / 2, .74, "MIXED\nchaotic sea\n+ frozen\nstates",
        ha="center", fontsize=9, transform=ax.get_xaxis_transform())
ax.text(0.62, .78, "SEQUENTIAL RECALL CYCLE\n"
        r"escape-limited $\longleftarrow$ crossover $t_{\rm esc}\!\approx\!\tau$"
        r" $\longrightarrow$ delay-clocked pacemaker ($T_1\!\approx\!\tau\!+\!0.8$)",
        ha="center", fontsize=10, transform=ax.get_xaxis_transform())
ax.annotate(r"$\lambda_c=0.2822$" + "\nstatic fold ($\\xi^1$)"
            "\n(per-pattern spread up to 0.3025)",
            xy=(LAM_C, 0.02), xytext=(LAM_C - 0.002, 0.06),
            fontsize=9, ha="right", transform=ax.get_xaxis_transform())
ax.annotate(r"$\lambda^*\approx0.327$" + "\ndepinning (SNIC @ bond 91)\n"
            "(2nd threshold: 0.322)",
            xy=(LAM_STAR, 0.02), xytext=(LAM_STAR + 0.004, 0.06),
            fontsize=9, transform=ax.get_xaxis_transform())
ax.set_xlim(0.20, 1.0); ax.set_yticks([])
ax.set_xlabel(r"$\lambda$", fontsize=12)
ax.set_title(r"Deterministic phase map — continuous-time DDE, "
             r"$\alpha=0.05,\ \tau=10,\ \beta=20,\ N=2000$ (seed 42)",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figD_phase_map.png"), dpi=170,
            bbox_inches="tight")
print("figures ->", OUT)
