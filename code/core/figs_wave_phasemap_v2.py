"""
Two presentation figures (English, dpi 170):
  figI_traveling_wave.png  — the recall cycle as a traveling wave on the pattern
      ring: full-period heatmap of a_nu(t) (diagonal stripe) + front handoff zoom
      (window located automatically at the front's passage over nu0).
      Data: saved cycle results/cycle/cycle_lam0.9_tau10.0_N2000.npz (no resim).
  figD_phase_map_v2.png    — redesigned 1-D phase map: full-range bar on top +
      connected zoom panel on the critical region [0.26, 0.36].
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")
FIG = os.path.join(BASE, "figures")

# ------------------------------------------------------------- traveling wave
d = np.load(os.path.join(BASE, "cycle_lam0.9_tau10.0_N2000.npz"))
a, dt, L, T, T1 = d["a_grid"], float(d["dt"]), int(d["L"]), float(d["T"]), float(d["T1_mean"])
P = a.shape[1]
t = np.arange(a.shape[0]) * dt - L * dt          # grid spans [-tau, T]
i0 = L                                            # index of t = 0
sub = 25                                          # 0.25 t.u. sampling for imshow

fig = plt.figure(figsize=(13.5, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)

ax = fig.add_subplot(gs[0, 0])
A = a[i0::sub].T
tt = t[i0::sub]
im = ax.imshow(A, aspect="auto", origin="lower", cmap="inferno",
               extent=[tt[0], tt[-1], 0.5, P + 0.5], vmin=0.0,
               interpolation="nearest")
ax.set_xlabel("t"); ax.set_ylabel(r"pattern index $\nu$")
ax.set_title(rf"One full period around the ring ($T = P\,T_1 = {T:.0f}$)")
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(r"$a_\nu(t)$")

# locate the front's passage over pattern nu0 (argmax of its overlap in [0,T])
nu0 = 50
tc = float(t[i0 + int(np.argmax(a[i0:, nu0 - 1]))])
ax2 = fig.add_subplot(gs[0, 1])
w = (t >= tc - 1.6 * T1) & (t <= tc + 2.0 * T1)
cols = ["#888888", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
ymax = 0.0
for j, nu in enumerate(range(nu0 - 2, nu0 + 3)):
    ax2.plot(t[w], a[w, nu - 1], lw=1.8, color=cols[j], label=rf"$a_{{{nu}}}$")
    ymax = max(ymax, float(a[w, nu - 1].max()))
t50 = float(t[i0 + int(np.argmax(a[i0:, nu0 - 1]))])
t51 = float(t[i0 + int(np.argmax(a[i0:, nu0]))])
ax2.axvline(t50, color="k", ls=":", lw=1)
ax2.axvline(t51, color="k", ls=":", lw=1)
yarr = 1.02 * ymax
ax2.annotate("", xy=(t51, yarr), xytext=(t50, yarr),
             arrowprops=dict(arrowstyle="<->", lw=1.2), annotation_clip=False)
ax2.text(0.5 * (t50 + t51), 1.05 * ymax, rf"$T_1={t51-t50:.2f}$",
         ha="center", fontsize=10)
ax2.set_ylim(None, 1.16 * ymax)
ax2.set_xlabel("t"); ax2.set_ylabel(r"$a_\nu(t)$")
ax2.set_title(r"Front handoff, clocked by the delay ($T_1\approx\tau+0.83$)")
ax2.legend(fontsize=9, ncol=2, loc="center left"); ax2.grid(alpha=.3)
fig.suptitle(r"The sequential-recall cycle is a disorder-rippled traveling wave "
             rf"($N=2000$, $\alpha=0.05$, $\tau=10$, $\lambda=0.9$)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fp1 = os.path.join(FIG, "figI_traveling_wave.png")
fig.savefig(fp1, dpi=170, bbox_inches="tight"); plt.close(fig)
print("wrote", fp1)

# ------------------------------------------------------------- phase map v2
LAM_C, LAM_S, LAM_2, LAM_DISP, LAM_DEP = 0.2822, 0.327, 0.322, 0.3025, 0.328
Z0, Z1 = 0.26, 0.36                    # zoom window
CBLUE, CORNG, CGREEN = "#a8c6e8", "#f5c97f", "#a9d8b8"

fig, (axt, axz) = plt.subplots(2, 1, figsize=(13.5, 5.6),
                               gridspec_kw=dict(height_ratios=[1.0, 1.35],
                                                hspace=0.55))
# --- top: full-range bar ------------------------------------------------------
axt.axvspan(0.20, LAM_C, color=CBLUE)
axt.axvspan(LAM_C, LAM_S, color=CORNG)
axt.axvspan(LAM_S, 1.00, color=CGREEN)
axt.axvline(LAM_C, color="0.25", ls="--", lw=1.4)
axt.axvline(LAM_S, color="crimson", lw=1.8)
axt.text(0.241, 0.5, "MEMORY", rotation=90, ha="center", va="center",
         fontsize=10.5, weight="bold")
axt.text(0.3045, 0.5, "MIXED", rotation=90, ha="center", va="center",
         fontsize=10.5, weight="bold")
axt.text(0.663, 0.66, "SEQUENTIAL RECALL CYCLE", ha="center", fontsize=13,
         weight="bold")
axt.text(0.663, 0.30,
         r"escape-limited ($t_{\rm esc}\!\gg\!\tau$) $\longleftarrow$ crossover "
         r"$t_{\rm esc}\!\approx\!\tau$ $\longrightarrow$ delay-clocked pacemaker "
         r"($T_1\simeq\tau+0.83$)", ha="center", fontsize=9.5)
axt.set_xlim(0.20, 1.00); axt.set_ylim(0, 1); axt.set_yticks([])
axt.set_xlabel(r"$\lambda$", labelpad=1)
axt.set_title(r"Deterministic phase map — continuous-time DDE "
              r"($\alpha=0.05$, $\tau=10$, $\beta=20$, $N=2000$, seed 42)",
              fontsize=12.5)
# mark the zoom window on the top bar
axt.plot([Z0, Z1], [0, 0], color="0.2", lw=3, solid_capstyle="butt",
         transform=axt.get_xaxis_transform(), clip_on=False)

# --- bottom: zoom on the critical region -------------------------------------
axz.axvspan(Z0, LAM_C, color=CBLUE)
axz.axvspan(LAM_C, LAM_S, color=CORNG)
axz.axvspan(LAM_S, Z1, color=CGREEN)
axz.set_xlim(Z0, Z1); axz.set_ylim(0, 1); axz.set_yticks([])
axz.set_xlabel(r"$\lambda$ (zoom)", labelpad=1)

# critical lines
axz.axvline(LAM_C, color="0.25", ls="--", lw=1.6)
axz.axvline(LAM_2, color="crimson", ls=":", lw=1.3)
axz.axvline(LAM_S, color="crimson", lw=2.0)

# region captions (roomy in the zoom)
axz.text(0.2705, 0.80, "memory\n(static retrieval, $m_1\\approx1$)",
         ha="center", fontsize=9.5)
axz.text(0.3045, 0.80, "chaotic sea ($\\lambda_L\\approx+0.034$)\n+ frozen pinned fronts",
         ha="center", fontsize=9.5)
axz.text(0.344, 0.80, "recall cycle\n(escape-limited)", ha="center", fontsize=9.5)

# coexistence strips
axz.axvspan(Z0, LAM_DEP, ymin=0.42, ymax=0.56, color="0.45", alpha=0.45, hatch="//")
axz.text(0.294, 0.60, "pinned fronts exist (exact fixed points)", ha="center",
         fontsize=9, color="0.25")
axz.plot([LAM_DEP], [0.49], marker="|", ms=14, color="0.2")
axz.text(LAM_DEP + 0.0012, 0.47, "depinning 0.328", fontsize=8.5, color="0.25")
axz.axvspan(LAM_C, LAM_DISP, ymin=0.16, ymax=0.30, color="#5b8ec4", alpha=0.6)
axz.text(0.293 + 0.0135, 0.21, "residual per-pattern memories (fold dispersion)",
         fontsize=9, color="#2b567f", va="center")

# labels under the zoom axis
axz.annotate(r"$\lambda_c=0.2822$: static fold of $\xi^1$",
             xy=(LAM_C, 0.0), xycoords=("data", "axes fraction"),
             xytext=(LAM_C - 0.004, -0.42), textcoords=("data", "axes fraction"),
             fontsize=10, ha="right",
             arrowprops=dict(arrowstyle="-", color="0.25", lw=1.1))
axz.annotate("2nd bond\nthreshold 0.322",
             xy=(LAM_2, 0.0), xycoords=("data", "axes fraction"),
             xytext=(LAM_2 - 0.002, -0.60), textcoords=("data", "axes fraction"),
             fontsize=9, ha="center", color="crimson",
             arrowprops=dict(arrowstyle="-", color="crimson", lw=0.9))
axz.annotate(r"$\lambda^*\approx0.327$: depinning SNIC at the weakest bond",
             xy=(LAM_S, 0.0), xycoords=("data", "axes fraction"),
             xytext=(LAM_S + 0.004, -0.42), textcoords=("data", "axes fraction"),
             fontsize=10, ha="left", color="crimson",
             arrowprops=dict(arrowstyle="-", color="crimson", lw=1.2))

# connectors top bar -> zoom panel
for x, side in [(Z0, 0), (Z1, 1)]:
    cp = ConnectionPatch(xyA=(x, 0.0), coordsA=axt.get_xaxis_transform(),
                         xyB=(x, 1.0), coordsB=axz.get_xaxis_transform(),
                         color="0.4", lw=1.0, ls="-")
    fig.add_artist(cp)

fig.tight_layout()
fp2 = os.path.join(FIG, "figD_phase_map_v2.png")
fig.savefig(fp2, dpi=170, bbox_inches="tight"); plt.close(fig)
print("wrote", fp2)
