"""E35-D0 figure: readout diagnostics on the stored JP_KP V0 trajectory.

Four panels (dpi 170, English labels):
 (a) PSNR per hidden frame for each decoder vs B1/B2 reference lines;
 (b) MSE vs PSNR per decoder x clock (shows the PSNR/MSE divergence of D5);
 (c) amplification |G^dag m|/|m| along one period, transition windows shaded;
 (d) paired per-transition mean Delta PSNR vs B1, with bootstrap 95% CI.

Colorblind-safe categorical palette (Okabe-Ito), fixed assignment.
"""
from __future__ import annotations

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results/8_mhn_video/data/E35_D0_diagnostics.npz")
OUT = os.path.join(ROOT, "results/8_mhn_video/figures/E35_D0_readouts.png")

d = np.load(NPZ, allow_pickle=True)
decs = [str(x) for x in d["decoders"]]
clks = [str(x) for x in d["clocks"]]
psnr = d["psnr"]; mse = d["mse"]
mean_psnr = d["mean_psnr"]; mean_mse = d["mean_mse"]
seg = d["hidden_segment"]
b1_psnr = d["B1_psnr"]; b2_psnr = d["B2_psnr"]; b0_psnr = d["B0_psnr"]
boot_vs_B1 = d["boot_ci_vs_B1"]

# Okabe-Ito, CVD-safe, fixed assignment per decoder entity.
CMAP = {
    "D1_Xa":            "#000000",
    "D2c_XGdm_clip":    "#56B4E9",
    "D2n_XGdm_noclip":  "#CC79A7",
    "D3_TSVD":          "#009E73",
    "D4_beta5":         "#E69F00",
    "D5_norm2":         "#0072B2",
}
LBL = {
    "D1_Xa": "D1  X.a (current)",
    "D2c_XGdm_clip": "D2c  X.G+m (clip)",
    "D2n_XGdm_noclip": "D2n  X.G+m (no clip)",
    "D3_TSVD": "D3  TSVD r*=7 (key-selected)",
    "D4_beta5": "D4  X.G+m readout beta*=5",
    "D5_norm2": "D5  2-comp renorm",
}
B1C, B2C, B0C = "#D55E00", "#7A4E9E", "#8A8A8A"

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
})
fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.4))
ci_u = clks.index("CLK_U")

# ---- (a) PSNR per hidden frame ----
a0 = ax[0, 0]
order = np.argsort(seg + d["hidden_offset"] / 10.0)
x = np.arange(len(order))
for dec in ["D1_Xa", "D2c_XGdm_clip", "D3_TSVD", "D5_norm2"]:
    di = decs.index(dec)
    a0.plot(x, psnr[di, ci_u][order], lw=1.4, color=CMAP[dec],
            marker="o", ms=2.5, label=LBL[dec])
a0.axhline(np.mean(b1_psnr), color=B1C, lw=2, ls="--", label="B1 linear (42.84)")
a0.axhline(np.mean(b2_psnr), color=B2C, lw=2, ls=":", label="B2 motion (44.74)")
a0.axhline(np.mean(b0_psnr), color=B0C, lw=1.5, ls="-.", label="B0 hold (30.06)")
a0.set_xlabel("hidden frame (ordered by transition, phase)")
a0.set_ylabel("PSNR (dB)")
a0.set_title("(a) PSNR per hidden frame, uniform-time clock", loc="left", fontweight="bold")
a0.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
a0.set_ylim(20, 48)

# ---- (b) MSE vs PSNR scatter ----
b0 = ax[0, 1]
mk = {"CLK_U": "o", "CLK_P": "s"}
for di, dec in enumerate(decs):
    for ci, clk in enumerate(clks):
        b0.scatter(mean_mse[di, ci], mean_psnr[di, ci], s=70,
                   color=CMAP[dec], marker=mk[clk], edgecolor="white",
                   linewidth=0.7, zorder=3)
    b0.annotate(dec.split("_")[0], (mean_mse[di, ci_u], mean_psnr[di, ci_u]),
                fontsize=7, xytext=(4, 3), textcoords="offset points")
b0.scatter(np.mean(d["B1_mse"]), np.mean(b1_psnr), s=90, marker="*",
           color=B1C, edgecolor="k", linewidth=0.5, zorder=4, label="B1 linear")
b0.scatter(np.mean(d["B2_mse"]), np.mean(b2_psnr), s=90, marker="*",
           color=B2C, edgecolor="k", linewidth=0.5, zorder=4, label="B2 motion")
b0.scatter(np.mean(d["B0_mse"]), np.mean(b0_psnr), s=70, marker="*",
           color=B0C, edgecolor="k", linewidth=0.5, zorder=4, label="B0 hold")
b0.set_xlabel("mean MSE (lower is better)")
b0.set_ylabel("mean PSNR (dB)")
b0.set_title("(b) MSE vs PSNR per decoder x clock (D5: PSNR up but MSE up)",
             loc="left", fontweight="bold")
b0.legend(fontsize=7, loc="lower left")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                  markersize=7, label="CLK-U"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="#555",
                  markersize=7, label="CLK-P")]
leg2 = b0.legend(handles=handles, fontsize=7, loc="upper right", title="clock")
b0.add_artist(leg2)

# ---- (c) amplification along one period ----
c0 = ax[1, 0]
t = d["amplification_time"]; amp = d["amplification_trace"]
tmask = d["amplification_transition_mask"]
# pick a window of ~one period in the analysed (second-half) regime
t_lo = 700.0; t_hi = 760.0
w = (t >= t_lo) & (t <= t_hi)
c0.plot(t[w], amp[w], lw=1.3, color="#333333")
# shade transition windows
in_trans = tmask[w]; tw = t[w]
starts = np.where(np.diff(in_trans.astype(int)) == 1)[0]
ends = np.where(np.diff(in_trans.astype(int)) == -1)[0]
if in_trans[0]:
    starts = np.r_[0, starts]
if in_trans[-1]:
    ends = np.r_[ends, len(tw) - 1]
for s, e in zip(starts, ends):
    c0.axvspan(tw[s], tw[e], color="#009E73", alpha=0.12)
c0.axhline(1.0, color="#999", lw=1, ls=":")
amp_med = np.median(amp)
c0.axhline(amp_med, color="#D55E00", lw=1.2, ls="--",
           label=f"median {amp_med:.2f}")
c0.set_xlabel("time t")
c0.set_ylabel(r"$\|G^{\dagger} m\| / \|m\|$")
c0.set_title("(c) Pseudoinverse amplification along one period (transitions shaded)",
             loc="left", fontweight="bold")
c0.legend(fontsize=7, loc="upper right")
c0.text(0.02, 0.04, f"max {np.max(amp):.2f}  |  cond(G)=5.9e5",
        transform=c0.transAxes, fontsize=7, color="#555")

# ---- (d) paired Delta PSNR vs B1, bootstrap CI ----
d0 = ax[1, 1]
ys = np.arange(len(decs))[::-1]
for di, dec in enumerate(decs):
    mean_, lo, hi = boot_vs_B1[di, ci_u, 0], boot_vs_B1[di, ci_u, 1], boot_vs_B1[di, ci_u, 2]
    d0.errorbar(mean_, ys[di], xerr=[[mean_ - lo], [hi - mean_]], fmt="o",
                color=CMAP[dec], ms=7, capsize=3, lw=1.6, ecolor=CMAP[dec])
d0.axvline(0.0, color=B1C, lw=2, ls="--")
d0.text(0.98, 0.97, "B1 = 0", color=B1C, fontsize=8, ha="right", va="top",
        transform=d0.transAxes)
d0.set_yticks(ys)
d0.set_yticklabels([dec.split("_")[0] for dec in decs])
d0.set_ylim(-0.6, len(decs) - 0.4)
d0.set_xlabel(r"mean $\Delta$PSNR vs B1 per transition (dB), bootstrap 95% CI")
d0.set_title("(d) All decoders remain far below B1 (V0 NO-GO conserved)",
             loc="left", fontweight="bold")
d0.text(0.02, 0.02, "0/16 favorable transitions vs B1 for every decoder",
        transform=d0.transAxes, fontsize=7, color="#555", va="bottom")

fig.suptitle("E35-D0  readout diagnostics on the stored JP_KP V0 trajectory "
             "(no new dataset; replay exact, max|m-m_stored|=0)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=170)
print("saved:", OUT)
