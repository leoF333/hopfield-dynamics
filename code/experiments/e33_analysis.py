"""
E33 aging analysis: two-time correlation collapse test + quench relaxation.
Figure figAC. Verdict: stationary (SRB, no aging) if C(t_w+t,t_w) collapses across
t_w; aging if it fans and collapses under t/t_w.
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIG = os.path.join(OUT, "figures")
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "E33_aging_mlx.npz")
d = np.load(fn)
tws = [int(t) for t in d["tws"]]
backend = "float32/GPU" if "mlx" in fn else "float64/CPU"
print(f"[E33an] {fn}  lam={float(d['lam'])} alpha={float(d['alpha'])} "
      f"seeds={list(d['seeds'])} kept~{int(d['nkeep_tot'])}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# (1) quench relaxation <|a|^2>(t) -> tau_r
tq, enq = d["tq"], d["enq"]
ax[0].plot(tq, enq, "-", color="#2b6cb0", lw=1.3)
plateau = np.median(enq[len(enq)//2:])
ax[0].axhline(plateau, color="k", ls=":", lw=1, label=f"plateau≈{plateau:.3f}")
ax[0].set_xlabel("t (no settle)"); ax[0].set_ylabel(r"$\langle|a|^2\rangle$ (ensemble)")
ax[0].set_title("Exp D: quench relaxation\n(approach to the attractor -> $\\tau_r$)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
# crude tau_r: time to reach within 5% of plateau
above = np.where(enq - plateau > 0.05 * plateau)[0]
tau_r = tq[above[-1]] if len(above) else 0.0
print(f"[E33an] quench: plateau={plateau:.3f}, crude tau_r≈{tau_r:.0f}")

# (2) two-time C normalized vs lag (collapse = stationary)
cols = plt.cm.viridis(np.linspace(0, 0.85, len(tws)))
Cnorm = {}
for tw, c in zip(tws, cols):
    if f"C_{tw}" not in d.files:
        continue
    lag, C = d[f"lag_{tw}"], d[f"C_{tw}"]
    ax[1].plot(lag, C / C[0], color=c, lw=1.4, label=f"$t_w$={tw}")
    Cnorm[tw] = (lag, C / C[0], C[0])
ax[1].set_xlabel("lag $t$"); ax[1].set_ylabel(r"$C(t_w{+}t,t_w)/C(t_w,t_w)$")
ax[1].set_xlim(0, 120)
ax[1].set_title("Two-time correlation vs lag\n(collapse = STATIONARY; fan = aging)")
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3)

# (3) same vs t/t_w (aging would collapse here)
for tw, c in zip(tws, cols):
    if tw == 0 or tw not in Cnorm:
        continue
    lag, cn, _ = Cnorm[tw]
    ax[2].plot(lag / tw, cn, color=c, lw=1.4, label=f"$t_w$={tw}")
ax[2].set_xlabel(r"$t/t_w$"); ax[2].set_ylabel(r"$C/C(t_w,t_w)$")
ax[2].set_xlim(0, 2)
ax[2].set_title("Aging scaling test ($t/t_w$)\n(collapse HERE = aging)")
ax[2].legend(fontsize=8.5); ax[2].grid(alpha=.3)

# verdict: compare the normalized curves across t_w>=50 (on-attractor) at a fixed lag
tws_on = [t for t in tws if t >= 50 and t in Cnorm]
spread = np.nan
if len(tws_on) >= 2:
    Lref = min(len(Cnorm[t][0]) for t in tws_on)
    il = min(np.searchsorted(Cnorm[tws_on[0]][0], 20), Lref - 1)   # lag~20
    vals = [Cnorm[t][1][il] for t in tws_on]
    spread = float(np.max(vals) - np.min(vals))
c0spread = np.nan
if len(tws_on) >= 2:
    c0s = [Cnorm[t][2] for t in tws_on]
    c0spread = float((np.max(c0s) - np.min(c0s)) / np.mean(c0s))
print(f"[E33an] on-attractor (t_w>=50): normalized-C spread at lag~20 = {spread:.3f} "
      f"(small=collapse=stationary); C(t_w,0) relative spread = {c0spread:.3f}")
verdict = ("STATIONARY (no aging): curves collapse" if (spread < 0.1 and c0spread < 0.15)
           else "t_w-DEPENDENCE present -> check aging (t/t_w panel) vs residual transient")
fig.suptitle(f"E33 -- aging test, mixture-phase chaos ($\\lambda$=0.31, {backend}, "
             f"pending float64 validation): {verdict}", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fp = os.path.join(FIG, "figAC_aging_twotime.png")
fig.savefig(fp, dpi=170, bbox_inches="tight")
print(f"[E33an] VERDICT: {verdict}")
print(f"[E33an] figure -> {fp}")
