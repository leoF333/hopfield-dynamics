"""
E17c: honest re-analysis of the E17 runs from the saved npz + E17b findings,
and regeneration of figH (v2, English labels).

Facts fed in from E17b (e17b_identify_final_state.py output, 2026-07-03):
  * at lam=0.20167 the branch state a* is WEAKLY STABLE: leading roots
    -0.00179 +/- 0.0239i (T_osc = 263 = 2.6 tau) and -0.0103 +/- 0.0805i
    (T_osc = 78 = 0.78 tau);
  * the trajectory converged to a SECOND stable memory-like fixed point a**
    at distance 1.0131e-2 (m1 = 0.99793, leading roots -0.038 +/- 0.036i).
This script fits the control decay rate and the oscillation period from the
npz and draws figH_hopf_birth_v2.png.
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

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
d = np.load(os.path.join(OUT, "E17_hopf_birth.npz"))
t_c, n_c = d["t_0.201"], d["nrm_0.201"]
t_t, n_t = d["t_0.20167"], d["nrm_0.20167"]
TAU = 100.0

# ---- control: late-time decay rate (slowest mode) + oscillation period ------
seg = (t_c > 250) & (t_c < 850) & (n_c > 1e-10)
p = np.polyfit(t_c[seg], np.log(n_c[seg]), 1)
rate_c = p[0]
# oscillation period from peaks of log-residual around the linear fit
resid = np.log(n_c[seg]) - np.polyval(p, t_c[seg])
ts = t_c[seg]
pk = [i for i in range(2, len(resid) - 2)
      if resid[i] > resid[i-1] and resid[i] > resid[i+1] and resid[i] > 0.2]
Tosc_c = float(np.median(np.diff(ts[pk]))) if len(pk) > 2 else np.nan
floor_c = float(np.median(n_c[t_c > 1500]))

# ---- test: plateau value + transition time -----------------------------------
plateau = float(np.median(n_t[t_t > 1000]))
i_dip = int(np.argmin(np.where(t_t > 50, n_t, np.inf)))
print(f"[E17c] control lam=0.2010 : rate = {rate_c:+.4f}  T_osc = {Tosc_c:.0f} "
      f"(T/tau = {Tosc_c/TAU:.2f})  floor = {floor_c:.1e}")
print(f"[E17c] test lam=0.20167  : dip to {n_t[i_dip]:.1e} at t={t_t[i_dip]:.0f}, "
      f"plateau = {plateau:.4e}  (E17b: |a**-a*| = 1.0131e-2)")

# ---- figure -------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.semilogy(t_c, n_c, color="steelblue", lw=1.1,
            label=rf"$\lambda=0.2010$ (control): decays, fit rate ${rate_c:+.4f}$")
a1.semilogy(t_t, n_t, color="crimson", lw=1.1,
            label=r"$\lambda=0.20167$: basin hop to twin state $a^{**}$")
a1.axhline(1.0131e-2, color="k", ls=":", lw=1,
           label=r"$\|a^{**}-a^*\|=1.013\times10^{-2}$ (E17b)")
a1.set_xlabel("t"); a1.set_ylabel(r"$\|a(t)-a^*\|$")
a1.set_title("Perturbation of the memory state near the fold\n"
             r"($\alpha=0.07$, $\tau=100$: no Hopf growth — bistability instead)")
a1.legend(fontsize=8.5, loc="center right"); a1.grid(alpha=.3)
a1.set_xlim(-100, 4100)

a2.semilogy(t_c, n_c, color="steelblue", lw=1.2)
a2.set_xlim(150, 900); a2.set_ylim(1e-11, 1e-4)
a2.set_xlabel("t"); a2.set_ylabel(r"$\|a(t)-a^*\|$")
a2.set_title(f"Control zoom: damped oscillation, $T_{{osc}}={Tosc_c:.0f}"
             rf"\approx{Tosc_c/TAU:.2f}\,\tau$" "\n"
             r"(delay-harmonic family; slow family $2.6\,\tau$ predicted by $T_P$)")
a2.grid(alpha=.3, which="both")
fig.suptitle(r"E17 — probing the delay-induced Hopf window of the memory "
             rf"($\alpha=0.07$, $\tau=100$, $N=2000$, seed 42)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fp = os.path.join(OUT, "figures", "figH_hopf_birth_v2.png")
fig.savefig(fp, dpi=170, bbox_inches="tight")
print(f"[E17c] fig -> {fp}")
