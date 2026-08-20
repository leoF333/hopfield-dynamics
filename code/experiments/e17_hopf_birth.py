"""
E17: birth of the slow oscillation of the MEMORY state just past lambda_Hopf
(alpha=0.07, tau=100 — the delay-induced Hopf regime found in the static study).

Protocol: exact memory fixed point on the branch (robust tracer + Newton polish),
perturbed by 1e-3 in-span noise, integrated with the exact P-dim DDE at tau=100.
  - control run at lam=0.2010 (below lambda_Hopf=0.20162): oscillation must DECAY;
  - test run at lam=0.20167 (inside the window (lambda_Hopf, fold=0.2017)):
    oscillation must GROW at rate ~Re z_cplx (~+0.003) with period ~2pi/omega_c
    (~310 = 3.1*tau), then either saturate (supercritical small cycle) or escape
    (subcritical -> fate: pinned front? itinerancy?).
Outputs: table + figH_hopf_birth.png + npz.
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from couplings import Couplings, make_patterns
from robust_branch import trace_branch, woodbury_newton
from cycle_reduced import ReducedDDE

N, ALPHA, BETA, T0, SEED = 2000, 0.07, 20.0, 1.0, 42
TAU, DT = 100.0, 0.02
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
rng = np.random.default_rng(11)

# ---- memory branch and target states ----------------------------------------
cfg = dict(lam_min=0.0, lam_max=0.6, ds=0.01, n_overlaps=5)
br, fold, st = trace_branch(coup, BETA, cfg)
i_pk = int(np.argmax(br["lam"]))
lam_up, u_up = br["lam"][:i_pk + 1], br["u_star"][:i_pk + 1]
print(f"[E17] memory branch: fold={fold:.5f} ({st})", flush=True)

runs = {}
for lam, t_tot in [(0.2010, 4000.0), (0.20167, 10000.0)]:
    j = int(np.argmin(np.abs(lam_up - lam)))
    u, ok = woodbury_newton(coup, u_up[j], lam, BETA)
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    a_star = np.linalg.lstsq(xi.T, u, rcond=None)[0]
    m1 = float(xi[0] @ np.tanh(BETA * u) / N)
    print(f"[E17] state lam={lam}: res={res:.1e} m1={m1:.4f}", flush=True)
    a0 = a_star + 1e-3 * rng.standard_normal(P)
    sysP = ReducedDDE(xi, BETA, lam, TAU, T0)
    t0w = time.time()
    sol = sysP.integrate(a0, t_tot, DT, record_every=25)
    print(f"[E17] sim lam={lam} t={t_tot:.0f} done ({time.time()-t0w:.0f}s)", flush=True)
    dA = sol["a"] - a_star[None, :]
    nrm = np.linalg.norm(dA, axis=1)
    runs[lam] = dict(t=sol["t"], dA=dA, nrm=nrm, a_star=a_star, a_end=sol["a"][-1])

# ---- analysis ----------------------------------------------------------------
out = {}
for lam, r in runs.items():
    t, nrm, dA = r["t"], r["nrm"], r["dA"]
    # growth/decay rate: fit log(nrm) on the middle exponential segment
    lo = nrm > 3 * nrm[1]; hi = nrm < 0.03 * np.nanmax(nrm) if lam > 0.2016 else nrm > 0
    seg = np.where((t > 200) & (nrm > 1e-4) & (nrm < 0.05))[0]
    if len(seg) > 10:
        p = np.polyfit(t[seg], np.log(nrm[seg]), 1); rate = p[0]
    else:
        rate = np.nan
    # oscillation period: component with max late variance, peak spacing
    late = slice(len(t) // 4, len(t) // 2)
    kosc = int(np.argmax(dA[late].var(axis=0)))
    x = dA[:, kosc] - np.convolve(dA[:, kosc], np.ones(41) / 41, mode="same")
    pk = [i for i in range(2, len(x) - 2)
          if x[i] > x[i-1] and x[i] > x[i+1] and abs(x[i]) > 0.2 * np.abs(x[late]).max()]
    Tosc = float(np.median(np.diff(t[pk]))) if len(pk) > 3 else np.nan
    a_end_sorted = np.sort(np.abs(r["a_end"]))[::-1][:4]
    out[lam] = dict(rate=rate, Tosc=Tosc, kosc=kosc, a_end=a_end_sorted)
    print(f"[E17] lam={lam}: taux={rate:+.5f}  T_osc={Tosc:.0f} (T/tau={Tosc/TAU:.2f})"
          f"  |a|_final tries={np.array2string(a_end_sorted, precision=2)}", flush=True)

# ---- figure -------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for lam, c in [(0.2010, "steelblue"), (0.20167, "crimson")]:
    r = runs[lam]
    a1.semilogy(r["t"], r["nrm"], color=c, lw=1.2,
                label=rf"$\lambda={lam}$ (taux fit {out[lam]['rate']:+.4f})")
a1.set_xlabel("t"); a1.set_ylabel(r"$\|a(t)-a^*\|$")
a1.set_title("Croissance/décroissance de la perturbation\n"
             r"(prédiction linéaire : $-0.015$ / $+0.003$)")
a1.legend(fontsize=9); a1.grid(alpha=.3)
r = runs[0.20167]; k = out[0.20167]["kosc"]
a2.plot(r["t"], r["dA"][:, k], color="crimson", lw=0.9)
a2.set_xlabel("t"); a2.set_ylabel(rf"$\delta a_{{{k}}}(t)$")
a2.set_title(f"L'oscillation lente naissante : T_osc={out[0.20167]['Tosc']:.0f}"
             f" ≈ {out[0.20167]['Tosc']/TAU:.2f}·τ  (préd. 2π/ω_c≈310)")
a2.grid(alpha=.3)
fig.suptitle(f"E17 — naissance du Hopf lent de la mémoire "
             f"(α={ALPHA}, τ={TAU:.0f}, N={N}, seed {SEED})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.91])
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
fig.savefig(os.path.join(OUT, "figures", "figH_hopf_birth.png"), dpi=170,
            bbox_inches="tight")
np.savez(os.path.join(OUT, "E17_hopf_birth.npz"),
         **{f"t_{lam}": runs[lam]["t"] for lam in runs},
         **{f"nrm_{lam}": runs[lam]["nrm"] for lam in runs})
print("[E17] fig -> figures/figH_hopf_birth.png", flush=True)
