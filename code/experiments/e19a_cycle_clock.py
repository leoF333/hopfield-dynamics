"""
E19a - Small-delay cycle existence and clock.

For tau in {0, 0.25, 0.5, 1, 2} (incl. the tau=0 ODE limit) at lam=0.9:
integrate the exact reduced DDE from a memory IC + small forward bias, long
enough for >=15 front handoffs. Measure:
  * T1 mean/std (per-pattern advance time) -> test pacemaker law T1 ~ tau + t_esc
  * waveform character: front width in patterns (# of a_nu above a threshold),
    single vs multi-front (number of local maxima of the a-profile)

Reuses ReducedDDE (exact P-dim reduction) for ALL integration (dim P=100).
Outputs: results/cycle/E19a_cycle_clock.npz + figL_smalltau_cycle.png.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from couplings import make_patterns
from cycle_reduced import ReducedDDE

N, ALPHA, BETA, T0, SEED, LAM = 2000, 0.05, 20.0, 1.0, 42, 0.9
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)

# small tau needs dt <= tau/20; keep dt=0.01 (>= that at all these tau except we
# floor tau=0 to a plain ODE) and long enough runs for >=15 handoffs.
TAUS = [0.0, 0.25, 0.5, 1.0, 2.0]
DT = 0.01
FRONT_THRESH = 0.30      # a_nu counted as "active front" above this

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIGDIR = os.path.join(OUT, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def run_tau(tau, dt=DT, min_handoff=18):
    """Integrate from memory IC + forward bias; return trajectory + clock/wave
    statistics. t_total scaled so we clear >= min_handoff handoffs."""
    sysP = ReducedDDE(xi, BETA, LAM, tau, T0)
    # per-step time ~ tau + t_esc(0.9) ~ tau + 0.83; budget generously
    T1_est = tau + 1.0
    t_total = max(60.0, (min_handoff + 8) * T1_est)
    a0 = np.zeros(P)
    a0[0] = 0.99
    a0[1] = 0.05
    t0w = time.time()
    sol = sysP.integrate(a0, t_total, dt, record_every=1)
    wall = time.time() - t0w
    A = sol["a"]                       # (n+1, P)
    t = sol["t"]
    lead = np.argmax(A, axis=1)
    chg = np.nonzero(np.diff(lead) != 0)[0] + 1
    step_times = chg * dt
    # discard first 40% as transient
    keep = step_times > 0.4 * t_total
    st = step_times[keep] if keep.sum() >= 5 else step_times[len(step_times)//2:]
    T1s = np.diff(st)
    # waveform character over the post-transient window
    i0 = int(0.4 * len(A))
    win = A[i0:]
    # peak amplitude of the leading component (recall strength)
    peak = float(win.max(axis=1).mean())
    peak_std = float(win.max(axis=1).std())
    # front width uses a RELATIVE threshold (fraction of the current peak),
    # so it is meaningful even when the recall is weak (low peak) at small tau
    peak_frame = win.max(axis=1, keepdims=True)
    rel = win / np.maximum(peak_frame, 1e-9)
    width = float((rel > 0.5).sum(axis=1).mean())
    width_std = float((rel > 0.5).sum(axis=1).std())
    # number of fronts: count local maxima of the (cyclic) profile above thresh,
    # averaged over frames
    def n_local_max(row):
        pk = row.max()
        above = row > 0.5 * pk           # relative threshold
        rp = np.roll(row, -1)
        rm = np.roll(row, 1)
        return int(np.sum(above & (row >= rp) & (row >= rm)))
    n_fronts = float(np.mean([n_local_max(win[k]) for k in
                              range(0, len(win), max(1, len(win)//400))]))
    return dict(tau=tau, t=t, A=A, lead=lead,
                T1_mean=float(T1s.mean()), T1_std=float(T1s.std()),
                n_handoff=int(len(chg)), n_used=int(len(T1s)),
                width=width, width_std=width_std, n_fronts=n_fronts,
                peak=peak, peak_std=peak_std, wall=wall)


print(f"[E19a] N={N} P={P} lam={LAM} beta={BETA}  taus={TAUS}\n")
results = {}
for tau in TAUS:
    r = run_tau(tau)
    results[tau] = r
    print(f"  tau={tau:>4}: T1={r['T1_mean']:.4f}+-{r['T1_std']:.4f}  "
          f"handoffs={r['n_handoff']:>3}  peak={r['peak']:.3f}+-{r['peak_std']:.3f}  "
          f"width={r['width']:.2f}  n_fronts={r['n_fronts']:.2f}  "
          f"[{r['wall']:.1f}s]", flush=True)

# t_esc from pacemaker law T1 = tau + t_esc (fit intercept)
taus = np.array(TAUS)
T1m = np.array([results[t]["T1_mean"] for t in TAUS])
T1s = np.array([results[t]["T1_std"] for t in TAUS])
# linear fit T1 = tau + c  -> c = mean(T1 - tau)
t_esc = float(np.mean(T1m - taus))
# also a free-slope fit for diagnosis
A_fit = np.vstack([taus, np.ones_like(taus)]).T
slope, intercept = np.linalg.lstsq(A_fit, T1m, rcond=None)[0]
print(f"\n[E19a] pacemaker-law check T1 = tau + t_esc:")
print(f"       fixed-slope intercept t_esc = {t_esc:.3f}")
print(f"       free fit: slope={slope:.3f} (expect ~1), intercept={intercept:.3f}")
print(f"       tau=10 reference: t_esc ~ 0.83, T1 ~ tau+0.83")

# ---- Table --------------------------------------------------------------------
print(f"\n{'tau':>5} {'T1_mean':>9} {'T1_std':>8} {'T1-tau':>8} "
      f"{'peak':>7} {'width':>7} {'n_fronts':>9} {'handoffs':>9}")
for t in TAUS:
    r = results[t]
    print(f"{t:>5} {r['T1_mean']:>9.4f} {r['T1_std']:>8.4f} "
          f"{r['T1_mean']-t:>8.4f} {r['peak']:>7.3f} {r['width']:>7.2f} "
          f"{r['n_fronts']:>9.2f} {r['n_handoff']:>9d}")

# ---- Figure: waveforms panel + T1(tau) ---------------------------------------
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

# top-left big: T1(tau) with pacemaker law
axT = fig.add_subplot(gs[0, 0])
axT.errorbar(taus, T1m, yerr=T1s, fmt="o", color="navy", ms=7, capsize=3,
             label="measured $T_1$")
tt = np.linspace(0, 2.2, 50)
axT.plot(tt, tt + t_esc, "-", color="crimson",
         label=rf"$\tau + t_{{\rm esc}}$, $t_{{\rm esc}}={t_esc:.2f}$")
axT.plot(tt, tt + 0.83, "--", color="gray",
         label=r"$\tau + 0.83$ ($\tau=10$ law)")
axT.set_xlabel(r"$\tau$")
axT.set_ylabel(r"$T_1$ (per-pattern handoff time)")
axT.set_title("(a) Small-$\\tau$ clock vs pacemaker law")
axT.legend(fontsize=8.5)
axT.grid(alpha=.3)

# top-mid: T1_std / T1_mean (order of the front)
axS = fig.add_subplot(gs[0, 1])
axS.plot(taus, T1s / T1m, "s-", color="teal", ms=7)
axS.set_xlabel(r"$\tau$")
axS.set_ylabel(r"$T_1$ scatter / mean")
axS.set_title("(b) Handoff-time disorder (front coherence)")
axS.grid(alpha=.3)

# top-right: recall strength (peak amplitude) + front width vs tau
axW = fig.add_subplot(gs[0, 2])
axW.errorbar(taus, [results[t]["peak"] for t in TAUS],
             yerr=[results[t]["peak_std"] for t in TAUS],
             fmt="o-", color="darkorange", ms=7, capsize=3,
             label="peak $a_\\nu$ (recall)")
axW.plot(taus, [results[t]["width"] for t in TAUS], "^--", color="purple",
         ms=7, label="front width (rel.)")
axW.set_xlabel(r"$\tau$")
axW.set_ylabel("amplitude / patterns")
axW.set_title("(c) Recall strength & front width")
axW.legend(fontsize=9)
axW.grid(alpha=.3)

# bottom: waveform kymographs (lead-centred) for 3 tau
for j, tau in enumerate([0.0, 0.5, 2.0]):
    ax = fig.add_subplot(gs[1, j])
    r = results[tau]
    A = r["A"]
    # show a window of ~8 handoffs after transient
    i0 = int(0.5 * len(A))
    win = A[i0:i0 + int(10 * (tau + 1.0) / DT)]
    t_ax = np.arange(win.shape[0]) * DT
    im = ax.imshow(win.T, aspect="auto", origin="lower",
                   extent=[t_ax[0], t_ax[-1], 0.5, P + 0.5],
                   cmap="magma", vmin=-0.2, vmax=float(win.max()),
                   interpolation="nearest")
    ax.set_xlabel("t")
    ax.set_ylabel(r"pattern index $\nu$")
    # zoom to the active band
    lead0 = r["lead"][i0]
    ax.set_ylim(lead0 - 4, lead0 + 12)
    ax.set_title(rf"(d) $\tau={tau}$ kymograph "
                 rf"($T_1={r['T1_mean']:.2f}$)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=r"$a_\nu$")

fig.suptitle(f"E19a - Sequential-recall cycle at small delay "
             f"($N={N}$, $\\alpha={ALPHA}$, $\\lambda={LAM}$, seed {SEED})",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fp = os.path.join(FIGDIR, "figL_smalltau_cycle.png")
fig.savefig(fp, dpi=175, bbox_inches="tight")
print(f"\n[E19a] figure -> {fp}")

np.savez(os.path.join(OUT, "E19a_cycle_clock.npz"),
         taus=taus, T1_mean=T1m, T1_std=T1s,
         peak=np.array([results[t]["peak"] for t in TAUS]),
         peak_std=np.array([results[t]["peak_std"] for t in TAUS]),
         width=np.array([results[t]["width"] for t in TAUS]),
         n_fronts=np.array([results[t]["n_fronts"] for t in TAUS]),
         n_handoff=np.array([results[t]["n_handoff"] for t in TAUS]),
         t_esc=t_esc, slope=slope, intercept=intercept)
print("[E19a] npz saved.")
