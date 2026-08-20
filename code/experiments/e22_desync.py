"""
E22 - Desynchronization of the sequential-recall cycle at small delay.

Turns E19a's coarse tau hints into a clean quantitative law at lam=0.9:
  E22a  dense tau sweep of the clock & front shape (T1, width W, peak recall,
        n_lobes) with error bars.
  E22b  a scalar coherence ORDER PARAMETER R(tau) in [0,1] (single sharp front
        -> R~1 ; delocalized multi-lobe -> R->0) and the crossover tau_c.
  E22c  small-tau period law: fit T1(tau), find where slope(T1 vs tau) -> 1,
        report the escape-time intercept. Optional lam re-synchronization check.

Reuses ReducedDDE (exact P-dim reduction, dim P=100), float64 CPU.

Order parameter (E22b): we report TWO complementary scalars per frame, averaged
over the post-transient window, and pick the participation-ratio one as primary.
  * Ring-Kuramoto coherence (primary):
        R_K = | sum_nu a_nu e^{i 2pi (nu - lead)/W_ring} | / sum_nu |a_nu|
    computed on the lead-centred, non-negative part of the profile with a ring
    of circumference set by the pattern count P. R_K ~ 1 for a single localized
    front (all mass at one phase), R_K -> 0 for mass spread around the ring.
  * Participation-ratio localization (secondary):
        PR    = (sum a_nu^2)^2 / (P sum a_nu^4)   in [1/P, 1]
        R_PR  = 1 - PR*P / W_broad ... reported directly as inverse participation
    We report L = 1/(P * PR) (number of participating patterns) and derive a
    normalized R_PR = (1/L - 1/L_max)/(1 - 1/L_max) style scalar; simpler: we
    report R_PR = 1/L_eff mapped so single-front->1. See code for exact def.

Usage:
    python e22_desync.py pilot         # one tau end-to-end + dt convergence
    python e22_desync.py sweep         # full E22a/b/c sweep + figure + npz
    python e22_desync.py lamcheck      # E22c optional lam re-sync check
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
from tqdm import tqdm

from couplings import make_patterns
from cycle_reduced import ReducedDDE

# ---- model / run constants ---------------------------------------------------
N, ALPHA, BETA, T0, SEED, LAM = 2000, 0.05, 20.0, 1.0, 42, 0.9
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)

FRONT_FRAC = 0.5          # a_nu counted "active" if > FRONT_FRAC * frame peak
MIN_HANDOFF = 25          # >=25 handoffs after warmup
WARMUP_FRAC = 0.40        # discard first 40% of trajectory as transient

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIGDIR = os.path.join(OUT, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def choose_dt(tau):
    """dt <= min(0.01, tau/25), and dt must divide tau exactly (integrator asserts).
    At tau=0 use the ODE branch dt=0.01."""
    if tau == 0.0:
        return 0.01
    dt_cap = min(0.01, tau / 25.0)
    # choose n = ceil(tau/dt_cap) so dt=tau/n <= dt_cap and divides tau exactly
    n = int(np.ceil(tau / dt_cap - 1e-12))
    return tau / n


def measure(A, lead, dt, t_total):
    """Compute clock + shape + coherence statistics on the post-transient window.
    A: (n+1, P) trajectory of overlaps a_nu(t). lead: argmax index per frame."""
    # ---- clock: per-pattern handoff times (lead advances) --------------------
    chg = np.nonzero(np.diff(lead) != 0)[0] + 1
    step_times = chg * dt
    keep = step_times > WARMUP_FRAC * t_total
    st = step_times[keep] if keep.sum() >= 5 else step_times[len(step_times) // 2:]
    # keep only forward single-step advances for a clean per-pattern time
    T1s = np.diff(st)
    T1s = T1s[T1s > 0]

    # ---- post-transient window ----------------------------------------------
    i0 = int(WARMUP_FRAC * len(A))
    win = A[i0:]                                   # (m, P)
    frames = range(0, len(win), max(1, len(win) // 600))

    # peak recall (leading component amplitude)
    peak_series = win.max(axis=1)
    peak = float(peak_series.mean()); peak_std = float(peak_series.std())

    # front WIDTH: # patterns with a_nu > FRONT_FRAC * frame-peak (relative)
    pk = win.max(axis=1, keepdims=True)
    rel = win / np.maximum(pk, 1e-9)
    width_series = (rel > FRONT_FRAC).sum(axis=1).astype(float)
    width = float(width_series.mean()); width_std = float(width_series.std())

    # n_lobes: local maxima of the cyclic profile above FRONT_FRAC*peak
    def n_local_max(row):
        p = row.max()
        above = row > FRONT_FRAC * p
        rp = np.roll(row, -1); rm = np.roll(row, 1)
        return int(np.sum(above & (row >= rp) & (row >= rm)))
    nlob = np.array([n_local_max(win[k]) for k in frames], float)
    n_lobes = float(nlob.mean()); n_lobes_std = float(nlob.std())

    # ---- E22b order parameters ----------------------------------------------
    # Ring-Kuramoto coherence on non-negative mass (primary).
    nu = np.arange(P)
    phase = np.exp(2j * np.pi * nu / P)
    def coherence(row):
        w = np.clip(row, 0.0, None)
        s = w.sum()
        if s < 1e-12:
            return 0.0
        return float(np.abs((w * phase).sum()) / s)
    Rk = np.array([coherence(win[k]) for k in frames], float)
    R_kura = float(Rk.mean()); R_kura_std = float(Rk.std())

    # Participation-ratio localization (secondary): L_eff = #participating patterns
    #   PR = (sum w^2)^2 / (P sum w^4);  L_eff = 1/(P*PR) in [1/P, 1] fraction,
    #   so N_part = L_eff * P = (sum w^2)^2 / sum w^4  patterns.
    def n_part(row):
        w = np.clip(row, 0.0, None)
        s2 = (w ** 2).sum(); s4 = (w ** 4).sum()
        if s4 < 1e-24:
            return float(P)
        return float(s2 * s2 / s4)
    Npart = np.array([n_part(win[k]) for k in frames], float)
    n_part_mean = float(Npart.mean()); n_part_std = float(Npart.std())
    # R_PR: map single participating pattern (N_part=1) -> 1, delocalized -> 0.
    # R_PR = 1/N_part  (in (0,1], =1 iff perfectly localized single pattern).
    R_pr_series = 1.0 / np.maximum(Npart, 1.0)
    R_pr = float(R_pr_series.mean()); R_pr_std = float(R_pr_series.std())

    return dict(
        T1_mean=float(T1s.mean()) if len(T1s) else np.nan,
        T1_std=float(T1s.std()) if len(T1s) else np.nan,
        n_handoff=int(len(chg)), n_used=int(len(T1s)),
        peak=peak, peak_std=peak_std,
        width=width, width_std=width_std,
        n_lobes=n_lobes, n_lobes_std=n_lobes_std,
        R_kura=R_kura, R_kura_std=R_kura_std,
        R_pr=R_pr, R_pr_std=R_pr_std,
        n_part=n_part_mean, n_part_std=n_part_std,
    )


def run_tau(tau, lam=LAM, dt=None, min_handoff=MIN_HANDOFF, return_traj=False):
    """Integrate from memory IC + forward bias; return clock/shape/coherence stats."""
    if dt is None:
        dt = choose_dt(tau)
    sysP = ReducedDDE(xi, BETA, lam, tau, T0)
    T1_est = tau + 1.2                                   # generous budget
    # need >= min_handoff after warmup -> total handoffs ~ min_handoff/(1-warmup)
    n_total = int(np.ceil((min_handoff + 6) / (1.0 - WARMUP_FRAC)))
    t_total = max(80.0, n_total * T1_est)
    a0 = np.zeros(P); a0[0] = 0.99; a0[1] = 0.05
    t0w = time.time()
    sol = sysP.integrate(a0, t_total, dt, record_every=1)
    wall = time.time() - t0w
    A = sol["a"]; lead = np.argmax(A, axis=1)
    stats = measure(A, lead, dt, t_total)
    stats.update(tau=tau, lam=lam, dt=dt, t_total=t_total, wall=wall)
    if return_traj:
        stats["A"] = A; stats["lead"] = lead
    return stats


# ==============================================================================
def pilot():
    """One tau end-to-end with timing + dt-convergence of T1."""
    tau = 0.5
    print(f"[E22 pilot] tau={tau}, lam={LAM}, P={P}\n")
    base_dt = choose_dt(tau)
    r = run_tau(tau, dt=base_dt)
    print(f"  dt={base_dt:.5f}  t_total={r['t_total']:.0f}  wall={r['wall']:.1f}s")
    print(f"  T1={r['T1_mean']:.4f}+-{r['T1_std']:.4f}  handoffs={r['n_handoff']} "
          f"used={r['n_used']}")
    print(f"  peak={r['peak']:.3f}  width={r['width']:.2f}  n_lobes={r['n_lobes']:.2f}")
    print(f"  R_kura={r['R_kura']:.3f}  R_pr={r['R_pr']:.3f}  N_part={r['n_part']:.2f}")

    print("\n[E22 pilot] dt-convergence of T1 (halve dt):")
    prev = None
    for dt in [base_dt, base_dt / 2, base_dt / 4]:
        rr = run_tau(tau, dt=dt, min_handoff=25)
        drift = "" if prev is None else f"  dT1={abs(rr['T1_mean']-prev):.4f}"
        print(f"    dt={dt:.5f}: T1={rr['T1_mean']:.4f}+-{rr['T1_std']:.4f} "
              f"width={rr['width']:.2f} R_kura={rr['R_kura']:.3f}{drift}")
        prev = rr['T1_mean']
    print("\n[E22 pilot] done.")


# ==============================================================================
def sweep():
    TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 5.0, 10.0]
    print(f"[E22a] dense tau sweep, lam={LAM}, N={N} P={P} beta={BETA}")
    print(f"       taus ({len(TAUS)}): {TAUS}\n")
    res = {}
    t_start = time.time()
    for tau in tqdm(TAUS, desc="E22a sweep"):
        r = run_tau(tau, return_traj=(tau in (0.0, 0.5, 2.0)))
        res[tau] = r
        tqdm.write(
            f"  tau={tau:>5}: dt={r['dt']:.4f} T1={r['T1_mean']:.3f}+-{r['T1_std']:.3f} "
            f"peak={r['peak']:.3f} W={r['width']:.2f} nlob={r['n_lobes']:.2f} "
            f"Rk={r['R_kura']:.3f} Rpr={r['R_pr']:.3f} Np={r['n_part']:.2f} "
            f"[{r['wall']:.0f}s]")
    print(f"\n[E22a] total sweep wall = {time.time()-t_start:.0f}s")

    taus = np.array(TAUS)
    T1m = np.array([res[t]["T1_mean"] for t in TAUS])
    T1s = np.array([res[t]["T1_std"] for t in TAUS])
    Wm = np.array([res[t]["width"] for t in TAUS])
    Ws = np.array([res[t]["width_std"] for t in TAUS])
    pk = np.array([res[t]["peak"] for t in TAUS])
    pks = np.array([res[t]["peak_std"] for t in TAUS])
    nl = np.array([res[t]["n_lobes"] for t in TAUS])
    nls = np.array([res[t]["n_lobes_std"] for t in TAUS])
    Rk = np.array([res[t]["R_kura"] for t in TAUS])
    Rks = np.array([res[t]["R_kura_std"] for t in TAUS])
    Rpr = np.array([res[t]["R_pr"] for t in TAUS])
    Np = np.array([res[t]["n_part"] for t in TAUS])

    # ---- E22c: small-tau period law -----------------------------------------
    # local slope of T1 vs tau (finite diff), find where it reaches ~1
    def local_slope(taus, y):
        s = np.full_like(y, np.nan)
        s[1:-1] = (y[2:] - y[:-2]) / (taus[2:] - taus[:-2])
        s[0] = (y[1] - y[0]) / (taus[1] - taus[0])
        s[-1] = (y[-1] - y[-2]) / (taus[-1] - taus[-2])
        return s
    slope_loc = local_slope(taus, T1m)
    # small-tau linear fit over tau in [0, 1]
    mlo = taus <= 1.0
    Alo = np.vstack([taus[mlo], np.ones(mlo.sum())]).T
    sl_lo, ic_lo = np.linalg.lstsq(Alo, T1m[mlo], rcond=None)[0]
    # large-tau pacemaker: fixed slope 1 intercept from tau>=2
    mhi = taus >= 2.0
    t_esc_hi = float(np.mean(T1m[mhi] - taus[mhi]))
    # tau where local slope first >= 0.9
    idx90 = np.nonzero(slope_loc >= 0.9)[0]
    tau_slope90 = float(taus[idx90[0]]) if len(idx90) else np.nan

    # ---- E22b: crossover tau_c from R(tau) ----------------------------------
    # define tau_c as where R_kura crosses the midpoint between its small-tau
    # floor and large-tau plateau (0.5*(R_min+R_max)), by linear interpolation.
    R_lo = float(Rk[taus <= 0.2].mean())
    R_hi = float(Rk[taus >= 3.0].mean())
    R_mid = 0.5 * (R_lo + R_hi)
    tau_c = np.nan
    for i in range(len(taus) - 1):
        if (Rk[i] - R_mid) * (Rk[i + 1] - R_mid) <= 0 and Rk[i] != Rk[i + 1]:
            f = (R_mid - Rk[i]) / (Rk[i + 1] - Rk[i])
            tau_c = float(taus[i] + f * (taus[i + 1] - taus[i]))
            break

    # ---- console tables ------------------------------------------------------
    print(f"\n{'tau':>5} {'T1':>8} {'+-':>7} {'slope':>6} {'W':>6} {'+-':>6} "
          f"{'peak':>6} {'nlob':>5} {'Rkura':>6} {'Rpr':>6} {'Npart':>6}")
    for i, t in enumerate(TAUS):
        print(f"{t:>5} {T1m[i]:>8.3f} {T1s[i]:>7.3f} {slope_loc[i]:>6.2f} "
              f"{Wm[i]:>6.2f} {Ws[i]:>6.2f} {pk[i]:>6.3f} {nl[i]:>5.2f} "
              f"{Rk[i]:>6.3f} {Rpr[i]:>6.3f} {Np[i]:>6.2f}")

    print(f"\n[E22c] small-tau fit (tau<=1): T1 = {sl_lo:.3f} tau + {ic_lo:.3f}")
    print(f"[E22c] large-tau pacemaker (tau>=2): T1 = tau + {t_esc_hi:.3f}")
    print(f"[E22c] local slope reaches 0.9 at tau ~ {tau_slope90}")
    print(f"[E22b] R_kura: floor(tau<=0.2)={R_lo:.3f} plateau(tau>=3)={R_hi:.3f} "
          f"mid={R_mid:.3f} -> tau_c={tau_c:.3f}")

    # ---- figure --------------------------------------------------------------
    make_figure(taus, T1m, T1s, Wm, Ws, pk, pks, nl, nls, Rk, Rks, Rpr, Np,
                slope_loc, sl_lo, ic_lo, t_esc_hi, tau_c, R_mid, res)

    # ---- npz -----------------------------------------------------------------
    npz_path = os.path.join(OUT, "E22_desync.npz")
    np.savez(npz_path,
             taus=taus, T1_mean=T1m, T1_std=T1s, width=Wm, width_std=Ws,
             peak=pk, peak_std=pks, n_lobes=nl, n_lobes_std=nls,
             R_kura=Rk, R_kura_std=Rks, R_pr=Rpr, n_part=Np,
             slope_local=slope_loc, sl_lo=sl_lo, ic_lo=ic_lo,
             t_esc_hi=t_esc_hi, tau_slope90=tau_slope90,
             tau_c=tau_c, R_lo=R_lo, R_hi=R_hi, R_mid=R_mid,
             dt=np.array([res[t]["dt"] for t in TAUS]))
    print(f"\n[E22] npz -> {npz_path}")
    return res


def make_figure(taus, T1m, T1s, Wm, Ws, pk, pks, nl, nls, Rk, Rks, Rpr, Np,
                slope_loc, sl_lo, ic_lo, t_esc_hi, tau_c, R_mid, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.5, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    tt = np.linspace(0, taus.max() * 1.02, 200)

    # (a) T1(tau) with small-tau law + pacemaker asymptote
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(taus, T1m, yerr=T1s, fmt="o", color="navy", ms=6, capsize=3,
                label="measured $T_1$", zorder=3)
    ax.plot(tt, sl_lo * tt + ic_lo, "-", color="crimson",
            label=rf"small-$\tau$: ${sl_lo:.2f}\tau+{ic_lo:.2f}$")
    ax.plot(tt, tt + t_esc_hi, "--", color="gray",
            label=rf"pacemaker $\tau+{t_esc_hi:.2f}$")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$T_1$ (per-pattern handoff)")
    ax.set_title("(a) Clock $T_1(\\tau)$ vs pacemaker law")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    # (b) local slope of T1 vs tau -> approaches 1
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(taus, slope_loc, "s-", color="teal", ms=6)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="slope = 1 (pacemaker)")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$dT_1/d\tau$ (local)")
    ax.set_title("(b) Clock slope $\\to 1$ (pacemaker onset)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    # (c) order parameter R(tau) with tau_c
    ax = fig.add_subplot(gs[0, 2])
    ax.errorbar(taus, Rk, yerr=Rks, fmt="o-", color="darkgreen", ms=6, capsize=3,
                label=r"$R_{\rm ring}$ (Kuramoto)")
    ax.plot(taus, Rpr, "^--", color="purple", ms=5, label=r"$R_{\rm PR}=1/N_{\rm part}$")
    ax.axhline(R_mid, color="gray", ls=":", lw=1)
    if not np.isnan(tau_c):
        ax.axvline(tau_c, color="crimson", ls="--", lw=1.3,
                   label=rf"$\tau_c={tau_c:.2f}$")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("coherence $R$")
    ax.set_title("(c) Desync order parameter $R(\\tau)$")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    # (d) width W(tau) and n_part
    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(taus, Wm, yerr=Ws, fmt="o-", color="orangered", ms=6, capsize=3,
                label="front width $W$ (patterns)")
    ax.plot(taus, Np, "d--", color="steelblue", ms=5,
            label=r"$N_{\rm part}$ (participation)")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("patterns")
    ax.set_title("(d) Front width & participation")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    # (e) peak recall + n_lobes
    ax = fig.add_subplot(gs[1, 1])
    l1 = ax.errorbar(taus, pk, yerr=pks, fmt="o-", color="darkorange", ms=6,
                     capsize=3, label=r"peak $a_\nu$ (recall)")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"peak $a_\nu$", color="darkorange")
    ax.tick_params(axis="y", labelcolor="darkorange")
    ax2 = ax.twinx()
    l2 = ax2.errorbar(taus, nl, yerr=nls, fmt="s--", color="indigo", ms=5,
                      capsize=2, label=r"$n_{\rm lobes}$")
    ax2.set_ylabel(r"$n_{\rm lobes}$", color="indigo")
    ax2.tick_params(axis="y", labelcolor="indigo")
    ax.set_title("(e) Recall strength & lobe count")
    lns = [l1, l2]; ax.legend(lns, [l.get_label() for l in lns], fontsize=8.5)
    ax.grid(alpha=.3)

    # (f) kymograph strip: tau=0 (broad multi-lobe) vs tau=2 (sharp)
    ax = fig.add_subplot(gs[1, 2])
    tau_k = 0.0 if 0.0 in res and "A" in res[0.0] else None
    if tau_k is not None:
        r = res[0.0]; A = r["A"]; dt = r["dt"]
        i0 = int(0.6 * len(A))
        win = A[i0:i0 + int(12 * (0.0 + 1.2) / dt)]
        t_ax = np.arange(win.shape[0]) * dt
        lead0 = int(np.argmax(A[i0]))
        rows = [(lead0 + k) % P for k in range(-4, 20)]
        im = ax.imshow(win[:, rows].T, aspect="auto", origin="lower",
                       extent=[t_ax[0], t_ax[-1], -4, 20], cmap="magma",
                       vmin=-0.1, vmax=float(win.max()), interpolation="nearest")
        ax.set_xlabel("t"); ax.set_ylabel(r"$\nu-\nu_0$")
        ax.set_title(r"(f) $\tau=0$ kymograph (broad multi-lobe)")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label=r"$a_\nu$")
    fig.suptitle(f"E22 - Desynchronization of the recall cycle at small delay "
                 f"($N={N}$, $\\alpha={ALPHA}$, $\\lambda={LAM}$, seed {SEED})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(FIGDIR, "figP_desync.png")
    fig.savefig(fp, dpi=175, bbox_inches="tight")
    print(f"[E22] figure -> {fp}")


# ==============================================================================
def lamcheck():
    """E22c optional: does raising lam re-synchronize the small-tau front?"""
    LAMS = [0.9, 0.95, 0.99]
    TAUS = [0.25, 1.0]
    print(f"[E22c lamcheck] lams={LAMS} taus={TAUS}\n")
    print(f"{'lam':>5} {'tau':>5} {'T1':>8} {'W':>6} {'peak':>6} {'Rk':>6} "
          f"{'Rpr':>6} {'nlob':>5} {'Np':>6}")
    rows = []
    for lam in LAMS:
        for tau in TAUS:
            r = run_tau(tau, lam=lam)
            rows.append((lam, tau, r))
            print(f"{lam:>5} {tau:>5} {r['T1_mean']:>8.3f} {r['width']:>6.2f} "
                  f"{r['peak']:>6.3f} {r['R_kura']:>6.3f} {r['R_pr']:>6.3f} "
                  f"{r['n_lobes']:>5.2f} {r['n_part']:>6.2f}")
    np.savez(os.path.join(OUT, "E22_lamcheck.npz"),
             lams=np.array([x[0] for x in rows]),
             taus=np.array([x[1] for x in rows]),
             T1=np.array([x[2]["T1_mean"] for x in rows]),
             width=np.array([x[2]["width"] for x in rows]),
             peak=np.array([x[2]["peak"] for x in rows]),
             R_kura=np.array([x[2]["R_kura"] for x in rows]),
             R_pr=np.array([x[2]["R_pr"] for x in rows]),
             n_lobes=np.array([x[2]["n_lobes"] for x in rows]),
             n_part=np.array([x[2]["n_part"] for x in rows]))
    print(f"\n[E22c lamcheck] npz -> {os.path.join(OUT, 'E22_lamcheck.npz')}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if mode == "pilot":
        pilot()
    elif mode == "sweep":
        sweep()
    elif mode == "lamcheck":
        lamcheck()
    else:
        print(f"unknown mode {mode}; use pilot|sweep|lamcheck")
