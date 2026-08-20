"""
E29 -- what does the limit cycle pass THROUGH just above lambda*?

Question (user): the recalled fixed points deform continuously from xi^mu into
~70/30 mixtures of (xi^mu, xi^mu+1) and die at saddle-nodes; the cycle born at
lambda* feels their ghosts. Does the running cycle pass through the PURE patterns
xi^mu, or through the tilted MIXTURES that the fixed points had become at death?

Protocol (N=2000, P=100, seed 42, tau=10, beta=20, exact P-dim reduction):
  - warm-start descent lambda = 0.90 -> lambda*+0.002 (lambda*=0.32763, bond 91);
  - at each lambda, integrate until the winner index argmax_mu a_mu(t) has
    completed >= 2 full forward loops (discard >=1, measure on the LAST loop);
  - per bond mu on the measured loop:
      * analog peak: max_t a_mu(t), and co-overlaps a_mu+1, a_mu-1, 3rd |a| there;
      * ghost test: the minimum-|da/dt| point inside mu's winner window,
        compared with the stored fold state (a_c(mu), a1_c(mu)) from E25;
      * binary readout: m_sign(t) = (1/N) xi^w(t) . sign(u(t)) -- peak value per
        bond and fraction of loop time with EXACT retrieval (m_sign = 1);
  - period T(lambda), speed profile along the loop (SNIC bottleneck at bond 91).
Output: figAA_cycle_vs_ghosts.png + E29_cycle_ghosts.npz + printed tables.

CLI: e29_cycle_vs_ghosts.py [--timing]
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from couplings import make_patterns
from cycle_reduced import ReducedDDE

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "cycle")
N, P, SEED = 2000, 100, 42
BETA, TAU, T0 = 20.0, 10.0, 1.0
DT = 0.01
REC = 5                       # record every REC steps -> dt_rec = 0.05
LAMS = [0.90, 0.70, 0.50, 0.40, 0.37, 0.35, 0.34, 0.335, 0.332, 0.330]
T_CHUNK = 200.0               # integrate in chunks, prune old records
T_MAX = 30000.0               # hard cap per lambda


def winner_advance(w):
    """Cumulative forward advance of the winner sequence (mod-P unwrapping)."""
    dw = np.diff(w)
    steps = ((dw + P // 2) % P) - P // 2       # signed circular increment
    return np.concatenate([[0], np.cumsum(steps)])


def run_lambda(sysP, hist, lam, fold_ac, fold_a1c, verbose=True):
    """Integrate at one lambda until >=2 full loops; measure the last loop."""
    sysP.lam = lam
    t_off, rec_t, rec_a = 0.0, [], []
    adv_last = 0
    t_start = time.time()
    while t_off < T_MAX:
        sol = sysP.integrate(hist, T_CHUNK, DT, record_every=REC)
        hist = sol["hist"]
        rec_t.append(sol["t"][1:] + t_off)     # drop duplicated t=0 sample
        rec_a.append(sol["a"][1:])
        t_off += T_CHUNK
        # loop counting on everything retained
        A = np.concatenate(rec_a)
        w = np.argmax(A, axis=1)
        adv = winner_advance(w)
        adv_last = adv[-1]
        if adv_last >= 2 * P + 5:
            break
        # prune: keep only records within the last 2.4 loops of advance
        keep_from = np.searchsorted(adv, adv_last - int(2.4 * P))
        tcat = np.concatenate(rec_t)
        rec_t, rec_a = [tcat[keep_from:]], [A[keep_from:]]
    tcat = np.concatenate(rec_t); A = np.concatenate(rec_a)
    w = np.argmax(A, axis=1); adv = winner_advance(w)
    wall = time.time() - t_start
    if adv_last < 2 * P + 5:
        print(f"  [lam={lam:.3f}] CAP HIT at T={t_off:.0f} "
              f"(advance {adv_last}/{2*P+5}) -- skipping")
        return hist, None
    # last full loop: advance from adv[-1]-P to adv[-1] (plus small margins)
    i1 = len(adv) - 1
    i0 = np.searchsorted(adv, adv[-1] - P)
    t_loop, A_loop = tcat[i0:i1 + 1], A[i0:i1 + 1]
    w_loop = w[i0:i1 + 1]
    T_per = t_loop[-1] - t_loop[0]
    # speed along the loop (finite differences of recorded samples)
    dtr = np.gradient(t_loop)
    speed = np.linalg.norm(np.gradient(A_loop, axis=0), axis=1) / dtr

    # ---- per-bond measurements -------------------------------------------
    peak_a = np.full(P, np.nan); peak_a1 = np.full(P, np.nan)
    peak_am1 = np.full(P, np.nan); peak_third = np.full(P, np.nan)
    slow_a = np.full(P, np.nan); slow_a1 = np.full(P, np.nan)
    dwell = np.full(P, np.nan); vmin = np.full(P, np.nan)
    for mu in range(P):
        win = np.where(w_loop == mu)[0]
        if len(win) == 0:
            continue
        # analog peak of a_mu over the WHOLE loop (not only its window)
        jp = int(np.argmax(A_loop[:, mu]))
        peak_a[mu] = A_loop[jp, mu]
        peak_a1[mu] = A_loop[jp, (mu + 1) % P]
        peak_am1[mu] = A_loop[jp, (mu - 1) % P]
        rest = np.abs(A_loop[jp]).copy()
        rest[[mu, (mu + 1) % P, (mu - 1) % P]] = 0.0
        peak_third[mu] = rest.max()
        # ghost test: slowest point inside mu's winner window
        js = win[int(np.argmin(speed[win]))]
        slow_a[mu] = A_loop[js, mu]; slow_a1[mu] = A_loop[js, (mu + 1) % P]
        vmin[mu] = speed[js]
        dwell[mu] = dtr[win].sum()

    # ---- binary readout on the loop (subsampled x4 + at each peak) --------
    sub = np.arange(0, len(t_loop), 4)
    peaks_idx = np.array([int(np.argmax(A_loop[:, mu])) for mu in range(P)])
    idx_sign = np.unique(np.concatenate([sub, peaks_idx]))
    U = sysP.xi.T @ A_loop[idx_sign].T              # (N, n_samples)
    Ssign = np.sign(U); Ssign[Ssign == 0] = 1.0
    Msign = (sysP.xi @ Ssign) / sysP.N              # (P, n_samples) sign overlaps
    w_s = w_loop[idx_sign]
    m_sign_w = Msign[w_s, np.arange(len(idx_sign))]  # retrieval wrt current winner
    frac_exact = float(np.mean(m_sign_w >= 1.0 - 1e-12))
    m_peak = np.full(P, np.nan)
    pos_in_idx = np.searchsorted(idx_sign, peaks_idx)
    for mu in range(P):
        m_peak[mu] = Msign[mu, pos_in_idx[mu]]

    if verbose:
        print(f"  [lam={lam:.3f}] T_loop={T_per:8.1f}  wall={wall:6.1f}s | "
              f"peak a_mu: med {np.nanmedian(peak_a):.3f} "
              f"[{np.nanmin(peak_a):.3f},{np.nanmax(peak_a):.3f}] | "
              f"co a_mu+1 at peak: med {np.nanmedian(peak_a1):.3f} | "
              f"m_sign peak: min {np.nanmin(m_peak):.4f} | "
              f"exact-readout time frac: {frac_exact:.3f}")
        sys.stdout.flush()
    out = dict(lam=lam, T=T_per, peak_a=peak_a, peak_a1=peak_a1,
               peak_am1=peak_am1, peak_third=peak_third,
               slow_a=slow_a, slow_a1=slow_a1, vmin=vmin, dwell=dwell,
               m_peak=m_peak, frac_exact=frac_exact,
               t_loop=t_loop - t_loop[0], w_loop=w_loop, speed=speed)
    # store the (a_mu, a_mu+1) projections of the loop for 3 reference bonds
    for mu in (91, 43, 28):
        out[f"proj_{mu}"] = A_loop[:, [mu, (mu + 1) % P]]
    return hist, out


def _experiment_main():
    xi, _ = make_patterns(N, P, SEED)
    d25 = np.load(os.path.join(BASE, "E25_deathshape.npz"))
    sel = np.asarray(d25["sel"], int)
    fold_ac = np.full(P, np.nan); fold_a1c = np.full(P, np.nan)
    fold_lc = np.full(P, np.nan)
    fold_ac[sel] = d25["a_c"]; fold_a1c[sel] = d25["a1_c"]
    fold_lc[sel] = d25["lam_c"]
    lam_star = np.nanmax(fold_lc)
    print(f"[E29] lambda* = {lam_star:.5f} (bond {np.nanargmax(fold_lc)}); "
          f"lambdas: {LAMS}")

    sysP = ReducedDDE(xi, BETA, LAMS[0], TAU, T0)
    if "--timing" in sys.argv:
        hist = 0.9 * np.eye(P)[0]
        t0c = time.time(); sysP.integrate(hist, 100.0, DT, record_every=REC)
        print(f"[E29] timing: 100 t.u. at P={P} in {time.time()-t0c:.1f} s")
        return

    hist = 0.9 * np.eye(P)[0]        # constant history on pattern 1
    # initial transient at lam=0.9 to land on the cycle
    sol = sysP.integrate(hist, 400.0, DT, record_every=REC)
    hist = sol["hist"]
    results = []
    for lam in LAMS:
        hist, out = run_lambda(sysP, hist, lam, fold_ac, fold_a1c)
        if out is not None:
            results.append(out)

    # ------------------------------------------------------------ save npz
    save = dict(lam_star=lam_star, fold_ac=fold_ac, fold_a1c=fold_a1c,
                fold_lc=fold_lc, lams=np.array([r["lam"] for r in results]),
                Ts=np.array([r["T"] for r in results]),
                frac_exact=np.array([r["frac_exact"] for r in results]))
    for r in results:
        tag = f"{r['lam']:.3f}".replace(".", "p")
        for k in ("peak_a", "peak_a1", "peak_am1", "peak_third", "slow_a",
                  "slow_a1", "vmin", "dwell", "m_peak"):
            save[f"{k}_{tag}"] = r[k]
    rc = results[-1]                  # closest to lambda*
    tagc = f"{rc['lam']:.3f}".replace(".", "p")
    for mu in (91, 43, 28):
        save[f"proj_{mu}_{tagc}"] = rc[f"proj_{mu}"]
    save[f"t_loop_{tagc}"] = rc["t_loop"]; save[f"w_loop_{tagc}"] = rc["w_loop"]
    save[f"speed_{tagc}"] = rc["speed"]
    np.savez(os.path.join(BASE, "E29_cycle_ghosts.npz"), **save)
    print("[E29] npz saved (checkpoint before figure)")

    # ------------------------------------------------------------ tables
    print("\n[E29] === peak analog overlap vs fold state, per lambda ===")
    print(f"{'lam':>6} {'T_loop':>8} {'<peak a>':>9} {'<co a+1>':>9} "
          f"{'<fold a_c>':>10} {'<fold a1_c>':>11} {'min m_peak':>10} "
          f"{'exact frac':>10}")
    for r in results:
        print(f"{r['lam']:6.3f} {r['T']:8.1f} {np.nanmedian(r['peak_a']):9.3f} "
              f"{np.nanmedian(r['peak_a1']):9.3f} {np.nanmedian(fold_ac):10.3f} "
              f"{np.nanmedian(fold_a1c):11.3f} {np.nanmin(r['m_peak']):10.4f} "
              f"{r['frac_exact']:10.3f}")
    # ghost test at the closest lambda
    ok = np.isfinite(rc["slow_a"]) & np.isfinite(fold_ac)
    d_ghost = np.hypot(rc["slow_a"] - fold_ac, rc["slow_a1"] - fold_a1c)[ok]
    mu_w = int(np.nanargmax(fold_lc))
    print(f"\n[E29] ghost test at lam={rc['lam']:.3f}: distance "
          f"(slow point) -> (fold state) per bond:")
    print(f"  median {np.median(d_ghost):.3f}  min {d_ghost.min():.3f}  "
          f"max {d_ghost.max():.3f}")
    print(f"  critical bond {mu_w}: slow ({rc['slow_a'][mu_w]:.3f},"
          f"{rc['slow_a1'][mu_w]:.3f}) vs fold ({fold_ac[mu_w]:.3f},"
          f"{fold_a1c[mu_w]:.3f})  dwell {rc['dwell'][mu_w]:.1f} / "
          f"T={rc['T']:.1f}  vmin {rc['vmin'][mu_w]:.2e}")
    dw = rc["dwell"][np.isfinite(rc["dwell"])]
    print(f"  dwell: critical bond {rc['dwell'][mu_w]:.1f} vs median "
          f"{np.median(dw):.1f} (x{rc['dwell'][mu_w]/np.median(dw):.1f})")

    # ------------------------------------------------------------ figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))

    ax = axes[0, 0]   # loop projections + fold points + pure-pattern corner
    cols = {91: "#c0392b", 43: "#2b6cb0", 28: "#2f855a"}
    for mu in (91, 43, 28):
        Cxy = rc[f"proj_{mu}"]
        ax.plot(Cxy[:, 0], Cxy[:, 1], color=cols[mu], lw=1.1,
                label=f"cycle, bond {mu} "
                      f"($\\lambda_c$={fold_lc[mu]:.3f})")
        ax.scatter([fold_ac[mu]], [fold_a1c[mu]], marker="*", s=240,
                   color=cols[mu], edgecolor="k", zorder=5)
    ax.scatter([], [], marker="*", s=140, color="w", edgecolor="k",
               label="fold state (dead fixed point)")
    a_pure = np.nanmax([r["peak_a"] for r in results[:1]])
    ax.scatter([1.0], [0.0], marker="s", s=90, color="gold", edgecolor="k",
               zorder=5, label=r"pure pattern $(a_\mu,a_{\mu+1})=(1,0)$")
    ax.set_xlabel(r"$a_\mu$"); ax.set_ylabel(r"$a_{\mu+1}$")
    ax.set_title(f"Cycle at $\\lambda={rc['lam']:.3f}$ in the "
                 r"$(a_\mu,a_{\mu+1})$ plane vs the dead fixed points")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[0, 1]   # peak analog overlap vs fold a_c, colored by lambda
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        c = cmap(i / max(len(results) - 1, 1))
        ax.scatter(fold_ac, r["peak_a"], s=22, color=c, edgecolor="none",
                   alpha=.75, label=f"$\\lambda$={r['lam']:.3f}"
                   if r["lam"] in (0.9, 0.5, 0.37, rc["lam"]) else None)
    ax.plot([0.6, 1.0], [0.6, 1.0], "k--", lw=1, label="peak = fold value")
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xlabel(r"fold state $a_c(\mu)$ (fixed point at death)")
    ax.set_ylabel(r"cycle peak $\max_t a_\mu(t)$")
    ax.set_title("Analog peaks: the cycle never reaches the pure patterns")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[1, 0]   # binary readout quality
    lams_arr = save["lams"]
    mins = [np.nanmin(r["m_peak"]) for r in results]
    meds = [np.nanmedian(r["m_peak"]) for r in results]
    ax.plot(lams_arr, meds, "o-", color="#2b6cb0", label=r"median$_\mu$ peak $m^{sign}_\mu$")
    ax.plot(lams_arr, mins, "s-", color="#c0392b", label=r"min$_\mu$ peak $m^{sign}_\mu$")
    ax.plot(lams_arr, save["frac_exact"], "d-", color="#2f855a",
            label=r"loop-time fraction with sign$(u)$ EXACTLY $=\xi^\mu$")
    ax.axvline(lam_star, color="k", ls=":", lw=1.2, label=r"$\lambda^*$")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("binary readout quality")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"Binarized recall: sign$(u(t))$ vs the stored patterns")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = axes[1, 1]   # speed along the loop at lambda closest to lambda*
    tl, sp, wl = rc["t_loop"], rc["speed"], rc["w_loop"]
    ax.semilogy(tl, sp, lw=0.9, color="#4a5568")
    j91 = np.where(wl == mu_w)[0]
    if len(j91):
        ax.semilogy(tl[j91], sp[j91], lw=1.6, color="#c0392b",
                    label=f"winner window of critical bond {mu_w}")
    ax.set_xlabel("time within one loop"); ax.set_ylabel(r"speed $|\dot a|$")
    ax.set_title(f"SNIC bottleneck at $\\lambda={rc['lam']:.3f}$: "
                 f"the cycle creeps past the ghost of bond {mu_w}")
    ax.legend(fontsize=9); ax.grid(alpha=.3, which="both")

    fig.suptitle("E29 -- the cycle just above $\\lambda^*$ passes by the DEAD "
                 "FIXED-POINT MIXTURES, while its binarized readout still "
                 "retrieves the patterns exactly ($N=2000$, seed 42)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(BASE, "figures", "figAA_cycle_vs_ghosts.png")
    fig.savefig(fp, dpi=175, bbox_inches="tight")
    print(f"\n[E29] figure -> {fp}")


if __name__ == "__main__":
    _experiment_main()
