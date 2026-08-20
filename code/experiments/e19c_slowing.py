"""
E19c - Critical-slowing law of the passage time at small tau.

At the tau where the small-delay cycle survives closest to lambda*=0.328 (chosen
from E19b; default tau=2), measure the passage time t_MAX at the WEAKEST bond as
lambda -> lambda* from above. t_MAX = max over bonds of the per-bond dwell time
in one traversal of the ring (E9 method). Fit t_MAX = C / sqrt(lambda - lambda*)
with lambda* FREE, and compare (C, lambda*) with the tau=10 reference
(C=2.3, lambda*=0.3275). If the law fails, characterise the replacement.

Integration is capped at t=2000 (t_MAX diverges near lambda*); non-passage of the
weakest bond within the cap is treated as CENSORED.

Outputs: results/cycle/E19c_slowing.npz + figM_smalltau_snic.png.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from couplings import make_patterns
from cycle_reduced import ReducedDDE

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIGDIR = os.path.join(OUT, "figures")

T_CAP = 2000.0          # integration cap (censor non-passage)


def dt_for(tau):
    if tau <= 0.0:
        return 0.01
    return min(0.01, tau / 25.0)


def bond_dwell_times(tau, lam, dt, t_cap=T_CAP):
    """Seed a travelling cycle, then let it run and record per-bond dwell times
    (time the front spends leading each pattern). Return the vector of dwell times
    for the ring bonds observed, and whether the weakest bond passed (not censored).
    The weakest bond (91) is where the front pins first -> its dwell time is t_MAX.
    """
    sysP = ReducedDDE(xi, BETA, lam, tau, T0)
    # seed from a memory IC + forward bias, warm up
    a0 = np.zeros(P); a0[0] = 0.99; a0[1] = 0.05
    warm = sysP.integrate(a0, max(120.0, 40 * (tau + 1)), dt)
    hist = warm["hist"]
    sol = sysP.integrate(hist, t_cap, dt)
    A = sol["a"]
    lead = np.argmax(A, axis=1)
    chg = np.nonzero(np.diff(lead) != 0)[0] + 1
    if len(chg) < 2:
        # front pinned during warmup already -> fully censored
        return None, True, lead[-1]
    times = chg * dt
    dwell = np.diff(times)              # dwell time preceding each handoff
    bonds = lead[chg[:-1]]             # which pattern the front sat on
    # per-bond max dwell over the run (weakest bond has the largest)
    t_max = float(dwell.max())
    weakest_bond = int(bonds[int(np.argmax(dwell))])
    # censored if the run ended still sitting on a bond much longer than t_max
    tail_dwell = (len(A) - 1) * dt - times[-1]
    censored = tail_dwell > t_max
    if censored:
        t_max = max(t_max, tail_dwell)
    return dict(t_max=t_max, weakest_bond=weakest_bond,
                dwell=dwell, bonds=bonds), censored, lead[-1]


def sweep(tau, lams):
    dt = dt_for(tau)
    rows = []
    for lam in lams:
        t0w = time.time()
        res, censored, lead_final = bond_dwell_times(tau, lam, dt)
        if res is None:
            rows.append(dict(lam=lam, t_max=np.nan, censored=True,
                             weakest=lead_final))
            print(f"  tau={tau} lam={lam:.4f}: PINNED in warmup (censored) "
                  f"lead={lead_final} [{time.time()-t0w:.0f}s]", flush=True)
            continue
        rows.append(dict(lam=lam, t_max=res["t_max"], censored=censored,
                         weakest=res["weakest_bond"]))
        print(f"  tau={tau} lam={lam:.4f}: t_MAX={res['t_max']:.2f} "
              f"{'(CENSORED)' if censored else ''} weakest_bond={res['weakest_bond']} "
              f"[{time.time()-t0w:.0f}s]", flush=True)
    return rows


def fit_sqrt(lams, tmax):
    """Fit t_MAX = C / sqrt(lam - lam_star), lam_star free."""
    def model(l, C, ls):
        return C / np.sqrt(np.maximum(l - ls, 1e-9))
    p0 = [2.3, min(lams) - 0.01]
    try:
        popt, pcov = curve_fit(model, lams, tmax, p0=p0, maxfev=20000,
                               bounds=([0.1, 0.0], [50.0, min(lams) - 1e-4]))
        pred = model(lams, *popt)
        ss_res = np.sum((tmax - pred) ** 2)
        ss_tot = np.sum((tmax - tmax.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-30)
        return popt[0], popt[1], r2
    except Exception as e:
        print("  fit failed:", e)
        return np.nan, np.nan, np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[2.0, 1.0])
    args = ap.parse_args()

    # lambda grid approaching lambda*~0.328 from above (8 points)
    LAMS = np.array([0.360, 0.350, 0.342, 0.336, 0.333, 0.331, 0.330, 0.329])

    print(f"[E19c] N={N} P={P}  lambda -> lambda* from above, taus={args.taus}\n")
    all_rows = {}
    fits = {}
    for tau in args.taus:
        print(f"=== tau={tau} (dt={dt_for(tau)}) ===", flush=True)
        rows = sweep(tau, LAMS)
        all_rows[tau] = rows
        lam_arr = np.array([r["lam"] for r in rows])
        tmax_arr = np.array([r["t_max"] for r in rows])
        cens = np.array([r["censored"] for r in rows])
        # fit on uncensored finite points only
        ok = np.isfinite(tmax_arr) & ~cens
        if ok.sum() >= 4:
            C, ls, r2 = fit_sqrt(lam_arr[ok], tmax_arr[ok])
        else:
            C, ls, r2 = np.nan, np.nan, np.nan
        fits[tau] = (C, ls, r2)
        print(f"  -> fit t_MAX = C/sqrt(lam-lam*):  C={C:.3f}  lam*={ls:.4f}  "
              f"R^2={r2:.4f}  (n_fit={ok.sum()})\n", flush=True)

    # ---- table ----
    print(f"\n{'tau':>6} {'C':>8} {'lambda*':>9} {'R^2':>8}   (tau=10 ref: C=2.3, lam*=0.3275)")
    for tau in args.taus:
        C, ls, r2 = fits[tau]
        print(f"{tau:>6} {C:>8.3f} {ls:>9.4f} {r2:>8.4f}")

    # ---- figure: t_MAX vs (lam - lam*) log-log with sqrt law ----
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(args.taus)))
    for tau, c in zip(args.taus, colors):
        rows = all_rows[tau]
        lam_arr = np.array([r["lam"] for r in rows])
        tmax_arr = np.array([r["t_max"] for r in rows])
        cens = np.array([r["censored"] for r in rows])
        C, ls, r2 = fits[tau]
        x = lam_arr - ls
        finite = np.isfinite(tmax_arr) & (x > 0)
        ax.loglog(x[finite & ~cens], tmax_arr[finite & ~cens], "o", color=c, ms=8,
                  label=rf"$\tau={tau}$: $C={C:.2f}$, $\lambda^*={ls:.4f}$, $R^2={r2:.3f}$")
        if np.any(cens & finite):
            ax.loglog(x[finite & cens], tmax_arr[finite & cens], "x", color=c,
                      ms=9, mew=2)
        if np.isfinite(C):
            xx = np.logspace(np.log10(max(x[finite].min(), 1e-4)),
                             np.log10(x[finite].max()), 50)
            ax.loglog(xx, C / np.sqrt(xx), "-", color=c, lw=1.5, alpha=.7)
    # tau=10 reference law
    xx = np.logspace(-3, -1.2, 50)
    ax.loglog(xx, 2.3 / np.sqrt(xx), "k--", lw=1.3,
              label=r"$\tau=10$ ref: $2.3/\sqrt{\lambda-0.3275}$")
    ax.set_xlabel(r"$\lambda - \lambda^*$")
    ax.set_ylabel(r"passage time $t_{\rm MAX}$ (weakest bond)")
    ax.set_title("E19c - Critical slowing of the front at the weakest bond, small $\\tau$\n"
                 "(x = censored; solid = $C/\\sqrt{\\lambda-\\lambda^*}$ fit)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fp = os.path.join(FIGDIR, "figM_smalltau_snic.png")
    fig.savefig(fp, dpi=175, bbox_inches="tight")
    print(f"\n[E19c] figure -> {fp}")

    np.savez(os.path.join(OUT, "E19c_slowing.npz"),
             taus=np.array(args.taus), lams=LAMS,
             **{f"tmax_tau{t}": np.array([r["t_max"] for r in all_rows[t]])
                for t in args.taus},
             **{f"cens_tau{t}": np.array([r["censored"] for r in all_rows[t]])
                for t in args.taus},
             **{f"weakest_tau{t}": np.array([r["weakest"] for r in all_rows[t]])
                for t in args.taus},
             fit_C=np.array([fits[t][0] for t in args.taus]),
             fit_lamstar=np.array([fits[t][1] for t in args.taus]),
             fit_r2=np.array([fits[t][2] for t in args.taus]))
    print("[E19c] npz saved.")
