"""
Descending-lambda Floquet scan of the pacemaker cycle (CYCLE_worklog §5.1).

For each lambda (descending from deep in the pacemaker phase):
  1. build the cycle from the exact P-dim reduction (memory IC);
  2. compute the leading Floquet multipliers (monodromy + Arnoldi);
  3. record T1, closure, mu_0 validation, leading nontrivial exponent Re z*.
Stops after the cycle fails to form at two consecutive lambda (simulation-side
boundary lambda_down). Outputs per-lambda rows, a summary figure and npz.

Detection targets (worklog §5.3):
  |mu*| -> 1 with arg 0 / pi / phi  => cyclic fold / period-doubling / Neimark-Sacker
  cycle formation fails while |mu*| < 1  => suspect fold-of-cycles or crisis
  (then hysteresis protocol, separate script).
"""
from __future__ import annotations

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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from couplings import make_patterns
from pacemaker_cycle import build_cycle
from floquet_monodromy import Monodromy

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lams", type=float, nargs="+",
                   default=[0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65,
                            0.60, 0.55, 0.50, 0.45, 0.40, 0.35])
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    P = round(args.alpha * args.N)
    xi, _ = make_patterns(args.N, P, args.seed)
    print(f"pacemaker scan: N={args.N} P={P} tau={args.tau} beta={args.beta} "
          f"lams={args.lams}", flush=True)

    rows = []; fails = 0
    t_all = time.time()
    for lam in args.lams:
        t0w = time.time()
        cyc = build_cycle(xi, args.beta, lam, args.tau, dt=args.dt,
                          verbose=False)
        if not cyc.get("ok"):
            # retry once with longer settling (critical slowing-down near edge)
            cyc = build_cycle(xi, args.beta, lam, args.tau, dt=args.dt,
                              verbose=False, settle_turns=2.5)
        if not cyc.get("ok"):
            fails += 1
            print(f"  lam={lam:.2f}: NO pacemaker from memory IC "
                  f"({cyc.get('reason','')[:60]})  [{time.time()-t0w:.0f}s]",
                  flush=True)
            rows.append(dict(lam=lam, ok=False))
            if fails >= 2:
                print("  two consecutive failures -> lambda_down bracketed; stop.",
                      flush=True)
                break
            continue
        fails = 0
        mon = Monodromy(xi, cyc, verbose=False)
        mu, vec = mon.leading_multipliers(k=args.k, tol=1e-6, maxiter=200)
        w = mon.phase_mode(cyc)
        align = abs(np.vdot(vec[:, 0] / np.linalg.norm(vec[:, 0]), w))
        mu0_err = abs(mu[0] - 1.0)
        nt = mu[1:]
        i_star = int(np.argmax(np.abs(nt))) if len(nt) else 0
        mu_star = nt[i_star] if len(nt) else 0.0
        rez = float(np.log(max(abs(mu_star), 1e-300)) / cyc["T"])
        rows.append(dict(lam=lam, ok=True, T=cyc["T"], T1=cyc["T1_mean"],
                         T1_std=cyc["T1_std"], closure=cyc["closure"],
                         mu=mu, mu0_err=mu0_err, align=align,
                         mu_star=mu_star, rez=rez))
        print(f"  lam={lam:.2f}: T1={cyc['T1_mean']:.3f}  "
              f"closure={cyc['closure']:.1e}  |mu0-1|={mu0_err:.1e} "
              f"align={align:.3f}  |mu*|={abs(mu_star):.4g} "
              f"arg={np.angle(mu_star):+.2f}  Re z*={rez:+.4f}  "
              f"[{time.time()-t0w:.0f}s]", flush=True)

    print(f"\nTotal wall time: {(time.time()-t_all)/60:.0f} min", flush=True)

    # ---- summary figure ---------------------------------------------------
    okr = [r for r in rows if r.get("ok")]
    if okr:
        lams = [r["lam"] for r in okr]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        ax = axes[0]
        ax.plot(lams, [r["rez"] for r in okr], "o-", color="navy", ms=6)
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\mathrm{Re}\,z^*$")
        ax.set_title("Leading nontrivial Floquet exponent")
        ax.grid(True, alpha=.3); ax.invert_xaxis()
        ax = axes[1]
        ax.plot(lams, [abs(r["mu_star"]) for r in okr], "s-", color="crimson", ms=6)
        ax.axhline(1, color="k", lw=.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$|\mu^*|$")
        ax.set_title("Leading nontrivial multiplier")
        ax.grid(True, alpha=.3); ax.invert_xaxis()
        ax = axes[2]
        ax.errorbar(lams, [r["T1"] for r in okr],
                    yerr=[r["T1_std"] for r in okr], fmt="o-", color="teal", ms=6)
        ax.axhline(args.tau + 1, color="k", ls=":", label=r"$\tau+1$")
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$T_1$")
        ax.set_title("Pacemaker step period")
        ax.legend(); ax.grid(True, alpha=.3); ax.invert_xaxis()
        fig.suptitle(f"Pacemaker cycle Floquet scan  ($N={args.N},\\ "
                     f"\\alpha={args.alpha},\\ \\tau={args.tau}$)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fp = os.path.join(OUT, f"floquet_scan_N{args.N}_tau{args.tau}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"[scan] figure -> {fp}", flush=True)

    np.savez(os.path.join(OUT, f"floquet_scan_N{args.N}_tau{args.tau}.npz"),
             lam=[r["lam"] for r in rows],
             ok=[r.get("ok", False) for r in rows],
             T1=[r.get("T1", np.nan) for r in rows],
             closure=[r.get("closure", np.nan) for r in rows],
             mu0_err=[r.get("mu0_err", np.nan) for r in rows],
             mu_star=[r.get("mu_star", np.nan) for r in rows],
             rez=[r.get("rez", np.nan) for r in rows])
    print("[scan] npz saved", flush=True)


if __name__ == "__main__":
    main()
