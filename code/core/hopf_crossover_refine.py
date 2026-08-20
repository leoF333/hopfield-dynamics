"""
Refined delay-induced Hopf analysis near the fold (alpha=0.07, large tau).

Fixes the sampling defect of the first pass: u* is solved at EXACT target lambda
values by Woodbury Newton warm-started from the nearest stable branch point (the
validated fig2 recipe), giving a fine distinct-lambda grid near the fold. At each
(lambda, tau) we record the rightmost REAL root and the rightmost COMPLEX pair
(exact, polished on T_P), then locate lambda_Hopf (complex upcross) and omega_c.

Outputs: printed table + results/stability/hopf_crossover_a0.07.npz + .png
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
from robust_branch import trace_branch, woodbury_newton, eigmax_M
from reduced_spectrum import physical_roots

N, ALPHA, BETA, T0, SEED = 2000, 0.07, 20.0, 1.0, 42
TAUS = [55.0, 70.0, 100.0, 130.0]
IM_TOL = 1e-3
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "stability")


def main():
    xi, xis = make_patterns(N, round(ALPHA * N), SEED)
    coup = Couplings(xi, xis); coup._use_np = True
    cfg = dict(lam_min=0.0, lam_max=0.6, ds=0.01, n_overlaps=5)
    br, fold, st = trace_branch(coup, BETA, cfg)
    i_pk = int(np.argmax(br["lam"]))
    lam_up, u_up = br["lam"][:i_pk + 1], br["u_star"][:i_pk + 1]
    print(f"alpha={ALPHA} N={N} fold={fold:.4f} ({st})", flush=True)

    # exact fine lambda grid near the fold (warm-started Woodbury Newton)
    lam_targets = np.concatenate([np.linspace(0.955 * fold, 0.998 * fold, 7),
                                  [fold]])
    states = []
    for lt in lam_targets:
        j = int(np.argmin(np.abs(lam_up - lt)))
        u, ok = woodbury_newton(coup, u_up[j], float(lt), BETA)
        em = eigmax_M(coup, u, float(lt), BETA)
        m1 = float(xi[0] @ np.tanh(BETA * u) / N)
        if ok and m1 > 0.9:
            states.append((float(lt), u, em))
    print(f"{len(states)} exact states on the branch "
          f"(lam: {states[0][0]:.4f} .. {states[-1][0]:.4f})", flush=True)

    data = {}
    for tau in TAUS:
        M = 64 if tau <= 100 else 80
        rows = []
        t0w = time.time()
        for lam, u, em in states:
            ph, _, _ = physical_roots(coup, u, BETA, lam, tau, T0,
                                      M=M, n_cand=60, k=12, polish=True)
            isc = np.abs(ph.imag) >= IM_TOL
            rr = float(ph.real[~isc].max()) if np.any(~isc) else np.nan
            rc, om = np.nan, np.nan
            if np.any(isc):
                kk = int(np.argmax(np.where(isc, ph.real, -np.inf)))
                rc, om = float(ph.real[kk]), float(abs(ph.imag[kk]))
            rows.append((lam, em, rr, rc, om))
            print(f"  tau={tau:>5.0f} lam={lam:.4f} eigM={em:+.4f} "
                  f"re_real={rr if not np.isnan(rr) else float('nan'):+.4f} "
                  f"re_cplx={rc:+.5f} omega={om:.4f}", flush=True)
        # locate the complex upcross
        lam_h, om_h = None, None
        for i in range(1, len(rows)):
            r0, r1 = rows[i - 1], rows[i]
            if np.isfinite(r0[3]) and np.isfinite(r1[3]) and r0[3] < 0 <= r1[3]:
                t = -r0[3] / (r1[3] - r0[3])
                lam_h = r0[0] + t * (r1[0] - r0[0])
                om_h = r0[4] + t * (r1[4] - r0[4])
                break
        if lam_h is not None:
            print(f"  => tau={tau:.0f}: lam_Hopf={lam_h:.5f} "
                  f"(fold-lam_Hopf={fold-lam_h:+.5f})  omega_c={om_h:.4f}  "
                  f"T_Hopf=2pi/w={2*np.pi/om_h:.1f}  T/tau={2*np.pi/om_h/tau:.2f}  "
                  f"({time.time()-t0w:.0f}s)", flush=True)
        else:
            print(f"  => tau={tau:.0f}: no complex upcross on the sampled arm "
                  f"({time.time()-t0w:.0f}s)", flush=True)
        data[tau] = dict(rows=np.array(rows), lam_h=lam_h, om_h=om_h)

    # ---- figure: re_real & re_cplx vs lambda per tau + omega_c(tau) --------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cols = plt.cm.plasma(np.linspace(0.1, 0.8, len(TAUS)))
    ax = axes[0]
    for tau, c in zip(TAUS, cols):
        R = data[tau]["rows"]
        ax.plot(R[:, 0], R[:, 3], "s-", color=c, ms=5, label=f"cplx, $\\tau={tau:.0f}$")
    R0 = data[TAUS[0]]["rows"]
    ax.plot(R0[:, 0], R0[:, 1], "k--o", ms=4, lw=1, label="eig_max(M) (fold mode)")
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\mathrm{Re}\,z$")
    ax.set_title(f"Rightmost complex pair vs fold mode near the fold "
                 f"($\\alpha={ALPHA}$)")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    ax = axes[1]
    ts = [t for t in TAUS if data[t]["om_h"]]
    if ts:
        oms = [data[t]["om_h"] for t in ts]
        ax.plot(ts, [2*np.pi/o for o in oms], "o-", color="crimson", ms=7,
                label=r"$T_{\rm Hopf}=2\pi/\omega_c$")
        ax.plot(ts, [3*t for t in ts], "k:", label=r"$3\tau$")
        ax.plot(ts, [2*t for t in ts], "k--", lw=.8, label=r"$2\tau$")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("period")
    ax.set_title("Emergent Hopf period vs delay")
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)
    fig.tight_layout()
    fp = os.path.join(OUT, "hopf_crossover_a0.07.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    np.savez(os.path.join(OUT, "hopf_crossover_a0.07.npz"),
             taus=TAUS, fold=fold,
             lam_h=[data[t]["lam_h"] or np.nan for t in TAUS],
             om_h=[data[t]["om_h"] or np.nan for t in TAUS],
             **{f"rows_tau{int(t)}": data[t]["rows"] for t in TAUS})
    print(f"\n[hopf refine] -> {fp}", flush=True)


if __name__ == "__main__":
    main()
