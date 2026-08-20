"""
Hopf margin vs tau: distance of the rightmost COMPLEX root to the imaginary axis,
evaluated at lambda_c (fold-adjacent sample), as a function of tau, per alpha.

Parses the '[max re_cplx=...]' diagnostics from lambda_c_grid_v2 run logs (the
margin equals the value at the fold since complex roots rise monotonically toward
it). Combines several logs (e.g. the tau<=15 grid and the large-tau extension).

Usage: python plot_hopf_margin.py LOG1 [LOG2 ...] [--out PATH]
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RE_ALPHA = re.compile(r"\[alpha=([0-9.]+)\] branch")
RE_TAU = re.compile(
    r"tau=\s*([0-9.]+): lam_c=([0-9.]+)\s+(.*?)\s+"
    r"(?:omega_c=([0-9.]+)\s+)?\[max re_cplx=([+\-0-9.inf]+)\]")


def parse(paths):
    rows = []
    for p in paths:
        alpha = None
        for line in open(p, errors="ignore"):
            m = RE_ALPHA.search(line)
            if m:
                alpha = float(m.group(1)); continue
            m = RE_TAU.search(line)
            if m and alpha is not None:
                margin = m.group(5)
                rows.append(dict(
                    alpha=alpha, tau=float(m.group(1)), lam_c=float(m.group(2)),
                    nature=("Hopf" if "Hopf" in m.group(3) else "fold"),
                    omega=float(m.group(4)) if m.group(4) else np.nan,
                    margin=(np.nan if "inf" in margin else float(margin))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = parse(a.logs)
    if not rows:
        sys.exit("no data parsed")

    # dedupe (alpha, tau) keeping the last occurrence
    seen = {}
    for r in rows:
        seen[(r["alpha"], r["tau"])] = r
    rows = sorted(seen.values(), key=lambda r: (r["alpha"], r["tau"]))

    alphas = sorted(set(r["alpha"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cols = plt.cm.viridis(np.linspace(0.1, 0.8, len(alphas)))
    for al, c in zip(alphas, cols):
        rs = [r for r in rows if r["alpha"] == al and np.isfinite(r["margin"])]
        taus = [r["tau"] for r in rs]
        marg = [-r["margin"] for r in rs]           # distance to axis = |Re|
        ax1.plot(taus, marg, "o-", color=c, ms=6, label=f"$\\alpha={al}$")
        ax2.semilogy(taus, marg, "o-", color=c, ms=6, label=f"$\\alpha={al}$")
        for r in rs:
            if r["nature"] == "Hopf":
                for ax in (ax1, ax2):
                    ax.plot(r["tau"], -r["margin"], "^", color="red", ms=12,
                            mec="k", zorder=5)
    for ax, ttl in ((ax1, "linear"), (ax2, "log")):
        ax.axhline(0 if ax is ax1 else 1e-3, color="k", lw=.8, ls=":")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$|\mathrm{Re}\,z_{\rm cplx}^{\rm rightmost}|$ at $\lambda_c$")
        ax.set_title(f"Hopf margin vs delay ({ttl})")
        ax.grid(True, alpha=.3); ax.legend(fontsize=9)
    fig.suptitle("Distance of the rightmost complex root to the imaginary axis "
                 "at $\\lambda_c$ — crossing 0 = delay-induced Hopf", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = a.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "stability", "hopf_margin_vs_tau.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[hopf margin] -> {out}")
    print(f"\n{'alpha':>6} {'tau':>6} {'margin |Re|':>12} {'nature':>8}")
    for r in rows:
        mg = f"{-r['margin']:.4f}" if np.isfinite(r["margin"]) else "  n/a"
        print(f"{r['alpha']:>6.2f} {r['tau']:>6} {mg:>12} {r['nature']:>8}")


if __name__ == "__main__":
    main()
