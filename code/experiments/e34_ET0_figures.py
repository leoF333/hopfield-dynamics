"""
Four-panel summary figure for gate E34-ET0 (edge-tracking, N=400 pilot).

Panels (English labels, dpi 170):
  (a) bisection convergence: Gram-metric bracket resolution d_G vs iteration,
      one curve per sentinel;
  (b) boundary-object spectrum in the complex plane, the single unstable root
      highlighted, with the a-priori spectral disk boundary;
  (c) lambda continuation of the tracked saddle (eigmax_M vs lambda) with the
      table fold marked and turning points annotated;
  (d) W^u omega-limit map: saddle_mu -> {omega_minus, omega_plus} against the
      pattern ring.

Everything is recomputed from the stored boundary states a_saddle / a_node, so
the figure is self-contained given the E34_ET0_results_*.npz.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from e34_lib import (
    NumpyLowRankCouplings, make_iid_patterns, gram_matrix, gram_distance,
    reduced_field_coefficients, solve_memory_node, approach_memory_fold,
    continue_node_fold_saddle, characteristic_spectral_bound)
from reduced_spectrum import GJ_GK, beyn_roots, sigmin_TP
from robust_branch import eigmax_M

DEFAULT_FIG = (
    REPO_DIR / "results" / "2_cycle_rappel_snic" / "figures" / "E34"
    / "figE34_ET0_edge_tracking.png")

COLORS = {14: "#1f77b4", 8: "#d62728", 2: "#2ca02c"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args(argv)

    data = dict(np.load(args.results, allow_pickle=True))
    summary = json.loads(args.summary.read_text())
    N = int(data["N"]); P = int(data["P"]); seed = int(data["seed"])
    beta = float(data["beta"]); tau = float(data["tau"]); t0 = float(data["t0"])
    links = [int(x) for x in np.atleast_1d(data["links"])]
    recs = {int(r["mu"]): r for r in summary["records"] if "mu" in r}

    xi, xis = make_iid_patterns(N, P, seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ---- (a) bisection convergence in d_G ---------------------------------
    for mu in links:
        key = f"mu{mu}_edge_history"
        if key not in data:
            continue
        hist = np.asarray(data[key], float)          # rows (s_mid, label)
        # reconstruct running bracket width from the bisection midpoints
        s_lo, s_hi = 0.0, 1.0
        widths = []
        for s_mid, label in hist[2:]:
            if label > 0.5:
                s_lo = s_mid
            else:
                s_hi = s_mid
            widths.append(s_hi - s_lo)
        widths = np.asarray(widths)
        a_node = data[f"mu{mu}_a_node"]
        succ = (mu + 1) % P
        u_succ, ok, _ = solve_memory_node(coup, succ, float(data[f"mu{mu}_lam_local"]), beta)
        if ok:
            a_far = reduced_field_coefficients(coup, u_succ, Q)
        else:
            a_far = reduced_field_coefficients(coup, 2.0 * xi[succ], Q)
        seg_len = float(gram_distance(a_node, a_far, Q))
        d_g = widths * seg_len
        ax_a.semilogy(np.arange(1, len(d_g) + 1), d_g, "o-", ms=3,
                      color=COLORS.get(mu, "k"), label=f"mu={mu}")
    ax_a.axhline(1e-8, ls=":", color="gray", lw=1)
    ax_a.set_xlabel("bisection iteration")
    ax_a.set_ylabel(r"bracket resolution $d_G$")
    ax_a.set_title("(a) Edge-tracking bisection convergence")
    ax_a.legend(fontsize=9)
    ax_a.grid(alpha=0.3)

    # ---- (b) boundary-object spectrum -------------------------------------
    for mu in links:
        key = f"mu{mu}_a_saddle"
        if key not in data:
            continue
        lam = float(data[f"mu{mu}_lam_local"])
        u_s = xi.T @ data[key]
        GJ, GK = GJ_GK(coup, u_s, beta, lam)
        rho = characteristic_spectral_bound(GJ, GK, lam)
        ev, _ = beyn_roots(GJ, GK, t0, tau, lam,
                           center=-1.0 / t0, R=rho / t0 + 0.15, Nq=400, L=90)
        # keep only genuine roots (Beyn returns spurious eigenvalues too)
        genuine = np.array([
            z for z in ev
            if sigmin_TP(z, GJ, GK, t0, tau, lam)
            / max(abs(t0 * z + 1.0), 1.0) < 5e-3])
        col = COLORS.get(mu, "k")
        ax_b.scatter(genuine.real, genuine.imag, s=16, color=col, alpha=0.6,
                     label=f"mu={mu}")
        z_u = float(data[f"mu{mu}_z_u"])
        ax_b.scatter([z_u], [0.0], s=140, facecolors="none",
                     edgecolors=col, linewidths=2.2)
    ax_b.axvline(0.0, color="k", lw=1)
    ax_b.set_xlabel(r"Re $z$")
    ax_b.set_ylabel(r"Im $z$")
    ax_b.set_title("(b) Boundary-object spectrum (circled: certified unstable root)")
    ax_b.legend(fontsize=9)
    ax_b.grid(alpha=0.3)
    ax_b.set_xlim(-1.6, 0.9)

    # ---- (c) lambda-continuation of the tracked saddle --------------------
    for mu in links:
        if f"mu{mu}_a_node" not in data:
            continue
        lam = float(data[f"mu{mu}_lam_local"])
        lam_c = float(data[f"mu{mu}_lam_c_table"])
        u_node, ok, _ = solve_memory_node(coup, mu, lam, beta)
        if not ok:
            continue
        near_node, near_lam, _ = approach_memory_fold(
            coup, mu, beta, u_node, lam, lam_limit=lam_c - 4e-3, eig_stop=1e9)
        pair = continue_node_fold_saddle(
            coup, mu, beta, u_node, lam, near_node, near_lam)
        params = pair.trace.parameters
        eigs = np.array([
            eigmax_M(coup, s, p, beta)
            for s, p in zip(pair.trace.states, params)])
        col = COLORS.get(mu, "k")
        ax_c.plot(params, eigs, "-", color=col, lw=1.3, label=f"mu={mu}")
        ax_c.axvline(lam_c, ls="--", color=col, lw=1, alpha=0.7)
        ax_c.plot([pair.fold_parameter], [0.0], "v", color=col, ms=9)
    ax_c.axhline(0.0, color="k", lw=0.8)
    ax_c.set_xlabel(r"$\lambda$")
    ax_c.set_ylabel(r"eig$_{\max} M(\lambda)$")
    ax_c.set_title("(c) Saddle continuation to the local fold "
                   "(dashed: table $\\lambda_c$, v: fold)")
    ax_c.legend(fontsize=9)
    ax_c.grid(alpha=0.3)

    # ---- (d) W^u omega-limit map ------------------------------------------
    ang = 2 * np.pi * np.arange(P) / P
    ring_x, ring_y = np.cos(ang), np.sin(ang)
    ax_d.scatter(ring_x, ring_y, s=22, color="0.6", zorder=1)
    for k in range(P):
        ax_d.annotate(str(k), (ring_x[k], ring_y[k]), fontsize=7,
                      ha="center", va="center",
                      xytext=(ring_x[k] * 1.12, ring_y[k] * 1.12))
    for mu in links:
        rec = recs.get(mu, {})
        om_m = rec.get("wu_minus", {}).get("omega")
        om_p = rec.get("wu_plus", {}).get("omega")
        col = COLORS.get(mu, "k")
        for om, style in ((om_m, "-"), (om_p, "--")):
            if isinstance(om, int) and 0 <= om < P:
                ax_d.annotate(
                    "", xy=(ring_x[om], ring_y[om]),
                    xytext=(ring_x[mu], ring_y[mu]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8,
                                    linestyle=style, alpha=0.85), zorder=2)
        ax_d.scatter([ring_x[mu]], [ring_y[mu]], s=70, marker="x",
                     color=col, zorder=3, label=f"saddle mu={mu}")
    ax_d.set_aspect("equal")
    ax_d.axis("off")
    ax_d.set_title("(d) W^u omega-limits on the ring "
                   "('-' solid, '+' dashed)")
    ax_d.legend(fontsize=8, loc="lower right")

    gate = str(data["gate"]) if "gate" in data else summary.get("gate", "?")
    fig.suptitle(
        f"E34-ET0 edge-tracking gate (N={N}, P={P}, seed={seed}, "
        f"beta={beta:g}, tau={tau:g})  ->  gate {gate}",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
