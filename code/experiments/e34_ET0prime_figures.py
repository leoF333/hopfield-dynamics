"""
ET0' amendment figure (mu=2 lower-lambda edge-tracking + upward continuation).

Panels (dpi 170, English labels):
  (a) mu=2 edge-tracking bisection convergence (d_G vs iteration) at lam';
  (b) boundary-object spectrum with the single certified unstable root circled;
  (c) upward lambda-continuation of the mu=2 memory branch from lam' to the
      local fold lam_c(2), turning points annotated, table fold marked;
  (d) GATE-DYN vs GATE-BRANCH scoreboard over the three sentinels (14, 8, 2).

Reads the ET0' npz/summary and (for the scoreboard) the ET0 production summary.
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

import e34b_edge_tracking as ET
from e34_lib import (
    NumpyLowRankCouplings, make_iid_patterns, gram_matrix, gram_distance,
    reduced_field_coefficients, solve_memory_node, approach_memory_fold,
    continue_node_fold_saddle, characteristic_spectral_bound,
    extract_null_mode, exponential_history)
from reduced_spectrum import GJ_GK, beyn_roots, sigmin_TP
from robust_branch import eigmax_M, woodbury_newton
from cycle_reduced import ReducedDDE

DEFAULT_FIG = (
    REPO_DIR / "results" / "2_cycle_rappel_snic" / "figures" / "E34"
    / "figE34_ET0prime.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--et0-summary", type=Path, required=True,
                        help="ET0 production summary for the scoreboard.")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args(argv)

    data = dict(np.load(args.results, allow_pickle=True))
    summ = json.loads(args.summary.read_text())
    et0 = json.loads(args.et0_summary.read_text())
    N = int(data["N"]); P = int(data["P"]); seed = int(data["seed"])
    beta = float(data["beta"]); tau = float(data["tau"]); t0 = float(data["t0"])
    rec2 = next(r for r in summ["records"] if int(r.get("mu", -1)) == 2)

    xi, xis = make_iid_patterns(N, P, seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    mu = 2
    col = "#2ca02c"
    lam_prime = float(data["mu2_lam_local"])
    lam_c2 = float(data["mu2_lam_c_table"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) bisection convergence -------------------------------------------
    hist = np.asarray(data["mu2_edge_history"], float)
    s_lo, s_hi = 0.0, 1.0
    widths = []
    for s_mid, label in hist[2:]:
        if label > 0.5:
            s_lo = s_mid
        else:
            s_hi = s_mid
        widths.append(s_hi - s_lo)
    a_node = data["mu2_a_node"]
    u3, ok, _ = solve_memory_node(coup, 3, lam_prime, beta)
    a_far = reduced_field_coefficients(coup, u3, Q)
    seg_len = float(gram_distance(a_node, a_far, Q))
    d_g = np.asarray(widths) * seg_len
    ax_a.semilogy(np.arange(1, len(d_g) + 1), d_g, "o-", ms=3, color=col)
    ax_a.axhline(1e-8, ls=":", color="gray", lw=1)
    ax_a.set_xlabel("bisection iteration")
    ax_a.set_ylabel(r"bracket resolution $d_G$")
    ax_a.set_title(f"(a) mu=2 edge-tracking at lam'={lam_prime:.4f} "
                   f"(node 2 vs node 3)")
    ax_a.grid(alpha=0.3)

    # (b) spectrum ---------------------------------------------------------
    if "mu2_a_saddle" in data:
        u_s = xi.T @ data["mu2_a_saddle"]
        GJ, GK = GJ_GK(coup, u_s, beta, lam_prime)
        rho = characteristic_spectral_bound(GJ, GK, lam_prime)
        ev, _ = beyn_roots(GJ, GK, t0, tau, lam_prime,
                           center=-1.0 / t0, R=rho / t0 + 0.15, Nq=400, L=90)
        genuine = np.array([
            z for z in ev
            if sigmin_TP(z, GJ, GK, t0, tau, lam_prime)
            / max(abs(t0 * z + 1.0), 1.0) < 5e-3])
        ax_b.scatter(genuine.real, genuine.imag, s=16, color=col, alpha=0.6)
        z_u = float(data["mu2_z_u"])
        ax_b.scatter([z_u], [0.0], s=150, facecolors="none",
                     edgecolors=col, linewidths=2.2,
                     label=f"z_u={z_u:.4f}")
        ax_b.legend(fontsize=9)
    ax_b.axvline(0.0, color="k", lw=1)
    ax_b.set_xlabel(r"Re $z$")
    ax_b.set_ylabel(r"Im $z$")
    ax_b.set_title("(b) mu=2 boundary-object spectrum (circled: unstable root)")
    ax_b.grid(alpha=0.3)
    ax_b.set_xlim(-1.6, 0.9)

    # (c) upward continuation to the fold ---------------------------------
    u_node, ok, _ = solve_memory_node(coup, mu, lam_prime, beta)
    near_node, near_lam, _ = approach_memory_fold(
        coup, mu, beta, u_node, lam_prime, lam_limit=lam_c2 - 4e-3,
        eig_stop=1e9)
    pair = continue_node_fold_saddle(
        coup, mu, beta, u_node, lam_prime, near_node, near_lam)
    params = pair.trace.parameters
    eigs = np.array([
        eigmax_M(coup, s, p, beta)
        for s, p in zip(pair.trace.states, params)])
    ax_c.plot(params, eigs, "-", color=col, lw=1.3)
    ax_c.scatter(params, eigs, s=6, color=col, alpha=0.4)
    ax_c.axvline(lam_c2, ls="--", color="k", lw=1, label=r"table $\lambda_c(2)$")
    ax_c.axvline(lam_prime, ls=":", color=col, lw=1, label="lam' (edge-track)")
    ax_c.plot([pair.fold_parameter], [0.0], "v", color="red", ms=11,
              label="fold reached")
    ax_c.axhline(0.0, color="k", lw=0.8)
    ax_c.set_xlabel(r"$\lambda$")
    ax_c.set_ylabel(r"eig$_{\max} M(\lambda)$")
    turns = int(data["mu2_turns_up"]) if "mu2_turns_up" in data else -1
    ax_c.set_title(f"(c) mu=2 upward continuation: fold={pair.fold_parameter:.5f}, "
                   f"{turns} turns")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3)

    # (d) W^u landing map with the -4 (mixture) branch trajectory ----------
    lamp = lam_prime
    a_saddle = data["mu2_a_saddle"]
    u_s = xi.T @ a_saddle
    GJ, GK = GJ_GK(coup, u_s, beta, lamp)
    S = np.roll(np.eye(P), 1, axis=0)
    char_red = lambda z: (t0 * z + 1.0) * np.eye(P) - (1.0 - lamp) * GJ \
        - lamp * np.exp(-z * tau) * (S @ GJ)
    z_u = float(data["mu2_z_u"])
    v = np.asarray(extract_null_mode(char_red, z_u, Q=Q, expect_real=True).vector, float)
    if v[3] < 0:
        v = -v
    sysP = ReducedDDE(xi, beta, lamp, tau, t0)
    arcs = {}
    for sign, name in ((-1.0, "minus"), (1.0, "plus")):
        _, hist, dhist = exponential_history(a_saddle, z_u, v, sign * 1e-3, tau, 0.01, Q=Q)
        out = ET.integrate_to_settle(
            sysP, hist, Q, dt=0.01, tau=tau, t_max=4000.0,
            record_every=50, da_hist0=dhist)
        arcs[name] = out["a"]
    a_node2 = data["mu2_a_node"]
    a_node3 = reduced_field_coefficients(coup, xi.T @ a_node2, Q) if False else \
        reduced_field_coefficients(coup, solve_memory_node(coup, 3, lamp, beta)[0], Q)
    a_mix = arcs["plus"][-1]
    # 2D projection: axes = Gram-orthonormalized {node3 - node2, mix - node2}
    basis0 = a_node3 - a_node2
    basis1 = a_mix - a_node2
    def gdot(x, y):
        return float(x @ Q @ y)
    e0 = basis0 / np.sqrt(gdot(basis0, basis0))
    b1 = basis1 - gdot(basis1, e0) * e0
    e1 = b1 / np.sqrt(gdot(b1, b1))
    def proj(a):
        da = a - a_node2
        return gdot(da, e0), gdot(da, e1)

    def pth(arc):
        xs, ys = zip(*[proj(a) for a in arc])
        return np.array(xs), np.array(ys)
    xm, ym = pth(arcs["minus"])
    xp, yp = pth(arcs["plus"])
    ax_d.plot(xm, ym, "-", color="#1f77b4", lw=1.6, label="W^u '-' -> node 2")
    ax_d.plot(xp, yp, "-", color="#d62728", lw=1.6,
              label="W^u '+' -> mixture (omega=-4)")
    for a, mk, cc, lab in ((a_node2, "o", "k", "node 2"),
                           (a_node3, "s", "0.4", "node 3"),
                           (a_saddle, "x", "green", "edge saddle"),
                           (a_mix, "*", "#d62728", "mixture (14/18/19)")):
        px, py = proj(a)
        ax_d.scatter([px], [py], marker=mk, s=110, color=cc, zorder=5, label=lab)
    ax_d.set_xlabel(r"Gram axis 1 (node3 $-$ node2)")
    ax_d.set_ylabel(r"Gram axis 2")
    ax_d.set_title("(d) mu=2 W^u landing: '+' misses node 3 -> mixture")
    ax_d.legend(fontsize=8, loc="best")
    ax_d.grid(alpha=0.3)

    fig.suptitle(
        f"E34-ET0' amendment (mu=2 lower-lambda, N={N}, seed={seed})",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
