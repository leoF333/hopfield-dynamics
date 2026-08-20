"""
ET1' summary figure (N=2000 delta-sweep GATE-DYN on 3 sentinels).

Panels (dpi 170, English labels):
  (a) d_G(edge object, arclength saddle) vs delta, per sentinel -- the
      discriminating GATE-DYN criterion (small = edge IS the necklace saddle);
  (b) W^u landing map vs delta: for each admissible cell, does the '+' branch
      reach the adjacent node (green) or miss it (red)?
  (c) an example saddle continuation (eigmax_M vs lambda) for one cell;
  (d) GATE-DYN / GATE-BRANCH scoreboard table over all cells.
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
    NumpyLowRankCouplings, make_iid_patterns, gram_matrix,
    solve_memory_node, approach_memory_fold, continue_node_fold_saddle)
from robust_branch import eigmax_M

DEFAULT_FIG = (
    REPO_DIR / "results" / "2_cycle_rappel_snic" / "figures" / "E34"
    / "figE34_ET1prime.png")
COLORS = {28: "#1f77b4", 0: "#d62728", 91: "#2ca02c"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args(argv)

    data = dict(np.load(args.results, allow_pickle=True))
    summ = json.loads(args.summary.read_text())
    N = int(data["N"]); P = int(data["P"]); seed = int(data["seed"])
    beta = float(data["beta"]); tau = float(data["tau"]); t0 = float(data["t0"])
    recs = [r for r in summ["records"] if "mu" in r]
    links = sorted(set(int(r["mu"]) for r in recs))
    deltas = sorted(set(float(r["delta"]) for r in recs))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) d_G(edge, arclength saddle) vs delta -----------------------------
    for mu in links:
        xs, ys = [], []
        for r in sorted((r for r in recs if int(r["mu"]) == mu),
                        key=lambda r: r["delta"]):
            if r.get("edge_vs_arclength_dG") is not None and np.isfinite(
                    r.get("edge_vs_arclength_dG", np.nan)):
                xs.append(r["delta"]); ys.append(r["edge_vs_arclength_dG"])
        if xs:
            ax_a.semilogy(xs, ys, "o-", color=COLORS.get(mu, "k"),
                          label=f"mu={mu}")
    ax_a.axhline(1e-4, ls="--", color="gray", lw=1,
                 label="GATE-DYN threshold 1e-4")
    ax_a.set_xlabel(r"$\delta = \lambda_c(\mu) - \lambda$")
    ax_a.set_ylabel(r"$d_G$(edge object, arclength saddle)")
    ax_a.set_title("(a) Edge object vs necklace saddle (discriminant)")
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3, which="both")

    # (b) W^u landing map vs delta -----------------------------------------
    ymap = {mu: i for i, mu in enumerate(links)}
    for r in recs:
        mu = int(r["mu"]); d = float(r["delta"])
        y = ymap[mu]
        adm = r.get("basin_admissible")
        if not adm:
            ax_b.scatter([d], [y], marker="s", s=90, facecolors="none",
                         edgecolors="0.5")
            ax_b.annotate("inadm.", (d, y), fontsize=6, ha="center",
                          va="bottom", color="0.5")
            continue
        adj = r.get("wu_adjacent")
        mn = r.get("wu_minus", {}).get("omega")
        pl = r.get("wu_plus", {}).get("omega")
        cc = "#2ca02c" if adj else "#d62728"
        ax_b.scatter([d], [y], marker="o", s=110, color=cc)
        ax_b.annotate(f"-:{mn} +:{pl}", (d, y), fontsize=6, ha="center",
                      va="bottom")
    ax_b.set_yticks(list(ymap.values()))
    ax_b.set_yticklabels([f"mu={mu}" for mu in links])
    ax_b.set_xlabel(r"$\delta$")
    ax_b.set_title("(b) W^u landing: green=reaches adjacent node, red=misses")
    ax_b.set_xlim(min(deltas) - 0.004, max(deltas) + 0.004)
    ax_b.grid(alpha=0.3)

    # (c) example continuation --------------------------------------------
    ex = None
    for r in recs:
        if r.get("basin_admissible") and r.get("a_saddle") is not None:
            ex = r; break
    ex = ex or next((r for r in recs if r.get("basin_admissible")), None)
    if ex is not None:
        mu = int(ex["mu"]); d = float(ex["delta"])
        lam = float(ex["lam_local"]); lam_c = float(data["lam_c_links"][links.index(mu)]) \
            if "lam_c_links" in data else float(ex["lam_local"]) + d
        xi, xis = make_iid_patterns(N, P, seed)
        coup = NumpyLowRankCouplings(xi, xis)
        u_node, ok, _ = solve_memory_node(coup, mu, lam, beta)
        if ok:
            near_node, near_lam, _ = approach_memory_fold(
                coup, mu, beta, u_node, lam, lam_limit=lam_c - 4e-3, eig_stop=1e9)
            pair = continue_node_fold_saddle(
                coup, mu, beta, u_node, lam, near_node, near_lam)
            params = pair.trace.parameters
            eigs = np.array([eigmax_M(coup, s, p, beta)
                             for s, p in zip(pair.trace.states, params)])
            ax_c.plot(params, eigs, "-", color=COLORS.get(mu, "k"), lw=1.3)
            ax_c.axvline(lam_c, ls="--", color="k", lw=1, label=r"$\lambda_c$")
            ax_c.plot([pair.fold_parameter], [0.0], "v", color="red", ms=10)
            turns = ex.get("turns_up", "-")
            ax_c.set_title(f"(c) Example continuation mu={mu}, delta={d}: "
                           f"{turns} turns")
    ax_c.axhline(0.0, color="k", lw=0.8)
    ax_c.set_xlabel(r"$\lambda$")
    ax_c.set_ylabel(r"eig$_{\max} M(\lambda)$")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3)

    # (d) scoreboard -------------------------------------------------------
    ax_d.axis("off")
    header = (f"{'mu':>4} {'delta':>6} {'adm':>4} {'edge=arc':>9} "
              f"{'idx':>3} {'W^u adj':>7} {'DYN':>4} {'BR':>3} {'turns':>5}")
    lines = [header, "-" * len(header)]
    for r in sorted(recs, key=lambda r: (int(r["mu"]), r["delta"])):
        mu = int(r["mu"]); d = r["delta"]
        adm = "Y" if r.get("basin_admissible") else "n"
        dga = r.get("edge_vs_arclength_dG")
        dgas = f"{dga:.1e}" if isinstance(dga, (int, float)) and np.isfinite(dga or np.nan) else "-"
        idx = r.get("n_unstable_saddle", "-")
        adj = "Y" if r.get("wu_adjacent") else ("n" if adm == "Y" else "-")
        dyn = "PASS" if r.get("gate_dyn") else "fail"
        br = "Y" if r.get("gate_branch") else "n"
        tu = r.get("turns_up", "-")
        lines.append(f"{mu:>4} {d:>6.3f} {adm:>4} {dgas:>9} {str(idx):>3} "
                     f"{adj:>7} {dyn:>4} {br:>3} {str(tu):>5}")
    n_dyn = sum(1 for r in recs if r.get("gate_dyn"))
    n_adm = sum(1 for r in recs if r.get("basin_admissible"))
    lines.append("-" * len(header))
    lines.append(f"admissible {n_adm}/{len(recs)}   GATE-DYN {n_dyn}/{len(recs)}")
    ax_d.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=9,
              va="top", transform=ax_d.transAxes)
    ax_d.set_title("(d) GATE-DYN / GATE-BRANCH scoreboard")

    fig.suptitle(
        f"E34-ET1' (N={N}, P={P}, seed={seed}, beta={beta:g}): "
        f"near-fold GATE-DYN vs delta", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
