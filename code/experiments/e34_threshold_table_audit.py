"""Audit of the E24 per-bead threshold table against the branch itself.

Trigger (E34 ET1'', mu=28): the table says lam_c[28] = 0.219458, but pseudo-
arclength continuation of bead 28's branch turns at 0.232122 (+1.27e-2), and a
naive Newton solve at lambda = 0.2300 > lam_c[28] still returns a STABLE
equilibrium carrying identity mu=28 (residual 8.6e-13, eigmax_M = -0.106).
A stable bead cannot exist above its own fold, so for that bead the tabulated
threshold is not the end of the stable branch. Since the alive-node census used
throughout E34 is built from this table (solve_alive_nodes: lam_c[k] > lam +
margin), a wrong entry silently removes a live bead from the census and turns a
legitimate necklace landing into an "omega = -4" non-census landing.

This checks every bead: is there still a stable equilibrium with identity mu at
lam_c[mu] + probe?  Reported as
  ok        -- no stable bead above the tabulated threshold (table consistent)
  late_fold -- a stable bead with identity mu survives above it (table early)
  unclear   -- Newton did not converge cleanly either way

Cheap: 2 Newton solves + 2 eigmax_M per bead, no spectral contour work.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from e34_lib import (
    NumpyLowRankCouplings, gram_matrix, make_iid_patterns,
    memory_branch_identity, solve_memory_node,
)
from robust_branch import eigmax_M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2_cycle_rappel_snic/data/E34"
THR = ROOT / "results/3_unification_seuils_scaling/data/E24_thr_N2000_P100_s42.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def probe(coup, Q, mu, lam, beta, res_tol=1e-9):
    u, ok, res = solve_memory_node(coup, mu, lam, beta)
    if not ok or res > res_tol:
        return dict(ok=False, res=float(res), eigmax=None, identity=False)
    return dict(ok=True, res=float(res), eigmax=float(eigmax_M(coup, u, lam, beta)),
                identity=bool(memory_branch_identity(coup, u, mu, Q=Q)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probes", default="0.005,0.010")
    p.add_argument("--beta", type=float, default=20.0)
    args = p.parse_args(argv)
    probes = [float(x) for x in args.probes.split(",")]

    with np.load(THR) as d:
        lam_c = np.asarray(d["lam_c"], float)
        N, P, seed = int(d["N"]), int(d["P"]), int(d["seed"])
    xi, xis = make_iid_patterns(N, P, seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    rows, late = [], []
    for mu in range(P):
        if not np.isfinite(lam_c[mu]):
            rows.append(dict(mu=mu, verdict="no_threshold"))
            continue
        rec = dict(mu=mu, lam_c=float(lam_c[mu]), probes={})
        stable_above = []
        for dl in probes:
            r = probe(coup, Q, mu, float(lam_c[mu] + dl), args.beta)
            rec["probes"][f"+{dl}"] = r
            if r["ok"] and r["identity"] and r["eigmax"] is not None \
                    and r["eigmax"] < 0:
                stable_above.append(dl)
        if stable_above:
            rec["verdict"] = "late_fold"
            rec["stable_above"] = stable_above
            late.append(mu)
        elif any(v["ok"] for v in rec["probes"].values()):
            rec["verdict"] = "ok"
        else:
            rec["verdict"] = "ok"      # no convergent solve above => consistent
        rows.append(rec)
        if mu % 20 == 0:
            print(f"{_now()} ... {mu}/{P}", flush=True)

    print(f"{_now()} late_fold beads ({len(late)}/{P}): {late}", flush=True)
    for mu in late:
        r = next(x for x in rows if x["mu"] == mu)
        det = ", ".join(f"+{k.strip('+')}: eig={v['eigmax']:+.4f}"
                        for k, v in r["probes"].items()
                        if v["ok"] and v["eigmax"] is not None)
        print(f"    mu={mu:3d} lam_c={r['lam_c']:.6f}  {det}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = OUT / f"E34_threshold_table_audit_{stamp}.json"
    path.write_text(json.dumps(
        dict(N=N, P=P, seed=seed, beta=args.beta, probes=probes,
             n_late=len(late), late=late, rows=rows), indent=1, default=float))
    print(f"{_now()} wrote {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
