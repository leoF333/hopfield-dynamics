"""Tune the argument-principle settings for P=100 saddle certification.

At N=400/P=20 the winding count converged with samples_per_edge=16,
max_refinements=7. At N=2000/P=100 it returns "integer count not yet stable"
even though the equilibrium is machine-exact (res ~1e-15), eigmax_M > 0 and a
positive rightmost REAL root exists -- i.e. the object is unstable, only the
exact count is unresolved. This reloads saddles already computed by
e34_necklace_wu_n2000 and escalates (samples_per_edge, max_refinements) until
the count stabilises, reporting the cheapest setting that works.
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
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from e34_lib import (
    NumpyLowRankCouplings, characteristic_spectral_bound,
    count_unstable_reduced_spectrum, gram_matrix, make_iid_patterns,
)
from reduced_spectrum import GJ_GK, rightmost_real_root
from robust_branch import eigmax_M

OUT = Path(__file__).resolve().parents[1] / "results/2_cycle_rappel_snic/data/E34"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--t0", type=float, default=1.0)
    args = p.parse_args(argv)

    npz_path = args.npz or sorted(OUT.glob("E34_ET1pp_wu_results_*.npz"))[-1]
    d = np.load(npz_path, allow_pickle=True)
    print(f"loaded {npz_path.name}", flush=True)

    xi, xis = make_iid_patterns(2000, 100, 42)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    settings = [(16, 7), (32, 9), (48, 11), (64, 13), (96, 15)]
    rows = []
    for i in range(len(d["mu"])):
        a_s = d["a_saddle"][i]
        if not np.all(np.isfinite(a_s)):
            continue
        mu, lam = int(d["mu"][i]), float(d["lam"][i])
        u = coup._xi_np.T @ a_s
        res = float(np.linalg.norm(coup.field_F(u, lam, args.beta))
                    / np.sqrt(coup.N))
        GJ, GK = GJ_GK(coup, u, args.beta, lam)
        rho = characteristic_spectral_bound(GJ, GK, lam)
        em = float(eigmax_M(coup, u, lam, args.beta))
        rr = rightmost_real_root(GJ, GK, args.t0, args.tau, lam,
                                 x_hi=rho / args.t0, x_lo=-0.5, n=400)
        print(f"\nmu={mu} lam={lam:.6f} res={res:.1e} eigmax={em:+.4f} "
              f"real_root={rr.real if rr is not None else float('nan'):.5f} "
              f"rho={rho:.3f}", flush=True)
        row = dict(mu=mu, lam=lam, res=res, eigmax=em,
                   real_root=float(rr.real) if rr is not None else np.nan)
        for spe, mr in settings:
            t0w = time.perf_counter()
            base = count_unstable_reduced_spectrum(
                GJ, GK, args.t0, args.tau, lam,
                samples_per_edge=spe, max_refinements=mr)
            eta = max(0.05, 0.02 * rho / args.t0)
            shifted = count_unstable_reduced_spectrum(
                GJ, GK, args.t0, args.tau, lam, eta=1.5 * eta,
                samples_per_edge=spe, max_refinements=mr)
            ok = bool(base.converged and shifted.converged
                      and base.count == shifted.count)
            w = time.perf_counter() - t0w
            print(f"   spe={spe:3d} refin={mr:2d} -> base={base.count} "
                  f"({base.reason}) shifted={shifted.count} stable={ok} "
                  f"[{w:.1f}s]", flush=True)
            row[f"spe{spe}_r{mr}"] = dict(
                count=base.count, shifted=shifted.count, stable=ok,
                reason=str(base.reason), wall=w)
            if ok:
                row["first_stable"] = [spe, mr, base.count]
                break
        rows.append(row)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    (OUT / f"E34_certif_tune_{stamp}.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote E34_certif_tune_{stamp}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
