"""Merge the sharded uniqueness runs and decide, over ALL starting patterns.

Each shard compares its own trajectories to its own first bead; the question is
global, so the phase-aligned waveforms of every shard must be compared to ONE
reference. This loads the per-shard npz files for a given lambda and reports the
worst deviation over all P starting patterns, together with the period spread.
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/2_cycle_rappel_snic/data/E34"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--pattern", default=None,
                   help="glob for the shard files (default: by lambda)")
    args = p.parse_args(argv)

    pat = args.pattern or f"E34_cycle_uniqueness_lam{args.lam:.6f}_L*"
    npzs = sorted(DATA.glob(pat + ".npz"))
    jsons = sorted(DATA.glob(pat + ".json"))
    if not npzs:
        raise SystemExit(f"no shard npz matching {pat}")
    W, rows = {}, []
    for f in npzs:
        d = np.load(f)
        for k in d.files:
            if k.startswith("fp_"):
                W[int(k[3:])] = d[k]
    for f in jsons:
        rows.extend(json.loads(f.read_text())["rows"])
    # a bead re-run to repair a diagnosed artifact appears twice; the files are
    # read in sorted order and the repair tag sorts last, so keeping the LAST
    # occurrence of each bead selects the repaired run for both the waveform
    # (overwritten in W above) and the row.
    rows = list({int(r["mu"]): r for r in rows}.values())
    rows.sort(key=lambda r: r["mu"])

    keys = sorted(W)
    ref_key = keys[0]
    ref = W[ref_key]
    dist = {k: float(np.max(np.linalg.norm(W[k] - ref, axis=1))) for k in keys}
    per = [r["T_refined"] for r in rows
           if np.isfinite(r.get("T_refined", np.nan))]
    steps = sorted({s for r in rows for s in r.get("ring_steps", [])})
    beads_visited = sorted({r.get("n_beads_visited") for r in rows})
    slowest = sorted({r.get("dwell_argmax") for r in rows
                      if r.get("dwell_argmax") is not None})
    stat = [r["mu"] for r in rows if r.get("stationary")]

    print(f"lambda = {args.lam}")
    print(f"  starting patterns analysed : {len(keys)} / {len(rows)}")
    print(f"  ring steps seen            : {steps}")
    print(f"  beads visited per orbit    : {beads_visited}")
    print(f"  slowest bead (dwell argmax): {slowest}")
    print(f"  starts that went STATIONARY: {stat if stat else 'none'}")
    if per:
        print(f"  refined period             : mean {np.mean(per):.4f}, "
              f"spread {np.max(per) - np.min(per):.2e}, "
              f"relative {(np.max(per) - np.min(per)) / np.mean(per):.2e}")
    worst = max(dist, key=dist.get)
    print(f"  waveform distance to ref bead {ref_key}: "
          f"max {dist[worst]:.3e} (bead {worst}), "
          f"median {np.median(list(dist.values())):.3e}")
    same = dist[worst] < 1e-2
    verdict = ("ONE cycle: all starting patterns converge to the SAME periodic "
               "orbit" if same else
               "SEVERAL attractors: some starting patterns reach a different "
               "orbit")
    print(f"  VERDICT {verdict}")

    out = DATA / f"E34_cycle_uniqueness_MERGED_lam{args.lam:.6f}.json"
    out.write_text(json.dumps(
        dict(lam=args.lam, n_patterns=len(keys), ring_steps=steps,
             beads_visited=beads_visited, slowest_beads=slowest,
             stationary_starts=stat,
             period_mean=float(np.mean(per)) if per else None,
             period_spread=float(np.max(per) - np.min(per)) if per else None,
             ref_bead=ref_key, dist_max=dist[worst], worst_bead=worst,
             dist_median=float(np.median(list(dist.values()))),
             dists=dist, verdict=verdict), indent=1, default=float))
    print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
