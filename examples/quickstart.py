#!/usr/bin/env python3
"""Replot archived thresholds and run a small, new sequential-recall experiment.

Run from any directory. Outputs always go to an explicit, separate output folder.
The small demo illustrates the dynamics; it is not a rerun of the N=2000 study.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "core"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycle_reduced import ReducedDDE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "quickstart")
    parser.add_argument("--data-only", action="store_true", help="Only replot shipped data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = ROOT / "data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz"
    with np.load(source, allow_pickle=False) as data:
        thresholds = data["lam_c"].copy()
        assert thresholds.shape == (100,) and np.isfinite(thresholds).all()
    summary = {"archived_data": {"N": 2000, "P": 100, "seed": 42,
        "min": float(thresholds.min()), "median": float(np.median(thresholds)),
        "max": float(thresholds.max()), "maximum_pattern_zero_based": int(thresholds.argmax()),
        "source": str(source.relative_to(ROOT))}}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    axes[0].scatter(np.arange(100), thresholds, s=17, color="#245c4d")
    axes[0].set(xlabel="Memory index (zero-based)", ylabel="Saddle-node threshold λc",
                title="One network, 100 local thresholds")
    axes[1].hist(thresholds, bins=16, color="#83a394", edgecolor="white")
    axes[1].axvline(np.median(thresholds), color="#245c4d", label="Median")
    axes[1].axvline(thresholds.max(), color="#ad6542", label="Maximum")
    axes[1].set(xlabel="Saddle-node threshold λc", ylabel="Memory count", title="Bulk and extreme of one ensemble")
    axes[1].legend(frameon=False)
    fig.savefig(args.output / "thresholds.png", dpi=180)
    plt.close(fig)
    if not args.data_only:
        N, P, seed, beta, lam, tau, dt, duration = 300, 9, 42, 20., .9, 10., .05, 180.
        xi = np.random.default_rng(seed).choice([-1., 1.], size=(P, N))
        model = ReducedDDE(xi, beta, lam, tau)
        history = np.zeros(P); history[0] = 1.
        solution = model.integrate(history, duration, dt, record_every=4, da_hist0=np.zeros(P))
        overlaps = np.array([model.m(a) for a in solution["a"]])
        if not np.isfinite(overlaps).all():
            raise RuntimeError("Non-finite recall trajectory")
        winners = overlaps.argmax(axis=1)
        visited = np.unique(winners[solution["t"] > tau]).tolist()
        summary["illustrative_simulation"] = dict(N=N, P=P, seed=seed, beta=beta,
            lam=lam, tau=tau, dt=dt, duration=duration, visited_patterns=visited,
            note="New small-network illustration, not a manuscript certification run")
        fig, ax = plt.subplots(figsize=(10, 3.7), layout="constrained")
        for i in range(P):
            ax.plot(solution["t"], overlaps[:, i], lw=1.2, label=f"Memory {i}")
        ax.set(xlabel="Time / t₀", ylabel="Pattern overlap", title="Sequential recall · N=300, P=9, λ=0.9, τ=10")
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper right")
        fig.savefig(args.output / "sequential-recall.png", dpi=180)
        plt.close(fig)
        np.savez_compressed(args.output / "recall.npz", t=solution["t"], overlaps=overlaps)
    summary["runtime_seconds"] = round(time.perf_counter() - started, 2)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
