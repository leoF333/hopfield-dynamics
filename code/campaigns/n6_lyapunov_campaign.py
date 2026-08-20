"""N6 multi-seed float64 Lyapunov spectrum and attractor geometry."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
from pathlib import Path

import numpy as np

from dynamics import relay_statistics
from lyapunov_stream import LyapunovConfig, run as run_lyapunov
from v5_paths import RUNS, environment_manifest, write_json

LAMBDAS = [0.29, 0.30, 0.31, 0.32, 0.325]
WINDOWS = [1000.0, 2000.0, 4000.0]


def kaplan_yorke(exponents: np.ndarray) -> float:
    ordered = np.sort(exponents)[::-1]
    cumulative = np.cumsum(ordered)
    nonnegative = np.flatnonzero(cumulative >= 0)
    if not len(nonnegative):
        return 0.0
    j = int(nonnegative[-1])
    if j == len(ordered) - 1:
        return float(len(ordered))
    return float(j + 1 + cumulative[j] / abs(ordered[j + 1]))


def geometry(npz_path: Path) -> dict:
    data = np.load(npz_path)
    A = np.asarray(data["record_a"], float)
    t = np.asarray(data["record_t"], float)
    centered = A - A.mean(axis=0)
    covariance = centered.T @ centered / max(len(A) - 1, 1)
    eig = np.linalg.eigvalsh(covariance)[::-1]
    pca_participation = float(eig.sum()**2 / np.maximum(eig @ eig, 1e-30))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vt[0]
    dt = float(np.median(np.diff(t)))
    frequency = np.fft.rfftfreq(len(pc1), dt)
    power = np.abs(np.fft.rfft(pc1 - pc1.mean()))**2
    level = float(np.median(A[:, 0]))
    crossings = np.flatnonzero(
        (A[:-1, 0] < level) & (A[1:, 0] >= level)
    ) + 1
    section = A[crossings, 1] if A.shape[1] > 1 else A[crossings, 0]
    relay = relay_statistics(t, np.argmax(A, axis=1), A.shape[1])
    out_path = npz_path.with_name(npz_path.stem + "_geometry.npz")
    np.savez_compressed(
        out_path, pca_eigenvalues=eig, frequency=frequency, power=power,
        section=section, return_x=section[:-1], return_y=section[1:],
        dwell=relay["dwell"],
    )
    return {
        "file": str(out_path),
        "pca_participation": pca_participation,
        "section_points": int(len(section)),
        "forward_fraction": relay["forward_fraction"],
        "tours": relay["tours"],
    }


def run_cell(N: int, P: int, seed: int, lam: float, dt: float,
             smoke: bool) -> dict:
    accumulation = 10.0 if smoke else 4000.0
    cfg = LyapunovConfig(
        N=N, P=P, seed=seed, lam=lam, dt=dt,
        transient=(10.0 if smoke else 400.0),
        accumulation=accumulation, k=(2 if smoke else 8),
        qr_time=(0.4 if smoke else 2.0),
        record_dt=(0.1 if smoke else 0.2),
    )
    tag = f"s{seed}_lam{lam:.3f}_dt{dt:g}".replace(".", "p")
    path = RUNS / "N6" / f"lyapunov_N{N}_P{P}_{tag}.npz"
    summary = run_lyapunov(cfg, path)
    data = np.load(path)
    cumulative = data["cumulative_exponents"]
    block_time = data["block_time"]
    estimates = {}
    requested = [accumulation] if smoke else WINDOWS
    for window in requested:
        index = int(np.argmin(np.abs(block_time - window)))
        estimates[str(window)] = cumulative[index].tolist()
    spectrum = np.asarray(summary["lyapunov"])
    geom = geometry(path)
    return {
        "N": N, "P": P, "seed": seed, "lambda": lam, "dt": dt,
        "file": str(path), "window_estimates": estimates,
        "spectrum": spectrum.tolist(),
        "positive_exponents": int(np.sum(spectrum > 0)),
        "kaplan_yorke_finite_spectrum": kaplan_yorke(spectrum),
        "geometry": geom,
        "wall_seconds": summary["wall_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    lambdas = [0.31] if args.smoke else LAMBDAS
    rows = []
    for seed in args.seeds:
        for lam in lambdas:
            rows.append(run_cell(args.N, args.P, seed, lam, 0.01, args.smoke))
            if not args.smoke and (seed == 42 or abs(lam - 0.31) < 1e-12):
                rows.append(run_cell(args.N, args.P, seed, lam, 0.005, False))
    summary_name = (
        "smoke.json" if args.smoke else
        ("summary.json" if args.seeds == [42, 43, 44, 45, 46]
         else "summary_seeds_" + "_".join(map(str, args.seeds)) + ".json")
    )
    write_json(
        RUNS / "N6" / summary_name,
        {
            "campaign": "N6", "rows": rows, "smoke": args.smoke,
            "environment": environment_manifest(),
        },
    )
    print(f"N6 completed {len(rows)} spectrum runs")


if __name__ == "__main__":
    main()
