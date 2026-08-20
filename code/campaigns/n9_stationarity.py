"""N9 float64 stationarity and operational ergodicity controls."""
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
from pathlib import Path

import numpy as np
from scipy import stats

from batch_dde import ReducedDDEBatch64
from couplings import make_patterns
from dynamics import periodic_closure
from v5_paths import RUNS, environment_manifest, write_json

WAITING_TIMES = [0.0, 50.0, 200.0, 800.0, 3200.0]


def participation(A: np.ndarray) -> np.ndarray:
    positive = np.clip(A, 0.0, None)
    s2 = np.sum(positive**2, axis=1)
    s4 = np.sum(positive**4, axis=1)
    return s2**2 / np.maximum(s4, 1e-30)


def analyse(seed: int, N: int, alpha: float, lam: float, dt: float,
            K: int, transient: float, observation: float,
            smoke: bool = False) -> dict:
    P = round(alpha * N)
    tag = f"a{alpha:.3f}_lam{lam:.4f}_s{seed}_dt{dt:g}".replace(".", "p")
    path = RUNS / "N9" / f"stationarity_N{N}_{tag}.npz"
    sidecar = path.with_suffix(".json")
    if (
        path.exists() and sidecar.exists()
        and os.environ.get("V5_FORCE", "0") != "1"
    ):
        saved_row = json.loads(sidecar.read_text(encoding="utf-8"))
        requested = {
            "N": N, "P": P, "alpha": alpha, "lambda": lam, "seed": seed,
            "dt": dt, "K_requested": K, "transient": transient,
            "observation": observation,
        }
        if any(saved_row.get(key) != value for key, value in requested.items()):
            raise RuntimeError(
                f"refusing to reuse {path}: saved N9 configuration differs"
            )
        return saved_row
    xi, _ = make_patterns(N, P, seed)
    rng = np.random.default_rng(90_000 + seed)
    system = ReducedDDEBatch64(xi, 20.0, lam, 10.0, 1.0)
    # K is the requested number of independently initialized trajectories that
    # remain in the wandering (non-fixed, non-periodic) ensemble.  Candidates
    # are generated in bounded batches so float64 memory does not scale with
    # unsuccessful basin trials.
    target_chaotic = K
    candidate_batch = max(4 if smoke else 32, K)
    accepted = []
    candidate_drifts = []
    candidate_closures = []
    attempted = 0
    t = None
    maximum_batches = 1 if smoke else 4
    for _ in range(maximum_batches):
        initial = 0.3 * rng.standard_normal((P, candidate_batch))
        settled = system.integrate(
            initial, transient, dt,
            record_every=max(1, round(0.5/dt)),
        )
        solution = system.integrate(
            settled["hist"], observation, dt,
            record_every=max(1, round(0.5/dt)),
            derivative_history=settled["dhist"],
        )
        t = solution["t"]
        candidate = solution["a"]  # time, P, candidate
        drift = np.linalg.norm(candidate[-1] - candidate[-2], axis=0) / np.maximum(
            np.linalg.norm(candidate[-1], axis=0), 1e-30
        )
        closure = np.array([
            periodic_closure(t, candidate[:, :, index], 10.0)[0]
            for index in range(candidate.shape[2])
        ])
        # This is an operational basin filter, not a Lyapunov diagnosis.  The
        # parameter cell itself must already have been selected as chaotic by
        # N8.  We remove fixed tails and clean periodic closures here.
        mask = (drift > 3e-3) & (closure > 1e-4)
        accepted.append(candidate[:, :, mask])
        candidate_drifts.extend(drift.tolist())
        candidate_closures.extend(closure.tolist())
        attempted += candidate.shape[2]
        if sum(chunk.shape[2] for chunk in accepted) >= target_chaotic:
            break
    if t is None:
        raise RuntimeError("no N9 candidate trajectories were integrated")
    A = np.concatenate(accepted, axis=2)
    if A.shape[2] > target_chaotic:
        A = A[:, :, :target_chaotic]
    K_kept = A.shape[2]
    if K_kept == 0:
        raise RuntimeError(
            f"N9 found no wandering histories for alpha={alpha}, lambda={lam}, "
            f"seed={seed}; the N8 parameter choice must be revised"
        )
    norm = np.sum(A**2, axis=1)
    part = participation(A)

    # A common PCA basis is fitted from a time/ensemble subsample.
    sample_t = np.linspace(0, len(t) - 1, min(len(t), 2000), dtype=int)
    samples = A[sample_t].transpose(0, 2, 1).reshape(-1, P)
    samples -= samples.mean(axis=0, keepdims=True)
    covariance = samples.T @ samples / max(len(samples) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    pc1 = eigenvectors[:, -1]
    coordinate = np.einsum("tpk,p->tk", A, pc1)

    midpoint = len(t) // 2
    ks = {}
    for name, values in {
        "pc1": coordinate, "participation": part, "norm": norm
    }.items():
        ks[name] = [
            float(stats.ks_2samp(values[:midpoint, k], values[midpoint:, k]).statistic)
            for k in range(K_kept)
        ]

    correlations = {}
    dt_record = float(t[1] - t[0])
    for tw in WAITING_TIMES:
        if tw >= observation:
            continue
        index = int(round(tw / dt_record))
        base = A[index]
        segment = A[index:]
        raw = np.einsum("tpk,pk->tk", segment, base)
        base_norm = np.sum(base**2, axis=0)
        normalized = raw / np.maximum(base_norm[None, :], 1e-30)
        correlations[str(tw)] = {
            "lag": (t[:len(segment)]).tolist(),
            "mean": np.mean(normalized, axis=1).tolist(),
            "seed_sem": (
                np.std(normalized, axis=1, ddof=1) / np.sqrt(K_kept)
                if K_kept > 1 else np.full(len(normalized), np.nan)
            ).tolist(),
        }

    # Time distribution of one long member against ensemble/time distribution of
    # the remaining members, with equal-size deterministic subsampling.
    ergodicity = {}
    for name, values in {
        "pc1": coordinate, "participation": part, "norm": norm
    }.items():
        one = values[:, 0]
        if K_kept > 1:
            ensemble = values[:, 1:].reshape(-1)
            indices = np.linspace(0, len(ensemble) - 1, len(one), dtype=int)
            ergodicity[name] = float(
                stats.ks_2samp(one, ensemble[indices]).statistic
            )
        else:
            ergodicity[name] = np.nan
    candidate_drifts = np.asarray(candidate_drifts)
    candidate_closures = np.asarray(candidate_closures)
    candidate_wandering = (
        (candidate_drifts > 3e-3) & (candidate_closures > 1e-4)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, t=t, norm=norm, participation=part, pc1=coordinate,
        pca_eigenvalues=eigenvalues,
        candidate_drift=candidate_drifts,
        candidate_periodic_closure=candidate_closures,
        candidate_wandering=candidate_wandering,
        N=N, P=P, alpha=alpha, lam=lam, seed=seed, dt=dt,
        transient=transient, observation=observation,
        attempted_histories=attempted, retained_histories=K_kept,
    )
    row = {
        "N": N, "P": P, "alpha": alpha, "lambda": lam, "seed": seed,
        "dt": dt, "K_requested": target_chaotic, "K_retained": K_kept,
        "transient": transient, "observation": observation,
        "attempted_histories": attempted, "file": str(path),
        "completion_32_chaotic": bool(smoke or K_kept >= 32),
        "escape_or_periodic_fraction": float(1.0 - np.mean(candidate_wandering)),
        "ks_early_late": ks,
        "ergodicity_ks": ergodicity,
        "two_time_correlations": correlations,
    }
    write_json(sidecar, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--lambda-a008", type=float)
    parser.add_argument("--lambda-a010", type=float)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and (args.lambda_a008 is None or args.lambda_a010 is None):
        raise SystemExit(
            "N9 production requires --lambda-a008 and --lambda-a010 selected "
            "from the completed N8 chaotic cells; no value is assumed."
        )
    points = (
        [(0.05, 0.31)] if args.smoke else
        [(0.05, 0.31), (0.08, args.lambda_a008), (0.10, args.lambda_a010)]
    )
    rows = []
    for alpha, lam in points:
        for seed in args.seeds:
            dts = [0.02] if args.smoke else [0.01] + ([0.005] if seed == 42 else [])
            for dt in dts:
                rows.append(analyse(
                    seed, args.N, alpha, lam, dt,
                    K=(4 if args.smoke else args.K),
                    transient=(4.0 if args.smoke else 400.0),
                    observation=(8.0 if args.smoke else 6000.0),
                    smoke=args.smoke,
                ))
    summary_name = (
        "smoke.json" if args.smoke else
        ("summary.json" if args.seeds == [42, 43, 44, 45, 46]
         else "summary_seeds_" + "_".join(map(str, args.seeds)) + ".json")
    )
    write_json(
        RUNS / "N9" / summary_name,
        {
            "campaign": "N9", "rows": rows, "smoke": args.smoke,
            "environment": environment_manifest(),
        },
    )
    print(f"N9 completed {len(rows)} seed/point/time-step cells")


if __name__ == "__main__":
    main()
