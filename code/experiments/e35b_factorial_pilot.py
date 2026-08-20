"""E35 T1: guarded J x K factorial pilot.

Examples
--------
Preview only (no allocation or files):
    python src/e35b_factorial_pilot.py --dry-run

Strict tiny smoke:
    python src/e35b_factorial_pilot.py --smoke --output /tmp/e35_t1_smoke

Future heavy-capable run (still requires the user's prior authorization):
    python src/e35b_factorial_pilot.py --production --confirm-heavy
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from e35_dynamics import MhnReducedDDE, classify_cycle, integrate_chunked
from e35_runtime import (
    LIMITATIONS,
    RunArtifacts,
    exclusive_compute_lock,
    json_safe,
    make_run_id,
)
from mhn_reduced import MhnContext, factorial_architectures


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "results" / "8_mhn_video"


def make_iid_patterns(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice((-1.0, 1.0), size=(p, n))


def make_correlated_ring(
    n: int,
    p: int,
    seed: int,
    *,
    modes: int = 2,
) -> np.ndarray:
    """Seamless correlated binary ring without importing the E28 script."""

    rng = np.random.default_rng(seed)
    theta = 2.0 * np.pi * np.arange(p) / p
    frequencies = np.arange(1, modes + 1)
    cosine = np.cos(np.outer(frequencies, theta))
    sine = np.sin(np.outer(frequencies, theta))
    amplitude_cos = rng.standard_normal((n, modes))
    amplitude_sin = rng.standard_normal((n, modes))
    latent = amplitude_cos @ cosine + amplitude_sin @ sine
    patterns = np.sign(latent.T)
    patterns[patterns == 0.0] = 1.0
    return patterns


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--production", action="store_true")
    parser.add_argument(
        "--confirm-heavy",
        action="store_true",
        help="Second guard; never use without the user's explicit authorization.",
    )
    parser.add_argument("--n", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--lam", type=float, default=0.9)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-total", type=float)
    parser.add_argument("--chunk-time", type=float)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ring-modes", type=int, default=2)
    parser.add_argument("--tsvd-rtol", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def resolve_config(args) -> dict[str, Any]:
    if args.production and not args.confirm_heavy:
        raise ValueError(
            "--production is locked: add --confirm-heavy only after explicit user approval"
        )
    mode = "production" if args.production else "smoke" if args.smoke else "dry_run"
    defaults = (
        dict(n=512, p=16, tau=10.0, dt=0.05, t_total=120.0, chunk_time=5.0)
        if mode in ("production", "dry_run")
        else dict(n=64, p=4, tau=0.5, dt=0.05, t_total=2.0, chunk_time=0.5)
    )
    config = {
        "experiment": "E35b_T1_factorial",
        "mode": mode,
        "n": defaults["n"] if args.n is None else args.n,
        "p": defaults["p"] if args.p is None else args.p,
        "beta": args.beta,
        "lam": args.lam,
        "tau": defaults["tau"] if args.tau is None else args.tau,
        "t0": args.t0,
        "dt": defaults["dt"] if args.dt is None else args.dt,
        "t_total": (
            defaults["t_total"] if args.t_total is None else args.t_total
        ),
        "chunk_time": (
            defaults["chunk_time"] if args.chunk_time is None else args.chunk_time
        ),
        "record_every": args.record_every,
        "seed": args.seed,
        "ring_modes": args.ring_modes,
        "tsvd_rtol": args.tsvd_rtol,
        "architectures": ["JH_KH", "JP_KH", "JH_KP", "JP_KP"],
        "datasets": ["iid", "correlated_ring"],
        "output": str(args.output),
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    n, p = int(config["n"]), int(config["p"])
    if not 1 < p < n:
        raise ValueError(f"require 1<P<N, got P={p}, N={n}")
    if config["ring_modes"] < 1:
        raise ValueError("ring_modes must be positive")
    if config["record_every"] < 1:
        raise ValueError("record_every must be positive")
    for name in ("beta", "tau", "t0", "dt", "t_total", "chunk_time"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for total_name in ("tau", "t_total", "chunk_time"):
        ratio = config[total_name] / config["dt"]
        if abs(ratio - round(ratio)) > 1e-10:
            raise ValueError(f"dt must divide {total_name}")
    if config["mode"] == "smoke":
        if n > 128 or p > 8:
            raise ValueError("smoke cap is N<=128 and P<=8")
        if config["tau"] > 1.0 or config["t_total"] > 4.0:
            raise ValueError("smoke cap is tau<=1 and t_total<=4")


def run(config: dict[str, Any], output: Path, heartbeat_seconds: float) -> int:
    run_id = make_run_id("E35b", config["mode"])
    datasets = {
        "iid": make_iid_patterns(config["n"], config["p"], config["seed"]),
        "correlated_ring": make_correlated_ring(
            config["n"],
            config["p"],
            config["seed"],
            modes=config["ring_modes"],
        ),
    }
    rows = []
    trajectories = []
    observable_trajectories = []
    with exclusive_compute_lock(output, run_id):
        with RunArtifacts(
            output,
            run_id,
            config,
            repository=REPOSITORY,
            heartbeat_interval=heartbeat_seconds,
        ) as artifacts:
            total_jobs = len(datasets) * len(config["architectures"])
            job_index = 0
            for dataset_name, patterns in datasets.items():
                context = MhnContext.from_patterns(
                    patterns,
                    config["beta"],
                    tsvd_rtol=config["tsvd_rtol"],
                )
                architectures = factorial_architectures(context)
                for architecture_name in config["architectures"]:
                    architecture = architectures[architecture_name]
                    job_index += 1
                    artifacts.update(
                        stage=f"{dataset_name}:{architecture_name}",
                        completed=job_index - 1,
                        total=total_jobs,
                        nan_count=0,
                    )
                    a0 = np.zeros(config["p"], dtype=np.float64)
                    a0[0] = 0.95
                    a0[1] = 0.05
                    system = MhnReducedDDE(
                        architecture,
                        config["lam"],
                        config["tau"],
                        config["t0"],
                    )

                    def on_chunk(state, name=architecture_name):
                        artifacts.update(
                            stage=f"{dataset_name}:{name}",
                            completed=job_index - 1
                            + state["completed_steps"] / state["total_steps"],
                            total=total_jobs,
                            nan_count=state["nan_count"],
                        )
                        artifacts.checkpoint(
                            hist=state["hist"],
                            dhist=state["dhist"],
                            architecture=f"{dataset_name}:{name}",
                            simulated_time=state["simulated_time"],
                            completed_steps=state["completed_steps"],
                        )

                    solution = integrate_chunked(
                        system,
                        a0,
                        da_hist0=np.zeros_like(a0),
                        t_total=config["t_total"],
                        dt=config["dt"],
                        record_every=config["record_every"],
                        chunk_time=config["chunk_time"],
                        on_chunk=on_chunk,
                    )
                    physical_overlap = context.physical_overlaps_batch(
                        solution["a"]
                    )
                    diagnosis = classify_cycle(
                        solution["t"], physical_overlap
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "architecture": architecture_name,
                            **context.diagnostics.to_dict(),
                            **diagnosis.to_dict(),
                        }
                    )
                    trajectories.append(solution["a"])
                    observable_trajectories.append(physical_overlap)
                    artifacts.event(
                        "configuration_finished",
                        dataset=dataset_name,
                        architecture=architecture_name,
                        diagnostics=diagnosis.to_dict(),
                        svd=context.diagnostics.to_dict(),
                    )
            artifacts.update(
                stage="finished",
                completed=total_jobs,
                total=total_jobs,
                nan_count=0,
            )
            artifacts.save_data(
                rows_json=np.array(
                    [json.dumps(json_safe(row), sort_keys=True) for row in rows]
                ),
                final_coefficients=np.array(
                    [trajectory[-1] for trajectory in trajectories]
                ),
                final_physical_overlaps=np.array(
                    [trajectory[-1] for trajectory in observable_trajectories]
                ),
                config_json=np.array(json.dumps(json_safe(config), sort_keys=True)),
            )
    print(
        json.dumps(
            json_safe({"run_id": run_id, "rows": rows}),
            indent=2,
            default=str,
        )
    )
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "dry_run_only_no_files_no_compute",
                    "config": config,
                    "limitations": LIMITATIONS,
                    "benchmark_status": "not_run",
                },
                indent=2,
                default=str,
            )
        )
        return 0
    return run(config, args.output, args.heartbeat_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
