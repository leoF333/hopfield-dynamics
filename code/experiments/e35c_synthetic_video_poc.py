"""E35 V0: guarded synthetic cyclic-video proof of concept.

The candidate architecture is supplied on the command line before any hidden
frame is scored.  The script never selects an architecture or phase on target
quality.
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

from e35_dynamics import (
    MhnReducedDDE,
    build_baseline_predictions,
    classify_cycle,
    decode_hidden_frames,
    evaluate_predictions,
    integrate_chunked,
)
from e35_runtime import (
    LIMITATIONS,
    RunArtifacts,
    exclusive_compute_lock,
    json_safe,
    make_run_id,
)
from e35_video_tools import (
    FramePreprocessor,
    make_cyclic_loop,
    seam_diagnostics,
    split_keyframes,
)
from mhn_reduced import MhnContext, factorial_architectures


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "results" / "8_mhn_video"
ARCHITECTURE_IDS = ("JH_KH", "JP_KH", "JH_KP", "JP_KP")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--production", action="store_true")
    parser.add_argument("--confirm-heavy", action="store_true")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--kind", choices=("orbit", "orbit_rotate"), default="orbit")
    parser.add_argument("--candidate", choices=ARCHITECTURE_IDS)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--lam", type=float, default=0.9)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-total", type=float)
    parser.add_argument("--chunk-time", type=float)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
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
        dict(
            height=32,
            width=32,
            p=16,
            k=4,
            tau=10.0,
            dt=0.05,
            t_total=400.0,
            chunk_time=5.0,
        )
        if mode in ("production", "dry_run")
        else dict(
            height=8,
            width=8,
            p=4,
            k=2,
            tau=0.5,
            dt=0.05,
            t_total=4.0,
            chunk_time=0.5,
        )
    )
    candidate = args.candidate
    architectures = ["JH_KH"]
    if candidate is not None and candidate != "JH_KH":
        architectures.append(candidate)
    config = {
        "experiment": "E35c_V0_synthetic_video",
        "mode": mode,
        "height": defaults["height"] if args.height is None else args.height,
        "width": defaults["width"] if args.width is None else args.width,
        "p": defaults["p"] if args.p is None else args.p,
        "k": defaults["k"] if args.k is None else args.k,
        "kind": args.kind,
        "candidate_predeclared": candidate,
        "architectures": architectures,
        "binary": bool(args.binary),
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
        "transient_fraction": args.transient_fraction,
        "seed": args.seed,
        "tsvd_rtol": args.tsvd_rtol,
        "output": str(args.output),
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    n = int(config["height"]) * int(config["width"])
    p = int(config["p"])
    if config["height"] < 8 or config["width"] < 8:
        raise ValueError("height and width must be at least 8")
    if not 1 < p < n:
        raise ValueError(f"require 1<P<N, got P={p}, N={n}")
    if config["k"] < 2:
        raise ValueError("k must be at least 2")
    if config["record_every"] < 1:
        raise ValueError("record_every must be positive")
    if not 0.0 <= config["transient_fraction"] < 1.0:
        raise ValueError("transient_fraction must lie in [0,1)")
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
    run_id = make_run_id("E35c", config["mode"])
    n_frames = config["p"] * config["k"]
    frames = make_cyclic_loop(
        n_frames,
        height=config["height"],
        width=config["width"],
        kind=config["kind"],
        seed=config["seed"],
    )
    split = split_keyframes(frames, config["k"])
    preprocessor = FramePreprocessor.fit_from_keyframes(
        split.keyframes, binary=config["binary"]
    )
    patterns = preprocessor.encode(split.keyframes).reshape(config["p"], -1)
    context = MhnContext.from_patterns(
        patterns,
        config["beta"],
        tsvd_rtol=config["tsvd_rtol"],
    )
    factorial = factorial_architectures(context)
    baselines = build_baseline_predictions(split)
    baseline_metrics = {
        name: evaluate_predictions(split.hidden_frames, prediction)
        for name, prediction in baselines.items()
    }

    cycle_rows = []
    metric_rows = []
    dynamic_predictions = []
    dynamic_valid = []
    dynamic_interval_start = []
    dynamic_interval_end = []
    physical_overlap_trajectories = []
    dynamic_metric_arrays = {
        name: [] for name in ("mse", "mae", "psnr", "ssim", "pearson")
    }
    with exclusive_compute_lock(output, run_id):
        with RunArtifacts(
            output,
            run_id,
            config,
            repository=REPOSITORY,
            heartbeat_interval=heartbeat_seconds,
        ) as artifacts:
            total_jobs = len(config["architectures"])
            for job_index, architecture_name in enumerate(config["architectures"]):
                architecture = factorial[architecture_name]
                system = MhnReducedDDE(
                    architecture,
                    config["lam"],
                    config["tau"],
                    config["t0"],
                )
                a0 = np.zeros(config["p"], dtype=np.float64)
                a0[0] = 0.95
                a0[1] = 0.05
                artifacts.update(
                    stage=architecture_name,
                    completed=job_index,
                    total=total_jobs,
                    nan_count=0,
                )

                def on_chunk(state, name=architecture_name):
                    artifacts.update(
                        stage=name,
                        completed=job_index
                        + state["completed_steps"] / state["total_steps"],
                        total=total_jobs,
                        nan_count=state["nan_count"],
                    )
                    artifacts.checkpoint(
                        hist=state["hist"],
                        dhist=state["dhist"],
                        architecture=name,
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
                cycle = classify_cycle(
                    solution["t"],
                    physical_overlap,
                    transient_fraction=config["transient_fraction"],
                )
                decoded = decode_hidden_frames(
                    solution["t"],
                    solution["a"],
                    physical_overlap,
                    split,
                    patterns,
                    preprocessor,
                    transient_fraction=config["transient_fraction"],
                )
                metrics = evaluate_predictions(
                    split.hidden_frames,
                    decoded.predictions,
                    valid=decoded.valid,
                )
                cycle_rows.append(
                    {"architecture": architecture_name, **cycle.to_dict()}
                )
                metric_rows.append(
                    {
                        "architecture": architecture_name,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key.startswith("mean_") or key == "valid_count"
                        },
                    }
                )
                dynamic_predictions.append(decoded.predictions)
                dynamic_valid.append(decoded.valid)
                dynamic_interval_start.append(decoded.interval_start)
                dynamic_interval_end.append(decoded.interval_end)
                physical_overlap_trajectories.append(physical_overlap)
                for metric_name in dynamic_metric_arrays:
                    dynamic_metric_arrays[metric_name].append(
                        metrics[metric_name]
                    )
                artifacts.event(
                    "architecture_finished",
                    architecture=architecture_name,
                    cycle=cycle.to_dict(),
                    metrics=metric_rows[-1],
                )

            artifacts.update(
                stage="finished",
                completed=total_jobs,
                total=total_jobs,
                nan_count=0,
            )
            artifacts.save_data(
                frames=frames,
                key_indices=split.key_indices,
                hidden_indices=split.hidden_indices,
                hidden_truth=split.hidden_frames,
                patterns=patterns,
                architecture=np.array(config["architectures"]),
                dynamic_predictions=np.array(dynamic_predictions),
                dynamic_valid=np.array(dynamic_valid),
                dynamic_interval_start=np.array(dynamic_interval_start),
                dynamic_interval_end=np.array(dynamic_interval_end),
                physical_overlaps=np.array(physical_overlap_trajectories),
                **{
                    f"dynamic_{name}": np.array(values)
                    for name, values in dynamic_metric_arrays.items()
                },
                B0_HOLD=baselines["B0_HOLD"],
                B1_LINEAR=baselines["B1_LINEAR"],
                B2_MOTION=baselines["B2_MOTION"],
                **{
                    f"{baseline}_{metric}": values[metric]
                    for baseline, values in baseline_metrics.items()
                    for metric in ("mse", "mae", "psnr", "ssim", "pearson")
                },
                cycle_json=np.array(
                    [json.dumps(json_safe(row), sort_keys=True) for row in cycle_rows]
                ),
                metric_json=np.array(
                    [json.dumps(json_safe(row), sort_keys=True) for row in metric_rows]
                ),
                baseline_metric_json=np.array(
                    [
                        json.dumps(
                            json_safe(
                                {
                                    "baseline": name,
                                    **{
                                        key: value
                                        for key, value in values.items()
                                        if key.startswith("mean_")
                                        or key == "valid_count"
                                    },
                                }
                            ),
                            sort_keys=True,
                        )
                        for name, values in baseline_metrics.items()
                    ]
                ),
                seam_json=np.array(
                    json.dumps(
                        json_safe(seam_diagnostics(split.frames)),
                        sort_keys=True,
                    )
                ),
                svd_json=np.array(
                    json.dumps(context.diagnostics.to_dict(), sort_keys=True)
                ),
                config_json=np.array(json.dumps(json_safe(config), sort_keys=True)),
            )
    print(
        json.dumps(
            json_safe({
                "run_id": run_id,
                "candidate_predeclared": config["candidate_predeclared"],
                "cycle": cycle_rows,
                "dynamic_metrics": metric_rows,
                "baseline_metrics": {
                    name: {
                        key: value
                        for key, value in metrics.items()
                        if key.startswith("mean_") or key == "valid_count"
                    }
                    for name, metrics in baseline_metrics.items()
                },
                "seam": seam_diagnostics(split.frames),
            }),
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
                    "target_selection": "forbidden",
                },
                indent=2,
                default=str,
            )
        )
        return 0
    return run(config, args.output, args.heartbeat_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
