#!/usr/bin/env python3
"""Verify and summarize the completed E36 campaigns without simulation."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

_MPL_CACHE = Path(tempfile.gettempdir()) / "e36_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from e36a2_delay import (
    analyze_delay_trajectory,
    load_physical_trajectory,
)


CAMPAIGNS = (
    "E36A_regimes_seed42",
    "E36A_dtcheck_lam1_seed42",
    "E36A2_delay_seed42",
    "E36A_reference_N2000_P100_K11_seed42",
)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_record_grid(times: np.ndarray, nominal: float) -> dict:
    """Validate a uniform grid, allowing one strictly shorter terminal step."""
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("record stream needs at least two times")
    differences = np.diff(times)
    tolerance = max(1e-12, 1e-8*nominal)
    if np.any(differences <= 0):
        raise ValueError("record times are not strictly increasing")
    bad = np.flatnonzero(np.abs(differences-nominal) > tolerance)
    short_terminal = False
    if len(bad):
        if (
            len(bad) == 1
            and bad[0] == len(differences)-1
            and 0 < differences[-1] < nominal-tolerance
        ):
            short_terminal = True
        else:
            raise ValueError(f"irregular record intervals: {bad.tolist()}")
    return {
        "records": len(times),
        "nominal_dt": nominal,
        "short_terminal_interval": short_terminal,
        "terminal_interval": float(differences[-1]),
    }


def _close(left: float, right: float, scale: float = 1.0) -> bool:
    return abs(left-right) <= 1e-10*max(scale, abs(left), abs(right))


def validate_run(run_dir: Path) -> dict:
    """Validate one A1/A2 stream, its chunks, checkpoint and final state."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in ("E36A-observables-v2", "E36A2-observables-v1"):
        raise ValueError(f"unsupported schema in {manifest_path}: {schema}")
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete run: {run_dir}")
    config = manifest["config"]
    K, N, P = (int(config[key]) for key in ("K", "N", "P"))
    required = {"t", "physical_overlaps", "saturation", "max_abs"}
    if schema == "E36A-observables-v2":
        required.add("raw_overlaps")
    listed_files = {item["file"] for item in manifest["chunks"]}
    disk_files = {
        str(path.relative_to(run_dir))
        for path in (run_dir / "chunks").glob("chunk_*.npz")
    }
    if listed_files != disk_files:
        raise ValueError(f"listed/disk chunk mismatch in {run_dir}")

    time_parts: list[np.ndarray] = []
    for item in manifest["chunks"]:
        path = run_dir / item["file"]
        with np.load(path, allow_pickle=False) as chunk:
            if not required.issubset(chunk.files):
                raise ValueError(f"missing arrays in {path}")
            times = np.asarray(chunk["t"], dtype=np.float64)
            count = len(times)
            expected_shapes = {
                "physical_overlaps": (count, K, P),
                "saturation": (count, K),
                "max_abs": (count, K),
            }
            if "raw_overlaps" in required:
                expected_shapes["raw_overlaps"] = (count, K, P)
            if item["n_records"] != count:
                raise ValueError(f"chunk count mismatch in {path}")
            if count < 1 or not _close(item["t_start"], float(times[0])):
                raise ValueError(f"chunk start mismatch in {path}")
            if not _close(item["t_end"], float(times[-1])):
                raise ValueError(f"chunk end mismatch in {path}")
            if not np.all(np.isfinite(times)):
                raise ValueError(f"non-finite time in {path}")
            for key, shape in expected_shapes.items():
                values = chunk[key]
                if values.shape != shape:
                    raise ValueError(
                        f"{key} shape {values.shape} != {shape} in {path}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite {key} in {path}")
        time_parts.append(times)

    times = np.concatenate(time_parts)
    nominal = float(config["dt"])*int(config["record_every"])
    grid = validate_record_grid(times, nominal)
    if not _close(float(manifest["last_time"]), float(times[-1])):
        raise ValueError(f"manifest last_time mismatch in {run_dir}")
    if not _close(float(times[-1]), float(config["t_final"])):
        raise ValueError(f"record end/t_final mismatch in {run_dir}")

    checkpoint_path = run_dir / "checkpoint.npz"
    final_path = run_dir / "final_state.npz"
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        checkpoint_state = np.asarray(checkpoint["state"])
        step = int(checkpoint["step"])
        checkpoint_dt = float(checkpoint["dt"])
        meta = json.loads(str(checkpoint["meta_json"]))
        if checkpoint_state.shape != (K, N):
            raise ValueError(f"checkpoint shape mismatch in {run_dir}")
        if not np.all(np.isfinite(checkpoint_state)):
            raise ValueError(f"non-finite checkpoint in {run_dir}")
        if not _close(checkpoint_dt, float(config["dt"])):
            raise ValueError(f"checkpoint dt mismatch in {run_dir}")
        if step != round(float(config["t_final"])/float(config["dt"])):
            raise ValueError(f"checkpoint step mismatch in {run_dir}")
        meta_expected = {
            "K": K,
            "N": N,
            "P": P,
            "beta": float(config["beta"]),
            "lam": float(config["lambda"]),
            "t0": float(config["t0"]),
        }
        if any(meta.get(key) != value for key, value in meta_expected.items()):
            raise ValueError(f"checkpoint metadata mismatch in {run_dir}")
    with np.load(final_path, allow_pickle=False) as final:
        final_state = np.asarray(final["state"])
        if final_state.shape != (K, N) or not np.all(np.isfinite(final_state)):
            raise ValueError(f"invalid final state in {run_dir}")
    final_error = float(np.max(np.abs(final_state-checkpoint_state)))
    if final_error != 0.0:
        raise ValueError(f"final/checkpoint mismatch in {run_dir}")

    summary_name = (
        "summary.json" if schema == "E36A-observables-v2"
        else "delay_summary.json"
    )
    summary = json.loads((run_dir / summary_name).read_text(encoding="utf-8"))
    if summary.get("config") != config:
        raise ValueError(f"summary/manifest config mismatch in {run_dir}")
    return {
        "path": str(run_dir),
        "schema": schema,
        "chunks": len(manifest["chunks"]),
        **grid,
        "checkpoint_step": step,
        "checkpoint_final_state_max_error": final_error,
        "status": "pass",
    }


def checksum_campaigns(data_root: Path) -> tuple[list[dict], str]:
    rows: list[dict] = []
    aggregate = hashlib.sha256()
    for campaign in CAMPAIGNS:
        campaign_root = data_root / campaign
        for path in sorted(item for item in campaign_root.rglob("*")
                           if item.is_file()):
            relative = path.relative_to(data_root).as_posix()
            digest = sha256_file(path)
            size = path.stat().st_size
            rows.append({"path": relative, "sha256": digest, "bytes": size})
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = intercept+slope*x
    residual_sum = float(np.sum((y-prediction)**2))
    total_sum = float(np.sum((y-np.mean(y))**2))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "R2": 1.0-residual_sum/total_sum,
    }


def compare_reference_to_pilot(
    reference_rows: list[dict],
    pilot_rows: list[dict],
) -> list[dict]:
    """Compare K=11 observables while retaining the metric-definition caveat."""
    pilot_by_lambda = {
        row["lambda"]: row for row in pilot_rows if row["K"] == 11
    }
    comparison = []
    for reference in reference_rows:
        pilot = pilot_by_lambda[reference["lambda"]]
        pilot_lag_per_layer = pilot["tau_relay"]/(pilot["K"]-1)
        matched = reference["matched_A2_analysis"]
        reference_lag_per_layer = matched["tau_relay"]/(pilot["K"]-1)
        comparison.append({
            "lambda": reference["lambda"],
            "initial_condition": reference["initial_condition"],
            "pilot_T1": pilot["T1_median"],
            "reference_T1": matched["T1_median"],
            "relative_T1_difference": (
                matched["T1_median"]-pilot["T1_median"]
            )/pilot["T1_median"],
            "pilot_tau_relay_per_layer": pilot_lag_per_layer,
            "reference_tau_relay_per_layer": reference_lag_per_layer,
            "relative_lag_difference": (
                reference_lag_per_layer-pilot_lag_per_layer
            )/pilot_lag_per_layer,
            "reference_A1_T1_grid": reference["T1_median"],
            "reference_A1_interlayer_lag_median": (
                reference["interlayer_lag_median"]
            ),
            "pilot_mean_saturation": pilot["mean_saturation"],
            "reference_mean_saturation": reference["mean_saturation"],
            "pilot_max_abs": pilot["max_abs"],
            "reference_max_abs": reference["max_abs"],
        })
    return comparison


def load_campaign(path: Path, schema: str) -> dict:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema") != schema or campaign.get("status") != "complete":
        raise ValueError(f"invalid campaign manifest: {path}")
    return campaign


def build_results(data_root: Path) -> tuple[dict, list[dict]]:
    main = load_campaign(
        data_root / CAMPAIGNS[0] / "campaign_manifest.json",
        "E36A-campaign-v1",
    )
    dt_lam1 = load_campaign(
        data_root / CAMPAIGNS[1] / "campaign_manifest.json",
        "E36A-campaign-v1",
    )
    a2 = load_campaign(
        data_root / CAMPAIGNS[2] / "campaign_manifest.json",
        "E36A2-campaign-v1",
    )
    reference = load_campaign(
        data_root / CAMPAIGNS[3] / "campaign_manifest.json",
        "E36A-campaign-v1",
    )
    if len(main["runs"]) != 16 or len(dt_lam1["runs"]) != 2:
        raise ValueError("unexpected A1 run count")
    if len(a2["entries"]) != 12 or len(a2["fits"]) != 3:
        raise ValueError("unexpected A2 entry/fit count")
    if len(reference["runs"]) != 6:
        raise ValueError("unexpected reference run count")
    if any(run["status"] != "complete" for run in main["runs"]):
        raise ValueError("incomplete A1 main result")
    if any(run["status"] != "complete" for run in dt_lam1["runs"]):
        raise ValueError("incomplete lambda=1 dt-check result")
    if any(
        entry["status"] != "complete" or not entry["analysis"]["valid"]
        for entry in a2["entries"]
    ):
        raise ValueError("incomplete or invalid A2 result")
    if any(run["status"] != "complete" for run in reference["runs"]):
        raise ValueError("incomplete reference result")

    run_dirs = sorted(
        path.parent for campaign in CAMPAIGNS
        for path in (data_root / campaign / "runs").rglob("manifest.json")
    )
    validations = [validate_run(path) for path in run_dirs]
    counts = {
        campaign: sum(campaign in str(path) for path in run_dirs)
        for campaign in CAMPAIGNS
    }
    expected_counts = {
        CAMPAIGNS[0]: 22,
        CAMPAIGNS[1]: 4,
        CAMPAIGNS[2]: 9,
        CAMPAIGNS[3]: 10,
    }
    if counts != expected_counts:
        raise ValueError(f"unexpected validated run counts: {counts}")

    checksum_rows, tree_sha256 = checksum_campaigns(data_root)
    a1_rows = []
    for run in main["runs"]:
        metrics = run["coarse"]
        a1_rows.append({
            "lambda": run["lambda"],
            "initial_condition": run["initial_condition"],
            "regime": metrics["regime"],
            "T1_median": metrics["T1_median"],
            "T1_cv": metrics["T1_cv"],
            "interlayer_lag_median": metrics["interlayer_lag_median"],
            "last_layer_transitions": metrics["n_transitions_last_layer"],
            "order_fraction": metrics["order_fraction"],
            "mean_saturation": float(np.mean(
                metrics["mean_saturation_by_layer"]
            )),
            "max_abs": float(np.max(metrics["max_abs_by_layer"])),
        })

    dt_rows = []
    for campaign_name, campaign in (
        (CAMPAIGNS[1], dt_lam1),
        (CAMPAIGNS[0], main),
        (CAMPAIGNS[3], reference),
    ):
        for run in campaign["runs"]:
            if run.get("dt_check") is not None:
                dt_rows.append({
                    "campaign": campaign_name,
                    "lambda": run["lambda"],
                    "initial_condition": run["initial_condition"],
                    **run["dt_check"],
                })
    if len(dt_rows) != 12 or any(row["status"] != "pass" for row in dt_rows):
        raise ValueError("dt convergence suite is incomplete or failed")
    dt_summary = {
        "checks": len(dt_rows),
        "all_pass": True,
        "all_class_match": all(row["class_match"] for row in dt_rows),
        "max_relative_T1_error": max(
            row["relative_T1_error"] for row in dt_rows
        ),
        "max_relative_final_state_error": max(
            row["relative_final_state_error"] for row in dt_rows
        ),
        "max_saturation_error": max(
            row["max_saturation_error"] for row in dt_rows
        ),
    }

    a2_rows = []
    for entry in a2["entries"]:
        analysis = entry["analysis"]
        a2_rows.append({
            "lambda": entry["lambda"],
            "K": entry["K"],
            "source": entry["source"],
            "T1_median": analysis["T1_median"],
            "T1_cv": analysis["T1_cv"],
            "tau_xcorr": analysis["xcorr"]["tau_eff"],
            "tau_relay": analysis["relay"]["tau_eff"],
            "relative_estimator_disagreement": (
                analysis["relative_estimator_disagreement"]
            ),
            "xcorr_unique_fraction": analysis["xcorr"]["unique_fraction"],
            "block_count": analysis["block_uncertainty"]["n_blocks"],
            "mean_saturation": float(np.mean(
                analysis["mean_saturation_by_layer"]
            )),
            "max_abs": float(np.max(analysis["max_abs_by_layer"])),
        })
    lag_fits = {}
    for lam in sorted({row["lambda"] for row in a2_rows}):
        rows = [row for row in a2_rows if row["lambda"] == lam]
        x = np.asarray([row["K"]-1 for row in rows], dtype=float)
        lag_fits[str(lam)] = {
            method: linear_fit(
                x, np.asarray([row[method] for row in rows], dtype=float)
            )
            for method in ("tau_xcorr", "tau_relay")
        }

    reference_rows = []
    for run in reference["runs"]:
        metrics = run["coarse"]
        run_dir = (
            data_root / CAMPAIGNS[3] / "runs"
            / run["run_id"] / "coarse"
        )
        trajectory = load_physical_trajectory(run_dir)
        matched_analysis = analyze_delay_trajectory(
            trajectory,
            P=reference["config"]["P"],
            transient_fraction=reference["config"]["transient_fraction"],
            T1_hint=metrics["T1_median"],
        )
        if not matched_analysis["valid"]:
            raise ValueError(
                f"matched A2 delay analysis failed for {run_dir}"
            )
        matched = {
            "T1_median": matched_analysis["T1_median"],
            "T1_cv": matched_analysis["T1_cv"],
            "tau_xcorr": matched_analysis["xcorr"]["tau_eff"],
            "tau_relay": matched_analysis["relay"]["tau_eff"],
            "relative_estimator_disagreement": (
                matched_analysis["relative_estimator_disagreement"]
            ),
            "xcorr_unique_fraction": (
                matched_analysis["xcorr"]["unique_fraction"]
            ),
            "block_count": matched_analysis["block_uncertainty"]["n_blocks"],
        }
        physical_wave_but_auto_go_false = bool(
            metrics["regime"] == "ordered_wave"
            and not metrics["go_A1_ordered_wave"]
        )
        reference_rows.append({
            "lambda": run["lambda"],
            "initial_condition": run["initial_condition"],
            "regime": metrics["regime"],
            "go_A1_ordered_wave": metrics["go_A1_ordered_wave"],
            "auto_go_false_due_to_transition_count": (
                physical_wave_but_auto_go_false
                and metrics["n_transitions_last_layer"] < 10
            ),
            "T1_median": metrics["T1_median"],
            "T1_cv": metrics["T1_cv"],
            "interlayer_lag_median": metrics["interlayer_lag_median"],
            "last_layer_transitions": metrics["n_transitions_last_layer"],
            "order_fraction": metrics["order_fraction"],
            "mean_saturation": float(np.mean(
                metrics["mean_saturation_by_layer"]
            )),
            "max_abs": float(np.max(metrics["max_abs_by_layer"])),
            "matched_A2_analysis": matched,
        })
    reference_comparison = compare_reference_to_pilot(
        reference_rows, a2_rows
    )
    reference_dt_rows = [
        row for row in dt_rows if row["campaign"] == CAMPAIGNS[3]
    ]
    if (
        len(reference_dt_rows) != 4
        or any(not row["class_match"] for row in reference_dt_rows)
    ):
        raise ValueError("reference dt-check suite is incomplete")
    reference_dt_summary = {
        "checks": len(reference_dt_rows),
        "all_pass": all(row["status"] == "pass" for row in reference_dt_rows),
        "max_relative_T1_error": max(
            row["relative_T1_error"] for row in reference_dt_rows
        ),
        "max_relative_final_state_error": max(
            row["relative_final_state_error"] for row in reference_dt_rows
        ),
        "max_saturation_error": max(
            row["max_saturation_error"] for row in reference_dt_rows
        ),
    }

    stationary = sorted({
        row["lambda"] for row in a1_rows if row["regime"] == "stationary"
    })
    waves = sorted({
        row["lambda"] for row in a1_rows
        if row["regime"] == "ordered_wave"
    })
    results = {
        "schema": "E36-curated-results-v1",
        "scope": {
            "N": 500,
            "P": 25,
            "beta": 20.0,
            "t0": 1.0,
            "seed": 42,
            "coarse_dt": 0.01,
            "fine_dt": 0.005,
            "new_simulations_in_postprocess": 0,
            "model": {
                "activation": "g(x)=tanh(beta*x)",
                "inner_layers": (
                    "t0*dU[k]/dt=-U[k]+J*g(U[k])+lambda*g(U[k-1]), "
                    "k=1,...,K-1"
                ),
                "closure_layer": (
                    "t0*dU[0]/dt=-U[0]+K_seq*g(U[K-1])"
                ),
            },
        },
        "verification": {
            "status": "pass",
            "campaigns": {
                name: {
                    "manifest_status": "complete",
                    "validated_run_streams": counts[name],
                }
                for name in CAMPAIGNS
            },
            "validated_run_streams": len(validations),
            "validated_chunks": sum(row["chunks"] for row in validations),
            "short_terminal_streams": [
                row["path"] for row in validations
                if row["short_terminal_interval"]
            ],
            "checkpoint_final_state_max_error": max(
                row["checkpoint_final_state_max_error"]
                for row in validations
            ),
            "checksummed_files": len(checksum_rows),
            "checksummed_bytes": sum(row["bytes"] for row in checksum_rows),
            "data_tree_sha256": tree_sha256,
            "run_details": validations,
        },
        "A1_regime_scan": {
            "rows": a1_rows,
            "stationary_lambdas": stationary,
            "ordered_wave_lambdas": waves,
            "sampled_transition_bracket": [max(stationary), min(waves)],
            "initial_condition_classification_matches": all(
                next(
                    other["regime"] for other in a1_rows
                    if other["lambda"] == row["lambda"]
                    and other["initial_condition"] != row["initial_condition"]
                ) == row["regime"]
                for row in a1_rows
            ),
        },
        "dt_convergence": {
            "comparison": "dt=0.01 versus dt=0.005",
            "rows": dt_rows,
            "summary": dt_summary,
        },
        "A2_delay_scaling": {
            "rows": a2_rows,
            "T1_fits_from_manifest": a2["fits"],
            "derived_effective_lag_fits": lag_fits,
        },
        "reference_N2000_P100_K11": {
            "scope": {
                "N": 2000,
                "P": 100,
                "P_over_N": 0.05,
                "K": 11,
                "lambdas": [1.0, 2.0, 4.0],
                "initial_conditions": ["coherent", "perturbed"],
                "seed": 42,
                "coarse_dt": 0.01,
                "fine_dt": 0.005,
            },
            "rows": reference_rows,
            "dt_convergence": reference_dt_summary,
            "comparison_to_N500_P25_K11": reference_comparison,
            "comparison_caveat": (
                "N and P both change by a factor four at fixed P/N=0.05, "
                "and the generated pattern realization also changes. The "
                "differences cannot be attributed to system size alone. "
                "Pilot/reference T1 and lag comparisons use the same A2 "
                "interpolated-crossing postprocessor on both data sets."
            ),
            "lambda1_auto_go_note": (
                "lambda=1 is classified ordered_wave with order_fraction=1. "
                "The automatic GO flag is false only because 8 last-layer "
                "transitions are below its hard threshold of 10."
            ),
        },
        "interpretation": {
            "regime_transition": (
                "On the sampled grid, both initial conditions are stationary "
                "through lambda=0.5 and form ordered waves from lambda=1."
            ),
            "K_scaling": (
                "T1 and both effective-lag estimators are nearly linear in K-1. "
                "The inferred delay quantum depends on lambda."
            ),
            "model_caveat": (
                "The finite-dimensional K-layer Markovian chain contains "
                "nonlinear intermediate states. Eliminating them gives a "
                "history-dependent, generally distributed/state-dependent "
                "memory, not an exact discrete-delay DDE. An Erlang/gamma kernel "
                "would require additional linearization/filter assumptions."
            ),
            "statistical_caveat": (
                "This is seed-42 exploration. Block intervals are descriptive "
                "within-trajectory uncertainty, not independent-seed confidence "
                "intervals."
            ),
        },
    }
    return json_safe(results), checksum_rows


def write_figures(results: dict, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })
    a1 = results["A1_regime_scan"]["rows"]
    dt_rows = results["dt_convergence"]["rows"]
    a2 = results["A2_delay_scaling"]["rows"]
    reference = results["reference_N2000_P100_K11"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ic, marker in (("coherent", "o"), ("perturbed", "s")):
        rows = [row for row in a1 if row["initial_condition"] == ic]
        wave = [row for row in rows if row["T1_median"] is not None]
        axes[0].plot(
            [row["lambda"] for row in wave],
            [row["T1_median"] for row in wave],
            marker=marker,
            label=ic,
        )
        stationary = [row for row in rows if row["T1_median"] is None]
        axes[0].scatter(
            [row["lambda"] for row in stationary],
            np.zeros(len(stationary)),
            marker=marker,
            facecolors="none",
        )
    axes[0].axvspan(0.5, 1.0, color="#f0a202", alpha=0.15)
    axes[0].set(
        xlabel=r"$\lambda$",
        ylabel=r"$T_1$ (cercles pleins) ; stationnaire à 0",
        title="A1 — transition de régime (K=6)",
    )
    axes[0].legend(frameon=False)
    labels = [
        (
            ("R" if "reference" in row["campaign"] else "P")
            + f" λ={row['lambda']:g}\n{row['initial_condition'][0]}"
        )
        for row in dt_rows
    ]
    errors = [row["relative_final_state_error"] for row in dt_rows]
    bar_colors = [
        "#e6550d" if "reference" in row["campaign"] else "#386cb0"
        for row in dt_rows
    ]
    axes[1].bar(np.arange(len(errors)), errors, color=bar_colors)
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(errors)), labels)
    axes[1].tick_params(axis="x", labelrotation=45, labelsize=7)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")
    axes[1].set(
        ylabel="erreur relative état final",
        title=r"Convergence $\Delta t$: 0.01 vs 0.005 (P=pilote, R=réf.)",
    )
    fig.tight_layout()
    regime_path = figure_dir / "E36_regimes_dt.png"
    fig.savefig(regime_path, bbox_inches="tight")
    plt.close(fig)

    colors = {1.0: "#d95f02", 2.0: "#1b9e77", 4.0: "#7570b3"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    for lam, color in colors.items():
        rows = [row for row in a2 if row["lambda"] == lam]
        K = np.asarray([row["K"] for row in rows])
        x = K-1
        T1 = np.asarray([row["T1_median"] for row in rows])
        fit = results["A2_delay_scaling"]["T1_fits_from_manifest"][str(lam)]
        axes[0, 0].plot(K, T1, "o", color=color, label=f"λ={lam:g}")
        axes[0, 0].plot(K, fit["intercept"]+fit["slope"]*x, color=color)
        axes[0, 1].plot(
            K, [row["tau_xcorr"] for row in rows],
            "o-", color=color, label=f"xcorr λ={lam:g}",
        )
        axes[0, 1].plot(
            K, [row["tau_relay"] for row in rows],
            "x--", color=color, label=f"relais λ={lam:g}",
        )
        axes[1, 0].plot(
            K, [row["mean_saturation"] for row in rows],
            "o-", color=color, label=f"λ={lam:g}",
        )
        axes[1, 1].plot(
            K, [row["max_abs"] for row in rows],
            "o-", color=color, label=f"λ={lam:g}",
        )
    axes[0, 0].set(
        xlabel="K", ylabel=r"$T_1$",
        title=r"A2 — $T_1=a+b(K-1)$",
    )
    axes[0, 1].set(
        xlabel="K", ylabel=r"$\tau_{\mathrm{eff}}$",
        title="Deux estimateurs indépendants",
    )
    axes[1, 0].set(
        xlabel="K", ylabel="fraction saturée moyenne",
        title=r"$|\beta U|>10$",
    )
    axes[1, 1].set(
        xlabel="K", ylabel=r"$\max |U|$",
        title="Amplitude maximale",
    )
    for axis in axes.flat:
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    scaling_path = figure_dir / "E36_delay_scaling.png"
    fig.savefig(scaling_path, bbox_inches="tight")
    plt.close(fig)

    comparison = reference["comparison_to_N500_P25_K11"]
    pilot = {
        row["lambda"]: row for row in comparison
        if row["initial_condition"] == "coherent"
    }
    ref_by_ic = {
        ic: [
            row for row in comparison if row["initial_condition"] == ic
        ]
        for ic in ("coherent", "perturbed")
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    lambda_values = sorted(pilot)
    pilot_rows = [pilot[lam] for lam in lambda_values]
    panels = (
        (
            "pilot_T1", "reference_T1", r"$T_1$",
            "Période de retour",
        ),
        (
            "pilot_tau_relay_per_layer",
            "reference_tau_relay_per_layer",
            "retard par couche",
            "Retard relais A2 par couche",
        ),
        (
            "pilot_mean_saturation", "reference_mean_saturation",
            "fraction saturée moyenne", r"$|\beta U|>10$",
        ),
        (
            "pilot_max_abs", "reference_max_abs",
            r"$\max |U|$", "Amplitude maximale",
        ),
    )
    reference_style = {
        "coherent": ("#1f78b4", "o"),
        "perturbed": ("#e31a1c", "s"),
    }
    for axis, (pilot_key, reference_key, ylabel, title) in zip(
        axes.flat, panels
    ):
        axis.plot(
            lambda_values,
            [row[pilot_key] for row in pilot_rows],
            "kD--",
            label="pilote 500/25 coh.",
        )
        for ic, rows in ref_by_ic.items():
            color, marker = reference_style[ic]
            axis.plot(
                [row["lambda"] for row in rows],
                [row[reference_key] for row in rows],
                marker=marker,
                color=color,
                label=f"référence 2000/100 {ic}",
            )
        axis.set(xlabel=r"$\lambda$", ylabel=ylabel, title=title)
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    reference_path = figure_dir / "E36_reference_comparison.png"
    fig.savefig(reference_path, bbox_inches="tight")
    plt.close(fig)
    return [regime_path, scaling_path, reference_path]


def markdown_report(results: dict) -> str:
    transition = results["A1_regime_scan"]
    dt_summary = results["dt_convergence"]["summary"]
    a2 = results["A2_delay_scaling"]
    reference = results["reference_N2000_P100_K11"]
    verification = results["verification"]
    lines = [
        "# E36 — Retard effectif dans la chaîne de Hopfield non réciproque",
        "",
        "## Résultat en bref",
        "",
        "Les quatre campagnes seed 42 sont complètes et leurs flux numériques ont "
        "été validés jusqu’aux chunks, checkpoints et états finaux. Sur la "
        "grille échantillonnée, la chaîne K=6 est stationnaire pour "
        r"$\lambda\leq0.5$ et devient une onde ordonnée pour "
        r"$\lambda\geq1$ : le seuil est donc seulement encadré par "
        r"$0.5<\lambda_c\leq1$. Les deux conditions initiales donnent la même "
        "classification à chaque valeur de λ.",
        "",
        "Dans le régime ondulatoire, la période de retour et les deux "
        "estimateurs indépendants du retard croissent presque linéairement "
        "avec K−1. Cette observation **étaye l’interprétation d’un retard "
        "effectif**, mais pas l’identité avec une DDE à retard discret.",
        "",
        "![Transition de régime et convergence temporelle]"
        "(figures/E36_regimes_dt.png)",
        "",
        "![Échelle avec K, retards et amplitudes]"
        "(figures/E36_delay_scaling.png)",
        "",
        "![Comparaison pilote et campagne de référence]"
        "(figures/E36_reference_comparison.png)",
        "",
        "## Périmètre numérique",
        "",
        "- Pilote : N=500, P=25, β=20, t₀=1, seed=42 ; intégrateur RK4.",
        "- A1 : K=6, λ={0, 0.125, 0.25, 0.5, 1, 2, 4, 8}, "
        "conditions initiales cohérente et perturbée, t=200.",
        "- A2 : K={4, 6, 8, 11}, λ={1, 2, 4} ; les K=6 sont réutilisés "
        "depuis A1.",
        "- Pas principal Δt=0.01 ; contrôles à Δt=0.005 pour λ={1,2,4,8}.",
        "- Référence : N=2000, P=100, K=11, λ={1,2,4}, mêmes deux "
        "initialisations et même rapport P/N=0.05.",
        "- Le post-traitement n’a lancé aucune simulation.",
        "",
        r"Le modèle intégré utilise $g(x)=\tanh(\beta x)$. Pour "
        r"$k=1,\ldots,K-1$, "
        r"$t_0\dot U_k=-U_k+Jg(U_k)+\lambda g(U_{k-1})$, tandis que la "
        r"fermeture est $t_0\dot U_0=-U_0+K_{\rm seq}g(U_{K-1})$. "
        r"$J$ est la matrice hebbienne et $K_{\rm seq}$ réalise le décalage "
        "cyclique des motifs.",
        "",
        "## A1 — transition de régime et robustesse à l’initialisation",
        "",
        "| λ | cohérente | perturbée | T₁ coh. | T₁ pert. | retard "
        r"intercouche coh. | saturation moyenne coh. | max \|U\| coh. |",
        "|---:|:---|:---|---:|---:|---:|---:|---:|",
    ]
    by_key = {
        (row["lambda"], row["initial_condition"]): row
        for row in transition["rows"]
    }
    for lam in sorted({row["lambda"] for row in transition["rows"]}):
        coherent = by_key[(lam, "coherent")]
        perturbed = by_key[(lam, "perturbed")]
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {lam:g} | {coherent['regime']} | {perturbed['regime']} | "
            f"{fmt(coherent['T1_median'])} | {fmt(perturbed['T1_median'])} | "
            f"{fmt(coherent['interlayer_lag_median'])} | "
            f"{coherent['mean_saturation']:.4f} | "
            f"{coherent['max_abs']:.4f} |"
        )
    lines += [
        "",
        "Le passage observé entre 0.5 et 1 n’est pas une estimation précise de "
        "λc : aucune valeur intermédiaire n’a été simulée. La coïncidence des "
        "deux initialisations exclut ici une forte dépendance au bassin initial, "
        "mais une seule réalisation des motifs est disponible.",
        "",
        "La fraction dite « saturée » mesure |βU|>10. Elle reste élevée dans les "
        "états stationnaires et ne constitue donc pas, seule, un diagnostic "
        "d’oscillation. L’amplitude maximale croît de 1.58 à λ=0 jusqu’à 10.03 "
        "à λ=8 ; aucune divergence numérique n’est observée.",
        "",
        "## Convergence temporelle",
        "",
        f"Les {dt_summary['checks']} comparaisons Δt=0.01/0.005 passent : "
        f"classification identique, erreur relative maximale sur T₁ "
        f"{dt_summary['max_relative_T1_error']:.3g}, sur l’état final "
        f"{dt_summary['max_relative_final_state_error']:.3e}, et écart maximal "
        f"de saturation {dt_summary['max_saturation_error']:.3e}. Cela valide "
        "le pas temporel pour les observables rapportées aux λ testés. Parmi "
        "elles, quatre contrôles appartiennent à la campagne N=2000/P=100.",
        "",
        "## A2 — période et retard en fonction de K",
        "",
        "| λ | K | T₁ | CV(T₁) | τ xcorr | τ relais | désaccord | "
        r"saturation | max \|U\| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in a2["rows"]:
        lines.append(
            f"| {row['lambda']:g} | {row['K']} | {row['T1_median']:.4f} | "
            f"{row['T1_cv']:.4f} | {row['tau_xcorr']:.4f} | "
            f"{row['tau_relay']:.4f} | "
            f"{100*row['relative_estimator_disagreement']:.2f}% | "
            f"{row['mean_saturation']:.4f} | {row['max_abs']:.4f} |"
        )
    lines += [
        "",
        "| λ | a | b par couche | R² | pic xcorr unique min. | verdict |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for lam in ("1.0", "2.0", "4.0"):
        fit = a2["T1_fits_from_manifest"][lam]
        lines.append(
            f"| {float(lam):g} | {fit['intercept']:.4f} | "
            f"{fit['slope']:.4f} | {fit['R2']:.6f} | "
            f"{fit['min_xcorr_unique_fraction']:.2f} | {fit['verdict']} |"
        )
    lines += [
        "",
        "Les pics de corrélation sont uniques pour toutes les couches et le "
        "désaccord entre estimateurs reste ≤1.439%. La linéarité avec K est "
        "très forte (R²≥0.9990), mais le quantum effectif dépend de λ : "
        "b=1.710 à λ=1, 1.192 à λ=2 et 0.950 à λ=4. Le cas λ=1 est donc "
        "GO-limited : il soutient la mémoire distribuée linéaire en profondeur, "
        "pas un retard unitaire universel.",
        "",
        "À λ fixé, saturation et amplitude augmentent modérément avec K. À K=6, "
        "l’augmentation de λ raccourcit T₁ (10.891, 6.938, 5.403) et le retard "
        "effectif, tandis que l’amplitude maximale croît (3.913, 4.405, 6.226).",
        "",
        "## Campagne de référence — N=2000, P=100, K=11",
        "",
        "| λ | initialisation | régime | GO automatique | transitions couche "
        "finale | T₁ | CV(T₁) | retard intercouche | saturation | max \\|U\\| |",
        "|---:|:---|:---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reference["rows"]:
        lines.append(
            f"| {row['lambda']:g} | {row['initial_condition']} | "
            f"{row['regime']} | {str(row['go_A1_ordered_wave']).lower()} | "
            f"{row['last_layer_transitions']} | {row['T1_median']:.4f} | "
            f"{row['T1_cv']:.4f} | {row['interlayer_lag_median']:.4f} | "
            f"{row['mean_saturation']:.4f} | {row['max_abs']:.4f} |"
        )
    reference_dt = reference["dt_convergence"]
    comparisons = {
        (row["lambda"], row["initial_condition"]): row
        for row in reference["comparison_to_N500_P25_K11"]
    }
    lines += [
        "",
        "### Réanalyse homogène avec les estimateurs A2",
        "",
        "Cette réanalyse utilise uniquement les overlaps enregistrés et applique "
        "au pilote et à la référence les mêmes croisements interpolés et la même "
        "corrélation vectorielle ; elle ne réintègre pas le modèle.",
        "",
        "| λ | initialisation | T₁ A2 | τ xcorr | τ relais | désaccord | "
        "pics uniques | blocs |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reference["rows"]:
        matched = row["matched_A2_analysis"]
        lines.append(
            f"| {row['lambda']:g} | {row['initial_condition']} | "
            f"{matched['T1_median']:.4f} | {matched['tau_xcorr']:.4f} | "
            f"{matched['tau_relay']:.4f} | "
            f"{100*matched['relative_estimator_disagreement']:.2f}% | "
            f"{matched['xcorr_unique_fraction']:.2f} | "
            f"{matched['block_count']} |"
        )
    lines += [
        "",
        "À λ=1, `regime=ordered_wave`, la fraction d’ordre vaut 1 et les deux "
        "initialisations donnent T₁=18.4. Le drapeau "
        "`go_A1_ordered_wave=false` n’est **pas un échec physique** : la fenêtre "
        "contient 8 transitions de la dernière couche, sous le seuil automatique "
        "fixe de 10. À λ=2 et 4, 12 et 15 transitions rendent ce même drapeau "
        "vrai.",
        "",
        f"Les quatre contrôles temporels de référence passent ; erreur maximale "
        f"sur T₁ {reference_dt['max_relative_T1_error']:.3g}, état final "
        f"{reference_dt['max_relative_final_state_error']:.3e}, saturation "
        f"{reference_dt['max_saturation_error']:.3e}.",
        "",
        "### Comparaison au pilote N=500/P=25/K=11",
        "",
        "| λ | T₁ pilote | T₁ réf. coh. | T₁ réf. pert. | écart réf. coh. | "
        "τ relais/couche pilote | τ relais/couche réf. coh. |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lam in (1.0, 2.0, 4.0):
        coherent = comparisons[(lam, "coherent")]
        perturbed = comparisons[(lam, "perturbed")]
        lines.append(
            f"| {lam:g} | {coherent['pilot_T1']:.4f} | "
            f"{coherent['reference_T1']:.4f} | "
            f"{perturbed['reference_T1']:.4f} | "
            f"{100*coherent['relative_T1_difference']:+.2f}% | "
            f"{coherent['pilot_tau_relay_per_layer']:.4f} | "
            f"{coherent['reference_tau_relay_per_layer']:.4f} |"
        )
    coherent_differences = [
        comparisons[(lam, "coherent")]["relative_T1_difference"]
        for lam in (1.0, 2.0, 4.0)
    ]
    difference_text = ", ".join(
        f"{100*value:+.2f}%" for value in coherent_differences
    )
    lines += [
        "",
        f"À estimateur A2 identique, les périodes de référence cohérentes "
        f"diffèrent du pilote de {difference_text} pour λ=1, 2 et 4. Les pics "
        "xcorr sont tous uniques et le désaccord entre les deux estimateurs de "
        "référence reste inférieur à 0.74%. Pour λ=1 et 2, la fenêtre ne fournit "
        "que deux blocs longs : aucune prétention d’intervalle de confiance par "
        "blocs n’est faite.",
        "",
        "Cette comparaison ne constitue pas une étude de taille à paramètre "
        "aléatoire fixé. N et P augmentent ensemble d’un facteur quatre à "
        "P/N=0.05 constant, mais la forme du tableau aléatoire change aussi avec "
        "ses dimensions malgré la seed identique. Les écarts mélangent donc "
        "effets de taille, de charge finie et de réalisation des motifs.",
        "",
        "## Vérification des artefacts",
        "",
        f"- {verification['validated_run_streams']} flux validés, "
        f"{verification['validated_chunks']} chunks NPZ vérifiés.",
        f"- {verification['checksummed_files']} fichiers, "
        f"{verification['checksummed_bytes']} octets couverts par SHA-256.",
        f"- Empreinte agrégée : `{verification['data_tree_sha256']}`.",
        "- États finaux et checkpoints identiques bit à bit ; métadonnées, "
        "dimensions, finitude et chronologie contrôlées.",
        "- Une seule grille comporte un dernier intervalle raccourci : λ=1, "
        "K=11 (0.01 après une cadence 0.05). Ce dernier callback est exclu des "
        "estimateurs ; aucune irrégularité intérieure n’est présente.",
        "- Empreintes détaillées : `E36_CHECKSUMS.sha256`.",
        "",
        "## Limites et conclusion",
        "",
        "La chaîne à K couches est markovienne dans son espace d’état étendu et "
        "contient des dynamiques intermédiaires non linéaires. Leur élimination "
        "formelle produit une dépendance à l’histoire généralement distribuée "
        "et dépendante de l’état, pas une DDE à retard ponctuel exacte. Une "
        "forme Erlang/gamma ne serait justifiée qu’après des hypothèses "
        "supplémentaires de linéarisation et de filtres identiques. Les résultats "
        "démontrent donc : (i) une "
        "transition stationnaire–onde ordonnée encadrée sur la grille testée, "
        "(ii) un retard effectif robuste à deux estimateurs, (iii) une croissance "
        "quasi linéaire de période et retard avec K, et (iv) une dépendance "
        "substantielle du quantum de retard à λ.",
        "",
        "Les intervalles par blocs sont descriptifs et corrélés au sein d’une "
        "même trajectoire. Avant une revendication statistique générale, les "
        "résultats déterminants doivent être reproduits sur plusieurs seeds.",
        "",
        "Données structurées : `E36_RESULTS.json`.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(results_root: Path, results: dict,
                  checksums: list[dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "E36_RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    checksum_lines = [
        f"{row['sha256']}  {row['path']}" for row in checksums
    ]
    (results_root / "E36_CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines)+"\n",
        encoding="utf-8",
    )
    (results_root / "E36_RESULTS.md").write_text(
        markdown_report(results),
        encoding="utf-8",
    )
    write_figures(results, results_root / "figures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    data_root = (
        args.data_root.resolve() if args.data_root is not None
        else repository / "results/7_tau_implicite/data"
    )
    output_root = (
        args.output_root.resolve() if args.output_root is not None
        else repository / "results/7_tau_implicite"
    )
    results, checksums = build_results(data_root)
    if not args.verify_only:
        write_outputs(output_root, results, checksums)
    print(json.dumps({
        "status": results["verification"]["status"],
        "validated_run_streams": (
            results["verification"]["validated_run_streams"]
        ),
        "validated_chunks": results["verification"]["validated_chunks"],
        "checksummed_files": results["verification"]["checksummed_files"],
        "data_tree_sha256": results["verification"]["data_tree_sha256"],
        "outputs_written": not args.verify_only,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
