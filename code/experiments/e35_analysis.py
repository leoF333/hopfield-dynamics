"""Rigorous post-processing of the selected E35 T1 and V0 production runs.

This script performs no DDE integration.  It verifies each run bundle, computes
descriptive factorial contrasts for T1, and applies a paired transition-block
bootstrap to the final synthetic-video V0 result.
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
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "e35_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from e35_runtime import atomic_json, json_safe, utc_now


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY / "results" / "8_mhn_video"
T1_RUNS = (
    "E35b_production_20260724T000314Z_35125",  # lambda=0.9
    "E35b_production_20260724T000357Z_35384",  # lambda=0.5
    "E35b_production_20260724T000435Z_35601",  # lambda=0.7
)
V0_LINEAGE = (
    "E35c_production_20260724T000531Z_35879",  # JH_KH only
    "E35c_production_20260724T000613Z_36011",  # then JP_KH
    "E35c_production_20260724T000634Z_36098",  # then JH_KP
    "E35c_production_20260724T000645Z_36158",  # finally JP_KP
)
V0_FINAL = V0_LINEAGE[-1]
ARCHITECTURES = ("JH_KH", "JP_KH", "JH_KP", "JP_KP")
BASELINES = ("B0_HOLD", "B1_LINEAR", "B2_MOTION")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL entry {path}:{line_number}")
        values.append(value)
    return values


def json_scalar(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"expected scalar JSON field, got shape {value.shape}")
    return json.loads(str(value.item()))


def verify_run_bundle(root: Path, run_id: str) -> dict[str, Any]:
    """Cross-check manifest, data, checkpoint, events and heartbeat."""

    paths = {
        "manifest": root / "manifests" / f"{run_id}.json",
        "data": root / "data" / f"{run_id}.npz",
        "checkpoint": root / "checkpoints" / f"{run_id}.npz",
        "log": root / "logs" / f"{run_id}.jsonl",
        "heartbeat": root / "logs" / f"{run_id}_heartbeat.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_id}: missing artifacts {missing}")

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{run_id}: manifest has no config object")
    with np.load(paths["data"], allow_pickle=False) as data:
        data_config = json_scalar(data["config_json"])
        data_fields = sorted(data.files)
    if data_config != config:
        raise ValueError(f"{run_id}: data config differs from manifest")

    with np.load(paths["checkpoint"], allow_pickle=False) as checkpoint:
        checkpoint_config = json_scalar(checkpoint["config_json"])
        hist = np.asarray(checkpoint["hist"])
        dhist = np.asarray(checkpoint["dhist"])
        completed_steps = int(checkpoint["completed_steps"])
        simulated_time = float(checkpoint["simulated_time"])
        checkpoint_architecture = str(checkpoint["architecture"].item())
    if checkpoint_config != config:
        raise ValueError(f"{run_id}: checkpoint config differs from manifest")
    if hist.shape != dhist.shape or hist.ndim != 2:
        raise ValueError(f"{run_id}: invalid hist/dhist shapes")
    p = int(config["p"])
    expected_history = int(round(float(config["tau"]) / float(config["dt"]))) + 1
    if hist.shape != (expected_history, p):
        raise ValueError(
            f"{run_id}: history shape {hist.shape}, expected {(expected_history, p)}"
        )
    if not np.isfinite(hist).all() or not np.isfinite(dhist).all():
        raise ValueError(f"{run_id}: non-finite checkpoint history")
    expected_steps = int(round(float(config["t_total"]) / float(config["dt"])))
    if completed_steps != expected_steps:
        raise ValueError(
            f"{run_id}: checkpoint steps {completed_steps}, expected {expected_steps}"
        )
    if not np.isclose(simulated_time, float(config["t_total"]), atol=1e-12):
        raise ValueError(f"{run_id}: checkpoint simulated time mismatch")

    events = read_jsonl(paths["log"])
    event_names = [event.get("event") for event in events]
    if not events or event_names[0] != "run_started":
        raise ValueError(f"{run_id}: log does not start with run_started")
    if event_names[-1] != "run_finished":
        raise ValueError(f"{run_id}: log does not end with run_finished")
    if "run_failed" in event_names:
        raise ValueError(f"{run_id}: run_failed present in log")
    if "data_saved" not in event_names:
        raise ValueError(f"{run_id}: data_saved absent from log")

    heartbeats = read_jsonl(paths["heartbeat"])
    if not heartbeats or heartbeats[-1].get("stage") != "finished":
        raise ValueError(f"{run_id}: final heartbeat is not finished")
    if int(heartbeats[-1].get("nan_count", -1)) != 0:
        raise ValueError(f"{run_id}: final heartbeat reports non-finite values")
    if int(heartbeats[-1].get("pid", -1)) != int(manifest.get("pid", -2)):
        raise ValueError(f"{run_id}: heartbeat PID differs from manifest")

    return {
        "run_id": run_id,
        "experiment": config["experiment"],
        "git_commit": manifest.get("git_commit", "unavailable"),
        "config": config,
        "checkpoint_architecture": checkpoint_architecture,
        "checkpoint_shape": list(hist.shape),
        "event_count": len(events),
        "heartbeat_count": len(heartbeats),
        "data_fields": data_fields,
        "sha256": {name: sha256(path) for name, path in paths.items()},
        "verified": True,
    }


def load_t1_rows(root: Path, run_ids: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        data_path = root / "data" / f"{run_id}.npz"
        with np.load(data_path, allow_pickle=False) as data:
            config = json_scalar(data["config_json"])
            run_rows = [json.loads(str(value)) for value in data["rows_json"]]
            if not np.isfinite(data["final_coefficients"]).all():
                raise ValueError(f"{run_id}: non-finite final coefficients")
            if not np.isfinite(data["final_physical_overlaps"]).all():
                raise ValueError(f"{run_id}: non-finite final physical overlaps")
        if len(run_rows) != 8:
            raise ValueError(f"{run_id}: expected 8 factorial rows")
        for row in run_rows:
            row = dict(row)
            row["lam"] = float(config["lam"])
            row["run_id"] = run_id
            rows.append(row)
    if len(rows) != 24:
        raise ValueError("selected T1 bundle must contain exactly 24 rows")
    return rows


def factorial_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive period contrasts; no inferential claim from one seed."""

    output = []
    lambdas = sorted({float(row["lam"]) for row in rows})
    datasets = sorted({str(row["dataset"]) for row in rows})
    for lam in lambdas:
        for dataset in datasets:
            cell = {
                row["architecture"]: row
                for row in rows
                if float(row["lam"]) == lam and row["dataset"] == dataset
            }
            if set(cell) != set(ARCHITECTURES):
                raise ValueError(f"incomplete factorial at lambda={lam}, {dataset}")
            periods = {
                architecture: float(cell[architecture]["period_mean"])
                for architecture in ARCHITECTURES
            }
            reference = periods["JH_KH"]
            output.append(
                {
                    "lam": lam,
                    "dataset": dataset,
                    "periods": periods,
                    "delta_seconds_vs_JH_KH": {
                        architecture: periods[architecture] - reference
                        for architecture in ARCHITECTURES[1:]
                    },
                    "delta_percent_vs_JH_KH": {
                        architecture: 100.0
                        * (periods[architecture] - reference)
                        / reference
                        for architecture in ARCHITECTURES[1:]
                    },
                    "J_effect_at_KH": periods["JP_KH"] - periods["JH_KH"],
                    "J_effect_at_KP": periods["JP_KP"] - periods["JH_KP"],
                    "K_effect_at_JH": periods["JH_KP"] - periods["JH_KH"],
                    "K_effect_at_JP": periods["JP_KP"] - periods["JP_KH"],
                    "interaction_seconds": (
                        periods["JP_KP"]
                        - periods["JP_KH"]
                        - periods["JH_KP"]
                        + periods["JH_KH"]
                    ),
                }
            )
    return output


def paired_block_bootstrap(
    difference: np.ndarray,
    block_ids: np.ndarray,
    *,
    n_bootstrap: int = 100_000,
    seed: int = 35042,
    favorable: str,
) -> dict[str, Any]:
    """Percentile bootstrap of a paired mean, resampling transition blocks."""

    values = np.asarray(difference, dtype=np.float64)
    blocks = np.asarray(block_ids)
    if values.ndim != 1 or blocks.shape != values.shape:
        raise ValueError("difference and block_ids must be aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("difference contains NaN or infinity")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    unique = np.unique(blocks)
    block_means = np.array([np.mean(values[blocks == block]) for block in unique])
    rng = np.random.default_rng(seed)
    selections = rng.integers(
        0, len(unique), size=(n_bootstrap, len(unique)), endpoint=False
    )
    replicates = np.mean(block_means[selections], axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975])
    if favorable == "negative":
        block_favorable = block_means < 0
        bootstrap_favorable = replicates < 0
    elif favorable == "positive":
        block_favorable = block_means > 0
        bootstrap_favorable = replicates > 0
    else:
        raise ValueError("favorable must be 'negative' or 'positive'")
    return {
        "estimate": float(np.mean(block_means)),
        "ci95_percentile": [float(low), float(high)],
        "n_blocks": int(len(unique)),
        "frames_per_block": {
            str(block): int(np.count_nonzero(blocks == block)) for block in unique
        },
        "block_means": block_means.tolist(),
        "favorable_block_count": int(np.count_nonzero(block_favorable)),
        "bootstrap_favorable_fraction": float(np.mean(bootstrap_favorable)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "favorable_direction": favorable,
    }


def recompute_mse_psnr(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    difference = np.asarray(prediction) - np.asarray(truth)
    mse = np.mean(difference * difference, axis=(1, 2))
    psnr = np.where(mse == 0.0, np.inf, -10.0 * np.log10(mse))
    return mse, psnr


def analyze_v0(
    root: Path,
    run_id: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    path = root / "data" / f"{run_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        config = json_scalar(data["config_json"])
        architectures = [str(value) for value in data["architecture"]]
        if architectures != ["JH_KH", "JP_KP"]:
            raise ValueError(f"unexpected final V0 arms {architectures}")
        candidate_index = architectures.index("JP_KP")
        truth = np.array(data["hidden_truth"])
        hidden_indices = np.array(data["hidden_indices"])
        candidate_prediction = np.array(data["dynamic_predictions"][candidate_index])
        candidate_valid = np.array(data["dynamic_valid"][candidate_index], dtype=bool)
        stored_dynamic_mse = np.array(data["dynamic_mse"][candidate_index])
        stored_dynamic_psnr = np.array(data["dynamic_psnr"][candidate_index])
        stored_dynamic_ssim = np.array(data["dynamic_ssim"][candidate_index])
        cycle_rows = [json.loads(str(value)) for value in data["cycle_json"]]
        metric_rows = [json.loads(str(value)) for value in data["metric_json"]]
        baseline_prediction = {
            baseline: np.array(data[baseline]) for baseline in BASELINES
        }
        stored_baseline = {
            baseline: {
                "mse": np.array(data[f"{baseline}_mse"]),
                "psnr": np.array(data[f"{baseline}_psnr"]),
                "ssim": np.array(data[f"{baseline}_ssim"]),
            }
            for baseline in BASELINES
        }
        seam = json_scalar(data["seam_json"])
        svd = json_scalar(data["svd_json"])

    if not np.all(candidate_valid) or len(truth) != 48:
        raise ValueError("final JP_KP V0 must have 48 valid hidden frames")
    dynamic_mse, dynamic_psnr = recompute_mse_psnr(
        truth, candidate_prediction
    )
    np.testing.assert_allclose(dynamic_mse, stored_dynamic_mse, rtol=1e-13)
    np.testing.assert_allclose(dynamic_psnr, stored_dynamic_psnr, rtol=1e-13)
    for baseline in BASELINES:
        mse, psnr = recompute_mse_psnr(truth, baseline_prediction[baseline])
        np.testing.assert_allclose(
            mse, stored_baseline[baseline]["mse"], rtol=1e-13
        )
        np.testing.assert_allclose(
            psnr, stored_baseline[baseline]["psnr"], rtol=1e-13
        )

    k = int(config["k"])
    block_ids = hidden_indices // k
    unique, counts = np.unique(block_ids, return_counts=True)
    if len(unique) != int(config["p"]) or not np.all(counts == k - 1):
        raise ValueError("hidden frames do not form equal transition blocks")

    comparisons = {}
    for baseline_index, baseline in enumerate(BASELINES):
        comparisons[baseline] = {
            "delta_mse_dynamic_minus_baseline": paired_block_bootstrap(
                dynamic_mse - stored_baseline[baseline]["mse"],
                block_ids,
                n_bootstrap=n_bootstrap,
                seed=seed + 10 * baseline_index,
                favorable="negative",
            ),
            "delta_psnr_dynamic_minus_baseline_db": paired_block_bootstrap(
                dynamic_psnr - stored_baseline[baseline]["psnr"],
                block_ids,
                n_bootstrap=n_bootstrap,
                seed=seed + 10 * baseline_index + 1,
                favorable="positive",
            ),
        }

    quality = {
        "JP_KP": {
            "mean_mse": float(np.mean(dynamic_mse)),
            "mean_psnr": float(np.mean(dynamic_psnr)),
            "mean_ssim": float(np.mean(stored_dynamic_ssim)),
        }
    }
    for baseline in BASELINES:
        quality[baseline] = {
            "mean_mse": float(np.mean(stored_baseline[baseline]["mse"])),
            "mean_psnr": float(np.mean(stored_baseline[baseline]["psnr"])),
            "mean_ssim": float(np.mean(stored_baseline[baseline]["ssim"])),
        }

    cycle_by_architecture = {
        row["architecture"]: row for row in cycle_rows
    }
    return {
        "run_id": run_id,
        "config": config,
        "cycle": cycle_by_architecture,
        "stored_metric_rows": metric_rows,
        "quality": quality,
        "paired_transition_block_bootstrap": comparisons,
        "block_definition": {
            "n_transition_blocks": int(len(unique)),
            "hidden_frames_per_transition": int(k - 1),
            "block_ids": block_ids.tolist(),
        },
        "seam": seam,
        "svd": svd,
        "decision": {
            "mechanism_gate": "GO_EXPLORATORY_POST_DIAGNOSTIC",
            "quality_gate": "NO_GO",
            "escalate_to_moving_mnist_or_real_video": False,
            "reason": (
                "JP_KP restores a valid ordered cycle, but its interpolation "
                "is worse than B1 and B2; candidate selection followed prior "
                "diagnostic failures and is not confirmatory."
            ),
        },
    }


def make_figure(
    factorial: list[dict[str, Any]],
    v0: dict[str, Any],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    colors = {"JP_KH": "#2b6cb0", "JH_KP": "#d97706", "JP_KP": "#7c3aed"}
    styles = {"iid": "--", "correlated_ring": "-"}
    for dataset in ("iid", "correlated_ring"):
        subset = sorted(
            [row for row in factorial if row["dataset"] == dataset],
            key=lambda row: row["lam"],
        )
        for architecture in ARCHITECTURES[1:]:
            axes[0].plot(
                [row["lam"] for row in subset],
                [
                    row["delta_percent_vs_JH_KH"][architecture]
                    for row in subset
                ],
                marker="o",
                color=colors[architecture],
                linestyle=styles[dataset],
                label=f"{architecture}, {dataset}",
            )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("period difference vs JH_KH (%)")
    axes[0].set_title("T1: descriptive timing contrasts (seed 42)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.25)

    comparison = v0["paired_transition_block_bootstrap"]
    x_positions = np.arange(len(BASELINES))
    for x, baseline in zip(x_positions, BASELINES):
        result = comparison[baseline][
            "delta_psnr_dynamic_minus_baseline_db"
        ]
        block_values = np.asarray(result["block_means"])
        jitter = np.linspace(-0.12, 0.12, len(block_values))
        axes[1].scatter(
            x + jitter,
            block_values,
            s=17,
            alpha=0.55,
            color="#64748b",
        )
        estimate = result["estimate"]
        low, high = result["ci95_percentile"]
        axes[1].errorbar(
            x,
            estimate,
            yerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color="#b91c1c" if high < 0 else "#334155",
            capsize=4,
            lw=2,
        )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xticks(x_positions, BASELINES)
    axes[1].set_ylabel(r"$\Delta$PSNR = JP_KP - baseline (dB)")
    axes[1].set_title("V0: paired transition blocks (16 blocks)")
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle(
        "E35 — exploratory cycle rescue; interpolation quality NO-GO",
        fontsize=12,
    )
    figure.tight_layout()
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, output)


def format_ci(value: dict[str, Any], digits: int = 4) -> str:
    low, high = value["ci95_percentile"]
    return (
        f"{value['estimate']:.{digits}f} "
        f"[{low:.{digits}f}, {high:.{digits}f}]"
    )


def make_markdown(
    verification: list[dict[str, Any]],
    t1_rows: list[dict[str, Any]],
    factorial: list[dict[str, Any]],
    v0: dict[str, Any],
    figure_relative: str,
) -> str:
    all_valid = sum(bool(row["valid_cycle"]) for row in t1_rows)
    max_cv = max(float(row["period_cv"]) for row in t1_rows)
    seam_deviation = max(
        abs(float(row["seam_duration_ratio"]) - 1.0) for row in t1_rows
    )
    commits = sorted({item["git_commit"] for item in verification})
    lines = [
        "# E35 — Analyse vérifiée T1/V0",
        "",
        f"> Généré le {utc_now()}. Aucun nouveau DDE n'a été simulé par cette analyse.",
        "",
        "## Intégrité et portée",
        "",
        f"- {len(verification)}/{len(verification)} bundles vérifiés : manifeste, "
        "configuration NPZ, checkpoint `hist+dhist`, fin de log et heartbeat final.",
        f"- Commit(s) des runs : `{', '.join(commits)}`.",
        "- Seed exploratoire unique : 42. Les contrastes T1 sont descriptifs, pas "
        "des estimations inter-seeds.",
        "- La lignée V0 a essayé successivement JH_KH, JP_KH, JH_KP puis JP_KP. "
        "Le succès mécanique de JP_KP est donc **exploratoire post-diagnostic**.",
        "",
        "## T1 — factoriel J×K",
        "",
        f"Les {all_valid}/24 configurations ont un cycle valide : couverture 1, "
        f"fraction forward 1. Le CV de période maximal est {max_cv:.3g}; "
        f"l'écart maximal du ratio de couture à 1 vaut {seam_deviation:.3f}.",
        "",
        "| λ | données | période JH_KH | Δ JP_KH | Δ JH_KP | Δ JP_KP | interaction |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(factorial, key=lambda item: (item["lam"], item["dataset"])):
        delta = row["delta_seconds_vs_JH_KH"]
        lines.append(
            f"| {row['lam']:.1f} | {row['dataset']} | "
            f"{row['periods']['JH_KH']:.3f} | {delta['JP_KH']:+.3f} | "
            f"{delta['JH_KP']:+.3f} | {delta['JP_KP']:+.3f} | "
            f"{row['interaction_seconds']:+.3f} |"
        )
    lines += [
        "",
        "Lecture prudente : dans cette corrélation modérée (`cond(G)=18.87`), "
        "JH_KH fonctionne déjà. T1 ne démontre donc pas une « rescousse » de capacité. "
        "Il montre seulement que les quatre bras conservent le cycle et modifient "
        "son timing de façon dépendante de λ, avec une interaction J×K visible.",
        "",
        "## V0 — boucle vidéo synthétique",
        "",
        "| méthode | cycle valide | MSE moyenne | PSNR moyenne (dB) | SSIM moyenne |",
        "|---|:---:|---:|---:|---:|",
    ]
    for method in ("JP_KP",) + BASELINES:
        quality = v0["quality"][method]
        cycle = (
            "oui"
            if method == "JP_KP" and v0["cycle"]["JP_KP"]["valid_cycle"]
            else "n/a"
        )
        lines.append(
            f"| {method} | {cycle} | {quality['mean_mse']:.6g} | "
            f"{quality['mean_psnr']:.3f} | {quality['mean_ssim']:.4f} |"
        )
    lines += [
        "",
        "Le contrôle JH_KH reste bloqué (couverture 0.0625, aucun relais). "
        "JP_KP produit un cycle ordonné (couverture 1, forward 1, 3.375 tours, "
        "ratio de couture dynamique 1.062). C'est un **GO mécanisme exploratoire**.",
        "",
        "### Bootstrap apparié par transition",
        "",
        "Un bloc est une transition entre deux frames-clés et contient ses trois "
        "frames cachées. Les 16 blocs sont rééchantillonnés avec remise; IC percentile "
        "95 %, 100 000 réplications. ΔMSE < 0 et ΔPSNR > 0 favoriseraient JP_KP.",
        "",
        "| baseline | ΔMSE JP_KP−baseline [IC95] | blocs favorables | "
        "ΔPSNR (dB) [IC95] | blocs favorables |",
        "|---|---:|---:|---:|---:|",
    ]
    for baseline in BASELINES:
        result = v0["paired_transition_block_bootstrap"][baseline]
        mse = result["delta_mse_dynamic_minus_baseline"]
        psnr = result["delta_psnr_dynamic_minus_baseline_db"]
        lines.append(
            f"| {baseline} | {format_ci(mse, 6)} | "
            f"{mse['favorable_block_count']}/16 | {format_ci(psnr, 3)} | "
            f"{psnr['favorable_block_count']}/16 |"
        )
    lines += [
        "",
        "**Verdict qualité : NO-GO.** JP_KP est inférieur à B1 et B2 avec des IC "
        "entièrement défavorables; il ne franchit pas non plus proprement B0. "
        "La porte interdit donc l'escalade vers Moving-MNIST ou une vraie vidéo.",
        "",
        "![Synthèse E35](" + figure_relative + ")",
        "",
        "## Limites et prochaine validation légitime",
        "",
        "- 16 blocs proviennent d'une seule boucle déterministe et d'une seule seed; "
        "le bootstrap mesure l'hétérogénéité des transitions, pas l'incertitude "
        "entre désordres ou vidéos.",
        f"- Le Gram vidéo est très conditionné (`cond={v0['svd']['condition_retained']:.3g}`) "
        f"et le résidu Moore–Penrose vaut {v0['svd']['moore_penrose_residual']:.3g}; "
        "la robustesse TSVD n'a pas été testée.",
        "- Le seul suivi défendable est une confirmation préspécifiée de l'effet "
        "mécanique JP_KP sur une autre seed/boucle. Elle ne doit pas être présentée "
        "comme une poursuite de l'axe interpolation vidéo après ce NO-GO.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=35042)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--figure-output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    json_output = args.json_output or root / "E35_ANALYSIS.json"
    markdown_output = args.markdown_output or root / "E35_RESULTS.md"
    figure_output = (
        args.figure_output
        or root / "figures" / "E35_analysis_summary.png"
    )

    selected_runs = T1_RUNS + V0_LINEAGE
    verification = [
        verify_run_bundle(root, run_id) for run_id in selected_runs
    ]
    t1_rows = load_t1_rows(root, T1_RUNS)
    factorial = factorial_contrasts(t1_rows)
    v0 = analyze_v0(
        root,
        V0_FINAL,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "new_simulations_run": False,
        "artifact_verification": verification,
        "t1": {
            "run_ids": list(T1_RUNS),
            "all_24_cycles_valid": all(
                bool(row["valid_cycle"]) for row in t1_rows
            ),
            "rows": t1_rows,
            "factorial_contrasts": factorial,
            "interpretation": (
                "Descriptive timing contrasts only: single seed and the "
                "Hebbian reference already cycles on the correlated ring."
            ),
        },
        "v0_lineage": list(V0_LINEAGE),
        "v0": v0,
    }
    atomic_json(json_output, report)
    make_figure(factorial, v0, figure_output)
    relative_figure = os.path.relpath(
        figure_output, start=markdown_output.parent
    )
    markdown = make_markdown(
        verification,
        t1_rows,
        factorial,
        v0,
        relative_figure,
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_output.with_suffix(markdown_output.suffix + ".tmp")
    temporary.write_text(markdown, encoding="utf-8")
    os.replace(temporary, markdown_output)
    print(
        json.dumps(
            {
                "json": str(json_output),
                "markdown": str(markdown_output),
                "figure": str(figure_output),
                "verified_runs": len(verification),
                "t1_valid_cycles": sum(
                    bool(row["valid_cycle"]) for row in t1_rows
                ),
                "v0_decision": v0["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
