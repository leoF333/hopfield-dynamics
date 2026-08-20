"""Curate and plot the first E34 N=400 gate results without new simulations.

The script is deliberately read-only with respect to simulation inputs. It
checks the four expected NPZ schemas and scientific invariants, then writes one
deterministic JSON summary and one compact four-panel PNG.
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
import tempfile

import numpy as np


REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (
    REPO_DIR / "results" / "2_cycle_rappel_snic" / "data" / "E34")
RESULT_DIR = REPO_DIR / "results" / "2_cycle_rappel_snic"
FIGURE_DIR = RESULT_DIR / "figures" / "E34"

DEFAULT_TABLE = (
    DATA_DIR / "E34_thresholds_N400_P20_s42_20260723T191104_587960.npz")
DEFAULT_LOCAL = DATA_DIR / "E34a_census_20260723T191129_596021.npz"
DEFAULT_REPLACEMENT = DATA_DIR / "E34a_census_20260723T191854_621183.npz"
DEFAULT_COMMON = DATA_DIR / "E34a_census_20260723T191922_621256.npz"
DEFAULT_JSON = RESULT_DIR / "E34_RESULTS.json"
DEFAULT_FIGURE = FIGURE_DIR / "figE34_first_results.png"

FOLD_CROSSCHECK_TOLERANCE = 2e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path) -> dict:
    try:
        display_path = str(path.relative_to(REPO_DIR))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _records(data: np.lib.npyio.NpzFile) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for index, value in enumerate(data["mu"]):
        mu = int(value)
        result[mu] = {
            "mu": mu,
            "raw_status": str(data["status"][index]),
            "target_lambda": float(data["lam_target"][index]),
            "fold_lambda": float(data["lam_fold"][index]),
            "node_unstable_count": int(data["unstable_count_node"][index]),
            "saddle_unstable_count": int(data["unstable_count_saddle"][index]),
            "node_eigmax": float(data["node_eigmax"][index]),
            "saddle_eigmax": float(data["saddle_eigmax"][index]),
            "node_residual": float(data["node_residual"][index]),
            "saddle_residual": float(data["saddle_residual"][index]),
        }
        for key in (
            "lam_fold_reference",
            "fold_mismatch",
            "fold_crosscheck_tol",
            "turning_count",
        ):
            if key in data:
                scalar = data[key][index]
                result[mu][key] = (
                    int(scalar) if key == "turning_count" else float(scalar))
    return result


def curate(
    table_path: Path,
    local_path: Path,
    replacement_path: Path,
    common_path: Path,
) -> dict:
    """Validate all inputs and return the deterministic scientific summary."""
    paths = {
        "threshold_table": table_path,
        "local_original": local_path,
        "local_replacement_mu14": replacement_path,
        "common_gate": common_path,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing {path}")

    with (
        np.load(table_path, allow_pickle=False) as table,
        np.load(local_path, allow_pickle=False) as local,
        np.load(replacement_path, allow_pickle=False) as replacement,
        np.load(common_path, allow_pickle=False) as common,
    ):
        if (
            int(table["N"]) != 400
            or int(table["P"]) != 20
            or int(table["seed"]) != 42
            or int(table["threshold_schema"]) != 2
        ):
            raise ValueError("unexpected threshold-table experiment or schema")
        threshold_status = np.asarray(table["threshold_status"], dtype=str)
        spectral = threshold_status == "spectral_fold"
        if int(spectral.sum()) != 19:
            raise ValueError(
                f"expected 19 spectral folds, found {int(spectral.sum())}")
        failed_mus = np.flatnonzero(~spectral).astype(int)
        if failed_mus.tolist() != [17]:
            raise ValueError(f"unexpected threshold failures: {failed_mus}")

        local_records = _records(local)
        replacement_records = _records(replacement)
        common_records = _records(common)
        if sorted(local_records) != [2, 3, 8]:
            raise ValueError("unexpected original local sentinels")
        if sorted(replacement_records) != [14]:
            raise ValueError("unexpected replacement sentinel")
        if sorted(common_records) != [2, 8, 14]:
            raise ValueError("unexpected common-gate sentinels")

        # Re-evaluate the original local file with the later fold-consistency
        # gate. Its raw "ok" predates the cross-check implementation.
        curated_local: dict[int, dict] = {}
        for mu in (3, 8, 2):
            record = dict(local_records[mu])
            reference = float(table["lam_c"][mu])
            mismatch = abs(record["fold_lambda"] - reference)
            record.update(
                fold_reference=reference,
                fold_mismatch=mismatch,
                fold_crosscheck_tol=FOLD_CROSSCHECK_TOLERANCE,
                curated_status=(
                    "fold_mismatch_indeterminate"
                    if mismatch > FOLD_CROSSCHECK_TOLERANCE else "ok"
                ),
            )
            curated_local[mu] = record

        replacement_record = dict(replacement_records[14])
        replacement_record["fold_reference"] = replacement_record[
            "lam_fold_reference"]
        replacement_record["curated_status"] = replacement_record["raw_status"]
        curated_local[14] = replacement_record
        valid_local_mus = [
            mu for mu in (14, 8, 2)
            if curated_local[mu]["curated_status"] == "ok"
            and curated_local[mu]["node_unstable_count"] == 0
            and curated_local[mu]["saddle_unstable_count"] == 1
        ]
        if valid_local_mus != [14, 8, 2]:
            raise ValueError(f"local gate invariant failed: {valid_local_mus}")

        expected_common_status = {
            14: "ok",
            8: "multifold_indeterminate",
            2: "multifold_indeterminate",
        }
        expected_turns = {14: 1, 8: 9, 2: 11}
        for mu in (14, 8, 2):
            if common_records[mu]["raw_status"] != expected_common_status[mu]:
                raise ValueError(f"unexpected common status for mu={mu}")
            if common_records[mu]["turning_count"] != expected_turns[mu]:
                raise ValueError(f"unexpected common turns for mu={mu}")

        values = np.asarray(table["lam_c"], dtype=float)[spectral]
        spectral_mus = np.flatnonzero(spectral)
        sorted_indices = np.argsort(values)
        sorted_thresholds = [
            {
                "mu": int(spectral_mus[index]),
                "lambda_c": float(values[index]),
            }
            for index in sorted_indices
        ]
        threshold_summary = {
            "N": int(table["N"]),
            "P": int(table["P"]),
            "seed": int(table["seed"]),
            "beta": float(table["beta"]),
            "method": str(table["threshold_method"]),
            "spectral_folds": int(spectral.sum()),
            "indeterminate": int((~spectral).sum()),
            "indeterminate_mus": failed_mus.tolist(),
            "indeterminate_reasons": {
                str(mu): str(table["fail_reason"][mu])
                for mu in failed_mus
            },
            "lambda_c_min": float(np.min(values)),
            "lambda_c_median": float(np.median(values)),
            "lambda_c_max": float(np.max(values)),
            "max_fold_bracket_width": float(
                np.nanmax(table["fold_bracket_width"][spectral])),
            "max_fold_fit_error": float(
                np.nanmax(table["fold_fit_error"][spectral])),
            "max_abs_threshold_side_eig": float(
                np.nanmax(np.abs(table["threshold_side_eig"][spectral]))),
            "max_threshold_side_residual": float(
                np.nanmax(table["threshold_side_residual"][spectral])),
            "sorted_thresholds": sorted_thresholds,
        }

        local_json = {
            str(mu): curated_local[mu]
            for mu in (3, 14, 8, 2)
        }
        common_json = {
            str(mu): common_records[mu]
            for mu in (14, 8, 2)
        }

    return {
        "analysis": "E34 first N=400 gate results",
        "inputs": {
            name: _input_record(path) for name, path in paths.items()
        },
        "thresholds": threshold_summary,
        "local_gate": {
            "crosscheck_tolerance": FOLD_CROSSCHECK_TOLERANCE,
            "records": local_json,
            "valid_mus": valid_local_mus,
            "excluded_mus": [3],
            "verdict": "GO on replacement set {14,8,2}",
        },
        "common_gate": {
            "lambda": common_records[14]["target_lambda"],
            "records": common_json,
            "simple_branches": [14],
            "multifold_mus": [8, 2],
            "verdict": "NO-GO for a simple three-link necklace",
        },
        "scope": {
            "N2000_run": False,
            "claim": (
                "N=400, P=20, seed=42 pilot evidence only; no finite-size "
                "or multiseed validation"
            ),
        },
        "overall_verdict": (
            "19/20 local spectral folds; mu=3 is multifold for the local "
            "pairing; local replacement set {14,8,2} passes index (0,1); "
            "the common-lambda simple necklace is NO-GO because mu=8 and "
            "mu=2 have 9 and 11 resolved turns"
        ),
    }


def _make_figure(summary: dict, output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "e34-matplotlib-cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = summary["thresholds"]["sorted_thresholds"]
    by_mu = {item["mu"]: item["lambda_c"] for item in thresholds}
    local = summary["local_gate"]["records"]
    common = summary["common_gate"]["records"]

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "figure.dpi": 180,
    })
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.6))

    axis = axes[0, 0]
    mus = np.arange(20)
    values = np.array([by_mu.get(mu, np.nan) for mu in mus])
    axis.scatter(mus[np.isfinite(values)], values[np.isfinite(values)],
                 s=24, color="#3274a1", label="Spectral fold")
    indeterminate_y = float(np.nanmin(values) - 0.006)
    axis.scatter([17], [indeterminate_y], marker="x", s=45, color="#c44e52",
                 label="Threshold indeterminate")
    for mu, marker, color, label in (
        (14, "o", "#55a868", None),
        (8, "s", "#55a868", None),
        (2, "^", "#55a868", None),
        (3, "D", "#dd8452", "Multifold after cross-check"),
    ):
        axis.scatter([mu], [by_mu[mu]], marker=marker, s=62, color=color,
                     edgecolor="black", linewidth=0.5, zorder=4, label=label)
        axis.annotate(f"μ{mu}", (mu, by_mu[mu]), xytext=(3, 4),
                      textcoords="offset points")
    axis.set(title="A  Threshold table: 19/20 spectral folds",
             xlabel="Pattern link μ", ylabel="λc")
    axis.set_xticks(np.arange(0, 20, 2))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="lower right")

    axis = axes[0, 1]
    local_order = [3, 14, 8, 2]
    table_folds = [float(local[str(mu)]["fold_reference"])
                   for mu in local_order]
    census_folds = [float(local[str(mu)]["fold_lambda"])
                    for mu in local_order]
    low = min(table_folds + census_folds) - 0.005
    high = max(table_folds + census_folds) + 0.005
    axis.plot([low, high], [low, high], color="0.55", linewidth=1)
    for mu, x_value, y_value in zip(
        local_order, table_folds, census_folds
    ):
        bad = mu == 3
        axis.scatter(
            x_value, y_value,
            marker="X" if bad else "o",
            s=62,
            color="#c44e52" if bad else "#55a868",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        offset = (-95, 5) if mu == 2 else (4, 4)
        axis.annotate(
            f"μ{mu}  Δ={abs(y_value - x_value):.5f}",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
        )
    axis.set(
        title="B  Fold cross-check exposes μ3",
        xlabel="Threshold-table fold",
        ylabel="Census-continuation fold",
        xlim=(low, high),
        ylim=(low, high),
    )
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    valid = [14, 8, 2]
    x_positions = np.arange(len(valid))
    node_counts = [local[str(mu)]["node_unstable_count"] for mu in valid]
    saddle_counts = [
        local[str(mu)]["saddle_unstable_count"] for mu in valid]
    axis.scatter(x_positions - 0.08, node_counts, marker="o", s=58,
                 color="#3274a1", label="Node")
    axis.scatter(x_positions + 0.08, saddle_counts, marker="^", s=62,
                 color="#dd8452", label="Saddle")
    axis.set(
        title="C  Local replacement set passes index test",
        xlabel="Validated sentinel",
        ylabel="Unstable-root count",
        xticks=x_positions,
        xticklabels=[f"μ{mu}" for mu in valid],
        yticks=[0, 1],
        ylim=(-0.2, 1.25),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="center right")

    axis = axes[1, 1]
    common_order = [14, 8, 2]
    turns = [common[str(mu)]["turning_count"] for mu in common_order]
    colors = ["#55a868" if value == 1 else "#c44e52" for value in turns]
    bars = axis.bar(
        [f"μ{mu}" for mu in common_order], turns, color=colors, width=0.62)
    axis.axhline(1, color="0.35", linewidth=1, linestyle="--")
    axis.bar_label(bars, labels=[str(value) for value in turns], padding=2)
    axis.set(
        title="D  Common λ: simple necklace NO-GO",
        xlabel="Sentinel at λ=0.23291",
        ylabel="Resolved λ-turns",
        ylim=(0, 12.5),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.98, 0.96, "simple branch requires 1 turn",
        transform=axis.transAxes, ha="right", va="top", color="0.35")

    figure.suptitle(
        "E34 pilot — N=400, P=20, seed=42 (no N=2000 run)",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.png")
    figure.savefig(temporary, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--replacement", type=Path, default=DEFAULT_REPLACEMENT)
    parser.add_argument("--common", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and invariants without writing outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = curate(
        args.table.resolve(),
        args.local.resolve(),
        args.replacement.resolve(),
        args.common.resolve(),
    )
    if not args.check_only:
        _atomic_json(args.json.resolve(), summary)
        _make_figure(summary, args.figure.resolve())
    print(json.dumps({
        "threshold_spectral_folds": summary["thresholds"]["spectral_folds"],
        "local_valid_mus": summary["local_gate"]["valid_mus"],
        "common_turns": {
            mu: summary["common_gate"]["records"][str(mu)]["turning_count"]
            for mu in (14, 8, 2)
        },
        "overall_verdict": summary["overall_verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
