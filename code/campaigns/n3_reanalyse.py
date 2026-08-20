"""Seed-aware, read-only reanalysis of the archived E30 campaign (task N3)."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from v5_paths import FIGURES, REFERENCE_ROOT, REPORTS, RUNS, environment_manifest, write_json

E30 = (
    REFERENCE_ROOT / "results" / "3_unification_seuils_scaling" /
    "data" / "E30_scaling"
)
NAME = re.compile(r"E30_N(?P<N>\d+)_P(?P<P>\d+)_s(?P<seed>\d+)\.npz")


def seed_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "n_resolved": int(np.isfinite(values).sum()),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "gap": float(np.ptp(values)),
        "skewness": float(stats.skew(values, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
    }


def load_records() -> list[dict]:
    records = []
    for path in sorted(E30.glob("E30_*.npz")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        data = np.load(path)
        values = np.asarray(data["lam_c"], float)
        values = values[np.isfinite(values)]
        if values.size < 3:
            continue
        N, P, seed = (int(match.group(k)) for k in ("N", "P", "seed"))
        record = {
            "path": str(path), "N": N, "P": P, "seed": seed,
            "alpha": P / N, "values": values,
            **seed_statistics(values),
        }
        records.append(record)
    return records


def aicc(residuals: np.ndarray, k: int) -> float:
    n = len(residuals)
    rss = max(float(residuals @ residuals), np.finfo(float).tiny)
    base = n * math.log(rss / n) + 2 * k
    return base + 2 * k * (k + 1) / max(n - k - 1, 1)


def fit_one(name: str, x: np.ndarray, y: np.ndarray, P: np.ndarray) -> dict:
    if name == "power":
        fun = lambda data, a, b: a * data[0] ** (-b)
        p0, bounds = (1.0, 0.5), ([0.0, 0.0], [np.inf, 3.0])
    elif name == "power_floor":
        fun = lambda data, a, b, floor: floor + a * data[0] ** (-b)
        p0, bounds = (1.0, 0.5, 0.0), ([0.0, 0.0, 0.0], [np.inf, 3.0, np.inf])
    elif name == "gaussian_extreme":
        fun = lambda data, c: c * np.sqrt(np.log(np.maximum(data[1], 2))) / np.sqrt(data[0])
        p0, bounds = (1.0,), ([0.0], [np.inf])
    else:
        raise ValueError(name)
    pars, covariance = optimize.curve_fit(
        fun, np.vstack([x, P]), y, p0=p0, bounds=bounds, maxfev=50_000
    )
    prediction = fun(np.vstack([x, P]), *pars)
    residual = y - prediction
    return {
        "model": name,
        "parameters": pars.tolist(),
        "parameter_se": np.sqrt(np.diag(covariance)).tolist(),
        "aicc": aicc(residual, len(pars)),
        "rss": float(residual @ residual),
        "residuals": residual.tolist(),
    }


def bootstrap_power(records: list[dict], metric: str, n_boot: int,
                    rng: np.random.Generator) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["N"]].append(row)
    exponents = []
    amplitudes = []
    for _ in range(n_boot):
        sample = []
        for size_rows in grouped.values():
            indices = rng.integers(0, len(size_rows), len(size_rows))
            sample.extend(size_rows[i] for i in indices)
        x = np.array([r["N"] for r in sample], float)
        y = np.array([r[metric] for r in sample], float)
        P = np.array([r["P"] for r in sample], float)
        try:
            fit = fit_one("power", x, y, P)
            amplitudes.append(fit["parameters"][0])
            exponents.append(fit["parameters"][1])
        except (RuntimeError, ValueError):
            continue
    return {
        "n_success": len(exponents),
        "amplitude_ci95": np.percentile(amplitudes, [2.5, 97.5]).tolist(),
        "exponent_ci95": np.percentile(exponents, [2.5, 97.5]).tolist(),
        "exponent_median": float(np.median(exponents)),
    }


def leave_one_size_out(records: list[dict], metric: str) -> list[dict]:
    output = []
    for omitted in sorted({r["N"] for r in records}):
        sample = [r for r in records if r["N"] != omitted]
        if len({r["N"] for r in sample}) < 2:
            continue
        x = np.array([r["N"] for r in sample], float)
        y = np.array([r[metric] for r in sample], float)
        P = np.array([r["P"] for r in sample], float)
        fit = fit_one("power", x, y, P)
        output.append({
            "omitted_N": omitted,
            "amplitude": fit["parameters"][0],
            "exponent": fit["parameters"][1],
        })
    return output


def circular_autocorrelation(z: np.ndarray, max_lag: int = 10) -> np.ndarray:
    centered = z - z.mean()
    denom = centered @ centered
    if denom <= 0:
        return np.full(max_lag, np.nan)
    return np.array([
        centered @ np.roll(centered, -lag) / denom
        for lag in range(1, max_lag + 1)
    ])


def distribution_diagnostics(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["N"], row["P"])].append(row)
    output = []
    for (N, P), rows in sorted(grouped.items()):
        if len(rows) < 8:
            continue
        standardized = []
        lag = []
        block_max = []
        block_len = max(2, round(math.sqrt(P)))
        for row in rows:
            z = (row["values"] - row["mean"]) / row["std"]
            standardized.append(z)
            lag.append(circular_autocorrelation(z))
            for start in range(0, len(z), block_len):
                block_max.append(float(np.max(z[start:start + block_len])))
        pooled = np.concatenate(standardized)
        qq_theory = stats.norm.ppf((np.arange(len(pooled)) + 0.5) / len(pooled))
        qq_data = np.sort(pooled)
        output.append({
            "N": N, "P": P, "n_seeds": len(rows),
            "n_motifs": int(sum(len(x) for x in standardized)),
            "ks_distance": float(stats.kstest(pooled, "norm").statistic),
            "skewness": float(stats.skew(pooled, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(pooled, fisher=True, bias=False)),
            "lag_mean": np.nanmean(lag, axis=0).tolist(),
            "lag1_seed_values": np.asarray(lag)[:, 0].tolist(),
            "block_length": block_len,
            "block_max_mean": float(np.mean(block_max)),
            "qq_rms": float(np.sqrt(np.mean((qq_data - qq_theory) ** 2))),
        })
    return output


def extreme_predictions(records: list[dict]) -> list[dict]:
    output = []
    for row in records:
        P = row["n_resolved"]
        q = stats.norm.ppf((P - 0.375) / (P + 0.25))
        gaussian_max = row["mean"] + row["std"] * q
        gaussian_min = row["mean"] - row["std"] * q
        try:
            shape, loc, scale = stats.skewnorm.fit(row["values"])
            skew_max = float(stats.skewnorm.ppf(
                (P - 0.375) / (P + 0.25), shape, loc=loc, scale=scale
            ))
            skew_min = float(stats.skewnorm.ppf(
                0.375 / (P + 0.25), shape, loc=loc, scale=scale
            ))
        except Exception:
            skew_max = skew_min = np.nan
        output.append({
            "N": row["N"], "P": row["P"], "seed": row["seed"],
            "observed_max": row["max"], "observed_min": row["min"],
            "gaussian_max": float(gaussian_max),
            "gaussian_min": float(gaussian_min),
            "gaussian_max_error": float(gaussian_max - row["max"]),
            "gaussian_min_error": float(gaussian_min - row["min"]),
            "skewnormal_max": skew_max, "skewnormal_min": skew_min,
            "skewnormal_max_error": float(skew_max - row["max"]),
            "skewnormal_min_error": float(skew_min - row["min"]),
        })
    return output


def analyse_series(name: str, records: list[dict], n_boot: int,
                   rng: np.random.Generator) -> dict:
    x = np.array([r["N"] for r in records], float)
    P = np.array([r["P"] for r in records], float)
    result = {
        "name": name,
        "n_seed_files": len(records),
        "sizes": {
            str(N): {
                "n_seeds": sum(r["N"] == N for r in records),
                "resolved_motifs": int(sum(r["n_resolved"] for r in records if r["N"] == N)),
            }
            for N in sorted({r["N"] for r in records})
        },
    }
    for metric in ("std", "gap"):
        y = np.array([r[metric] for r in records], float)
        models = [
            fit_one(model, x, y, P)
            for model in ("power", "power_floor", "gaussian_extreme")
            if metric == "gap" or model != "gaussian_extreme"
        ]
        best = min(m["aicc"] for m in models)
        for model in models:
            model["delta_aicc"] = model["aicc"] - best
        result[metric] = {
            "models": models,
            "bootstrap": bootstrap_power(records, metric, n_boot, rng),
            "leave_one_size_out": leave_one_size_out(records, metric),
        }
    result["distribution"] = distribution_diagnostics(records)
    result["extremes"] = extreme_predictions(records)
    return result


def write_seed_table(path: Path, records: list[dict]) -> None:
    fields = [
        "N", "P", "alpha", "seed", "n_resolved", "mean", "median", "std",
        "iqr", "min", "max", "gap", "skewness", "excess_kurtosis", "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fields})


def make_figure(results: dict, records_by_series: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    colors = {"fixed_alpha_0p05": "#2458a6", "fixed_P_100": "#bd3f32"}
    for name, rows in records_by_series.items():
        color = colors[name]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["N"]].append(row)
        N = np.array(sorted(grouped), float)
        std = np.array([np.mean([r["std"] for r in grouped[n]]) for n in N])
        gap = np.array([np.mean([r["gap"] for r in grouped[n]]) for n in N])
        axes[0, 0].loglog(N, std, "o-", color=color, label=name)
        axes[0, 1].loglog(N, gap, "o-", color=color, label=name)
        ext = results[name]["extremes"]
        axes[1, 0].scatter(
            [x["observed_max"] for x in ext],
            [x["gaussian_max"] for x in ext],
            s=14, alpha=0.6, color=color, label=name,
        )
        diagnostics = results[name]["distribution"]
        axes[1, 1].plot(
            [x["N"] for x in diagnostics],
            [np.median(x["lag1_seed_values"]) for x in diagnostics],
            "o-", color=color, label=name,
        )
    axes[0, 0].set(xlabel="$N$", ylabel=r"within-seed $\sigma_{\lambda_c}$",
                   title="Seed-aware threshold narrowing")
    axes[0, 1].set(xlabel="$N$", ylabel=r"$\max\lambda_c-\min\lambda_c$",
                   title="Extreme interval")
    lo, hi = axes[1, 0].get_xlim()
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1, 0].set(xlabel="observed maximum", ylabel="Gaussian-order prediction",
                   title="Finite-$P$ extreme prediction")
    axes[1, 1].axhline(0, color="k", lw=1)
    axes[1, 1].set(xscale="log", xlabel="$N$",
                   ylabel="median seed lag-one correlation",
                   title="Circular dependence diagnostic")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "N3_E30_seed_aware.png", dpi=600)
    fig.savefig(FIGURES / "N3_E30_seed_aware.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    records = load_records()
    series = {
        "fixed_alpha_0p05": [
            row for row in records if abs(row["alpha"] - 0.05) < 5e-4
        ],
        "fixed_P_100": [row for row in records if row["P"] == 100],
    }
    rng = np.random.default_rng(args.seed)
    result = {
        name: analyse_series(name, rows, args.bootstrap, rng)
        for name, rows in series.items()
    }
    result["environment"] = environment_manifest()
    result["archived_input_directory"] = str(E30)
    result["input_files"] = len(records)
    out = RUNS / "N3" / "E30_seed_aware_analysis.json"
    write_json(out, result)
    write_seed_table(REPORTS / "N3_E30_seed_statistics.csv", records)
    make_figure(result, series)
    print(f"N3 analysed {len(records)} archived files -> {out}")


if __name__ == "__main__":
    main()
