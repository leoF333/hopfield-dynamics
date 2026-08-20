"""Independent-history control for completed N1 dynamic brackets."""
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

from couplings import make_patterns
from dynamics import DynamicConfig, integrate_and_classify
from v5_paths import RUNS, activate_reference_modules, write_json

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402


def scalars(result: dict) -> dict:
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in result.items()
        if np.isscalar(value) or isinstance(value, str)
    }


def run(seed: int, N: int = 2000, P: int = 100) -> list[dict]:
    path = RUNS / "N1" / f"dynamic_N{N}_P{P}_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    low, high = float(data["dynamic_low"]), float(data["dynamic_high"])
    xi, _ = make_patterns(N, P, seed)
    config = DynamicConfig(
        N=N, P=P, seed=seed, dt=0.01, required_tours=5,
        max_time=20000.0, no_tour_arrest_time=5000.0,
    )
    system = ReducedDDE(xi, 20.0, 0.4, 10.0, 1.0)
    second = np.zeros(P)
    second[P//2] = 0.99
    second[(P//2 + 1) % P] = 0.05
    start_result, second_history, second_derivative = integrate_and_classify(
        system, second, None, config,
    )
    if start_result["label"] != "ordered_cycle":
        raise RuntimeError(
            f"seed {seed}: independent cycle failed: {start_result['label']}"
        )
    histories = [
        ("primary", data["last_cycle_history"], data["last_cycle_dhistory"]),
        ("independent_memory", second_history, second_derivative),
    ]
    rows = []
    for name, history, derivative in histories:
        for lam in (high, low):
            system.lam = lam
            result, _, _ = integrate_and_classify(
                system, history, derivative, config,
            )
            rows.append({
                "seed": seed, "history": name, "lambda": lam, **scalars(result)
            })
    summary_path = path.with_suffix(".json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["history_repeats"] = rows
    summary["independent_history_control_passed"] = (
        all(
            row["label"] == "ordered_cycle"
            for row in rows if row["lambda"] == high
        )
        and all(
            row["label"] in {"stationary_arrest", "noncycling_periodic"}
            for row in rows if row["lambda"] == low
        )
    )
    write_json(summary_path, summary)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for seed in args.seeds:
        print(run(seed))


if __name__ == "__main__":
    main()

