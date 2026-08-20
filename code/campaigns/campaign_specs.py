"""Machine-readable grids and rationale copied from NUMERICAL_WORKPLAN.md."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


SPECS = {
    "N1": {
        "priority": "blocking",
        "why": "Test the multi-seed equality between the largest static fold and the independent dynamic onset.",
        "grid": {
            "N": 2000, "P": 100, "seeds": list(range(42, 62)),
            "beta": 20, "tau": 10, "static_tolerance": 2e-4,
            "dynamic_step": 0.002, "dynamic_tolerance": 5e-4,
            "required_tours": 5, "maximum_time": 20000,
        },
        "estimated_production_time": "8–12 h for dynamics on CPU; static part <3 min",
    },
    "N2": {
        "priority": "blocking",
        "why": "Resolve the dynamic and static critical laws and test the local saddle-node normal form.",
        "grid": {
            "seeds": "42 plus four N1-admissible seeds spanning extreme gaps",
            "delta": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            "dt": [0.01, 0.005],
        },
        "estimated_production_time": "several hours to >1 day near delta=1e-5",
    },
    "N3": {
        "priority": "blocking",
        "why": "Add seed-aware uncertainty, model comparison, dependence diagnostics and finite-P extreme tests.",
        "grid": {"archived_files": 303, "bootstrap_replicates": 5000},
        "estimated_production_time": "3–10 min; no new simulation",
    },
    "N4": {
        "priority": "blocking",
        "why": "Recompose the central publication figures from raw/derived tables.",
        "grid": {"figures": [1, 2, 3, 4]},
        "estimated_production_time": "<1 min after N1/N3",
    },
    "N5A": {
        "priority": "strongly recommended",
        "why": "Resolve seed variation of the short-delay coherence and cycle–chaos boundary.",
        "grid": {
            "seeds": [42, 43, 44, 45, 46],
            "tau_coherence": [0.25, 0.5, 0.75, 1, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.8, 2, 3, 5, 10],
            "tau_boundary": [0.5, 1, 1.2, 1.3, 1.4, 2, 3, 5, 10],
            "lambda_bracket": [0.30, 0.36], "lambda_resolution": 0.002,
        },
        "estimated_production_time": "1–3 days on CPU including Lyapunov points",
    },
    "N5B": {
        "priority": "supplementary",
        "why": "Decide whether long-delay complex softening is a resolved Hopf or only a stable near-marginal pair.",
        "grid": {
            "N": 2000, "alpha": 0.07, "seeds": [42, 43, 44, 45, 46],
            "tau": [40, 45, 50, 52, 55, 70, 100, 130],
            "spectral_resolutions": [48, 64],
        },
        "estimated_production_time": "many hours to several days",
    },
    "N6": {
        "priority": "strongly recommended",
        "why": "Test disorder, time-window and step-size robustness of the partial Lyapunov spectrum.",
        "grid": {
            "seeds": [42, 43, 44, 45, 46],
            "lambda": [0.29, 0.30, 0.31, 0.32, 0.325],
            "windows": [1000, 2000, 4000], "dt": [0.01, 0.005], "k": 8,
        },
        "estimated_production_time": "several days on CPU",
    },
    "N7": {
        "priority": "strongly recommended",
        "why": "Test seed robustness of the Markov seam and smooth periodic controls.",
        "grid": {
            "seeds": [42, 43, 44, 45, 46],
            "markov_c": [0, 0.2, 0.4, 0.6, 0.8],
            "gp_kappa": [0.8, 0.4, 0.2, 0.1, 0.05],
        },
        "estimated_production_time": "8–24 h on CPU",
    },
    "N8": {
        "priority": "supplementary",
        "why": "Separate periodic from chaotic high-load moving states.",
        "grid": {
            "alpha": [0.05, 0.09, 0.13, 0.15],
            "lambda": [0.35, 0.40, 0.50, 0.70, 0.90],
            "seeds": [42, 43, 44, 45, 46],
        },
        "estimated_production_time": "1–3 days on CPU",
    },
    "N9": {
        "priority": "supplementary",
        "why": "Test float64 stationarity and equality of time/ensemble distributions.",
        "grid": {
            "points": [[0.05, 0.31], [0.08, "from N8"], [0.10, "from N8"]],
            "seeds": [42, 43, 44, 45, 46], "histories_per_seed": 32,
            "transient": 400, "observation": 6000, "dt": [0.01, 0.005],
        },
        "estimated_production_time": "several days on CPU; N8 choices required",
    },
    "N10": {
        "priority": "release gate",
        "why": "Re-run independent numerical checks and build point-level provenance.",
        "grid": {"full_reduced_N": [500, 2000], "reference_thresholds": 3},
        "estimated_production_time": "hours, dominated by Floquet and Lyapunov reruns",
    },
}

