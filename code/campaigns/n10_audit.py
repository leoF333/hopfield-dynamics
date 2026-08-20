"""N10 numerical and provenance audit for the final figure pipeline."""
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
import tempfile
from pathlib import Path

import numpy as np

from couplings import make_patterns
from dynamics import DynamicConfig, integrate_and_classify
from lyapunov_stream import LyapunovConfig, run as run_lyapunov
from static_thresholds import StaticConfig, run as run_static
from v5_paths import (
    MANIFESTS, REFERENCE_ROOT, ROOT, RUNS, activate_reference_modules,
    environment_manifest, reference_snapshot, write_json,
)

activate_reference_modules()
from cycle_reduced import ReducedDDE, integrate_fullN  # noqa: E402
from floquet_monodromy import Monodromy  # noqa: E402
from pacemaker_cycle import build_cycle  # noqa: E402

REFERENCE_FILES = [
    "src/couplings.py", "src/cycle_reduced.py", "src/robust_branch.py",
    "src/reduced_spectrum.py", "src/pacemaker_cycle.py",
    "src/floquet_monodromy.py", "src/e23a_chaos_nature.py",
]


def full_reduced_check(N: int, P: int, seed: int = 42) -> dict:
    xi, _ = make_patterns(N, P, seed)
    rng = np.random.default_rng(seed + 5)
    a0 = 0.1 * rng.standard_normal(P)
    system = ReducedDDE(xi, 20.0, 0.4, 1.0, 1.0)
    reduced = system.integrate(a0, 4.0, 0.01)
    full = integrate_fullN(xi, 20.0, 0.4, 1.0, 1.0, xi.T @ a0, 4.0, 0.01)
    error = np.linalg.norm(xi.T @ reduced["a"][-1] - full[-1])
    scale = max(np.linalg.norm(full[-1]), 1e-30)
    return {"N": N, "P": P, "relative_error": float(error / scale),
            "passed": bool(error / scale < 1e-10)}


def forbidden_scan() -> dict:
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py", ".json", ".md", ".tex", ".csv"
        }:
            continue
        if path.name == "n10_audit.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "9_obsolete_ne_pas_citer" in text or "invalid float32 static" in text.lower():
            hits.append(str(path))
    return {"hits": hits, "passed": not hits}


def dynamic_recheck(N: int, P: int, seed: int, smoke: bool) -> dict | None:
    path = RUNS / "N1" / f"dynamic_N{N}_P{P}_seed{seed}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    lam = float(data["dynamic_high"])
    system = ReducedDDE(
        make_patterns(N, P, seed)[0], 20.0, lam, 10.0, 1.0
    )
    cfg = DynamicConfig(
        N=N, P=P, seed=seed, dt=0.01,
        required_tours=(1 if smoke else 5),
        max_time=(3000.0 if smoke else 20000.0),
    )
    result, _, _ = integrate_and_classify(
        system, data["last_cycle_history"],
        data["last_cycle_dhistory"], cfg,
    )
    return {
        "lambda": lam, "label": result["label"],
        "forward_fraction": result["forward_fraction"],
        "passed": result["label"] == "ordered_cycle",
    }


def floquet_recheck(N: int, P: int, smoke: bool) -> dict:
    xi, _ = make_patterns(N, P, 42)
    cycle = build_cycle(
        xi, 20.0, 0.9 if smoke else 0.4, 2.0 if smoke else 10.0,
        dt=0.02 if smoke else 0.01, verbose=False,
        settle_turns=1.0,
    )
    if not cycle.get("ok"):
        return {"passed": False, "reason": cycle.get("reason")}
    monodromy = Monodromy(xi, cycle, mem_guard_gb=4.0, verbose=False)
    multipliers, vectors = monodromy.leading_multipliers(
        k=2 if smoke else 4, tol=1e-7, maxiter=100 if smoke else 300
    )
    phase = monodromy.phase_mode(cycle)
    alignments = [
        abs(np.vdot(vectors[:, j] / np.linalg.norm(vectors[:, j]), phase))
        for j in range(vectors.shape[1])
    ]
    phase_index = int(np.argmax(alignments))
    phase_multiplier = multipliers[phase_index]
    nontrivial = np.delete(np.abs(multipliers), phase_index)
    return {
        "multipliers": [[float(z.real), float(z.imag)] for z in multipliers],
        "phase_alignment": float(alignments[phase_index]),
        "phase_error": float(abs(phase_multiplier - 1.0)),
        "max_nontrivial_modulus": float(np.max(nontrivial)),
        "passed": bool(abs(phase_multiplier - 1.0) < 5e-2),
    }


def provenance_index() -> dict:
    figures = {}
    for figure in sorted((ROOT / "figures").glob("*")):
        sources = []
        stem_tokens = set(figure.stem.lower().replace("-", "_").split("_"))
        for result in RUNS.rglob("*.npz"):
            if stem_tokens & set(result.stem.lower().replace("-", "_").split("_")):
                sources.append(str(result))
        figures[figure.name] = {"sources": sources}
    return figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-floquet", action="store_true")
    parser.add_argument("--skip-lyapunov", action="store_true")
    args = parser.parse_args()
    sizes = [(80, 4)] if args.smoke else [(500, 25), (2000, 100)]
    trajectory = [full_reduced_check(N, P) for N, P in sizes]

    threshold_path = RUNS / "N10" / (
        "threshold_smoke.npz" if args.smoke else "threshold_reference.npz"
    )
    threshold = run_static(
        StaticConfig(
            N=(120 if args.smoke else 2000),
            P=(6 if args.smoke else 100),
            seed=42, relaxed_identity=args.smoke,
            bracket_tol=(2e-3 if args.smoke else 2e-4),
        ),
        threshold_path, nproc=1,
        motifs=([0, 1, 2] if args.smoke else [0, 43, 91]),
    )
    dynamic = dynamic_recheck(2000, 100, 42, args.smoke)
    floquet = None if args.skip_floquet else floquet_recheck(
        120 if args.smoke else 2000,
        6 if args.smoke else 100, args.smoke,
    )
    lyapunov = None
    if not args.skip_lyapunov:
        path = RUNS / "N10" / ("lyap_smoke.npz" if args.smoke else "lyap_reference.npz")
        lyapunov = run_lyapunov(
            LyapunovConfig(
                N=(120 if args.smoke else 2000),
                P=(6 if args.smoke else 100), seed=42, lam=0.31,
                dt=(0.02 if args.smoke else 0.01),
                transient=(2.0 if args.smoke else 400.0),
                accumulation=(2.0 if args.smoke else 1000.0),
                k=(2 if args.smoke else 8),
                qr_time=(0.2 if args.smoke else 2.0),
            ),
            path,
        )

    baseline_path = MANIFESTS / "reference_baseline.json"
    current_snapshot = reference_snapshot(REFERENCE_FILES)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        unchanged = baseline["sha256"] == current_snapshot
    else:
        write_json(baseline_path, {
            "reference_root": str(REFERENCE_ROOT), "sha256": current_snapshot
        })
        unchanged = True
    figure_provenance = provenance_index()
    write_json(MANIFESTS / "figure_provenance.json", figure_provenance)
    payload = {
        "campaign": "N10", "smoke": args.smoke,
        "full_reduced": trajectory, "threshold": threshold,
        "dynamic": dynamic, "floquet": floquet, "lyapunov": lyapunov,
        "forbidden_source_scan": forbidden_scan(),
        "reference_source_unchanged": unchanged,
        "reference_sha256": current_snapshot,
        "figure_provenance_file": str(MANIFESTS / "figure_provenance.json"),
        "environment": environment_manifest(),
    }
    write_json(RUNS / "N10" / ("smoke.json" if args.smoke else "audit.json"), payload)
    print(payload)


if __name__ == "__main__":
    main()
