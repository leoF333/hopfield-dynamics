"""N2 two-sided local critical laws and fold normal-form diagnostic."""
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
import math
import os
from pathlib import Path

import numpy as np
from scipy import optimize

from arclength import trace_through_fold
from couplings import Couplings, make_patterns
from dynamics import DynamicConfig, integrate_and_classify
from fold_refinement import refine_fold_from_full_state
from n5b_long_delay import roots_at
from v5_paths import RUNS, activate_reference_modules, environment_manifest, write_json

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402
from reduced_spectrum import (  # noqa: E402
    GJ_GK,
    rightmost_real_root,
    sigmin_TP,
)
from robust_branch import woodbury_newton  # noqa: E402

DELTAS = np.array([
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4,
    5e-4, 1e-3, 2e-3, 5e-3, 1e-2,
])
REAL_ROOT_METHOD = "real_axis_sigmin_scan_n300"


def amplitudes(xi: np.ndarray, state: np.ndarray) -> np.ndarray:
    gram = xi @ xi.T / xi.shape[1]
    overlap = xi @ state / xi.shape[1]
    return np.linalg.lstsq(gram, overlap, rcond=1e-12)[0]


def reduced_static_matrix(system: ReducedDDE, a: np.ndarray, lam: float) -> np.ndarray:
    S = np.roll(np.eye(system.P), 1, axis=0)
    return -np.eye(system.P) + ((1.0 - lam) * np.eye(system.P) + lam * S) @ system.Dm(a)


def resample_history(history: np.ndarray, derivative: np.ndarray,
                     tau: float, source_dt: float,
                     target_dt: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(source_dt - target_dt) < 1e-14:
        return history.copy(), derivative.copy()
    old_t = np.linspace(-tau, 0.0, len(history))
    new_t = np.linspace(-tau, 0.0, int(round(tau / target_dt)) + 1)
    new_history = np.stack([
        np.interp(new_t, old_t, history[:, j]) for j in range(history.shape[1])
    ], axis=1)
    new_derivative = np.stack([
        np.interp(new_t, old_t, derivative[:, j]) for j in range(derivative.shape[1])
    ], axis=1)
    return new_history, new_derivative


def critical_passages(t: np.ndarray, A: np.ndarray, motif: int) -> np.ndarray:
    winner = np.argmax(A, axis=1)
    passages = []
    starts = np.flatnonzero(
        (winner == motif) & np.r_[True, winner[:-1] != motif]
    )
    for start in starts:
        after = np.flatnonzero(winner[start:] != motif)
        if after.size:
            passages.append(t[start + after[0]] - t[start])
    return np.asarray(passages)


def aicc(residual: np.ndarray, k: int) -> float:
    n = len(residual)
    rss = max(float(residual @ residual), np.finfo(float).tiny)
    return n * math.log(rss / n) + 2*k + 2*k*(k+1)/max(n-k-1, 1)


def fit_models(delta: np.ndarray, value: np.ndarray, dynamic: bool) -> list[dict]:
    valid = np.isfinite(value)
    x, y = delta[valid], value[valid]
    if len(x) < 4:
        return []
    if dynamic:
        models = {
            "free_power": (
                lambda d, B, A, nu: B + A * d**(-nu),
                [np.min(y), max(np.ptp(y), 1e-6), 0.5],
                ([-np.inf, 0.0, 0.0], [np.inf, np.inf, 2.0]),
            ),
            "half_power": (
                lambda d, B, A: B + A * d**(-0.5),
                [np.min(y), max(np.ptp(y), 1e-6)],
                ([-np.inf, 0.0], [np.inf, np.inf]),
            ),
            "logarithmic": (
                lambda d, B, A: B - A * np.log(d),
                [np.min(y), 1.0],
                ([-np.inf, 0.0], [np.inf, np.inf]),
            ),
        }
    else:
        models = {
            "free_power": (
                lambda d, C, nu: -C * d**nu,
                [1.0, 0.5], ([0.0, 0.0], [np.inf, 2.0]),
            ),
            "half_power": (
                lambda d, C: -C * np.sqrt(d),
                [1.0], ([0.0], [np.inf]),
            ),
        }
    results = []
    for name, (function, guess, bounds) in models.items():
        pars, covariance = optimize.curve_fit(
            function, x, y, p0=guess, bounds=bounds, maxfev=100_000
        )
        residual = y - function(x, *pars)
        results.append({
            "model": name, "parameters": pars.tolist(),
            "parameter_se": np.sqrt(np.diag(covariance)).tolist(),
            "aicc": aicc(residual, len(pars)),
            "residuals": residual.tolist(),
        })
    best = min(row["aicc"] for row in results)
    for row in results:
        row["delta_aicc"] = row["aicc"] - best
    return results


def run_seed(
    seed: int, N: int, P: int, smoke: bool,
    refresh_static: bool = False,
) -> dict:
    stage = "smoke" if smoke else "production"
    seed_summary = (
        RUNS / "N2" /
        f"seed_N{N}_P{P}_s{seed}_{stage}.json"
    )
    if (
        seed_summary.exists()
        and os.environ.get("V5_FORCE", "0") != "1"
        and not refresh_static
    ):
        return json.loads(seed_summary.read_text(encoding="utf-8"))
    n1_static = RUNS / "N1" / f"static_N{N}_P{P}_seed{seed}.npz"
    n1_dynamic = RUNS / "N1" / f"dynamic_N{N}_P{P}_seed{seed}.npz"
    if not n1_static.exists():
        raise FileNotFoundError(f"N2 requires completed N1 static file: {n1_static}")
    static_data = np.load(n1_static)
    valid = np.flatnonzero(static_data["resolved"])
    motif = int(valid[np.argmax(static_data["lambda_low"][valid])])
    approximate_fold = 0.5 * (
        float(static_data["lambda_low"][motif])
        + float(static_data["lambda_high"][motif])
    )
    state_terminal = static_data["state"][motif]

    xi, xis = make_patterns(N, P, seed)
    coup = Couplings(xi, xis)
    # A fixed offset of 0.03 can cross another basin/branch for some disorder
    # realizations (seed 52 is an explicit example).  Arclength only needs one
    # regular point immediately below the fold, so select the first converged
    # residual-gated offset without changing branch identity.
    state_start = None
    start_lam = np.nan
    start_offset = np.nan
    for candidate_offset in (0.005, 0.002, 0.001, 0.01, 0.02, 0.03):
        candidate_lam = approximate_fold - candidate_offset
        candidate_state, ok = woodbury_newton(
            coup, state_terminal, candidate_lam, 20.0, tol=1e-12
        )
        candidate_residual = (
            np.linalg.norm(coup.field_F(candidate_state, candidate_lam, 20.0))
            / np.sqrt(N)
        )
        candidate_identity = int(
            np.argmax(np.abs(amplitudes(xi, candidate_state)))
        )
        if ok and candidate_residual < 1e-11 and candidate_identity == motif:
            state_start = candidate_state
            start_lam = candidate_lam
            start_offset = candidate_offset
            break
    if state_start is None:
        raise RuntimeError(
            "could not seed arclength below the selected fold without "
            "changing branch identity"
        )
    identity = lambda u: int(np.argmax(np.abs(amplitudes(xi, u)))) == motif
    arc = trace_through_fold(
        coup, state_start, start_lam, 20.0, identity,
        ds0=(0.01 if smoke else 0.002),
        ds_min=(1e-5 if smoke else 1e-8),
        ds_max=(0.02 if smoke else 0.004),
        max_arclength=(0.5 if smoke else 2.0),
    )
    arclength_fold = float(arc["fold_lambda"])
    j_fold = int(np.argmax(arc["lambda"]))
    arclength_fold_state = arc["state"][j_fold]
    # The static endpoint is already much closer to the singular point than a
    # deliberately coarse smoke arclength sample.  Use it to initialize the
    # exact augmented solve; retain the arclength trace as an independent
    # through-fold branch/geometric control.
    refined_fold = refine_fold_from_full_state(
        xi, state_terminal, approximate_fold
    )
    if (
        not refined_fold.converged
        or refined_fold.fixed_residual >= 1e-11
        or refined_fold.null_residual >= 1e-9
    ):
        raise RuntimeError(
            "augmented fold refinement failed: "
            f"{refined_fold.message}; fixed={refined_fold.fixed_residual:.3e}; "
            f"null={refined_fold.null_residual:.3e}"
        )
    fold = float(refined_fold.lambda_fold)
    fold_a = refined_fold.coefficients
    fold_state = xi.T @ fold_a
    system_fold = ReducedDDE(xi, 20.0, fold, 10.0, 1.0)
    matrix_fold = reduced_static_matrix(system_fold, fold_a, fold)
    eigenvalues, right_vectors = np.linalg.eig(matrix_fold)
    order = np.argsort(-eigenvalues.real)
    null_vector = right_vectors[:, order[0]].real
    null_vector /= np.linalg.norm(null_vector)
    stable_vector = right_vectors[:, order[1]].real
    stable_vector -= null_vector * (null_vector @ stable_vector)
    stable_vector /= np.linalg.norm(stable_vector)
    arc_path = (
        RUNS / "N2" /
        f"arclength_N{N}_P{P}_s{seed}_mu{motif}_{stage}.npz"
    )
    arc_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arc_path, **arc, motif=motif, null_vector=null_vector,
                        stable_vector=stable_vector,
                        arclength_fold_lambda=arclength_fold,
                        refined_fold_lambda=fold,
                        refined_fold_state=fold_state,
                        refined_fold_coefficients=fold_a,
                        arclength_start_lambda=start_lam,
                        arclength_start_offset=start_offset,
                        refined_fold_fixed_residual=refined_fold.fixed_residual,
                        refined_fold_null_residual=refined_fold.null_residual)

    deltas = np.array([1e-3, 1e-2]) if smoke else DELTAS
    static_rows = []
    # At a saddle-node, Newton iteration initialized exactly at the fold can
    # converge to either local branch.  N2B requires the stable,
    # memory-connected branch, so approach the fold from the known stable N1
    # terminal state and continue outward from there.
    static_branch_source = "stable_N1_terminal_continuation"
    current = state_terminal.copy()
    for delta in deltas:
        lam = fold - delta
        point_tag = f"{delta:g}".replace(".", "p")
        point_path = (
            RUNS / "N2" /
            f"static_N{N}_P{P}_s{seed}_d{point_tag}_{stage}.json"
        )
        if point_path.exists() and os.environ.get("V5_FORCE", "0") != "1":
            row = json.loads(point_path.read_text(encoding="utf-8"))
            reusable = (
                row.get("branch_source") == static_branch_source
                and bool(row.get("converged", False))
                and float(row.get("residual", np.inf)) < 1e-11
                and int(row.get("branch_identity", -1)) == motif
            )
            if reusable:
                current = np.asarray(row["state"], float)
                # The reduced-IG candidate list is optimized for the rightmost
                # spectrum as a whole and can omit a purely real fold mode.
                # Upgrade older stable-branch checkpoints with the dedicated
                # real-axis search without recomputing their expensive complex
                # spectrum or any dynamic trajectory.
                if row.get("leading_real_method") != REAL_ROOT_METHOD:
                    GJ, GK = GJ_GK(coup, current, 20.0, lam)
                    real_root = rightmost_real_root(
                        GJ, GK, 20.0, 10.0, lam,
                        x_hi=0.05, x_lo=-0.8, n=300,
                    )
                    if real_root is None:
                        raise RuntimeError(
                            f"dedicated real-root scan failed at delta={delta:g}"
                        )
                    real_residual = (
                        sigmin_TP(real_root.real, GJ, GK, 20.0, 10.0, lam)
                        / max(abs(20.0 * real_root.real + 1.0), 1e-8)
                    )
                    row["leading_real_arpack_candidate"] = row.get(
                        "leading_real"
                    )
                    row["leading_real"] = float(real_root.real)
                    row["leading_real_method"] = REAL_ROOT_METHOD
                    row["leading_real_scaled_residual"] = float(real_residual)
                    write_json(point_path, row)
                row.pop("state")
                static_rows.append(row)
                continue
        current, ok = woodbury_newton(coup, current, lam, 20.0, tol=1e-13)
        residual = np.linalg.norm(coup.field_F(current, lam, 20.0)) / np.sqrt(N)
        a = np.linalg.lstsq(xi.T, current, rcond=1e-12)[0]
        matrix = reduced_static_matrix(
            ReducedDDE(xi, 20.0, lam, 10.0, 1.0), a, lam
        )
        zero_frequency_spectrum = np.linalg.eigvals(matrix)
        zero_real = zero_frequency_spectrum[
            np.abs(zero_frequency_spectrum.imag) < 1e-8
        ]
        dde_spectrum = roots_at(coup, current, lam, 10.0, 48)
        dde_real_candidate = dde_spectrum["leading_real"]
        dde_complex = dde_spectrum["leading_complex"]
        GJ, GK = GJ_GK(coup, current, 20.0, lam)
        dde_real = rightmost_real_root(
            GJ, GK, 20.0, 10.0, lam,
            x_hi=0.05, x_lo=-0.8, n=300,
        )
        if dde_real is None:
            raise RuntimeError(
                f"dedicated real-root scan failed at delta={delta:g}"
            )
        real_residual = (
            sigmin_TP(dde_real.real, GJ, GK, 20.0, 10.0, lam)
            / max(abs(20.0 * dde_real.real + 1.0), 1e-8)
        )
        resolution64_difference = np.nan
        if seed == 42 and any(
            np.isclose(delta, target) for target in (1e-4, 1e-3, 1e-2)
        ):
            dde_spectrum64 = roots_at(coup, current, lam, 10.0, 64)
            root64 = dde_spectrum64["leading_real"]
            if (
                np.isfinite(dde_real_candidate.real)
                and np.isfinite(root64.real)
            ):
                resolution64_difference = float(
                    abs(dde_real_candidate - root64)
                )
        row = {
            "delta": float(delta), "lambda": float(lam), "converged": bool(ok),
            "residual": float(residual),
            "branch_source": static_branch_source,
            "branch_identity": int(np.argmax(np.abs(amplitudes(xi, current)))),
            "leading_real": float(dde_real.real),
            "leading_real_method": REAL_ROOT_METHOD,
            "leading_real_scaled_residual": float(real_residual),
            "leading_real_arpack_candidate": (
                float(dde_real_candidate.real)
                if np.isfinite(dde_real_candidate.real) else np.nan
            ),
            "leading_complex_margin": (
                float(dde_complex.real)
                if np.isfinite(dde_complex.real) else np.nan
            ),
            "zero_frequency_leading_real": (
                float(zero_real.real.max()) if len(zero_real) else np.nan
            ),
            "spectral_residual_gate": dde_spectrum["all_residuals_pass"],
            "spectral_max_scaled_residual": (
                float(np.max(dde_spectrum["residual"]))
                if len(dde_spectrum["residual"]) else np.nan
            ),
            "M48_M64_leading_real_difference": resolution64_difference,
            "null_projection": float(null_vector @ (a - fold_a)),
        }
        write_json(point_path, {**row, "state": current.tolist()})
        static_rows.append(row)

    dynamic_rows = []
    normal_form_samples = []
    if n1_dynamic.exists():
        dynamic_data = np.load(n1_dynamic)
        base_history = dynamic_data["last_cycle_history"]
        base_derivative = dynamic_data["last_cycle_dhistory"]
        source_dt = 0.01
        for delta in deltas:
            dts = [0.01]
            if not smoke and delta <= 5e-4:
                dts.append(0.005)
            for dt in dts:
                lam = fold + delta
                tag = f"d{delta:g}_dt{dt:g}".replace(".", "p")
                path = (
                    RUNS / "N2" /
                    f"dynamic_N{N}_P{P}_s{seed}_{tag}_{stage}.npz"
                )
                point_path = path.with_suffix(".json")
                if (
                    path.exists()
                    and point_path.exists()
                    and os.environ.get("V5_FORCE", "0") != "1"
                ):
                    row = json.loads(point_path.read_text(encoding="utf-8"))
                    dynamic_rows.append(row)
                    if seed == 42 and dt == 0.01:
                        saved = np.load(path)
                        coeff = np.asarray(saved["a"], float)
                        saved_t = np.asarray(saved["t"], float)
                        if len(coeff) > 2:
                            distance = np.linalg.norm(coeff - fold_a, axis=1)
                            x = (coeff - fold_a) @ null_vector
                            y = (coeff - fold_a) @ stable_vector
                            xdot = np.gradient(x, saved_t)
                            nearest = np.argsort(distance)[:min(300, len(distance))]
                            normal_form_samples.extend(
                                (
                                    float(delta), float(x[j]), float(y[j]),
                                    float(xdot[j]),
                                )
                                for j in nearest
                            )
                    continue
                history, derivative = resample_history(
                    base_history, base_derivative, 10.0, source_dt, dt
                )
                system = ReducedDDE(xi, 20.0, lam, 10.0, 1.0)
                result, _, _ = integrate_and_classify(
                    system, history, derivative,
                    DynamicConfig(
                        N=N, P=P, seed=seed, dt=dt, required_tours=3,
                        forward_fraction=0.99,
                        chunk_time=(40.0 if smoke else 200.0),
                        max_time=(200.0 if smoke else 100_000.0),
                        no_tour_arrest_time=(150.0 if smoke else 5000.0),
                        record_dt=max(0.05, dt),
                    ),
                    keep_trajectory=True,
                )
                passages = critical_passages(
                    result["record_t"], result["record_a"], motif
                )
                np.savez_compressed(
                    path, t=result["record_t"], a=result["record_a"],
                    passages=passages, delta=delta, lam=lam, dt=dt, motif=motif,
                    censored=(result["label"] != "ordered_cycle"),
                )
                closest = np.nan
                minimum_speed = np.nan
                closest_bond = -1
                if len(result["record_a"]):
                    coeff = result["record_a"]
                    distance = np.linalg.norm(coeff - fold_a, axis=1)
                    closest = float(np.min(distance))
                    index = int(np.argmin(distance))
                    speed = np.linalg.norm(
                        np.gradient(coeff, result["record_t"], axis=0), axis=1
                    )
                    minimum_speed = float(np.min(speed))
                    closest_bond = int(np.argmax(coeff[index]))
                    if seed == 42 and dt == 0.01:
                        x = (coeff - fold_a) @ null_vector
                        y = (coeff - fold_a) @ stable_vector
                        xdot = np.gradient(x, result["record_t"])
                        nearest = np.argsort(distance)[:min(300, len(distance))]
                        normal_form_samples.extend(
                            (float(delta), float(x[j]), float(y[j]), float(xdot[j]))
                            for j in nearest
                        )
                row = {
                    "delta": float(delta), "lambda": float(lam), "dt": dt,
                    "label": result["label"], "censored": result["label"] != "ordered_cycle",
                    "critical_passages": passages.tolist(),
                    "critical_median": float(np.median(passages)) if len(passages) else np.nan,
                    "critical_std": float(np.std(passages)) if len(passages) else np.nan,
                    "closest_fold_distance": closest,
                    "minimum_speed": minimum_speed,
                    "closest_bond": closest_bond,
                    "file": str(path),
                }
                write_json(point_path, row)
                dynamic_rows.append(row)

    normal_form = None
    if normal_form_samples:
        sample = np.asarray(normal_form_samples)
        design = np.c_[sample[:, 0], sample[:, 1]**2]
        parameters, _, _, _ = np.linalg.lstsq(design, sample[:, 3], rcond=None)
        residual = sample[:, 3] - design @ parameters
        normal_form = {
            "a_delta": float(parameters[0]), "b_x2": float(parameters[1]),
            "rms_residual": float(np.sqrt(np.mean(residual**2))),
            "n_samples": len(sample),
        }

    base_dynamic = [
        row for row in dynamic_rows if row["dt"] == 0.01 and not row["censored"]
    ]
    dynamic_fit = fit_models(
        np.array([row["delta"] for row in base_dynamic]),
        np.array([row["critical_median"] for row in base_dynamic]),
        dynamic=True,
    ) if base_dynamic else []
    static_fit = fit_models(
        np.array([row["delta"] for row in static_rows]),
        np.array([row["leading_real"] for row in static_rows]),
        dynamic=False,
    )
    payload = {
        "seed": seed, "N": N, "P": P, "selected_motif": motif,
        "arclength_file": str(arc_path), "arclength_turned": bool(arc["turned"]),
        "arclength_start_lambda": start_lam,
        "arclength_start_offset": start_offset,
        "arclength_fold_lambda": arclength_fold,
        "fold_lambda": fold,
        "fold_fixed_residual": refined_fold.fixed_residual,
        "fold_null_residual": refined_fold.null_residual,
        "static_rows": static_rows,
        "dynamic_rows": dynamic_rows, "static_fits": static_fit,
        "dynamic_fits": dynamic_fit, "normal_form": normal_form,
    }
    write_json(seed_summary, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--refresh-static", action="store_true",
        help=(
            "rebuild N2B from the stable N1 terminal branch while reusing "
            "completed dynamic trajectories"
        ),
    )
    args = parser.parse_args()
    rows = [
        run_seed(
            seed, args.N, args.P, args.smoke,
            refresh_static=args.refresh_static,
        )
        for seed in args.seeds
    ]
    summary_name = (
        "smoke.json" if args.smoke else
        ("summary.json" if args.seeds == [42, 60, 47, 52, 51]
         else "summary_seeds_" + "_".join(map(str, args.seeds)) + ".json")
    )
    write_json(
        RUNS / "N2" / summary_name,
        {
            "campaign": "N2", "rows": rows, "smoke": args.smoke,
            "environment": environment_manifest(),
        },
    )
    print(f"N2 completed {len(rows)} selected folds")


if __name__ == "__main__":
    main()
