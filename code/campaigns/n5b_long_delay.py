"""N5B resolution-stable long-delay spectral scan and nonlinear eigenmode test."""
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
import os
from pathlib import Path

import numpy as np

from couplings import Couplings, make_patterns
from dynamics import periodic_closure, reduced_fixed_residual
from static_thresholds import StaticConfig, run as run_static
from v5_paths import RUNS, activate_reference_modules, environment_manifest, write_json

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402
from reduced_spectrum import (  # noqa: E402
    GJ_GK, T_P, physical_roots, sigmin_TP,
)
from robust_branch import woodbury_newton  # noqa: E402

TAUS = [40, 45, 50, 52, 55, 70, 100, 130]
PINNED_TAUS = [10, 20, 30, 50, 70, 100]
OFFSETS = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]


def coefficients(xi: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(xi.T, state, rcond=1e-12)[0]


def roots_at(coup: Couplings, state: np.ndarray, lam: float,
             tau: float, M: int) -> dict:
    roots, candidates, _ = physical_roots(
        coup, state, 20.0, lam, tau, 1.0,
        M=M, n_cand=max(80, coup.P // 2), k=16,
    )
    GJ, GK = GJ_GK(coup, state, 20.0, lam)
    residual = np.array([
        sigmin_TP(root, GJ, GK, 1.0, tau, lam)
        / max(abs(root + 1.0), 1e-8)
        for root in roots
    ])
    real = roots[np.abs(roots.imag) < 1e-3]
    complex_roots = roots[np.abs(roots.imag) >= 1e-3]
    leading_real = real[np.argmax(real.real)] if len(real) else np.nan + 0j
    leading_complex = (
        complex_roots[np.argmax(complex_roots.real)]
        if len(complex_roots) else np.nan + 1j*np.nan
    )
    return {
        "roots": roots, "residual": residual,
        "leading_real": leading_real, "leading_complex": leading_complex,
        "all_residuals_pass": bool(len(residual) and np.max(residual) < 1e-10),
    }


def tracked_complex(spectrum: dict, reference: complex | None) -> tuple[complex, float]:
    """Select one complex root continuously instead of re-ranking at each point."""
    roots = spectrum["roots"]
    indices = np.flatnonzero(np.abs(roots.imag) >= 1e-3)
    if not len(indices):
        return np.nan + 1j*np.nan, np.nan
    candidates = roots[indices]
    if reference is None or not np.isfinite(reference.real):
        local = int(np.argmax(candidates.real))
    else:
        local = int(np.argmin(np.abs(candidates - reference)))
    index = int(indices[local])
    return complex(roots[index]), float(spectrum["residual"][index])


def conjugacy_aware_difference(left: complex, right: complex) -> float:
    """Compare real-system eigenvalues modulo the conjugate-pair convention."""
    if not (
        np.isfinite(left.real) and np.isfinite(left.imag)
        and np.isfinite(right.real) and np.isfinite(right.imag)
    ):
        return np.nan
    return float(min(abs(left - right), abs(left - np.conj(right))))


def nonlinear_probe(xi: np.ndarray, state: np.ndarray, lam: float, tau: float,
                    root: complex, amplitude: float, side: int,
                    output: Path) -> dict:
    if output.exists() and os.environ.get("V5_FORCE", "0") != "1":
        saved = np.load(output)
        return {
            "file": str(output),
            "measured_growth": float(saved["measured_growth"]),
            "measured_frequency": float(saved["measured_frequency"]),
            "predicted_growth": float(saved["predicted_root"].real),
            "predicted_frequency": float(abs(saved["predicted_root"].imag)),
            "fixed_residual": float(saved["fixed_residual"]),
            "periodic_closure": float(saved["periodic_closure"]),
            "candidate_period": float(saved["candidate_period"]),
            "final_classification": str(saved["final_classification"]),
        }
    system = ReducedDDE(xi, 20.0, lam, tau, 1.0)
    coup = Couplings(xi, np.roll(xi, -1, axis=0))
    GJ, GK = GJ_GK(coup, state, 20.0, lam)
    _, _, vh = np.linalg.svd(T_P(root, GJ, GK, 1.0, tau, lam))
    vector = vh[-1].conj()
    vector /= np.linalg.norm(vector)
    base = coefficients(xi, state)
    dt = 0.01
    L = int(round(tau / dt))
    history_time = np.linspace(-tau, 0.0, L + 1)
    mode = np.real(np.exp(root * history_time[:, None]) * vector[None, :])
    history = base[None, :] + amplitude * mode
    derivative = amplitude * np.real(
        root * np.exp(root * history_time[:, None]) * vector[None, :]
    )
    period = 2.0 * np.pi / max(abs(root.imag), 1e-8)
    total = 20.0 * period
    solution = system.integrate(
        history, total, dt, record_every=max(1, round(0.05 / dt)),
        da_hist0=derivative,
    )
    deviation = solution["a"] - base
    projection = deviation @ vector.conj()
    envelope = np.abs(projection)
    fit_mask = (
        (solution["t"] > period)
        & (solution["t"] < min(8.0 * period, total))
        & (envelope > 1e-14)
    )
    slope = np.polyfit(
        solution["t"][fit_mask], np.log(envelope[fit_mask]), 1
    )[0] if fit_mask.sum() > 5 else np.nan
    phase = np.unwrap(np.angle(projection))
    frequency = np.polyfit(
        solution["t"][fit_mask], phase[fit_mask], 1
    )[0] if fit_mask.sum() > 5 else np.nan
    fixed_residual = reduced_fixed_residual(system, solution["hist"])
    period_closure, candidate_period = periodic_closure(
        solution["t"], solution["a"], tau
    )
    if fixed_residual < 1e-10:
        final_classification = "fixed_point"
    elif period_closure < 1e-5:
        final_classification = "periodic_candidate"
    else:
        final_classification = "nonperiodic_or_unresolved"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, t=solution["t"], projection=projection, envelope=envelope,
        predicted_root=root, measured_growth=slope, measured_frequency=frequency,
        fixed_residual=fixed_residual, periodic_closure=period_closure,
        candidate_period=candidate_period, lam=lam, tau=tau,
        amplitude=amplitude, side=side,
        final_classification=np.array(final_classification),
    )
    return {
        "file": str(output), "measured_growth": float(slope),
        "measured_frequency": float(frequency),
        "predicted_growth": float(root.real),
        "predicted_frequency": float(abs(root.imag)),
        "fixed_residual": float(fixed_residual),
        "periodic_closure": float(period_closure),
        "candidate_period": float(candidate_period),
        "final_classification": final_classification,
    }


def memory_scan(seed: int, N: int, smoke: bool) -> dict:
    checkpoint = (
        RUNS / "N5B" /
        f"memory_scan_N{N}_s{seed}_{'smoke' if smoke else 'production'}.json"
    )
    if (
        checkpoint.exists()
        and os.environ.get("V5_FORCE", "0") != "1"
    ):
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    P = round(0.07 * N)
    xi, xis = make_patterns(N, P, seed)
    # Smoke and production configurations use different static tolerances and
    # spectral resolutions.  Keep their checkpoints in separate namespaces so
    # that a validated smoke file can never be mistaken for production data.
    variant = "_smoke" if smoke else ""
    static_path = (
        RUNS / "N5B" /
        f"memory_static_N{N}_P{P}_s{seed}{variant}.npz"
    )
    static = run_static(
        StaticConfig(
            N=N, P=P, seed=seed, bracket_tol=(2e-3 if smoke else 1e-6)
        ),
        static_path, nproc=1, motifs=[0],
    )
    data = np.load(static_path)
    fold = 0.5 * (float(data["lambda_low"][0]) + float(data["lambda_high"][0]))
    state = data["state"][0]
    coup = Couplings(xi, xis)
    taus = [50] if smoke else TAUS
    offsets = [1e-3] if smoke else OFFSETS
    rows = []
    crossings = []
    tau_anchor48 = None
    tau_anchor64 = None
    for tau in taus:
        previous = None
        previous48 = tau_anchor48
        previous64 = tau_anchor64
        for offset in offsets:
            lam = fold - offset
            point_tag = f"{offset:g}".replace(".", "p")
            point_checkpoint = (
                RUNS / "N5B" /
                f"spectrum_N{N}_s{seed}_tau{tau:g}_off{point_tag}"
                f"{variant}.json"
            )
            if (
                point_checkpoint.exists()
                and os.environ.get("V5_FORCE", "0") != "1"
            ):
                row = json.loads(point_checkpoint.read_text(encoding="utf-8"))
            else:
                solved, ok = woodbury_newton(
                    coup, state, lam, 20.0, tol=1e-12
                )
                residual = (
                    np.linalg.norm(coup.field_F(solved, lam, 20.0))
                    / np.sqrt(N)
                )
                if not ok or residual >= 1e-11:
                    row = {
                        "tau": tau, "offset": offset, "lambda": lam,
                        "resolved": False, "residual": float(residual),
                    }
                    rows.append(row)
                    write_json(point_checkpoint, row)
                    continue
                resolutions = {
                    str(M): roots_at(coup, solved, lam, tau, M)
                    for M in ((16, 20) if smoke else (48, 64))
                }
                root48, selected_residual48 = tracked_complex(
                    resolutions[str(16 if smoke else 48)], previous48
                )
                root64, selected_residual64 = tracked_complex(
                    resolutions[str(20 if smoke else 64)], previous64
                )
                agreement = conjugacy_aware_difference(root48, root64)
                row = {
                    "tau": tau, "offset": offset, "lambda": lam,
                    "resolved": True, "residual": float(residual),
                    "leading_real_M1": [
                        resolutions[str(16 if smoke else 48)]["leading_real"].real,
                        resolutions[str(16 if smoke else 48)]["leading_real"].imag,
                    ],
                    "leading_complex_M1": [root48.real, root48.imag],
                    "leading_real_M2": [
                        resolutions[str(20 if smoke else 64)]["leading_real"].real,
                        resolutions[str(20 if smoke else 64)]["leading_real"].imag,
                    ],
                    "leading_complex_M2": [root64.real, root64.imag],
                    "resolution_agreement": agreement,
                    "selected_residual_M1": selected_residual48,
                    "selected_residual_M2": selected_residual64,
                    "residual_gate_M1": resolutions[
                        str(16 if smoke else 48)
                    ]["all_residuals_pass"],
                    "residual_gate_M2": resolutions[
                        str(20 if smoke else 64)
                    ]["all_residuals_pass"],
                }
                write_json(point_checkpoint, row)
            if not row.get("resolved", False):
                rows.append(row)
                continue
            root48 = complex(*row["leading_complex_M1"])
            root64 = complex(*row["leading_complex_M2"])
            if offset == offsets[0]:
                tau_anchor48, tau_anchor64 = root48, root64
            previous48, previous64 = root48, root64
            rows.append(row)
            if previous is not None:
                prev_root = complex(*previous["leading_complex_M2"])
                if np.isfinite(root64.real) and prev_root.real * root64.real <= 0:
                    crossings.append({
                        "tau": tau,
                        "lambda_bracket": [previous["lambda"], lam],
                        "root_left": previous["leading_complex_M2"],
                        "root_right": [root64.real, root64.imag],
                    })
            previous = row
    refined_crossings = []
    if not smoke:
        for crossing in crossings:
            lo, hi = sorted(crossing["lambda_bracket"])
            state_lo, ok_lo = woodbury_newton(
                coup, state, lo, 20.0, tol=1e-12
            )
            state_hi, ok_hi = woodbury_newton(
                coup, state, hi, 20.0, tol=1e-12
            )
            if not (ok_lo and ok_hi):
                continue
            root_lo, _ = tracked_complex(
                roots_at(coup, state_lo, lo, crossing["tau"], 64),
                complex(*crossing["root_left"]),
            )
            root_hi, _ = tracked_complex(
                roots_at(coup, state_hi, hi, crossing["tau"], 64),
                complex(*crossing["root_right"]),
            )
            if (
                not np.isfinite(root_lo.real)
                or not np.isfinite(root_hi.real)
                or root_lo.real * root_hi.real > 0
            ):
                continue
            for _ in range(32):
                if hi - lo <= 1e-6:
                    break
                mid = 0.5 * (lo + hi)
                state_mid, ok_mid = woodbury_newton(
                    coup, state_lo, mid, 20.0, tol=1e-12
                )
                if not ok_mid:
                    break
                root_mid, _ = tracked_complex(
                    roots_at(coup, state_mid, mid, crossing["tau"], 64),
                    0.5 * (root_lo + root_hi),
                )
                if not np.isfinite(root_mid.real):
                    break
                if root_lo.real * root_mid.real <= 0:
                    hi, state_hi, root_hi = mid, state_mid, root_mid
                else:
                    lo, state_lo, root_lo = mid, state_mid, root_mid
            lam_cross = 0.5 * (lo + hi)
            state_cross, ok_cross = woodbury_newton(
                coup, state_lo, lam_cross, 20.0, tol=1e-12
            )
            if not ok_cross:
                continue
            reference = 0.5 * (root_lo + root_hi)
            root64, residual64 = tracked_complex(
                roots_at(coup, state_cross, lam_cross, crossing["tau"], 64),
                reference,
            )
            root48, residual48 = tracked_complex(
                roots_at(coup, state_cross, lam_cross, crossing["tau"], 48),
                reference,
            )
            transversality = (root_hi.real - root_lo.real) / max(hi - lo, 1e-30)
            refined_crossings.append({
                **crossing,
                "lambda_crossing": lam_cross,
                "lambda_width": hi - lo,
                "root_M48": [root48.real, root48.imag],
                "root_M64": [root64.real, root64.imag],
                "resolution_difference": conjugacy_aware_difference(
                    root48, root64
                ),
                "selected_residual_M48": residual48,
                "selected_residual_M64": residual64,
                "transversality": float(transversality),
            })

    probes = []
    if not smoke:
        for crossing in refined_crossings:
            lam = float(crossing["lambda_crossing"])
            solved, ok = woodbury_newton(coup, state, lam, 20.0, tol=1e-12)
            crossing_reference = complex(*crossing["root_M64"])
            root, _ = tracked_complex(
                roots_at(coup, solved, lam, crossing["tau"], 64),
                crossing_reference,
            )
            if ok and np.isfinite(root.real):
                for side in (-1, 1):
                    lam_side = lam + side * 1e-6
                    side_state, side_ok = woodbury_newton(
                        coup, solved, lam_side, 20.0, tol=1e-12
                    )
                    if not side_ok:
                        continue
                    side_root, _ = tracked_complex(
                        roots_at(
                            coup, side_state, lam_side, crossing["tau"], 64
                        ),
                        root,
                    )
                    if not np.isfinite(side_root.real):
                        continue
                    for amplitude in (1e-6, 1e-5, 1e-4):
                        path = (
                            RUNS / "N5B" /
                            f"probe_s{seed}_tau{crossing['tau']}_lam{lam_side:.7f}"
                            f"_eps{amplitude:g}.npz"
                        )
                        probes.append(nonlinear_probe(
                            xi, side_state, lam_side, crossing["tau"],
                            side_root, amplitude, side, path,
                        ))
    result = {
        "seed": seed, "N": N, "P": P, "static": static,
        "fold": fold, "spectral_rows": rows,
        "candidate_crossings": crossings,
        "refined_crossings": refined_crossings,
        "nonlinear_probes": probes,
    }
    write_json(checkpoint, result)
    return result


def pinned_controls(
    N: int, smoke: bool, seeds: list[int] | None = None
) -> list[dict]:
    rows = []
    seeds = ([42] if smoke else list(range(42, 62))) if seeds is None else seeds
    variant = "_smoke" if smoke else ""
    for seed in seeds:
        path = RUNS / "N1" / f"static_N{N}_P100_seed{seed}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        valid = np.flatnonzero(data["resolved"])
        selected = int(valid[np.argmax(data["lambda_low"][valid])])
        fold = 0.5 * (
            float(data["lambda_low"][selected])
            + float(data["lambda_high"][selected])
        )
        state = data["state"][selected]
        xi, xis = make_patterns(N, 100, seed)
        coup = Couplings(xi, xis)
        for tau in ([10] if smoke else PINNED_TAUS):
            for relative in ([1e-3] if smoke else [1e-3, 5e-4, 2e-4]):
                tag = f"{relative:g}".replace(".", "p")
                checkpoint = (
                    RUNS / "N5B" /
                    f"pinned_N{N}_s{seed}_tau{tau:g}_rel{tag}"
                    f"{variant}.json"
                )
                if (
                    checkpoint.exists()
                    and os.environ.get("V5_FORCE", "0") != "1"
                ):
                    rows.append(json.loads(
                        checkpoint.read_text(encoding="utf-8")
                    ))
                    continue
                lam = fold - relative
                solved, ok = woodbury_newton(coup, state, lam, 20.0, tol=1e-12)
                if not ok:
                    continue
                spectrum = roots_at(coup, solved, lam, tau, 24 if smoke else 64)
                root = spectrum["leading_complex"]
                row = {
                    "seed": seed, "motif": selected, "tau": tau,
                    "relative_below_fold": relative,
                    "complex_margin": float(root.real),
                    "complex_frequency": float(abs(root.imag)),
                    "residual_gate": spectrum["all_residuals_pass"],
                }
                rows.append(row)
                write_json(checkpoint, row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", choices=["memory", "pinned", "all"], default="all")
    parser.add_argument("--pinned-seeds", type=int, nargs="*")
    args = parser.parse_args()
    rows = (
        [memory_scan(seed, args.N, args.smoke) for seed in args.seeds]
        if args.mode in {"memory", "all"} else []
    )
    controls = (
        pinned_controls(args.N, args.smoke, args.pinned_seeds)
        if args.mode in {"pinned", "all"} else []
    )
    if args.smoke:
        summary_name = "smoke.json"
    else:
        memory_tag = "_".join(map(str, args.seeds)) if rows else "none"
        pinned_tag = (
            "_".join(map(str, args.pinned_seeds))
            if args.pinned_seeds else ("all" if controls else "none")
        )
        summary_name = (
            f"summary_{args.mode}_memory_{memory_tag}_pinned_{pinned_tag}.json"
        )
    write_json(
        RUNS / "N5B" / summary_name,
        {
            "campaign": "N5B", "memory_scans": rows,
            "pinned_controls": controls, "smoke": args.smoke,
            "environment": environment_manifest(),
        },
    )
    print(f"N5B completed {len(rows)} memory-branch scans and {len(controls)} controls")


if __name__ == "__main__":
    main()
