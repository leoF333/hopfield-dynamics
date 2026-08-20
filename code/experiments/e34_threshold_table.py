"""
Safe NumPy-only per-pattern fold table for the E34 pilot.

This replaces E24's MLX-dependent startup path for E34. It is strictly serial,
uses float64 low-rank operators, and never forms an N x N coupling matrix.

Modes
-----
--dry-run
    Resolve and print the plan without writing files.
--smoke
    Hard-capped at N<=128 and P<=8.
--production --confirm-heavy
    Unlock the intended N=400, P=20 pilot only after explicit approval.
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

# Set thread limits before importing NumPy/BLAS.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from e34_lib import (
    NumpyLowRankCouplings,
    gram_matrix,
    make_iid_patterns,
    memory_branch_identity,
    reduced_field_coefficients,
    solve_memory_node,
    trace_upper_fold,
)
from robust_branch import eigmax_M, woodbury_newton, woodbury_solve


DEFAULT_OUTPUT_DIR = (
    REPO_DIR / "results" / "2_cycle_rappel_snic" / "data" / "E34")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_metadata() -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unavailable", None


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ThresholdRecorder:
    """Manifest, JSONL log, heartbeat and checkpoint for a serial threshold run."""

    def __init__(self, output_dir: Path, run_id: str, plan: dict):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.run_id = run_id
        self.manifest_path = output_dir / f"E34thr_manifest_{run_id}.json"
        self.log_path = output_dir / f"E34thr_events_{run_id}.jsonl"
        self.heartbeat_path = output_dir / f"E34thr_heartbeat_{run_id}.json"
        self.checkpoint_path = output_dir / f"E34thr_checkpoint_{run_id}.npz"
        commit, dirty = _git_metadata()
        self.manifest = dict(
            plan,
            run_id=run_id,
            git_commit=commit,
            git_dirty=dirty,
            numpy_version=np.__version__,
            started_at=_utc_now(),
            status="running",
            command=sys.argv,
        )
        _atomic_json(self.manifest_path, self.manifest)
        self.heartbeat("initialized")

    def event(self, event: str, **fields) -> None:
        payload = dict(timestamp=_utc_now(), event=event, **fields)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def heartbeat(self, stage: str, **fields) -> None:
        _atomic_json(
            self.heartbeat_path,
            dict(timestamp=_utc_now(), pid=os.getpid(), stage=stage, **fields),
        )

    def checkpoint(self, payload: dict) -> None:
        _atomic_npz(self.checkpoint_path, payload)

    def finish(self, status: str, **fields) -> None:
        self.manifest.update(
            status=status,
            finished_at=_utc_now(),
            **fields,
        )
        _atomic_json(self.manifest_path, self.manifest)
        self.heartbeat(status, **fields)


def _parse_links(text: str, P: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(P))
    try:
        links = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--links must be 'all' or comma-separated integers") from error
    if not links or len(set(links)) != len(links):
        raise ValueError("--links must be non-empty and contain no duplicates")
    if any(mu < 0 or mu >= P for mu in links):
        raise ValueError(f"all links must lie in [0,{P})")
    return links


def _resolve_plan(args: argparse.Namespace) -> dict:
    mode = "dry-run" if args.dry_run else "smoke" if args.smoke else "production"
    N = args.N if args.N is not None else (64 if mode != "production" else 400)
    P = args.P if args.P is not None else (4 if mode != "production" else 20)
    links = _parse_links(args.links or "all", P)
    if N <= 0 or P <= 0 or P > N:
        raise ValueError("require 0 < P <= N")
    if mode == "smoke" and (N > 128 or P > 8):
        raise ValueError("--smoke enforces N<=128 and P<=8")
    if mode == "production" and not args.confirm_heavy:
        raise ValueError(
            "--production is locked; add --confirm-heavy only after approval")
    if not (0 <= args.lam_start < args.lam_limit):
        raise ValueError("require 0 <= lam-start < lam-limit")
    if not (0 < args.step <= args.lam_limit - args.lam_start):
        raise ValueError("--step must be positive and fit inside the lambda range")
    if args.bisect_tol <= 0 or args.bisect_tol >= args.step:
        raise ValueError("require 0 < bisect-tol < step")
    return dict(
        mode=mode,
        N=N,
        P=P,
        alpha=P / N,
        seed=args.seed,
        beta=args.beta,
        lam_start=args.lam_start,
        lam_limit=args.lam_limit,
        step=args.step,
        bisect_tol=args.bisect_tol,
        links=links,
        store_curves=args.store_curves,
        serial=True,
        blas_threads=1,
        output_dir=str(args.output_dir.resolve()),
        safety_caps=(
            dict(N_max=128, P_max=8) if mode == "smoke" else None),
    )


def _node_identity(coup, u: np.ndarray, mu: int) -> bool:
    """Node-side identity: mu must remain the leading field coefficient."""
    coefficients = reduced_field_coefficients(coup, u)
    return bool(
        int(np.argmax(np.abs(coefficients))) == mu
        and memory_branch_identity(coup, u, mu)
    )


def _try_node(
    coup,
    u_start: np.ndarray,
    mu: int,
    lam: float,
    beta: float,
) -> tuple[np.ndarray | None, str, float, float]:
    u, ok = woodbury_newton(
        coup, np.asarray(u_start, float), lam, beta, tol=1e-11, max_iter=80)
    residual = float(
        np.linalg.norm(coup.field_F(u, lam, beta)) / np.sqrt(coup.N))
    eigenvalue = float(eigmax_M(coup, u, lam, beta))
    if not ok:
        return None, "newton", residual, eigenvalue
    if residual >= 1e-9:
        return None, "residual", residual, eigenvalue
    if not _node_identity(coup, u, mu):
        return None, "identity", residual, eigenvalue
    if eigenvalue >= 0:
        return None, "static_unstable", residual, eigenvalue
    return u, "", residual, eigenvalue


def _indeterminate_result(
    mu: int,
    status: str,
    detail: str,
    rows: list[list[float]],
    *,
    high_reason: str = "",
    operational_low: float = np.nan,
    operational_high: float = np.nan,
    operational_high_eig: float = np.nan,
    operational_high_residual: float = np.nan,
    threshold_side_eig: float = np.nan,
    threshold_side_residual: float = np.nan,
    arc_rows: np.ndarray | None = None,
) -> dict:
    """Return an explicit non-fold record; never promote a Newton edge to lam_c."""
    width = (
        operational_high - operational_low
        if np.isfinite(operational_low) and np.isfinite(operational_high)
        else np.nan
    )
    return dict(
        mu=mu,
        lam_c=np.nan,
        lam_extrap=np.nan,
        threshold_status=status,
        fail=status,
        fail_detail=detail,
        high_reason=high_reason,
        operational_low=operational_low,
        operational_high=operational_high,
        operational_width=width,
        operational_high_eig=operational_high_eig,
        operational_high_residual=operational_high_residual,
        threshold_side_eig=threshold_side_eig,
        threshold_side_residual=threshold_side_residual,
        fold_bracket_low=np.nan,
        fold_bracket_high=np.nan,
        fold_bracket_width=np.nan,
        fold_fit_error=np.nan,
        fold_sample_lambda=np.nan,
        fold_sample_eig=np.nan,
        rows=np.asarray(rows, dtype=float).reshape((-1, 5)),
        arc_rows=(
            np.empty((0, 3), dtype=float)
            if arc_rows is None else np.asarray(arc_rows, dtype=float)
        ),
    )


def trace_threshold(
    coup,
    mu: int,
    *,
    beta: float,
    lam_start: float,
    lam_limit: float,
    step_initial: float,
    bisect_tol: float,
    progress=None,
) -> dict:
    """Locate a fold by branch-following and a two-sided spectral crossing.

    Warm-start Newton is used only to approach the fold. A failed Newton solve is
    recorded as an operational edge, never as ``lam_c``. The final threshold is
    accepted only after pseudo-arclength continuation turns in lambda and the
    rightmost static eigenvalue changes sign between the stable and saddle sides.
    """
    u, ok, seed_residual = solve_memory_node(
        coup, mu, lam_start, beta, tolerance=1e-11)
    if not ok:
        return _indeterminate_result(
            mu,
            "seed_failed",
            f"residual={seed_residual:.3e}",
            [],
        )

    lam = float(lam_start)
    rows: list[list[float]] = []

    def record(current_u, current_lam):
        coefficients = reduced_field_coefficients(coup, current_u)
        eigenvalue = float(eigmax_M(coup, current_u, current_lam, beta))
        residual = float(
            np.linalg.norm(coup.field_F(current_u, current_lam, beta))
            / np.sqrt(coup.N))
        rows.append([
            current_lam,
            coefficients[mu],
            coefficients[(mu + 1) % coup.P],
            eigenvalue,
            residual,
        ])
        if progress is not None:
            progress(current_lam, eigenvalue, residual)
        return eigenvalue, residual

    eigenvalue, residual = record(u, lam)
    step = float(step_initial)
    step_min = max(1e-5, bisect_tol / 4.0)
    approach_status = "lam_limit"
    operational_low = np.nan
    operational_high = np.nan
    operational_high_eig = np.nan
    operational_high_residual = np.nan
    high_reason = ""

    while lam < lam_limit - 1e-14:
        trial = min(lam + step, lam_limit)
        candidate, reason, trial_residual, trial_eigenvalue = _try_node(
            coup, u, mu, trial, beta)
        # A sharp reversal away from zero is the signature seen when ordinary
        # Newton jumps to another stable root after passing the desired fold.
        spectral_reversal = bool(
            candidate is not None
            and eigenvalue > -0.5
            and trial_eigenvalue < eigenvalue - 0.06
        )
        if candidate is None or spectral_reversal:
            operational_low = lam
            operational_high = trial
            operational_high_eig = trial_eigenvalue
            operational_high_residual = trial_residual
            if spectral_reversal:
                reason = "spectral_reversal"
            high_reason = reason
            step *= 0.5
            if step < step_min:
                approach_status = "operational_edge"
                break
            continue

        u = candidate
        lam = trial
        eigenvalue, residual = record(u, lam)
        if eigenvalue >= -0.15:
            approach_status = "spectral_near_fold"
            break
        if lam >= lam_limit - 1e-14:
            break
        # eig^2 scales linearly with distance to a generic saddle-node. It is
        # used only to choose the next approach step, not to define the fold.
        step = float(np.clip(
            0.0625 * max(eigenvalue * eigenvalue, bisect_tol),
            step_min,
            step_initial,
        ))

    if approach_status == "lam_limit":
        return _indeterminate_result(
            mu,
            "lam_limit",
            f"stable branch reached {lam_limit:.6f}",
            rows,
            threshold_side_eig=eigenvalue,
            threshold_side_residual=residual,
        )

    Q = gram_matrix(coup._xi_np)

    def residual_function(state, parameter):
        return coup.field_F(state, parameter, beta)

    def linear_solve(state, parameter, rhs):
        return woodbury_solve(
            coup, coup.compute_gain(state, beta), parameter, rhs)

    def parameter_derivative(state, parameter):
        del parameter
        return coup.dF_dlam(state, beta)

    arc_step = min(0.002, max(5e-4, step_initial / 10.0))
    trace = trace_upper_fold(
        u,
        lam,
        residual_function,
        linear_solve,
        parameter_derivative,
        identity_check=lambda state, parameter: memory_branch_identity(
            coup, state, mu, Q=Q),
        stop_after_turn_parameter=lam,
        step_initial=arc_step,
        step_min=min(1e-5, arc_step / 20.0),
        step_max=min(0.004, 2.0 * arc_step),
        max_steps=1200,
        tolerance=1e-11,
    )
    arc_eigenvalues = np.asarray([
        eigmax_M(coup, state, parameter, beta)
        for state, parameter in zip(trace.states, trace.parameters)
    ])
    arc_rows = np.column_stack(
        (trace.parameters, arc_eigenvalues, trace.residuals))
    stable_indices = np.flatnonzero(arc_eigenvalues < -1e-8)
    unstable_indices = np.flatnonzero(arc_eigenvalues > 1e-8)
    if not trace.turned or not len(stable_indices) or not len(unstable_indices):
        return _indeterminate_result(
            mu,
            "arclength_no_spectral_fold",
            (
                f"trace_status={trace.status}, turned={trace.turned}, "
                f"eig_min={arc_eigenvalues.min():.3e}, "
                f"eig_max={arc_eigenvalues.max():.3e}"
            ),
            rows,
            high_reason=high_reason,
            operational_low=operational_low,
            operational_high=operational_high,
            operational_high_eig=operational_high_eig,
            operational_high_residual=operational_high_residual,
            threshold_side_eig=eigenvalue,
            threshold_side_residual=residual,
            arc_rows=arc_rows,
        )

    near = np.abs(arc_eigenvalues) < 0.35
    fit_indices = np.flatnonzero(near)
    if (
        len(fit_indices) < 4
        or np.sum(arc_eigenvalues[fit_indices] < 0) < 2
        or np.sum(arc_eigenvalues[fit_indices] > 0) < 2
    ):
        return _indeterminate_result(
            mu,
            "arclength_insufficient_spectral_points",
            f"near-fold spectral points={len(fit_indices)}",
            rows,
            high_reason=high_reason,
            operational_low=operational_low,
            operational_high=operational_high,
            operational_high_eig=operational_high_eig,
            operational_high_residual=operational_high_residual,
            threshold_side_eig=eigenvalue,
            threshold_side_residual=residual,
            arc_rows=arc_rows,
        )

    eig_squared = arc_eigenvalues[fit_indices] ** 2
    design = np.column_stack((eig_squared, np.ones(len(fit_indices))))
    slope, intercept = np.linalg.lstsq(
        design, trace.parameters[fit_indices], rcond=None)[0]
    fitted = design @ np.array([slope, intercept])
    fit_error = float(np.max(np.abs(
        fitted - trace.parameters[fit_indices])))
    sample_index = int(np.argmax(trace.parameters))
    sample_lambda = float(trace.parameters[sample_index])
    sample_eigenvalue = float(arc_eigenvalues[sample_index])
    fit_tolerance = max(2e-3, 10.0 * bisect_tol)
    fit_valid = bool(
        slope < 0
        and np.isfinite(intercept)
        and fit_error <= fit_tolerance
        and intercept >= sample_lambda - fit_tolerance
        and intercept <= sample_lambda + 0.01
    )
    if not fit_valid:
        return _indeterminate_result(
            mu,
            "arclength_spectral_fit_failed",
            (
                f"slope={slope:.3e}, intercept={intercept:.6f}, "
                f"sample_max={sample_lambda:.6f}, fit_error={fit_error:.3e}"
            ),
            rows,
            high_reason=high_reason,
            operational_low=operational_low,
            operational_high=operational_high,
            operational_high_eig=operational_high_eig,
            operational_high_residual=operational_high_residual,
            threshold_side_eig=eigenvalue,
            threshold_side_residual=residual,
            arc_rows=arc_rows,
        )

    stable_near_index = stable_indices[
        np.argmin(np.abs(arc_eigenvalues[stable_indices]))]
    uncertainty = max(
        bisect_tol,
        fit_error,
        abs(float(intercept) - sample_lambda),
    )
    lam_c = float(intercept)
    return dict(
        mu=mu,
        lam_c=lam_c,
        lam_extrap=lam_c,
        threshold_status="spectral_fold",
        fail="",
        fail_detail="",
        high_reason=high_reason,
        operational_low=operational_low,
        operational_high=operational_high,
        operational_width=(
            operational_high - operational_low
            if np.isfinite(operational_low) and np.isfinite(operational_high)
            else np.nan
        ),
        operational_high_eig=operational_high_eig,
        operational_high_residual=operational_high_residual,
        threshold_side_eig=float(arc_eigenvalues[stable_near_index]),
        threshold_side_residual=float(trace.residuals[stable_near_index]),
        fold_bracket_low=lam_c - uncertainty,
        fold_bracket_high=lam_c + uncertainty,
        fold_bracket_width=2.0 * uncertainty,
        fold_fit_error=fit_error,
        fold_sample_lambda=sample_lambda,
        fold_sample_eig=sample_eigenvalue,
        rows=np.asarray(rows),
        arc_rows=arc_rows,
    )


def _payload(
    plan: dict,
    lam_c: np.ndarray,
    lam_extrap: np.ndarray,
    fail_reason: np.ndarray,
    threshold_status: np.ndarray,
    high_reason: np.ndarray,
    fail_detail: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    completed: np.ndarray,
    curves: dict[int, np.ndarray],
    arc_curves: dict[int, np.ndarray],
) -> dict:
    failures = np.flatnonzero(
        completed & (threshold_status != "spectral_fold")).astype(int)
    payload = dict(
        lam_c=lam_c,
        lam_extrap=lam_extrap,
        N=np.array(plan["N"]),
        P=np.array(plan["P"]),
        seed=np.array(plan["seed"]),
        beta=np.array(plan["beta"]),
        mus=np.asarray(plan["links"], dtype=int),
        fails=failures,
        fail_reason=fail_reason,
        fail_detail=fail_detail,
        threshold_status=threshold_status,
        high_reason=high_reason,
        completed=completed,
        bisect_tol=np.array(plan["bisect_tol"]),
        threshold_schema=np.array(2),
        threshold_method=np.array("pseudo_arclength_spectral_v2"),
        **diagnostics,
    )
    if plan["store_curves"]:
        for mu, rows in curves.items():
            payload[f"curve_{mu}"] = rows
        for mu, rows in arc_curves.items():
            payload[f"arc_curve_{mu}"] = rows
    return payload


def _run(plan: dict) -> int:
    start = time.perf_counter()
    output_dir = Path(plan["output_dir"])
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    recorder = ThresholdRecorder(output_dir, run_id, plan)
    xi, xi_shift = make_iid_patterns(plan["N"], plan["P"], plan["seed"])
    coup = NumpyLowRankCouplings(xi, xi_shift)

    lam_c = np.full(plan["P"], np.nan)
    lam_extrap = np.full(plan["P"], np.nan)
    fail_reason = np.full(plan["P"], "not_requested", dtype="U64")
    threshold_status = np.full(
        plan["P"], "not_requested", dtype="U64")
    high_reason = np.full(plan["P"], "", dtype="U64")
    fail_detail = np.full(plan["P"], "", dtype="U256")
    diagnostic_fields = (
        "operational_low",
        "operational_high",
        "operational_width",
        "operational_high_eig",
        "operational_high_residual",
        "threshold_side_eig",
        "threshold_side_residual",
        "fold_bracket_low",
        "fold_bracket_high",
        "fold_bracket_width",
        "fold_fit_error",
        "fold_sample_lambda",
        "fold_sample_eig",
    )
    diagnostics = {
        field: np.full(plan["P"], np.nan)
        for field in diagnostic_fields
    }
    completed = np.zeros(plan["P"], dtype=bool)
    curves: dict[int, np.ndarray] = {}
    arc_curves: dict[int, np.ndarray] = {}

    for index, mu in enumerate(plan["links"]):
        branch_start = time.perf_counter()
        recorder.event(
            "branch_start", mu=mu, index=index, total=len(plan["links"]))

        last_heartbeat = -np.inf

        def progress(lam, eigenvalue, residual):
            nonlocal last_heartbeat
            if lam - last_heartbeat >= 0.02:
                recorder.heartbeat(
                    "tracing",
                    mu=mu,
                    lam=lam,
                    eigmax=eigenvalue,
                    residual=residual,
                    completed=int(completed.sum()),
                    total=len(plan["links"]),
                )
                last_heartbeat = lam

        result = trace_threshold(
            coup,
            mu,
            beta=plan["beta"],
            lam_start=plan["lam_start"],
            lam_limit=plan["lam_limit"],
            step_initial=plan["step"],
            bisect_tol=plan["bisect_tol"],
            progress=progress,
        )
        lam_c[mu] = result["lam_c"]
        lam_extrap[mu] = result["lam_extrap"]
        fail_reason[mu] = result["fail"] or ""
        threshold_status[mu] = result["threshold_status"]
        high_reason[mu] = result["high_reason"]
        fail_detail[mu] = result["fail_detail"]
        for field in diagnostic_fields:
            diagnostics[field][mu] = result[field]
        completed[mu] = True
        if plan["store_curves"]:
            curves[mu] = result["rows"]
            arc_curves[mu] = result["arc_rows"]
        checkpoint = _payload(
            plan,
            lam_c,
            lam_extrap,
            fail_reason,
            threshold_status,
            high_reason,
            fail_detail,
            diagnostics,
            completed,
            curves,
            arc_curves,
        )
        recorder.checkpoint(checkpoint)
        recorder.event(
            "branch_complete",
            mu=mu,
            lam_c=(
                float(result["lam_c"])
                if np.isfinite(result["lam_c"]) else None),
            threshold_status=result["threshold_status"],
            fail=result["fail"],
            fail_detail=result["fail_detail"],
            high_reason=result["high_reason"],
            threshold_side_eig=result["threshold_side_eig"],
            threshold_side_residual=result["threshold_side_residual"],
            operational_low=result["operational_low"],
            operational_high=result["operational_high"],
            operational_width=result["operational_width"],
            fold_bracket_width=result["fold_bracket_width"],
            elapsed_seconds=time.perf_counter() - branch_start,
        )

    payload = _payload(
        plan,
        lam_c,
        lam_extrap,
        fail_reason,
        threshold_status,
        high_reason,
        fail_detail,
        diagnostics,
        completed,
        curves,
        arc_curves,
    )
    final_path = output_dir / (
        f"E34_thresholds_N{plan['N']}_P{plan['P']}_"
        f"s{plan['seed']}_{run_id}.npz")
    _atomic_npz(final_path, payload)
    elapsed = time.perf_counter() - start
    failures = payload["fails"].tolist()
    status = "complete" if not failures else "partial"
    recorder.finish(
        status,
        elapsed_seconds=elapsed,
        completed=int(completed.sum()),
        requested=len(plan["links"]),
        failures=failures,
        result_file=str(final_path),
    )
    recorder.event(
        "run_complete",
        status=status,
        elapsed_seconds=elapsed,
        failures=failures,
    )
    finite = lam_c[np.asarray(plan["links"], dtype=int)]
    finite = finite[np.isfinite(finite)]
    summary = dict(
        status=status,
        elapsed_seconds=elapsed,
        completed=int(completed.sum()),
        requested=len(plan["links"]),
        failures=failures,
        result_file=str(final_path),
        manifest=str(recorder.manifest_path),
        lam_c_min=float(np.min(finite)) if len(finite) else None,
        lam_c_median=float(np.median(finite)) if len(finite) else None,
        lam_c_max=float(np.max(finite)) if len(finite) else None,
    )
    print(json.dumps(summary, indent=2))
    return 0 if status == "complete" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--production", action="store_true")
    parser.add_argument("--confirm-heavy", action="store_true")
    parser.add_argument("--N", type=int)
    parser.add_argument("--P", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--lam-start", type=float, default=0.05)
    parser.add_argument("--lam-limit", type=float, default=0.8)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--bisect-tol", type=float, default=2e-4)
    parser.add_argument("--links")
    parser.add_argument("--store-curves", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = _resolve_plan(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _run(plan)


if __name__ == "__main__":
    raise SystemExit(main())
