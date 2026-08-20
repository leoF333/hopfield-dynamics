"""
E34a node/fold/saddle census with explicit safety gates.

Modes
-----
--dry-run
    Validate and print the resolved plan. No files and no numerical solve.
--smoke
    Hard-capped at N<=128, P<=8 and at most three bonds.
--production --confirm-heavy
    Unlock larger parameters. The explicit confirmation flag is mandatory.

The implementation is serial by design for G0/G1. Parallel production is not
enabled until per-bond timing and memory have been benchmarked.
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
from datetime import datetime, timezone
import json
import os
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
    approach_memory_fold,
    characteristic_spectral_bound,
    continue_node_fold_saddle,
    count_unstable_reduced_spectrum,
    extract_null_mode,
    gram_matrix,
    localize_roots,
    make_iid_patterns,
    NumpyLowRankCouplings,
    reduced_field_coefficients,
    solve_memory_node,
)


DEFAULT_OUTPUT = (
    REPO_DIR / "results" / "2_cycle_rappel_snic" / "data" / "E34")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class RunRecorder:
    """Minimal manifest, event log, heartbeat and atomic checkpoint writer."""

    def __init__(self, output_dir: Path, run_id: str, manifest: dict):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.manifest_path = output_dir / f"E34a_manifest_{run_id}.json"
        self.log_path = output_dir / f"E34a_events_{run_id}.jsonl"
        self.heartbeat_path = output_dir / f"E34a_heartbeat_{run_id}.json"
        self.checkpoint_path = output_dir / f"E34a_checkpoint_{run_id}.npz"
        self.manifest = dict(manifest)
        self.manifest.update(
            run_id=run_id,
            started_at=_utc_now(),
            status="running",
        )
        _atomic_json(self.manifest_path, self.manifest)
        self.heartbeat(stage="initialized")

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

    def checkpoint(self, records: list[dict], P: int) -> None:
        temporary = self.checkpoint_path.with_name(
            f".{self.checkpoint_path.name}.{os.getpid()}.tmp")
        count = len(records)
        a_node = np.full((count, P), np.nan)
        a_saddle = np.full((count, P), np.nan)
        m_node = np.full((count, P), np.nan)
        m_saddle = np.full((count, P), np.nan)
        unstable_mode = np.full((count, P), np.nan)
        for index, record in enumerate(records):
            for key, destination in (
                ("a_node", a_node),
                ("a_saddle", a_saddle),
                ("m_node", m_node),
                ("m_saddle", m_saddle),
                ("unstable_mode", unstable_mode),
            ):
                value = record.get(key)
                if value is not None:
                    destination[index] = value
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                mu=np.array([record["mu"] for record in records], dtype=int),
                status=np.array(
                    [record["status"] for record in records], dtype="U64"),
                target_source=np.array(
                    [record.get("target_source", "unknown")
                     for record in records],
                    dtype="U64",
                ),
                lam_target=np.array(
                    [record["lam_target"] for record in records]),
                lam_fold=np.array(
                    [record.get("lam_fold", np.nan) for record in records]),
                lam_fold_reference=np.array([
                    record.get("lam_fold_reference", np.nan)
                    for record in records
                ]),
                fold_mismatch=np.array([
                    record.get("fold_mismatch", np.nan)
                    for record in records
                ]),
                fold_crosscheck_tol=np.array([
                    record.get("fold_crosscheck_tol", np.nan)
                    for record in records
                ]),
                turning_count=np.array([
                    record.get("turning_count", -1)
                    for record in records
                ], dtype=int),
                node_residual=np.array(
                    [record.get("node_residual", np.nan) for record in records]),
                saddle_residual=np.array(
                    [record.get("saddle_residual", np.nan) for record in records]),
                node_eigmax=np.array(
                    [record.get("node_eigmax", np.nan) for record in records]),
                saddle_eigmax=np.array(
                    [record.get("saddle_eigmax", np.nan) for record in records]),
                unstable_count_node=np.array(
                    [record.get("unstable_count_node", -1) for record in records],
                    dtype=int,
                ),
                unstable_count_saddle=np.array(
                    [record.get("unstable_count_saddle", -1) for record in records],
                    dtype=int,
                ),
                root_real=np.array(
                    [record.get("root_real", np.nan) for record in records]),
                root_imag=np.array(
                    [record.get("root_imag", np.nan) for record in records]),
                root_residual=np.array(
                    [record.get("root_residual", np.nan) for record in records]),
                mode_residual=np.array(
                    [record.get("mode_residual", np.nan) for record in records]),
                a_node=a_node,
                a_saddle=a_saddle,
                m_node=m_node,
                m_saddle=m_saddle,
                unstable_mode=unstable_mode,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.checkpoint_path)

    def finish(self, status: str, **fields) -> None:
        self.manifest.update(
            status=status,
            finished_at=_utc_now(),
            **fields,
        )
        _atomic_json(self.manifest_path, self.manifest)
        self.heartbeat(stage=status, **fields)


def _parse_links(text: str, P: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(P))
    try:
        links = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--links must be 'all' or comma-separated integers") from error
    if not links:
        raise ValueError("--links resolved to an empty list")
    if len(set(links)) != len(links):
        raise ValueError("--links contains duplicates")
    if any(mu < 0 or mu >= P for mu in links):
        raise ValueError(f"all links must lie in [0,{P})")
    return links


def _count_parameter_turns(
    parameters: np.ndarray,
    *,
    difference_tolerance: float = 1e-8,
) -> int:
    """Count resolved changes of continuation direction in lambda."""
    differences = np.diff(np.asarray(parameters, dtype=float))
    signs = np.sign(differences[np.abs(differences) > difference_tolerance])
    if not len(signs):
        return 0
    compressed = signs[np.r_[True, signs[1:] != signs[:-1]]]
    return int(np.sum(compressed[1:] != compressed[:-1]))


def _resolved_plan(args: argparse.Namespace) -> dict:
    mode = "dry-run" if args.dry_run else "smoke" if args.smoke else "production"
    N = args.N if args.N is not None else (64 if mode != "production" else 2000)
    P = args.P if args.P is not None else (
        4 if mode != "production" else max(1, round(args.alpha * N)))
    lam_target = args.lam_target
    if lam_target is None:
        lam_target = (
            0.05
            if mode != "production" or args.target_offset is not None
            else 0.21
        )
    links_text = args.links
    if links_text is None:
        links_text = "0" if mode != "production" else "all"
    links = _parse_links(links_text, P)

    if mode == "smoke":
        if N > 128 or P > 8:
            raise ValueError("--smoke enforces N<=128 and P<=8")
        if len(links) > 3:
            raise ValueError("--smoke enforces at most three links")
    if mode == "production" and not args.confirm_heavy:
        raise ValueError(
            "--production is locked; add --confirm-heavy only after approval")
    if N <= 0 or P <= 0 or P > N:
        raise ValueError("require 0 < P <= N")
    if not (0 <= lam_target < args.lam_limit):
        raise ValueError("require 0 <= lam-target < lam-limit")
    if args.target_offset is not None:
        if not (4e-3 < args.target_offset < args.lam_limit):
            raise ValueError(
                "--target-offset must exceed the 0.004 fold margin "
                "and remain below lam-limit")
    if args.fold_crosscheck_tol <= 0:
        raise ValueError("--fold-crosscheck-tol must be positive")

    return dict(
        mode=mode,
        N=N,
        P=P,
        alpha=P / N,
        seed=args.seed,
        beta=args.beta,
        tau=args.tau,
        t0=args.t0,
        lam_target=lam_target,
        target_mode=(
            "local_offset" if args.target_offset is not None else "common_lambda"),
        target_offset=args.target_offset,
        fold_crosscheck_tol=args.fold_crosscheck_tol,
        lam_limit=args.lam_limit,
        links=links,
        thresholds=str(args.thresholds) if args.thresholds else None,
        output_dir=str(args.output_dir.resolve()),
        serial=True,
        safety_caps=(
            dict(N_max=128, P_max=8, links_max=3)
            if mode == "smoke" else None
        ),
    )


def _spectral_analysis(
    coup,
    u: np.ndarray,
    beta: float,
    lam: float,
    tau: float,
    t0: float,
    Q: np.ndarray,
    *,
    localize: bool,
) -> dict:
    from reduced_spectrum import GJ_GK, T_P

    GJ, GK = GJ_GK(coup, u, beta, lam)
    rho = characteristic_spectral_bound(GJ, GK, lam)
    eta = max(0.05, 0.02 * rho / t0)
    count = count_unstable_reduced_spectrum(
        GJ,
        GK,
        t0,
        tau,
        lam,
        samples_per_edge=16,
        max_refinements=7,
    )
    shifted_count = count_unstable_reduced_spectrum(
        GJ,
        GK,
        t0,
        tau,
        lam,
        eta=1.5 * eta,
        samples_per_edge=16,
        max_refinements=7,
    )
    shifted_agreement = bool(
        count.converged
        and shifted_count.converged
        and count.count == shifted_count.count
    )
    result = dict(
        count=count.count if count.count is not None else -1,
        count_ok=shifted_agreement,
        count_reason=(
            count.reason
            if shifted_agreement
            else f"base={count.reason}; shifted={shifted_count.reason}"
        ),
        shifted_agreement=shifted_agreement,
    )
    if not localize or count.count != 1 or not shifted_agreement:
        return result

    identity = np.eye(GJ.shape[0], dtype=complex)

    def characteristic(z):
        return T_P(z, GJ, GK, t0, tau, lam)

    def derivative(z):
        return (
            t0 * identity
            + lam * tau * np.exp(-z * tau) * GK
        )

    localization = localize_roots(
        characteristic,
        count.rectangle,
        derivative=derivative,
        parent_result=count,
        target_width=5e-4,
        target_height=5e-4,
        max_depth=16,
        argument_kwargs=dict(
            samples_per_edge=12,
            max_refinements=7,
        ),
        polish_residual_tol=1e-9,
    )
    result.update(
        localization_ok=localization.converged,
        localization_reason=localization.reason,
    )
    if not localization.converged or len(localization.roots) != 1:
        return result
    root = localization.roots[0]
    mode = extract_null_mode(
        characteristic,
        root.value,
        Q=Q,
        expect_real=True,
    )
    result.update(
        root=root.value,
        root_residual=root.residual,
        mode=mode.vector,
        mode_residual=mode.residual,
        mode_ok=mode.converged,
    )
    return result


def _run(plan: dict) -> int:
    start = time.perf_counter()
    output_dir = Path(plan["output_dir"])
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    manifest = dict(plan)
    manifest.update(
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        numpy_version=np.__version__,
        command=sys.argv,
    )
    recorder = RunRecorder(output_dir, run_id, manifest)
    recorder.event("run_start", plan=plan)

    xi, xi_shift = make_iid_patterns(plan["N"], plan["P"], plan["seed"])
    coup = NumpyLowRankCouplings(xi, xi_shift)
    Q = gram_matrix(xi)
    thresholds = None
    threshold_status = None
    threshold_bracket_width = None
    if plan["thresholds"]:
        with np.load(plan["thresholds"]) as data:
            thresholds = np.asarray(data["lam_c"], dtype=float)
            if "threshold_status" in data:
                threshold_status = np.asarray(
                    data["threshold_status"], dtype=str)
            elif "completed" in data and "fail_reason" in data:
                raise ValueError(
                    "unsafe pre-v2 E34 threshold table: its lam_c values are "
                    "operational Newton edges, not spectrally validated folds")
            if "fold_bracket_width" in data:
                threshold_bracket_width = np.asarray(
                    data["fold_bracket_width"], dtype=float)
            for key, expected in (
                ("N", plan["N"]),
                ("P", plan["P"]),
                ("seed", plan["seed"]),
            ):
                if key in data and int(data[key]) != expected:
                    raise ValueError(
                        f"threshold table {key}={int(data[key])}, "
                        f"expected {expected}")
        if thresholds.shape != (plan["P"],):
            raise ValueError(
                f"threshold table has shape {thresholds.shape}, "
                f"expected {(plan['P'],)}")
        if (
            threshold_status is not None
            and threshold_status.shape != (plan["P"],)
        ):
            raise ValueError(
                f"threshold_status has shape {threshold_status.shape}, "
                f"expected {(plan['P'],)}")
        if (
            threshold_bracket_width is not None
            and threshold_bracket_width.shape != (plan["P"],)
        ):
            raise ValueError(
                f"fold_bracket_width has shape "
                f"{threshold_bracket_width.shape}, expected {(plan['P'],)}")

    records: list[dict] = []
    successes = 0
    for mu in plan["links"]:
        link_start = time.perf_counter()
        recorder.event("link_start", mu=mu)
        target_offset = plan["target_offset"]
        target_source = "common_lambda"
        near_node = None
        near_lam = np.nan
        approach_status = "not_started"

        if target_offset is not None and thresholds is None:
            recorder.heartbeat(stage="fold_probe_node", mu=mu)
            probe_node, probe_ok, probe_residual = solve_memory_node(
                coup, mu, plan["lam_target"], plan["beta"])
            if not probe_ok:
                record = dict(
                    mu=mu,
                    lam_target=np.nan,
                    target_source="preapproach_failed",
                    node_residual=probe_residual,
                    status="node_failed",
                )
                records.append(record)
                recorder.event(
                    "link_failed", mu=mu, stage="fold_probe_node",
                    residual=probe_residual)
                recorder.checkpoint(records, plan["P"])
                continue
            near_node, near_lam, approach_status = approach_memory_fold(
                coup,
                mu,
                plan["beta"],
                probe_node,
                plan["lam_target"],
                lam_limit=plan["lam_limit"],
            )
            if approach_status not in {"eig_stop", "newton_edge"}:
                record = dict(
                    mu=mu,
                    lam_target=np.nan,
                    target_source=f"preapproach_{approach_status}",
                    node_residual=probe_residual,
                    status="fold_not_bracketed",
                )
                records.append(record)
                recorder.event(
                    "link_failed", mu=mu, stage="fold_preapproach",
                    status=approach_status, near_lam=near_lam)
                recorder.checkpoint(records, plan["P"])
                continue
            lam_target = float(near_lam - target_offset)
            target_source = f"preapproach_{approach_status}"
        elif target_offset is not None:
            if (
                not np.isfinite(thresholds[mu])
                or (
                    threshold_status is not None
                    and threshold_status[mu] != "spectral_fold"
                )
            ):
                record = dict(
                    mu=mu,
                    lam_target=np.nan,
                    target_source="threshold_table_indeterminate",
                    node_residual=np.nan,
                    status="threshold_indeterminate",
                )
                records.append(record)
                recorder.event(
                    "link_failed",
                    mu=mu,
                    stage="target_resolution",
                    threshold_status=(
                        str(threshold_status[mu])
                        if threshold_status is not None else "nonfinite"),
                )
                recorder.checkpoint(records, plan["P"])
                continue
            lam_target = float(thresholds[mu] - target_offset)
            target_source = "threshold_table_offset"
        else:
            lam_target = float(plan["lam_target"])

        if not (0 <= lam_target < plan["lam_limit"]):
            record = dict(
                mu=mu,
                lam_target=lam_target,
                target_source=target_source,
                node_residual=np.nan,
                status="invalid_local_target",
            )
            records.append(record)
            recorder.event(
                "link_failed", mu=mu, stage="target_resolution",
                lam_target=lam_target, target_source=target_source)
            recorder.checkpoint(records, plan["P"])
            continue

        recorder.heartbeat(
            stage="node",
            mu=mu,
            lam_target=lam_target,
            target_source=target_source,
        )
        node, node_ok, node_residual = solve_memory_node(
            coup,
            mu,
            lam_target,
            plan["beta"],
        )
        record = dict(
            mu=mu,
            lam_target=lam_target,
            target_source=target_source,
            node_residual=node_residual,
            status="node_failed",
        )
        if not node_ok:
            records.append(record)
            recorder.event(
                "link_failed", mu=mu, stage="node", residual=node_residual)
            recorder.checkpoint(records, plan["P"])
            continue

        if near_node is not None:
            pass
        elif thresholds is None:
            near_node, near_lam, approach_status = approach_memory_fold(
                coup,
                mu,
                plan["beta"],
                node,
                lam_target,
                lam_limit=plan["lam_limit"],
            )
        else:
            near_limit = float(thresholds[mu] - 4e-3)
            if near_limit <= lam_target:
                raise ValueError(
                    f"lambda_c[{mu}] leaves no 0.004 margin above lam_target")
            near_node, near_lam, approach_status = approach_memory_fold(
                coup,
                mu,
                plan["beta"],
                node,
                lam_target,
                lam_limit=near_limit,
                eig_stop=1e9,
            )
        recorder.event(
            "fold_approach",
            mu=mu,
            near_lam=near_lam,
            lam_target=lam_target,
            target_source=target_source,
            status=approach_status,
        )

        last_heartbeat_step = -10

        def progress(step, lam, residual, turned):
            nonlocal last_heartbeat_step
            if step - last_heartbeat_step >= 10 or turned:
                recorder.heartbeat(
                    stage="arclength",
                    mu=mu,
                    step=step,
                    lam=lam,
                    residual=residual,
                    turned=turned,
                )
                last_heartbeat_step = step

        pair = continue_node_fold_saddle(
            coup,
            mu,
            plan["beta"],
            node,
            lam_target,
            near_node,
            near_lam,
            progress_callback=progress,
        )
        turning_count = _count_parameter_turns(pair.trace.parameters)
        fold_reference = (
            float(thresholds[mu]) if thresholds is not None else np.nan)
        fold_tolerance = float(plan["fold_crosscheck_tol"])
        if (
            threshold_bracket_width is not None
            and np.isfinite(threshold_bracket_width[mu])
        ):
            fold_tolerance = max(
                fold_tolerance,
                2.5 * float(threshold_bracket_width[mu]),
            )
        fold_mismatch = (
            abs(float(pair.fold_parameter) - fold_reference)
            if np.isfinite(fold_reference) else np.nan
        )
        record.update(
            lam_fold=pair.fold_parameter,
            lam_fold_reference=fold_reference,
            fold_mismatch=fold_mismatch,
            fold_crosscheck_tol=fold_tolerance,
            turning_count=turning_count,
            node_eigmax=pair.node_eigmax,
            saddle_eigmax=(
                pair.saddle_eigmax
                if pair.saddle_eigmax is not None else np.nan),
            a_node=reduced_field_coefficients(coup, node, Q),
            m_node=coup.overlap_raw(np.tanh(plan["beta"] * node)),
        )
        if not pair.converged or pair.saddle is None:
            identity_lost_after_turn = bool(
                pair.trace.turned and pair.trace.status == "identity_failure")
            if plan["target_mode"] == "common_lambda" and identity_lost_after_turn:
                record["status"] = "identity_lost_common"
            else:
                record["status"] = "continuation_failed_local"
            records.append(record)
            recorder.event(
                "link_failed",
                mu=mu,
                stage="continuation",
                reason=pair.reason,
                trace_status=pair.trace.status,
                turned=pair.trace.turned,
            )
            recorder.checkpoint(records, plan["P"])
            continue

        saddle = pair.saddle
        saddle_residual = float(
            np.linalg.norm(
                coup.field_F(saddle, lam_target, plan["beta"]))
            / np.sqrt(plan["N"])
        )
        record.update(
            a_saddle=reduced_field_coefficients(coup, saddle, Q),
            m_saddle=coup.overlap_raw(np.tanh(plan["beta"] * saddle)),
            saddle_residual=saddle_residual,
        )
        mismatch_failed = bool(
            np.isfinite(fold_mismatch) and fold_mismatch > fold_tolerance)
        multifold_failed = turning_count != 1
        if mismatch_failed or multifold_failed:
            record["status"] = (
                "fold_mismatch_indeterminate"
                if mismatch_failed else "multifold_indeterminate"
            )
            records.append(record)
            recorder.event(
                "link_failed",
                mu=mu,
                stage="fold_crosscheck",
                lam_fold=pair.fold_parameter,
                lam_fold_reference=fold_reference,
                mismatch=fold_mismatch,
                tolerance=fold_tolerance,
                turning_count=turning_count,
                status=record["status"],
            )
            recorder.checkpoint(records, plan["P"])
            continue

        recorder.heartbeat(stage="spectral_node", mu=mu)
        node_spectrum = _spectral_analysis(
            coup,
            node,
            plan["beta"],
            lam_target,
            plan["tau"],
            plan["t0"],
            Q,
            localize=False,
        )
        recorder.heartbeat(stage="spectral_saddle", mu=mu)
        saddle_spectrum = _spectral_analysis(
            coup,
            saddle,
            plan["beta"],
            lam_target,
            plan["tau"],
            plan["t0"],
            Q,
            localize=True,
        )
        record.update(
            unstable_count_node=node_spectrum["count"],
            unstable_count_saddle=saddle_spectrum["count"],
            a_saddle=reduced_field_coefficients(coup, saddle, Q),
            m_saddle=coup.overlap_raw(np.tanh(plan["beta"] * saddle)),
            saddle_residual=saddle_residual,
        )
        root = saddle_spectrum.get("root")
        if root is not None:
            record.update(
                root_real=float(np.real(root)),
                root_imag=float(np.imag(root)),
                root_residual=saddle_spectrum["root_residual"],
                mode_residual=saddle_spectrum["mode_residual"],
                unstable_mode=saddle_spectrum["mode"],
            )
        spectral_ok = bool(
            node_spectrum["count_ok"]
            and node_spectrum["count"] == 0
            and saddle_spectrum["count_ok"]
            and saddle_spectrum["count"] == 1
            and saddle_spectrum.get("localization_ok", False)
            and saddle_spectrum.get("mode_ok", False)
        )
        record["status"] = "ok" if spectral_ok else "spectral_indeterminate"
        if spectral_ok:
            successes += 1
        records.append(record)
        recorder.checkpoint(records, plan["P"])
        recorder.event(
            "link_complete",
            mu=mu,
            status=record["status"],
            elapsed_seconds=time.perf_counter() - link_start,
        )

    elapsed = time.perf_counter() - start
    final_path = output_dir / f"E34a_census_{run_id}.npz"
    os.replace(recorder.checkpoint_path, final_path)
    final_status = "complete" if successes == len(plan["links"]) else "partial"
    recorder.finish(
        final_status,
        elapsed_seconds=elapsed,
        successes=successes,
        requested=len(plan["links"]),
        result_file=str(final_path),
    )
    recorder.event(
        "run_complete",
        status=final_status,
        elapsed_seconds=elapsed,
        successes=successes,
    )
    print(json.dumps(dict(
        status=final_status,
        elapsed_seconds=elapsed,
        successes=successes,
        requested=len(plan["links"]),
        result_file=str(final_path),
        manifest=str(recorder.manifest_path),
    ), indent=2))
    return 0 if successes == len(plan["links"]) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--production", action="store_true")
    parser.add_argument(
        "--confirm-heavy",
        action="store_true",
        help="Mandatory explicit unlock for --production.",
    )
    parser.add_argument("--N", type=int)
    parser.add_argument("--P", type=int)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument(
        "--lam-target",
        type=float,
        help=(
            "Common scientific target. With --target-offset and no threshold "
            "table, this is only the low-lambda seed for the fold pre-approach."
        ),
    )
    parser.add_argument(
        "--target-offset",
        type=float,
        help=(
            "Use a per-link local target lambda_c(mu)-offset. lambda_c comes "
            "from --thresholds or a safe pre-approach. Recommended G1-local: 0.02."
        ),
    )
    parser.add_argument(
        "--fold-crosscheck-tol",
        type=float,
        default=2e-3,
        help=(
            "Maximum allowed |fold from census continuation - fold from "
            "threshold table|. The table uncertainty can only enlarge this."
        ),
    )
    parser.add_argument("--lam-limit", type=float, default=0.8)
    parser.add_argument("--links")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = _resolved_plan(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _run(plan)


if __name__ == "__main__":
    raise SystemExit(main())
