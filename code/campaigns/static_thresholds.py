"""Per-motif static fold thresholds with the N1 publication diagnostics."""
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
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from couplings import Couplings, make_patterns
from v5_paths import RUNS, activate_reference_modules, environment_manifest, write_json

activate_reference_modules()
from robust_branch import eigmax_M, woodbury_newton  # noqa: E402

_CTX: dict = {}


@dataclass
class StaticConfig:
    N: int = 2000
    P: int = 100
    seed: int = 42
    beta: float = 20.0
    initial_step: float = 0.02
    bracket_tol: float = 2e-4
    lambda_max: float = 0.60
    relaxed_identity: bool = False


def _initialize_worker(config: StaticConfig, xi_override: np.ndarray | None = None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if xi_override is None:
        xi, xis = make_patterns(config.N, config.P, config.seed)
    else:
        xi = np.asarray(xi_override, dtype=np.float64)
        xis = np.roll(xi, -1, axis=0)
    coup = Couplings(xi, xis)
    gram = xi @ xi.T / config.N
    _CTX.clear()
    _CTX.update(config=config, xi=xi, coup=coup, gram=gram)


def _amplitudes(u: np.ndarray) -> np.ndarray:
    gram = _CTX["gram"]
    overlap = _CTX["coup"].overlap_raw(u)
    return np.linalg.lstsq(gram, overlap, rcond=1e-12)[0]


def _branch_identity(u: np.ndarray, mu: int) -> tuple[bool, np.ndarray]:
    cfg: StaticConfig = _CTX["config"]
    a = _amplitudes(u)
    order = np.argsort(np.abs(a))[::-1]
    if int(order[0]) != mu:
        return False, a
    if cfg.relaxed_identity:
        return True, a
    second_ok = (
        int(order[1]) == (mu + 1) % cfg.P
        or np.abs(a[order[1]]) < 0.1
    )
    third = np.abs(a[order[2]]) if len(order) > 2 else 0.0
    return bool(second_ok and third < 0.15), a


def _solve_connected(u_start: np.ndarray, mu: int, lam: float):
    cfg: StaticConfig = _CTX["config"]
    coup: Couplings = _CTX["coup"]
    state, converged = woodbury_newton(coup, u_start, lam, cfg.beta)
    if not converged:
        return None
    identity, amp = _branch_identity(state, mu)
    residual = np.linalg.norm(coup.field_F(state, lam, cfg.beta)) / np.sqrt(cfg.N)
    if not identity or residual >= 1e-9:
        return None
    return state, amp, residual


def _annealed_seed(mu: int) -> tuple[np.ndarray | None, str]:
    cfg: StaticConfig = _CTX["config"]
    coup: Couplings = _CTX["coup"]
    state = 0.99 * _CTX["xi"][mu].copy()
    for beta in (5.0, 8.0, 12.0, 16.0, cfg.beta):
        state, ok = woodbury_newton(coup, state, 0.0, beta)
        if not ok:
            return None, f"annealing_failed_beta_{beta:g}"
    identity, _ = _branch_identity(state, mu)
    return (state, "") if identity else (None, "identity_failed_at_lambda0")


def _low_rank_spectrum(u: np.ndarray, lam: float) -> np.ndarray:
    cfg: StaticConfig = _CTX["config"]
    coup: Couplings = _CTX["coup"]
    gain = coup.compute_gain(u, cfg.beta)
    Ut = (1.0 - lam) * coup._xi_np + lam * coup._xis_np
    V = coup._xi_np * gain[None, :]
    return -1.0 + np.linalg.eigvals(V @ Ut.T / cfg.N)


def trace_motif(mu: int) -> dict:
    cfg: StaticConfig = _CTX["config"]
    coup: Couplings = _CTX["coup"]
    state, failure = _annealed_seed(mu)
    empty = {
        "mu": mu, "resolved": False, "failure": failure,
        "lambda_low": np.nan, "lambda_high": np.nan,
        "state": np.full(cfg.N, np.nan), "amplitudes": np.full(cfg.P, np.nan),
        "binary_overlaps": np.full(cfg.P, np.nan), "residual": np.nan,
        "leading_real": np.nan, "leading_complex_real": np.nan,
        "twin_warning": False, "twin_separation": np.nan,
        "largest_step_jump": np.nan,
    }
    if state is None:
        return empty

    lam = 0.0
    step = cfg.initial_step
    last_amp = _amplitudes(state)
    largest_jump = 0.0
    # Connected warm-start walk. A failed step is the upper bracket candidate.
    while lam < cfg.lambda_max:
        trial_lam = min(lam + step, cfg.lambda_max)
        result = _solve_connected(state, mu, trial_lam)
        if result is not None:
            new_state, new_amp, _ = result
            jump = np.linalg.norm(new_amp - last_amp)
            largest_jump = max(largest_jump, float(jump))
            state, last_amp, lam = new_state, new_amp, trial_lam
            step = min(cfg.initial_step, 1.15 * step)
            if lam >= cfg.lambda_max:
                empty["failure"] = "no_fold_below_lambda_max"
                empty["largest_step_jump"] = largest_jump
                return empty
        else:
            if step <= cfg.bracket_tol:
                break
            step *= 0.5

    low = lam
    high = min(cfg.lambda_max, lam + max(2.0 * step, cfg.bracket_tol))
    # Ensure high really fails; the branch can sometimes survive a shrunken step.
    while high < cfg.lambda_max and _solve_connected(state, mu, high) is not None:
        low = high
        accepted = _solve_connected(state, mu, high)
        state, last_amp, _ = accepted
        high = min(cfg.lambda_max, high + max(step, cfg.bracket_tol))
    if high >= cfg.lambda_max and _solve_connected(state, mu, high) is not None:
        empty["failure"] = "no_failed_upper_bracket"
        empty["largest_step_jump"] = largest_jump
        return empty

    for _ in range(32):
        if high - low <= cfg.bracket_tol:
            break
        mid = 0.5 * (low + high)
        result = _solve_connected(state, mu, mid)
        if result is None:
            high = mid
        else:
            new_state, new_amp, _ = result
            largest_jump = max(largest_jump, float(np.linalg.norm(new_amp - last_amp)))
            low, state, last_amp = mid, new_state, new_amp

    residual = np.linalg.norm(coup.field_F(state, low, cfg.beta)) / np.sqrt(cfg.N)
    spectrum = _low_rank_spectrum(state, low)
    real_like = spectrum[np.abs(spectrum.imag) < 1e-8].real
    complex_like = spectrum[np.abs(spectrum.imag) >= 1e-8]
    leading_real = float(real_like.max()) if real_like.size else np.nan
    leading_complex_real = (
        float(complex_like.real.max()) if complex_like.size else np.nan
    )
    binary = coup.overlap_raw(np.tanh(cfg.beta * state))
    # Independent local seeds at the same terminal lambda probe the documented
    # twin micro-structure. A distinct connected solution is flagged, never used
    # to assign or modify the primary threshold.
    twin_separation = 0.0
    direction = (
        _CTX["xi"][(mu + 1) % cfg.P] - _CTX["xi"][mu]
    ) / np.sqrt(cfg.N)
    for amplitude in (-0.05, -0.02, 0.02, 0.05):
        candidate = _solve_connected(
            state + amplitude * direction, mu, low
        )
        if candidate is not None:
            separation = np.linalg.norm(candidate[0] - state) / np.sqrt(cfg.N)
            twin_separation = max(twin_separation, float(separation))
    # A 1e-5 state-space separation is far above the Newton tolerance and the
    # residual gate; it is only a review flag, not a bifurcation criterion.
    twin_warning = bool(twin_separation > 1e-5 or largest_jump > 0.10)
    return {
        "mu": mu, "resolved": True, "failure": "",
        "lambda_low": float(low), "lambda_high": float(high),
        "state": state, "amplitudes": last_amp,
        "binary_overlaps": binary, "residual": float(residual),
        "leading_real": leading_real,
        "leading_complex_real": leading_complex_real,
        "twin_warning": bool(twin_warning),
        "twin_separation": float(twin_separation),
        "largest_step_jump": float(largest_jump),
    }


def _worker(mu: int) -> dict:
    return trace_motif(mu)


def run(config: StaticConfig, output: Path, nproc: int = 1,
        xi_override: np.ndarray | None = None, motifs: list[int] | None = None) -> dict:
    motifs = list(range(config.P)) if motifs is None else motifs
    sidecar = output.with_suffix(".json")
    if (
        output.exists() and sidecar.exists()
        and os.environ.get("V5_FORCE", "0") != "1"
    ):
        with np.load(output) as saved:
            has_current_schema = "twin_separation" in saved.files
        summary = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            has_current_schema
            and summary.get("config") == asdict(config)
            and summary.get("attempted") == len(motifs)
        ):
            return summary
        if has_current_schema:
            raise RuntimeError(
                f"refusing to reuse {output}: saved static configuration differs"
            )
    started = time.perf_counter()
    if nproc == 1:
        _initialize_worker(config, xi_override)
        rows = [trace_motif(mu) for mu in motifs]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            nproc, initializer=_initialize_worker,
            initargs=(config, xi_override)
        ) as pool:
            rows = list(pool.imap_unordered(_worker, motifs))
    rows.sort(key=lambda item: item["mu"])

    n = len(rows)
    mu = np.array([r["mu"] for r in rows], dtype=int)
    resolved = np.array([r["resolved"] for r in rows], dtype=bool)
    lambda_low = np.array([r["lambda_low"] for r in rows])
    lambda_high = np.array([r["lambda_high"] for r in rows])
    arrays = {
        "mu": mu,
        "resolved": resolved,
        "lambda_low": lambda_low,
        "lambda_high": lambda_high,
        "state": np.stack([r["state"] for r in rows]),
        "amplitudes": np.stack([r["amplitudes"] for r in rows]),
        "binary_overlaps": np.stack([r["binary_overlaps"] for r in rows]),
        "residual": np.array([r["residual"] for r in rows]),
        "leading_real": np.array([r["leading_real"] for r in rows]),
        "leading_complex_real": np.array([r["leading_complex_real"] for r in rows]),
        "twin_warning": np.array([r["twin_warning"] for r in rows]),
        "twin_separation": np.array([r["twin_separation"] for r in rows]),
        "largest_step_jump": np.array([r["largest_step_jump"] for r in rows]),
        "failure": np.array([r["failure"] for r in rows], dtype="U80"),
        "N": np.array(config.N), "P": np.array(config.P),
        "seed": np.array(config.seed), "beta": np.array(config.beta),
        "wall_seconds": np.array(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    valid = np.flatnonzero(resolved)
    maximum_position = (
        int(valid[np.nanargmax(lambda_low[valid])]) if valid.size else None
    )
    summary = {
        "config": asdict(config),
        "output": str(output),
        "wall_seconds": float(arrays["wall_seconds"]),
        "resolved": int(resolved.sum()),
        "attempted": n,
        "admissible_n1": bool(resolved.sum() >= min(98, config.P)),
        "maximum_motif": (
            int(mu[maximum_position]) if maximum_position is not None else None
        ),
        "maximum_static_low": (
            float(lambda_low[maximum_position])
            if maximum_position is not None else None
        ),
        "maximum_static_high": (
            float(lambda_high[maximum_position])
            if maximum_position is not None else None
        ),
        "environment": environment_manifest(),
    }
    write_json(sidecar, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--tol", type=float, default=2e-4)
    parser.add_argument("--motifs", type=int, nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = StaticConfig(
        N=args.N, P=args.P, seed=args.seed, beta=args.beta,
        bracket_tol=args.tol,
    )
    output = args.out or (
        RUNS / "N1" / f"static_N{args.N}_P{args.P}_seed{args.seed}.npz"
    )
    summary = run(cfg, output, args.nproc, motifs=args.motifs)
    print(summary)


if __name__ == "__main__":
    main()
