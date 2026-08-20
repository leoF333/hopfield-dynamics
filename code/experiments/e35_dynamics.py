"""E35 DDE adapter, cycle diagnostics, phase decoding and image metrics.

This module does not alter :mod:`cycle_reduced`.  ``MhnReducedDDE`` subclasses
its corrected RK4/Hermite integrator and only replaces the reduced fields.
Restarts always carry both ``hist`` and ``dhist``.
"""

from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from cycle_reduced import ReducedDDE
from e35_video_tools import (
    FramePreprocessor,
    KeyframeSplit,
    baseline_hold,
    baseline_linear,
    baseline_motion_translation,
)
from mhn_reduced import MhnArchitecture


class MhnReducedDDE(ReducedDDE):
    """Use an :class:`MhnArchitecture` with the shared DDE integrator."""

    def __init__(
        self,
        architecture: MhnArchitecture,
        lam: float,
        tau: float,
        t0: float = 1.0,
    ) -> None:
        context = architecture.context
        super().__init__(context.xi, context.beta, lam, tau, t0)
        self.architecture = architecture

    def m(self, a: np.ndarray) -> np.ndarray:
        """Retain the historical name for the Hebbian primitive."""

        return self.architecture.context.hebb_field(a)

    def rhs(self, a_now: np.ndarray, a_del: np.ndarray) -> np.ndarray:
        return self.architecture.rhs(
            a_now, a_del, self.lam, t0=self.t0
        )

    def Dm(self, a: np.ndarray) -> np.ndarray:
        """Instantaneous-storage Jacobian; use ``DK`` for the sequence field."""

        return self.architecture.store_jacobian(a)

    def DK(self, a: np.ndarray) -> np.ndarray:
        return self.architecture.sequence_jacobian(a)


@dataclass(frozen=True)
class CycleDiagnostics:
    sample_count: int
    lead_changes: int
    forward_transitions: int
    forward_fraction: float
    coverage_fraction: float
    tours_estimate: float
    period_count: int
    period_mean: float
    period_cv: float
    seam_transition_count: int
    seam_mean_duration: float
    median_link_duration: float
    seam_duration_ratio: float
    valid_cycle: bool
    reason: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class PeakInterval:
    source: int
    target: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class DynamicDecode:
    predictions: np.ndarray
    valid: np.ndarray
    interval_start: np.ndarray
    interval_end: np.ndarray


def integrate_chunked(
    system: MhnReducedDDE,
    a_hist0: np.ndarray,
    *,
    da_hist0: np.ndarray,
    t_total: float,
    dt: float,
    record_every: int = 1,
    chunk_time: Optional[float] = None,
    on_chunk: Optional[Callable[[dict], None]] = None,
) -> dict[str, np.ndarray]:
    """Integrate in restart-safe chunks and expose checkpoints/heartbeats.

    ``da_hist0`` is mandatory: a caller specifying a constant history should
    pass zeros, while a restart must pass the previous ``dhist``.
    """

    if not np.isfinite(t_total) or t_total <= 0:
        raise ValueError("t_total must be finite and positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if record_every < 1:
        raise ValueError("record_every must be at least one")
    chunk = t_total if chunk_time is None else float(chunk_time)
    if not np.isfinite(chunk) or chunk <= 0:
        raise ValueError("chunk_time must be finite and positive")

    total_steps = int(round(t_total / dt))
    if abs(total_steps * dt - t_total) > 1e-12:
        raise ValueError("dt must divide t_total")
    chunk_steps = max(1, int(round(chunk / dt)))
    if abs(chunk_steps * dt - chunk) > 1e-12:
        raise ValueError("dt must divide chunk_time")
    if chunk_steps % record_every:
        raise ValueError("record_every must divide the number of steps per chunk")
    history = np.asarray(a_hist0, dtype=np.float64)
    derivative_history = np.asarray(da_hist0, dtype=np.float64)
    recorded_t: list[np.ndarray] = []
    recorded_a: list[np.ndarray] = []
    completed_steps = 0

    while completed_steps < total_steps:
        steps = min(chunk_steps, total_steps - completed_steps)
        duration = steps * dt
        solution = system.integrate(
            history,
            duration,
            dt,
            record_every=record_every,
            da_hist0=derivative_history,
        )
        local_t = solution["t"] + completed_steps * dt
        local_a = solution["a"]
        if recorded_t:
            local_t = local_t[1:]
            local_a = local_a[1:]
        recorded_t.append(local_t)
        recorded_a.append(local_a)
        completed_steps += steps
        history = solution["hist"]
        derivative_history = solution["dhist"]
        if on_chunk is not None:
            on_chunk(
                {
                    "completed_steps": completed_steps,
                    "total_steps": total_steps,
                    "simulated_time": completed_steps * dt,
                    "hist": history,
                    "dhist": derivative_history,
                    "nan_count": int(
                        np.size(history) - np.count_nonzero(np.isfinite(history))
                    ),
                }
            )

    return {
        "t": np.concatenate(recorded_t),
        "a": np.concatenate(recorded_a, axis=0),
        "hist": np.array(history, copy=True),
        "dhist": np.array(derivative_history, copy=True),
    }


def classify_cycle(
    times: np.ndarray,
    observables: np.ndarray,
    *,
    transient_fraction: float = 0.5,
) -> CycleDiagnostics:
    """Classify ordered recall from an explicit physical observable.

    E35 callers must pass the physical overlap trajectory ``m(t)``, not the
    coefficient trajectory ``a(t)``.  Keeping this argument generic makes the
    event convention explicit and testable.
    """

    t, observable = _validate_trajectory(times, observables)
    p = observable.shape[1]
    if not 0.0 <= transient_fraction < 1.0:
        raise ValueError("transient_fraction must lie in [0,1)")
    start = min(len(t) - 2, int(np.floor(transient_fraction * len(t))))
    t = t[start:]
    observable = observable[start:]
    lead = np.argmax(observable, axis=1)
    change_indices = np.flatnonzero(np.diff(lead) != 0) + 1
    if change_indices.size == 0:
        return CycleDiagnostics(
            sample_count=len(t),
            lead_changes=0,
            forward_transitions=0,
            forward_fraction=0.0,
            coverage_fraction=len(np.unique(lead)) / p,
            tours_estimate=0.0,
            period_count=0,
            period_mean=np.nan,
            period_cv=np.nan,
            seam_transition_count=0,
            seam_mean_duration=np.nan,
            median_link_duration=np.nan,
            seam_duration_ratio=np.nan,
            valid_cycle=False,
            reason="no_lead_change",
        )

    event_indices = np.concatenate(([0], change_indices))
    event_lead = lead[event_indices]
    source = event_lead[:-1]
    target = event_lead[1:]
    forward = (target - source) % p == 1
    forward_count = int(np.count_nonzero(forward))
    forward_fraction = forward_count / len(source)
    coverage = len(np.unique(lead)) / p
    tours = forward_count / p

    durations = np.diff(t[event_indices])
    transition_source = source
    seam_mask = (transition_source == p - 1) & (target == 0)
    seam_durations = durations[seam_mask]
    forward_durations = durations[forward]
    median_link = (
        float(np.median(forward_durations))
        if forward_durations.size
        else np.nan
    )
    seam_mean = (
        float(np.mean(seam_durations)) if seam_durations.size else np.nan
    )
    seam_ratio = (
        seam_mean / median_link
        if np.isfinite(seam_mean) and np.isfinite(median_link) and median_link > 0
        else np.nan
    )

    reference_entries = t[change_indices[target == 0]]
    periods = np.diff(reference_entries)
    period_mean = float(np.mean(periods)) if periods.size else np.nan
    period_cv = (
        float(np.std(periods, ddof=1) / period_mean)
        if periods.size >= 2 and period_mean > 0
        else np.nan
    )
    valid = bool(
        coverage == 1.0
        and forward_fraction >= 0.95
        and tours >= 3.0
        and periods.size >= 2
        and period_cv <= 0.05
    )
    failed = []
    if coverage < 1.0:
        failed.append("incomplete_coverage")
    if forward_fraction < 0.95:
        failed.append("non_forward_transitions")
    if tours < 3.0:
        failed.append("fewer_than_three_tours")
    if periods.size < 2:
        failed.append("insufficient_periods")
    elif period_cv > 0.05:
        failed.append("period_cv_above_0.05")
    return CycleDiagnostics(
        sample_count=len(t),
        lead_changes=len(source),
        forward_transitions=forward_count,
        forward_fraction=float(forward_fraction),
        coverage_fraction=float(coverage),
        tours_estimate=float(tours),
        period_count=int(periods.size),
        period_mean=period_mean,
        period_cv=period_cv,
        seam_transition_count=int(np.count_nonzero(seam_mask)),
        seam_mean_duration=seam_mean,
        median_link_duration=median_link,
        seam_duration_ratio=seam_ratio,
        valid_cycle=valid,
        reason="ok" if valid else ",".join(failed),
    )


def extract_peak_intervals(
    times: np.ndarray,
    observables: np.ndarray,
    *,
    transient_fraction: float = 0.5,
) -> list[PeakInterval]:
    """Find ordered observable peak-to-peak links without hidden frames.

    For E35 the observable is the physical overlap ``m(t)``.  The coefficient
    trajectory is intentionally not accepted under an ambiguous name.
    """

    t, observable = _validate_trajectory(times, observables)
    p = observable.shape[1]
    start = max(1, int(np.floor(transient_fraction * len(t))))
    lead = np.argmax(observable, axis=1)
    candidates: list[tuple[int, int]] = []
    for mu in range(p):
        values = observable[:, mu]
        peaks = np.flatnonzero(
            (values[1:-1] >= values[:-2])
            & (values[1:-1] > values[2:])
        ) + 1
        for index in peaks:
            if index >= start and lead[index] == mu:
                candidates.append((int(index), mu))
    candidates.sort()

    intervals: list[PeakInterval] = []
    for (left_index, source), (right_index, target) in zip(
        candidates[:-1], candidates[1:]
    ):
        if target == (source + 1) % p and right_index > left_index:
            intervals.append(
                PeakInterval(
                    source=source,
                    target=target,
                    start_index=left_index,
                    end_index=right_index,
                    start_time=float(t[left_index]),
                    end_time=float(t[right_index]),
                )
            )
    return intervals


def decode_hidden_frames(
    times: np.ndarray,
    coefficients: np.ndarray,
    event_observables: np.ndarray,
    split: KeyframeSplit,
    patterns: np.ndarray,
    preprocessor: FramePreprocessor,
    *,
    transient_fraction: float = 0.5,
) -> DynamicDecode:
    """Decode hidden phases from the latest complete physical-overlap link.

    ``event_observables`` selects events (normally ``m(t)``), while
    ``coefficients`` is interpolated to reconstruct ``X a(t)``.  Hidden target
    pixels are not read by this function.
    """

    t, a = _validate_trajectory(times, coefficients)
    event_t, observable = _validate_trajectory(times, event_observables)
    if not np.array_equal(t, event_t) or observable.shape != a.shape:
        raise ValueError(
            "event_observables must share times and shape with coefficients"
        )
    xi = np.asarray(patterns, dtype=np.float64)
    p, n = xi.shape
    if p != len(split.keyframes):
        raise ValueError("patterns and keyframe split have different P")
    frame_shape = split.keyframes.shape[1:]
    if int(np.prod(frame_shape)) != n:
        raise ValueError("flattened frame size does not match pattern N")

    intervals = extract_peak_intervals(
        t, observable, transient_fraction=transient_fraction
    )
    latest: dict[int, PeakInterval] = {}
    for interval in intervals:
        latest[interval.source] = interval

    prediction = np.full(
        (len(split.hidden_indices),) + frame_shape,
        np.nan,
        dtype=np.float64,
    )
    valid = np.zeros(len(split.hidden_indices), dtype=bool)
    starts = np.full(len(split.hidden_indices), np.nan)
    ends = np.full(len(split.hidden_indices), np.nan)
    for output_index, (mu, offset) in enumerate(
        zip(split.hidden_segment, split.hidden_offset)
    ):
        interval = latest.get(int(mu))
        if interval is None:
            continue
        q = float(offset) / split.k
        target_time = interval.start_time + q * (
            interval.end_time - interval.start_time
        )
        coeff = np.array(
            [np.interp(target_time, t, a[:, column]) for column in range(p)]
        )
        model_frame = (xi.T @ coeff).reshape(frame_shape)
        prediction[output_index] = preprocessor.decode(model_frame)
        valid[output_index] = True
        starts[output_index] = interval.start_time
        ends[output_index] = interval.end_time
    return DynamicDecode(
        predictions=prediction,
        valid=valid,
        interval_start=starts,
        interval_end=ends,
    )


def build_baseline_predictions(
    split: KeyframeSplit,
) -> dict[str, np.ndarray]:
    """Return B0/B1/B2 predictions aligned with ``split.hidden_indices``."""

    shape = split.hidden_frames.shape
    output = {
        "B0_HOLD": np.empty(shape, dtype=np.float64),
        "B1_LINEAR": np.empty(shape, dtype=np.float64),
        "B2_MOTION": np.empty(shape, dtype=np.float64),
    }
    p = len(split.keyframes)
    for index, (mu, offset) in enumerate(
        zip(split.hidden_segment, split.hidden_offset)
    ):
        previous = split.keyframes[int(mu)]
        following = split.keyframes[(int(mu) + 1) % p]
        q = float(offset) / split.k
        output["B0_HOLD"][index] = baseline_hold(previous, q)
        output["B1_LINEAR"][index] = baseline_linear(previous, following, q)
        output["B2_MOTION"][index] = baseline_motion_translation(
            previous, following, q
        )
    return output


def evaluate_predictions(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    valid: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray | float | int]:
    """Per-frame MSE/MAE/PSNR/SSIM/Pearson and simple aggregates."""

    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if target.shape != estimate.shape or target.ndim != 3:
        raise ValueError("truth and predictions must share shape (frames,H,W)")
    mask = (
        np.ones(len(target), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if mask.shape != (len(target),):
        raise ValueError("valid must have one entry per frame")
    mask &= np.isfinite(estimate).all(axis=(1, 2))

    mse = np.full(len(target), np.nan)
    mae = np.full(len(target), np.nan)
    psnr = np.full(len(target), np.nan)
    ssim = np.full(len(target), np.nan)
    pearson = np.full(len(target), np.nan)
    for index in np.flatnonzero(mask):
        difference = estimate[index] - target[index]
        mse[index] = float(np.mean(difference * difference))
        mae[index] = float(np.mean(np.abs(difference)))
        psnr[index] = (
            np.inf if mse[index] == 0.0 else -10.0 * np.log10(mse[index])
        )
        ssim[index] = _gaussian_ssim(target[index], estimate[index])
        centered_truth = target[index].ravel() - np.mean(target[index])
        centered_estimate = estimate[index].ravel() - np.mean(estimate[index])
        denominator = np.linalg.norm(centered_truth) * np.linalg.norm(
            centered_estimate
        )
        if denominator > np.finfo(float).eps:
            pearson[index] = float(
                centered_truth @ centered_estimate / denominator
            )

    finite_psnr = psnr[np.isfinite(psnr)]
    return {
        "valid": mask,
        "valid_count": int(np.count_nonzero(mask)),
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "pearson": pearson,
        "mean_mse": _finite_mean(mse),
        "mean_mae": _finite_mean(mae),
        "mean_psnr": (
            float(np.mean(finite_psnr)) if finite_psnr.size else np.nan
        ),
        "mean_ssim": _finite_mean(ssim),
        "mean_pearson": _finite_mean(pearson),
    }


def _gaussian_ssim(
    truth: np.ndarray,
    estimate: np.ndarray,
    *,
    sigma: float = 1.5,
) -> float:
    """Gaussian-window SSIM with fixed ``data_range=1``."""

    first = np.asarray(truth, dtype=np.float64)
    second = np.asarray(estimate, dtype=np.float64)
    c1, c2 = 0.01**2, 0.03**2
    mu_first = gaussian_filter(first, sigma=sigma, mode="reflect", truncate=3.5)
    mu_second = gaussian_filter(second, sigma=sigma, mode="reflect", truncate=3.5)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_cross = mu_first * mu_second
    variance_first = gaussian_filter(
        first * first, sigma=sigma, mode="reflect", truncate=3.5
    ) - mu_first_sq
    variance_second = gaussian_filter(
        second * second, sigma=sigma, mode="reflect", truncate=3.5
    ) - mu_second_sq
    covariance = gaussian_filter(
        first * second, sigma=sigma, mode="reflect", truncate=3.5
    ) - mu_cross
    numerator = (2.0 * mu_cross + c1) * (2.0 * covariance + c2)
    denominator = (
        (mu_first_sq + mu_second_sq + c1)
        * (variance_first + variance_second + c2)
    )
    return float(np.mean(numerator / denominator))


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _validate_trajectory(
    times: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64)
    a = np.asarray(coefficients, dtype=np.float64)
    if t.ndim != 1 or a.ndim != 2 or len(t) != len(a) or len(t) < 3:
        raise ValueError("trajectory requires t:(samples,), a:(samples,P)")
    if not np.isfinite(t).all() or not np.isfinite(a).all():
        raise ValueError("trajectory contains NaN or infinity")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    return t, a
