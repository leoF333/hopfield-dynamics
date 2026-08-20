"""Synthetic cyclic-video data and leakage-safe E35 baselines.

The tools in this file are deliberately independent of the DDE integrator.
They prepare a controlled proof of concept and its non-redundant baselines:

``B0`` hold, ``B1`` linear pixels, and ``B2`` deterministic translation-aware
interpolation.  Hidden frames are never accepted by the preprocessor fit API.
"""

from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


LoopKind = Literal["orbit", "orbit_rotate"]


@dataclass(frozen=True)
class KeyframeSplit:
    """A cyclic sequence split into stored keys and hidden evaluation frames."""

    frames: np.ndarray
    keyframes: np.ndarray
    key_indices: np.ndarray
    hidden_frames: np.ndarray
    hidden_indices: np.ndarray
    hidden_segment: np.ndarray
    hidden_offset: np.ndarray
    k: int


@dataclass(frozen=True)
class FramePreprocessor:
    """Preprocessor fitted from keyframes only.

    Inputs and decoded outputs use ``[0,1]``.  Model patterns use ``[-1,1]``.
    """

    binary: bool
    threshold: Optional[float]

    @classmethod
    def fit_from_keyframes(
        cls,
        keyframes: np.ndarray,
        *,
        binary: bool = False,
        bins: int = 256,
    ) -> "FramePreprocessor":
        keys = _validate_frames(keyframes)
        threshold = _otsu_threshold(keys, bins=bins) if binary else None
        return cls(binary=bool(binary), threshold=threshold)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        values = _validate_frames(frames)
        if self.binary:
            assert self.threshold is not None
            return np.where(values >= self.threshold, 1.0, -1.0)
        return 2.0 * values - 1.0

    @staticmethod
    def decode(model_frames: np.ndarray) -> np.ndarray:
        values = np.asarray(model_frames, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("model_frames contains NaN or infinity")
        return np.clip(0.5 * (values + 1.0), 0.0, 1.0)


def make_cyclic_loop(
    n_frames: int = 32,
    *,
    height: int = 32,
    width: int = 32,
    kind: LoopKind = "orbit",
    seed: int = 42,
    include_endpoint: bool = False,
) -> np.ndarray:
    """Render a deterministic smooth loop in ``[0,1]``.

    ``include_endpoint=True`` returns ``n_frames+1`` frames and forces the last
    frame to equal the first bit-for-bit.  Production storage should remove
    that duplicate endpoint; it exists only to certify periodic closure.
    """

    if n_frames < 4:
        raise ValueError("n_frames must be at least 4")
    if height < 8 or width < 8:
        raise ValueError("height and width must both be at least 8")
    if kind not in ("orbit", "orbit_rotate"):
        raise ValueError(f"unknown loop kind {kind!r}")

    rng = np.random.default_rng(seed)
    # Seeded but fixed for the whole loop: no frame-wise random noise.
    lobe_weight = float(rng.uniform(0.45, 0.65))
    lobe_offset = float(rng.uniform(0.24, 0.34))
    sigma_long = float(rng.uniform(0.16, 0.20))
    sigma_short = float(rng.uniform(0.075, 0.105))

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    phases = np.linspace(
        0.0,
        2.0 * np.pi,
        n_frames + int(include_endpoint),
        endpoint=include_endpoint,
    )
    frames = np.empty((phases.size, height, width), dtype=np.float64)
    for index, phase in enumerate(phases):
        center_x = 0.28 * np.cos(phase)
        center_y = 0.22 * np.sin(phase)
        angle = phase if kind == "orbit_rotate" else 0.35
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = xx - center_x, yy - center_y
        local_x = cos_a * dx + sin_a * dy
        local_y = -sin_a * dx + cos_a * dy
        first = np.exp(
            -0.5
            * (
                (local_x / sigma_long) ** 2
                + (local_y / sigma_short) ** 2
            )
        )
        second = np.exp(
            -0.5
            * (
                ((local_x - lobe_offset) / (0.65 * sigma_long)) ** 2
                + ((local_y + 0.10) / (0.80 * sigma_short)) ** 2
            )
        )
        frames[index] = np.clip(first + lobe_weight * second, 0.0, 1.0)

    if include_endpoint:
        frames[-1] = frames[0]
    return frames


def make_nonlinear_advection_loop(
    n_frames: int = 256,
    *,
    height: int = 64,
    width: int = 64,
    winding: tuple[int, int] = (2, 1),
    curve_amplitude: tuple[float, float] = (6.0, 4.0),
    speed_modulation: float = 0.35,
    texture_sigma: float = 2.5,
    contrast: float = 1.0,
    seed: int = 1,
    include_endpoint: bool = False,
) -> np.ndarray:
    """E35-V1 generator: nonlinear advection of a textured field on a torus.

    A band-limited random texture (Gaussian-smoothed white noise of correlation
    length ``texture_sigma``, squashed by ``tanh`` into ``[0,1]``) is advected
    along a closed curve of the periodic ``height x width`` domain.  The curve
    combines an integer ``winding`` drift (large inter-keyframe displacement)
    with a sinusoidal transverse excursion (curvature), and the arclength
    parameter itself is modulated sinusoidally (non-uniform speed).  Curvature
    and non-uniform speed are what make the pixel chord ``B1`` and the
    constant-shift warp ``B2`` structurally suboptimal.

    A full-frame texture, rather than a sparse bright object on a dark ground,
    keeps the keyframe patterns from sharing a dominant background component:
    with a sparse object every pattern is ``~-1`` almost everywhere and the
    Gram matrix becomes near-singular, which suppresses ordered recall.

    Advection uses an exact Fourier phase shift, so translation is exact at
    sub-pixel accuracy and the loop closes exactly: the drift over one turn is
    an integer number of domain widths/heights.  ``include_endpoint=True``
    returns ``n_frames+1`` frames whose last frame equals the first bit-for-bit.
    """

    if n_frames < 4:
        raise ValueError("n_frames must be at least 4")
    if height < 8 or width < 8:
        raise ValueError("height and width must both be at least 8")
    w_x, w_y = int(winding[0]), int(winding[1])
    if w_x < 0 or w_y < 0 or (w_x == 0 and w_y == 0):
        raise ValueError("winding must be non-negative and not both zero")
    a_x, a_y = float(curve_amplitude[0]), float(curve_amplitude[1])
    if not np.isfinite([a_x, a_y]).all() or a_x < 0.0 or a_y < 0.0:
        raise ValueError("curve_amplitude must be finite and non-negative")
    if not 0.0 <= float(speed_modulation) < 0.5:
        raise ValueError("speed_modulation must lie in [0,0.5) to stay monotone")
    if not np.isfinite(texture_sigma) or texture_sigma <= 0.0:
        raise ValueError("texture_sigma must be finite and positive")
    if not np.isfinite(contrast) or contrast <= 0.0:
        raise ValueError("contrast must be finite and positive")

    rng = np.random.default_rng(seed)
    # Seeded once per loop: the texture never carries frame-wise noise.
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
    noise = rng.standard_normal((height, width))

    ky = 2.0 * np.pi * np.fft.fftfreq(height)
    kx = 2.0 * np.pi * np.fft.fftfreq(width)
    k_squared = ky[:, None] ** 2 + kx[None, :] ** 2
    envelope = np.exp(-0.5 * (texture_sigma**2) * k_squared)
    # Suppress the Nyquist rows/columns so the real part of every shifted
    # field stays an exact translation of the same band-limited texture.
    if height % 2 == 0:
        envelope[height // 2, :] = 0.0
    if width % 2 == 0:
        envelope[:, width // 2] = 0.0
    envelope[0, 0] = 0.0  # zero mean texture: no dominant DC component
    spectrum = np.fft.fft2(noise) * envelope
    reference = np.real(np.fft.ifft2(spectrum))
    scale = float(np.std(reference))
    if scale <= np.finfo(float).eps:
        raise ValueError("texture_sigma leaves no spectral content")
    spectrum = spectrum / scale

    thetas = np.linspace(
        0.0,
        2.0 * np.pi,
        n_frames + int(include_endpoint),
        endpoint=include_endpoint,
    )
    frames = np.empty((thetas.size, height, width), dtype=np.float64)
    for index, theta in enumerate(thetas):
        # Non-uniform speed along the closed curve (monotone, 2pi-periodic).
        s = theta + speed_modulation * np.sin(2.0 * theta)
        shift_x = width * w_x * s / (2.0 * np.pi) + a_x * np.sin(s + phase0)
        shift_y = height * w_y * s / (2.0 * np.pi) + a_y * np.sin(2.0 * s + phase0)
        phase = np.exp(
            -1j * (ky[:, None] * shift_y + kx[None, :] * shift_x)
        )
        field = np.real(np.fft.ifft2(spectrum * phase))
        frames[index] = 0.5 * (1.0 + np.tanh(contrast * field))

    if include_endpoint:
        frames[-1] = frames[0]
    return np.clip(frames, 0.0, 1.0)


def loop_displacement_statistics(
    n_frames: int,
    k: int,
    *,
    height: int,
    width: int,
    winding: tuple[int, int],
    curve_amplitude: tuple[float, float],
    speed_modulation: float,
) -> dict[str, float]:
    """Geometric inter-keyframe displacement of the advection curve (pixels).

    Purely geometric: it never touches rendered frames, so it can order
    generator settings before any image metric is computed.
    """

    thetas = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    s = thetas + speed_modulation * np.sin(2.0 * thetas)
    a_x, a_y = float(curve_amplitude[0]), float(curve_amplitude[1])
    x = width * int(winding[0]) * s / (2.0 * np.pi) + a_x * np.sin(s)
    y = height * int(winding[1]) * s / (2.0 * np.pi) + a_y * np.sin(2.0 * s)
    keys = np.arange(0, n_frames, k)
    x_key, y_key = x[keys], y[keys]
    dx = np.diff(np.append(x_key, x[0] + width * int(winding[0])))
    dy = np.diff(np.append(y_key, y[0] + height * int(winding[1])))
    steps = np.hypot(dx, dy)
    return {
        "mean_keyframe_displacement": float(np.mean(steps)),
        "min_keyframe_displacement": float(np.min(steps)),
        "max_keyframe_displacement": float(np.max(steps)),
        "path_length": float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
    }


def split_keyframes(
    frames: np.ndarray,
    k: int,
    *,
    drop_duplicate_endpoint: bool = True,
) -> KeyframeSplit:
    """Split a closed sequence without exposing hidden frames to fitting code."""

    values = _validate_frames(frames)
    if k < 2:
        raise ValueError("k must be at least 2 to define hidden frames")
    if (
        drop_duplicate_endpoint
        and len(values) > 1
        and np.array_equal(values[0], values[-1])
    ):
        values = values[:-1]
    n_frames = len(values)
    if n_frames % k:
        raise ValueError(f"number of unique frames {n_frames} must be divisible by k={k}")

    all_indices = np.arange(n_frames, dtype=np.int64)
    key_indices = all_indices[::k]
    key_mask = np.zeros(n_frames, dtype=bool)
    key_mask[key_indices] = True
    hidden_indices = all_indices[~key_mask]
    hidden_segment = hidden_indices // k
    hidden_offset = hidden_indices % k
    return KeyframeSplit(
        frames=np.array(values, copy=True),
        keyframes=np.array(values[key_indices], copy=True),
        key_indices=key_indices,
        hidden_frames=np.array(values[hidden_indices], copy=True),
        hidden_indices=hidden_indices,
        hidden_segment=hidden_segment,
        hidden_offset=hidden_offset,
        k=int(k),
    )


def baseline_hold(key_previous: np.ndarray, q: float) -> np.ndarray:
    """B0: hold the previous keyframe."""

    _validate_phase(q)
    return np.array(_validate_image(key_previous), copy=True)


def baseline_linear(
    key_previous: np.ndarray,
    key_next: np.ndarray,
    q: float,
) -> np.ndarray:
    """B1: direct linear interpolation in pixel space."""

    phase = _validate_phase(q)
    previous, following = _validate_image_pair(key_previous, key_next)
    return (1.0 - phase) * previous + phase * following


def estimate_translation_phase_correlation(
    key_previous: np.ndarray,
    key_next: np.ndarray,
    *,
    subpixel: bool = True,
) -> np.ndarray:
    """Estimate periodic ``(dy,dx)`` mapping the previous key to the next.

    This is a deterministic, non-learned endpoint-only estimator.  Mean
    subtraction and normalized cross-power make it insensitive to a global
    brightness offset.
    """

    previous, following = _validate_image_pair(key_previous, key_next)
    first = np.fft.fft2(previous - np.mean(previous))
    second = np.fft.fft2(following - np.mean(following))
    cross = second * np.conj(first)
    magnitude = np.abs(cross)
    cross /= np.where(magnitude > np.finfo(float).eps, magnitude, 1.0)
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)

    shifts = np.array(peak, dtype=np.float64)
    for axis, size in enumerate(correlation.shape):
        if subpixel:
            before_index = list(peak)
            after_index = list(peak)
            before_index[axis] = (peak[axis] - 1) % size
            after_index[axis] = (peak[axis] + 1) % size
            before = float(correlation[tuple(before_index)])
            center = float(correlation[peak])
            after = float(correlation[tuple(after_index)])
            denominator = before - 2.0 * center + after
            if abs(denominator) > np.finfo(float).eps:
                shifts[axis] += 0.5 * (before - after) / denominator
        if shifts[axis] > size / 2.0:
            shifts[axis] -= size
    return shifts


def warp_translation_periodic(
    image: np.ndarray,
    shift_yx: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """Periodic bilinear translation by a possibly fractional ``(dy,dx)``."""

    values = _validate_image(image)
    shift = np.asarray(shift_yx, dtype=np.float64)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("shift_yx must contain two finite values")
    height, width = values.shape
    source_y = (np.arange(height)[:, None] - shift[0]) % height
    source_x = (np.arange(width)[None, :] - shift[1]) % width
    y0 = np.floor(source_y).astype(np.int64)
    x0 = np.floor(source_x).astype(np.int64)
    y1 = (y0 + 1) % height
    x1 = (x0 + 1) % width
    wy = source_y - y0
    wx = source_x - x0
    return (
        (1.0 - wy) * (1.0 - wx) * values[y0, x0]
        + (1.0 - wy) * wx * values[y0, x1]
        + wy * (1.0 - wx) * values[y1, x0]
        + wy * wx * values[y1, x1]
    )


def baseline_motion_translation(
    key_previous: np.ndarray,
    key_next: np.ndarray,
    q: float,
    *,
    estimated_shift: Optional[np.ndarray] = None,
) -> np.ndarray:
    """B2: endpoint-only phase correlation and bidirectional warping."""

    phase = _validate_phase(q)
    previous, following = _validate_image_pair(key_previous, key_next)
    shift = (
        estimate_translation_phase_correlation(previous, following)
        if estimated_shift is None
        else np.asarray(estimated_shift, dtype=np.float64)
    )
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("estimated_shift must contain two finite values")
    from_previous = warp_translation_periodic(previous, phase * shift)
    from_following = warp_translation_periodic(
        following, -(1.0 - phase) * shift
    )
    return np.clip(
        (1.0 - phase) * from_previous + phase * from_following,
        0.0,
        1.0,
    )


def cyclic_step_errors(frames: np.ndarray) -> np.ndarray:
    """Mean absolute error for every cyclic bond, including the seam last->first."""

    values = _validate_frames(frames)
    following = np.roll(values, -1, axis=0)
    axes = tuple(range(1, values.ndim))
    return np.mean(np.abs(values - following), axis=axes)


def seam_diagnostics(frames: np.ndarray) -> dict[str, float]:
    """Report seam error relative to the internal transition median."""

    errors = cyclic_step_errors(frames)
    seam = float(errors[-1])
    internal = errors[:-1]
    median_internal = float(np.median(internal)) if len(internal) else 0.0
    scale = max(median_internal, np.finfo(float).eps)
    return {
        "seam_error": seam,
        "median_internal_error": median_internal,
        "seam_ratio": seam / scale,
        "max_step_error": float(np.max(errors)),
    }


def _otsu_threshold(frames: np.ndarray, *, bins: int) -> float:
    if bins < 8:
        raise ValueError("Otsu requires at least 8 bins")
    values = np.asarray(frames, dtype=np.float64).ravel()
    if np.ptp(values) <= np.finfo(float).eps:
        return float(values[0])
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probability = histogram.astype(np.float64)
    probability /= np.sum(probability)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_left = np.cumsum(probability)
    mean_left_numerator = np.cumsum(probability * centers)
    mean_total = mean_left_numerator[-1]
    denominator = weight_left * (1.0 - weight_left)
    between = np.zeros_like(denominator)
    valid = denominator > np.finfo(float).eps
    between[valid] = (
        mean_total * weight_left[valid] - mean_left_numerator[valid]
    ) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between))])


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("frames must have shape (time,height,width)")
    if not np.isfinite(values).all():
        raise ValueError("frames contains NaN or infinity")
    if np.min(values) < 0.0 or np.max(values) > 1.0:
        raise ValueError("frames must be in [0,1]")
    return values


def _validate_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("image contains NaN or infinity")
    if np.min(values) < 0.0 or np.max(values) > 1.0:
        raise ValueError("image must be in [0,1]")
    return values


def _validate_image_pair(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous = _validate_image(first)
    following = _validate_image(second)
    if previous.shape != following.shape:
        raise ValueError("keyframes must have identical shapes")
    return previous, following


def _validate_phase(q: float) -> float:
    phase = float(q)
    if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
        raise ValueError("q must lie in [0,1]")
    return phase

