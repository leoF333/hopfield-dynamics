"""
Numerical foundations shared by the E34 heteroclinic-necklace experiments.

This module deliberately contains only small, independently testable operations:

* the physical Gram metric of the pattern-span reduction;
* analytic exponential histories and their derivatives;
* distances and Hausdorff distances between DDE history segments;
* an argument-principle root counter with explicit refinement diagnostics.

The argument counter provides controlled numerical evidence, not interval
arithmetic. A result is marked converged only when the contour stays away from
singularities, phase increments are resolved, the integer residual is small, and
the same count survives successive refinements.
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
from typing import Callable

import numpy as np


Array = np.ndarray
Rectangle = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# NumPy-only low-rank backend

def make_iid_patterns(
    N: int,
    P: int,
    seed: int = 42,
) -> tuple[Array, Array]:
    """Generate the project's iid +/-1 patterns without initializing MLX/Metal."""
    if N <= 0 or P <= 0 or P > N:
        raise ValueError("require 0 < P <= N")
    rng = np.random.default_rng(seed)
    xi = rng.choice(np.array([-1.0, 1.0]), size=(P, N))
    return xi, np.roll(xi, shift=-1, axis=0)


class NumpyLowRankCouplings:
    """
    Float64 CPU implementation of the Couplings interface used by E34.

    It intentionally implements only low-rank operations and never constructs an
    N x N coupling matrix. robust_branch works with this class by duck typing.
    """

    def __init__(self, xi: Array, xi_shift: Array):
        xi = np.asarray(xi, dtype=float)
        xi_shift = np.asarray(xi_shift, dtype=float)
        if xi.ndim != 2 or xi_shift.shape != xi.shape:
            raise ValueError("xi and xi_shift must share shape (P, N)")
        self._xi_np = xi
        self._xis_np = xi_shift
        self.P, self.N = xi.shape
        self._use_np = True

    def overlap_raw(self, x: Array) -> Array:
        return self._xi_np @ np.asarray(x, dtype=float) / self.N

    def apply_J(self, x: Array) -> Array:
        overlaps = self.overlap_raw(x)
        return self._xi_np.T @ overlaps

    def apply_K(self, x: Array) -> Array:
        overlaps = self.overlap_raw(x)
        return self._xis_np.T @ overlaps

    def apply_C(self, x: Array, lam: float) -> Array:
        overlaps = self.overlap_raw(x)
        return (
            (1.0 - lam) * (self._xi_np.T @ overlaps)
            + lam * (self._xis_np.T @ overlaps)
        )

    def field_F(self, u: Array, lam: float, beta: float) -> Array:
        u = np.asarray(u, dtype=float)
        return -u + self.apply_C(np.tanh(beta * u), lam)

    def dF_dlam(self, u: Array, beta: float) -> Array:
        activation = np.tanh(beta * np.asarray(u, dtype=float))
        return self.apply_K(activation) - self.apply_J(activation)

    @staticmethod
    def compute_gain(u: Array, beta: float) -> Array:
        activation = np.tanh(beta * np.asarray(u, dtype=float))
        return beta * (1.0 - activation * activation)


# ---------------------------------------------------------------------------
# Gram geometry

def gram_matrix(xi: Array) -> Array:
    """Return Q = xi xi.T / N for patterns stored with shape (P, N)."""
    xi = np.asarray(xi, dtype=float)
    if xi.ndim != 2 or xi.shape[1] == 0:
        raise ValueError("xi must have shape (P, N) with N > 0")
    return (xi @ xi.T) / xi.shape[1]


def _validate_gram(Q: Array, dimension: int) -> Array:
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (dimension, dimension):
        raise ValueError(
            f"Q must have shape {(dimension, dimension)}, got {Q.shape}")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q contains NaN or infinite values")
    if not np.allclose(Q, Q.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Q must be symmetric")
    return Q


def gram_squared_norm(delta: Array, Q: Array) -> Array:
    """
    Return delta.T Q delta along the last axis.

    Tiny negative values caused by roundoff are clipped. A materially negative
    value signals that Q is not positive semidefinite and raises ValueError.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim == 0:
        raise ValueError("delta must have a coefficient axis")
    Q = _validate_gram(Q, delta.shape[-1])
    value = np.einsum("...i,ij,...j->...", delta, Q, delta, optimize=True)
    scale = max(1.0, float(np.max(np.abs(value))) if value.size else 1.0)
    if np.any(value < -1e-12 * scale):
        raise ValueError("Q is not positive semidefinite on the supplied vectors")
    return np.maximum(value, 0.0)


def gram_distance(a: Array, b: Array, Q: Array) -> Array:
    """Physical field distance sqrt((a-b).T Q (a-b))."""
    return np.sqrt(gram_squared_norm(np.asarray(a) - np.asarray(b), Q))


def normalize_mode_gram(v: Array, Q: Array) -> Array:
    """Return a copy of v normalized to unit Gram norm."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("v must be one-dimensional")
    norm = float(gram_distance(v, np.zeros_like(v), Q))
    if not np.isfinite(norm) or norm <= 1e-14:
        raise ValueError("v has zero or ill-conditioned Gram norm")
    return v / norm


# ---------------------------------------------------------------------------
# DDE histories

def exponential_history(
    a_s: Array,
    z_u: float | complex,
    v: Array,
    eps: float,
    tau: float,
    dt: float,
    Q: Array | None = None,
) -> tuple[Array, Array, Array]:
    """
    Build a_s + eps exp(z_u theta) v and its analytic derivative on [-tau, 0].

    If Q is supplied, v is first normalized in the Gram metric. E34 uses a real
    simple unstable root; a complex z_u is rejected unless its imaginary part is
    negligible.
    """
    a_s = np.asarray(a_s, dtype=float)
    v = np.asarray(v, dtype=float)
    if a_s.ndim != 1 or v.shape != a_s.shape:
        raise ValueError("a_s and v must be one-dimensional with the same shape")
    if tau <= 0 or dt <= 0:
        raise ValueError("tau and dt must be positive")
    if not np.isfinite(eps) or eps == 0:
        raise ValueError("eps must be finite and nonzero")
    z_complex = complex(z_u)
    if abs(z_complex.imag) > 1e-12 * max(1.0, abs(z_complex.real)):
        raise ValueError("E34 exponential_history expects a real unstable root")
    z = float(z_complex.real)
    if not np.isfinite(z):
        raise ValueError("z_u must be finite")
    if Q is not None:
        v = normalize_mode_gram(v, Q)

    L = int(round(tau / dt))
    if abs(L * dt - tau) > 1e-12:
        raise ValueError("dt must divide tau")
    theta = dt * np.arange(L + 1, dtype=float) - tau
    amplitude = eps * np.exp(z * theta)
    hist = a_s[None, :] + amplitude[:, None] * v[None, :]
    dhist = (z * amplitude)[:, None] * v[None, :]
    return theta, hist, dhist


def history_distance(
    phi: Array,
    psi: Array,
    Q: Array,
    *,
    kind: str = "linf",
    theta: Array | None = None,
) -> float:
    """
    Distance between two histories sampled on the same grid.

    kind="linf" implements max_theta d_G. kind="l2" implements the normalized
    time integral of d_G squared (trapezoidal rule).
    """
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if phi.ndim != 2 or psi.shape != phi.shape:
        raise ValueError("phi and psi must have the same shape (L+1, P)")
    sq = np.asarray(gram_squared_norm(phi - psi, Q), dtype=float)
    if kind == "linf":
        return float(np.sqrt(np.max(sq)))
    if kind != "l2":
        raise ValueError("kind must be 'linf' or 'l2'")
    if len(sq) == 1:
        return float(np.sqrt(sq[0]))

    if theta is None:
        theta = np.arange(len(sq), dtype=float)
    else:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(sq),):
            raise ValueError(f"theta must have shape {(len(sq),)}")
        if not np.all(np.diff(theta) > 0):
            raise ValueError("theta must be strictly increasing")
    duration = float(theta[-1] - theta[0])
    if duration <= 0:
        raise ValueError("history grid must span a positive duration")
    return float(np.sqrt(np.trapezoid(sq, theta) / duration))


@dataclass(frozen=True)
class HistoryHausdorffResult:
    """Symmetric and directed Hausdorff distances between two history sets."""

    distance: float
    directed_a_to_b: float
    directed_b_to_a: float
    kind: str


def history_hausdorff(
    histories_a: Array,
    histories_b: Array,
    Q: Array,
    *,
    kind: str = "l2",
    theta: Array | None = None,
) -> HistoryHausdorffResult:
    """
    Exact finite-set Hausdorff distance between two sampled history collections.

    The implementation is intentionally memory-bounded: it does not materialize
    an (n_a, n_b, L, P) tensor. Production callers should subsample histories to
    the resolution justified by their convergence check.
    """
    histories_a = np.asarray(histories_a, dtype=float)
    histories_b = np.asarray(histories_b, dtype=float)
    if histories_a.ndim != 3 or histories_b.ndim != 3:
        raise ValueError("history collections must have shape (M, L+1, P)")
    if histories_a.shape[1:] != histories_b.shape[1:]:
        raise ValueError("history collections must share their (L+1, P) shape")
    if len(histories_a) == 0 or len(histories_b) == 0:
        raise ValueError("history collections must be non-empty")

    def directed(source: Array, target: Array) -> float:
        largest_minimum = 0.0
        for phi in source:
            nearest = min(
                history_distance(phi, psi, Q, kind=kind, theta=theta)
                for psi in target
            )
            largest_minimum = max(largest_minimum, nearest)
        return largest_minimum

    a_to_b = directed(histories_a, histories_b)
    b_to_a = directed(histories_b, histories_a)
    return HistoryHausdorffResult(
        distance=max(a_to_b, b_to_a),
        directed_a_to_b=a_to_b,
        directed_b_to_a=b_to_a,
        kind=kind,
    )


# ---------------------------------------------------------------------------
# Argument-principle counting

@dataclass(frozen=True)
class WindingIteration:
    """Diagnostics from one uniform refinement of a rectangular contour."""

    samples_per_edge: int
    winding: float
    rounded_count: int
    integer_residual: float
    max_phase_step: float
    min_boundary_measure: float


@dataclass(frozen=True)
class WindingResult:
    """Result and audit trail of argument_count."""

    count: int | None
    converged: bool
    reason: str
    rectangle: Rectangle
    iterations: tuple[WindingIteration, ...]


@dataclass(frozen=True)
class RootBox:
    """A rectangular root enclosure carrying its argument-principle count."""

    rectangle: Rectangle
    count: int
    depth: int


@dataclass(frozen=True)
class PolishedRoot:
    """A numerically polished nonlinear eigenvalue and its residual."""

    value: complex
    residual: float
    converged: bool
    iterations: int
    source_box: Rectangle


@dataclass(frozen=True)
class RootLocalizationResult:
    """Root boxes, polished roots and count-conservation diagnostics."""

    total_count: int | None
    boxes: tuple[RootBox, ...]
    roots: tuple[PolishedRoot, ...]
    converged: bool
    reason: str
    count_conserved: bool


@dataclass(frozen=True)
class ModeResult:
    """Null mode of a characteristic matrix at a polished root."""

    vector: Array
    residual: float
    singular_value: float
    realification_error: float
    converged: bool


@dataclass(frozen=True)
class ArclengthResult:
    """Accepted points from a pseudo-arclength trace through an upper fold."""

    states: Array
    parameters: Array
    residuals: Array
    tangents: Array
    turned: bool
    fold_parameter: float
    status: str


@dataclass(frozen=True)
class FoldPairResult:
    """Node/fold/saddle result for one project memory branch."""

    node: Array
    saddle: Array | None
    lam_target: float
    fold_parameter: float
    node_eigmax: float
    saddle_eigmax: float | None
    trace: ArclengthResult
    converged: bool
    reason: str


def rectangle_contour(rectangle: Rectangle, samples_per_edge: int) -> Array:
    """Counter-clockwise rectangular contour without duplicated corner samples."""
    x_left, x_right, y_bottom, y_top = map(float, rectangle)
    if not (x_left < x_right and y_bottom < y_top):
        raise ValueError(
            "rectangle must be (x_left, x_right, y_bottom, y_top)")
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be >= 2")

    n = int(samples_per_edge)
    bottom = np.linspace(x_left, x_right, n, endpoint=False) + 1j * y_bottom
    right = x_right + 1j * np.linspace(y_bottom, y_top, n, endpoint=False)
    top = np.linspace(x_right, x_left, n, endpoint=False) + 1j * y_top
    left = x_left + 1j * np.linspace(y_top, y_bottom, n, endpoint=False)
    return np.concatenate((bottom, right, top, left))


def _phase_and_boundary_measure(value: complex | Array) -> tuple[float, float]:
    """
    Return determinant phase and a non-singularity measure.

    Scalars use |f|. Matrices use sigma_min/max(sigma_max, 1): this detects a
    uniformly small matrix (important for the 1x1 case) while remaining relative
    for matrices whose natural scale exceeds one. No raw determinant is formed.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = complex(array)
        if not (np.isfinite(scalar.real) and np.isfinite(scalar.imag)):
            return np.nan, np.nan
        return float(np.angle(scalar)), float(abs(scalar))
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("contour function must return a scalar or square matrix")
    if not np.all(np.isfinite(array)):
        return np.nan, np.nan
    sign, _ = np.linalg.slogdet(array)
    singular_values = np.linalg.svd(array, compute_uv=False)
    if sign == 0 or singular_values[0] == 0:
        return 0.0, 0.0
    relative_sigma = float(
        singular_values[-1] / max(float(singular_values[0]), 1.0))
    return float(np.angle(sign)), relative_sigma


def argument_count(
    function: Callable[[complex], complex | Array],
    rectangle: Rectangle,
    *,
    samples_per_edge: int = 16,
    max_refinements: int = 9,
    phase_step_max: float = np.pi / 4,
    boundary_tol: float = 1e-8,
    integer_tol: float = 1e-7,
    stable_refinements: int = 2,
) -> WindingResult:
    """
    Count zeros of an analytic scalar determinant (or matrix determinant).

    The contour is doubled until:
      1. every principal phase increment is <= phase_step_max;
      2. the boundary non-singularity measure exceeds boundary_tol;
      3. the winding lies within integer_tol of an integer;
      4. the same integer is obtained for stable_refinements successive grids.

    The routine assumes the supplied function has no poles inside the rectangle.
    It cannot establish analyticity or replace interval arithmetic.
    """
    if max_refinements < 1:
        raise ValueError("max_refinements must be >= 1")
    if not (0 < phase_step_max < np.pi):
        raise ValueError("phase_step_max must lie in (0, pi)")
    if boundary_tol <= 0 or integer_tol <= 0:
        raise ValueError("boundary_tol and integer_tol must be positive")
    if stable_refinements < 2:
        raise ValueError("stable_refinements must be >= 2")

    audit: list[WindingIteration] = []
    previous_count: int | None = None
    stable_count = 0
    n = int(samples_per_edge)
    last_reason = "maximum refinements reached"

    for _ in range(max_refinements):
        contour = rectangle_contour(rectangle, n)
        evaluated = [_phase_and_boundary_measure(function(z)) for z in contour]
        phases = np.array([item[0] for item in evaluated], dtype=float)
        measures = np.array([item[1] for item in evaluated], dtype=float)
        if not np.all(np.isfinite(phases)) or not np.all(np.isfinite(measures)):
            return WindingResult(
                count=None,
                converged=False,
                reason="non-finite contour evaluation",
                rectangle=rectangle,
                iterations=tuple(audit),
            )
        if np.any(measures == 0):
            return WindingResult(
                count=None,
                converged=False,
                reason="zero or singular matrix sampled on contour",
                rectangle=rectangle,
                iterations=tuple(audit),
            )

        phase_delta = np.angle(
            np.exp(1j * (np.roll(phases, -1) - phases)))
        winding = float(np.sum(phase_delta) / (2 * np.pi))
        rounded = int(np.rint(winding))
        residual = abs(winding - rounded)
        max_step = float(np.max(np.abs(phase_delta)))
        min_measure = float(np.min(measures))
        audit.append(WindingIteration(
            samples_per_edge=n,
            winding=winding,
            rounded_count=rounded,
            integer_residual=residual,
            max_phase_step=max_step,
            min_boundary_measure=min_measure,
        ))

        resolved = max_step <= phase_step_max
        separated = min_measure > boundary_tol
        integer_like = residual <= integer_tol
        if resolved and separated and integer_like:
            if rounded == previous_count:
                stable_count += 1
            else:
                stable_count = 1
            previous_count = rounded
            if stable_count >= stable_refinements:
                if rounded < 0:
                    return WindingResult(
                        count=None,
                        converged=False,
                        reason="negative winding: contour orientation or analyticity error",
                        rectangle=rectangle,
                        iterations=tuple(audit),
                    )
                return WindingResult(
                    count=rounded,
                    converged=True,
                    reason="count stable under contour refinement",
                    rectangle=rectangle,
                    iterations=tuple(audit),
                )
            last_reason = "integer count not yet stable"
        elif not separated:
            previous_count = None
            stable_count = 0
            last_reason = "contour too close to a zero or singular matrix"
        elif not resolved:
            previous_count = None
            stable_count = 0
            last_reason = "phase increments remain under-resolved"
        else:
            previous_count = None
            stable_count = 0
            last_reason = "winding is not sufficiently close to an integer"
        n *= 2

    return WindingResult(
        count=None,
        converged=False,
        reason=last_reason,
        rectangle=rectangle,
        iterations=tuple(audit),
    )


def _split_rectangle(
    rectangle: Rectangle,
    x_fraction: float,
    y_fraction: float,
) -> tuple[Rectangle, Rectangle, Rectangle, Rectangle]:
    x_left, x_right, y_bottom, y_top = rectangle
    x_mid = x_left + x_fraction * (x_right - x_left)
    y_mid = y_bottom + y_fraction * (y_top - y_bottom)
    return (
        (x_left, x_mid, y_bottom, y_mid),
        (x_mid, x_right, y_bottom, y_mid),
        (x_mid, x_right, y_mid, y_top),
        (x_left, x_mid, y_mid, y_top),
    )


def _root_residual(value: complex | Array) -> float:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(abs(complex(array)))
    singular_values = np.linalg.svd(array, compute_uv=False)
    return float(
        singular_values[-1] / max(float(singular_values[0]), 1.0))


def _finite_difference_derivative(
    function: Callable[[complex], complex | Array],
    z: complex,
) -> complex | Array:
    step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(z))
    return (function(z + step) - function(z - step)) / (2.0 * step)


def polish_root(
    function: Callable[[complex], complex | Array],
    z0: complex,
    *,
    derivative: Callable[[complex], complex | Array] | None = None,
    residual_tol: float = 1e-10,
    step_tol: float = 1e-12,
    max_iterations: int = 20,
    source_box: Rectangle = (0.0, 1.0, 0.0, 1.0),
) -> PolishedRoot:
    """
    Polish a scalar zero or nonlinear matrix eigenvalue.

    Matrix-valued functions use the two-sided smallest-singular-vector Newton
    correction u^H T v / (u^H T' v). The derivative is analytic when supplied
    and otherwise a centered finite difference.
    """
    z = complex(z0)
    for iteration in range(max_iterations + 1):
        value = function(z)
        residual = _root_residual(value)
        if not np.isfinite(residual):
            return PolishedRoot(
                z, np.inf, False, iteration, source_box)
        if residual <= residual_tol:
            return PolishedRoot(
                z, residual, True, iteration, source_box)
        if iteration == max_iterations:
            break

        derivative_value = (
            derivative(z)
            if derivative is not None
            else _finite_difference_derivative(function, z)
        )
        array = np.asarray(value)
        if array.ndim == 0:
            denominator = complex(np.asarray(derivative_value))
            numerator = complex(array)
        else:
            left, _, vh = np.linalg.svd(array)
            u = left[:, -1]
            v = vh[-1].conj()
            numerator = np.vdot(u, array @ v)
            denominator = np.vdot(u, np.asarray(derivative_value) @ v)
        if abs(denominator) <= 1e-14 * max(1.0, abs(numerator)):
            break
        step = numerator / denominator
        if not (np.isfinite(step.real) and np.isfinite(step.imag)):
            break
        z_new = z - step
        if abs(step) <= step_tol * max(1.0, abs(z_new)):
            z = z_new
            final_residual = _root_residual(function(z))
            return PolishedRoot(
                z,
                final_residual,
                bool(final_residual <= residual_tol),
                iteration + 1,
                source_box,
            )
        z = z_new
    return PolishedRoot(
        z,
        _root_residual(function(z)),
        False,
        max_iterations,
        source_box,
    )


def localize_roots(
    function: Callable[[complex], complex | Array],
    rectangle: Rectangle,
    *,
    derivative: Callable[[complex], complex | Array] | None = None,
    parent_result: WindingResult | None = None,
    target_width: float = 1e-3,
    target_height: float = 1e-3,
    max_depth: int = 16,
    split_fractions: tuple[float, ...] = (0.5, 0.47, 0.53, 0.43, 0.57),
    argument_kwargs: dict | None = None,
    polish_residual_tol: float = 1e-9,
) -> RootLocalizationResult:
    """
    Localize all counted roots by count-preserving rectangular subdivision.

    Internal split lines can pass through a root. Several deterministic split
    fractions are therefore tried; a split is accepted only when all four child
    counts converge and sum exactly to the parent count.
    """
    if target_width <= 0 or target_height <= 0 or max_depth < 1:
        raise ValueError("target sizes and max_depth must be positive")
    kwargs = dict(argument_kwargs or {})
    parent = parent_result or argument_count(function, rectangle, **kwargs)
    if not parent.converged or parent.count is None:
        return RootLocalizationResult(
            total_count=None,
            boxes=(),
            roots=(),
            converged=False,
            reason=f"parent count unavailable: {parent.reason}",
            count_conserved=False,
        )
    if parent.count == 0:
        return RootLocalizationResult(
            total_count=0,
            boxes=(),
            roots=(),
            converged=True,
            reason="parent rectangle contains no roots",
            count_conserved=True,
        )

    pending = [RootBox(rectangle, parent.count, 0)]
    terminal: list[RootBox] = []
    failure_reason: str | None = None
    while pending:
        box = pending.pop()
        x_left, x_right, y_bottom, y_top = box.rectangle
        small_enough = (
            (x_right - x_left) <= target_width
            and (y_top - y_bottom) <= target_height
        )
        if box.count == 1 and small_enough:
            terminal.append(box)
            continue
        if box.depth >= max_depth:
            terminal.append(box)
            if box.count != 1:
                failure_reason = (
                    f"depth limit left a box containing {box.count} roots")
            continue

        accepted_children: list[RootBox] | None = None
        for x_fraction in split_fractions:
            for y_fraction in split_fractions:
                child_results: list[RootBox] = []
                valid = True
                for child_rectangle in _split_rectangle(
                        box.rectangle, x_fraction, y_fraction):
                    child_count = argument_count(
                        function, child_rectangle, **kwargs)
                    if not child_count.converged or child_count.count is None:
                        valid = False
                        break
                    if child_count.count:
                        child_results.append(RootBox(
                            child_rectangle,
                            child_count.count,
                            box.depth + 1,
                        ))
                if valid and sum(child.count for child in child_results) == box.count:
                    accepted_children = child_results
                    break
            if accepted_children is not None:
                break
        if accepted_children is None:
            terminal.append(box)
            failure_reason = (
                f"could not conserve count while subdividing depth {box.depth}")
        else:
            pending.extend(accepted_children)

    terminal.sort(key=lambda item: (
        item.rectangle[0], item.rectangle[2], item.depth))
    count_conserved = sum(box.count for box in terminal) == parent.count
    polished: list[PolishedRoot] = []
    for box in terminal:
        if box.count != 1:
            continue
        x_left, x_right, y_bottom, y_top = box.rectangle
        start = complex(
            0.5 * (x_left + x_right),
            0.5 * (y_bottom + y_top),
        )
        root = polish_root(
            function,
            start,
            derivative=derivative,
            residual_tol=polish_residual_tol,
            source_box=box.rectangle,
        )
        x_tolerance = 1e-8 * max(1.0, abs(x_left), abs(x_right))
        y_tolerance = 1e-8 * max(1.0, abs(y_bottom), abs(y_top))
        inside_box = (
            x_left - x_tolerance <= root.value.real <= x_right + x_tolerance
            and y_bottom - y_tolerance <= root.value.imag <= y_top + y_tolerance
        )
        if root.converged and not inside_box:
            root = PolishedRoot(
                value=root.value,
                residual=root.residual,
                converged=False,
                iterations=root.iterations,
                source_box=root.source_box,
            )
        polished.append(root)

    all_polished = (
        len(polished) == parent.count
        and all(root.converged for root in polished)
    )
    converged = count_conserved and failure_reason is None and all_polished
    reason = (
        "all roots localized, count conserved and residuals polished"
        if converged
        else failure_reason or "one or more roots failed polishing"
    )
    return RootLocalizationResult(
        total_count=parent.count,
        boxes=tuple(terminal),
        roots=tuple(polished),
        converged=converged,
        reason=reason,
        count_conserved=count_conserved,
    )


def extract_null_mode(
    characteristic_matrix: Callable[[complex], Array],
    root: complex,
    *,
    Q: Array | None = None,
    expect_real: bool = True,
    residual_tol: float = 1e-9,
    realification_tol: float = 1e-8,
) -> ModeResult:
    """Extract and normalize the right null mode of T(root) using an SVD."""
    matrix = np.asarray(characteristic_matrix(root))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("characteristic_matrix must return a square matrix")
    _, singular_values, vh = np.linalg.svd(matrix)
    vector = vh[-1].conj()
    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) == 0:
        raise ValueError("null vector is numerically zero")
    vector = vector * np.exp(-1j * np.angle(vector[pivot]))
    realification_error = float(
        np.linalg.norm(vector.imag) / max(np.linalg.norm(vector), 1e-300))
    if expect_real:
        if realification_error > realification_tol:
            return ModeResult(
                vector=vector,
                residual=np.inf,
                singular_value=float(singular_values[-1]),
                realification_error=realification_error,
                converged=False,
            )
        vector = vector.real
    if Q is None:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-14:
            raise ValueError("mode has zero Euclidean norm")
        vector = vector / norm
    elif np.iscomplexobj(vector):
        Q = _validate_gram(Q, len(vector))
        norm_sq = np.vdot(vector, Q @ vector).real
        if norm_sq <= 1e-28:
            raise ValueError("mode has zero Gram norm")
        vector = vector / np.sqrt(norm_sq)
    else:
        vector = normalize_mode_gram(vector, Q)
    residual = float(
        np.linalg.norm(matrix @ vector)
        / max(np.linalg.norm(matrix, 2) * np.linalg.norm(vector), 1.0)
    )
    return ModeResult(
        vector=vector,
        residual=residual,
        singular_value=float(singular_values[-1]),
        realification_error=realification_error,
        converged=bool(residual <= residual_tol),
    )


# ---------------------------------------------------------------------------
# Generic pseudo-arclength and project-specific low-rank wrappers

def _arclength_tangent(
    state: Array,
    parameter: float,
    previous: Array,
    linear_solve: Callable[[Array, float, Array], Array],
    parameter_derivative: Callable[[Array, float], Array],
) -> Array:
    derivative = parameter_derivative(state, parameter)
    state_part = -linear_solve(state, parameter, derivative)
    tangent = np.concatenate((np.asarray(state_part, float), [1.0]))
    tangent /= np.linalg.norm(tangent)
    if np.dot(tangent, previous) < 0:
        tangent = -tangent
    return tangent


def _bordered_corrector(
    predicted_state: Array,
    predicted_parameter: float,
    base_state: Array,
    base_parameter: float,
    tangent: Array,
    step: float,
    residual_function: Callable[[Array, float], Array],
    linear_solve: Callable[[Array, float, Array], Array],
    parameter_derivative: Callable[[Array, float], Array],
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Array, float, bool, float]:
    state = np.asarray(predicted_state, float).copy()
    parameter = float(predicted_parameter)
    tangent_state = tangent[:-1]
    tangent_parameter = float(tangent[-1])
    dimension_scale = np.sqrt(len(state))

    def merit(candidate_state: Array, candidate_parameter: float) -> float:
        residual = residual_function(candidate_state, candidate_parameter)
        constraint = (
            np.dot(tangent_state, candidate_state - base_state)
            + tangent_parameter * (candidate_parameter - base_parameter)
            - step
        )
        return float(np.linalg.norm(residual) / dimension_scale + abs(constraint))

    for _ in range(max_iterations):
        residual = residual_function(state, parameter)
        constraint = (
            np.dot(tangent_state, state - base_state)
            + tangent_parameter * (parameter - base_parameter)
            - step
        )
        fp_residual = float(np.linalg.norm(residual) / dimension_scale)
        current_merit = fp_residual + abs(constraint)
        if current_merit < tolerance:
            return state, parameter, True, fp_residual

        derivative = parameter_derivative(state, parameter)
        try:
            direct = linear_solve(state, parameter, -residual)
            response = linear_solve(state, parameter, derivative)
        except np.linalg.LinAlgError:
            return state, parameter, False, fp_residual
        denominator = tangent_parameter - np.dot(tangent_state, response)
        if abs(denominator) < 1e-14:
            return state, parameter, False, fp_residual
        delta_parameter = (
            -constraint - np.dot(tangent_state, direct)) / denominator
        delta_state = direct - delta_parameter * response

        accepted = False
        damping = 1.0
        for _ in range(24):
            candidate_state = state + damping * delta_state
            candidate_parameter = parameter + damping * delta_parameter
            if merit(candidate_state, candidate_parameter) < current_merit:
                state = candidate_state
                parameter = candidate_parameter
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            return state, parameter, False, fp_residual
    residual = residual_function(state, parameter)
    return (
        state,
        parameter,
        False,
        float(np.linalg.norm(residual) / dimension_scale),
    )


def trace_upper_fold(
    state0: Array,
    parameter0: float,
    residual_function: Callable[[Array, float], Array],
    linear_solve: Callable[[Array, float, Array], Array],
    parameter_derivative: Callable[[Array, float], Array],
    *,
    identity_check: Callable[[Array, float], bool] | None = None,
    stop_after_turn_parameter: float | None = None,
    step_initial: float = 0.01,
    step_min: float = 1e-5,
    step_max: float = 0.03,
    max_steps: int = 500,
    tolerance: float = 1e-10,
    max_corrector_iterations: int = 35,
    turn_tolerance: float = 1e-6,
    progress_callback: Callable[[int, float, float, bool], None] | None = None,
) -> ArclengthResult:
    """
    Follow a branch toward increasing parameter, through one upper fold.

    Only solves Jacobian systems through the supplied linear_solve callback.
    Project code supplies the exact Woodbury solve, so no N x N matrix is formed.
    """
    if not (0 < step_min <= step_initial <= step_max):
        raise ValueError("require 0 < step_min <= step_initial <= step_max")
    state = np.asarray(state0, dtype=float).copy()
    parameter = float(parameter0)
    initial_residual = residual_function(state, parameter)
    states = [state.copy()]
    parameters = [parameter]
    residuals = [
        float(np.linalg.norm(initial_residual) / np.sqrt(len(state)))]
    previous = np.zeros(len(state) + 1)
    previous[-1] = 1.0
    tangents = [previous.copy()]
    step = float(step_initial)
    maximum_parameter = parameter
    turned = False
    status = "max_steps"

    for _ in range(max_steps):
        try:
            tangent = _arclength_tangent(
                state,
                parameter,
                previous,
                linear_solve,
                parameter_derivative,
            )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            step *= 0.5
            if step < step_min:
                status = "tangent_failure"
                break
            continue
        predicted_state = state + step * tangent[:-1]
        predicted_parameter = parameter + step * tangent[-1]
        corrected_state, corrected_parameter, ok, corrected_residual = (
            _bordered_corrector(
                predicted_state,
                predicted_parameter,
                state,
                parameter,
                tangent,
                step,
                residual_function,
                linear_solve,
                parameter_derivative,
                tolerance=tolerance,
                max_iterations=max_corrector_iterations,
            )
        )
        identity_ok = (
            identity_check is None
            or identity_check(corrected_state, corrected_parameter)
        )
        if not ok or not identity_ok:
            step *= 0.5
            if step < step_min:
                status = "corrector_failure" if not ok else "identity_failure"
                break
            continue

        state = corrected_state
        parameter = corrected_parameter
        previous = tangent
        states.append(state.copy())
        parameters.append(parameter)
        residuals.append(corrected_residual)
        tangents.append(tangent.copy())
        if parameter > maximum_parameter:
            maximum_parameter = parameter
        elif parameter < maximum_parameter - turn_tolerance:
            turned = True
        if progress_callback is not None:
            progress_callback(
                len(states) - 1,
                parameter,
                corrected_residual,
                turned,
            )
        step = min(step * 1.2, step_max)

        if (
            turned
            and stop_after_turn_parameter is not None
            and parameter <= stop_after_turn_parameter
        ):
            status = "target_reached"
            break
    else:
        status = "max_steps"

    return ArclengthResult(
        states=np.asarray(states),
        parameters=np.asarray(parameters),
        residuals=np.asarray(residuals),
        tangents=np.asarray(tangents),
        turned=turned,
        fold_parameter=float(maximum_parameter),
        status=status,
    )


def reduced_field_coefficients(coup, u: Array, Q: Array | None = None) -> Array:
    """Return a in u=xi.T a without forming an N x N matrix."""
    if Q is None:
        Q = gram_matrix(coup._xi_np)
    return np.linalg.solve(Q, coup.overlap_raw(np.asarray(u, float)))


def memory_branch_identity(
    coup,
    u: Array,
    mu: int,
    *,
    Q: Array | None = None,
    third_max: float | None = None,
) -> bool:
    """Loose identity guard suitable near a fold of memory branch mu."""
    coefficients = reduced_field_coefficients(coup, u, Q)
    order = np.argsort(np.abs(coefficients))[::-1]
    successor = (mu + 1) % coup.P
    if third_max is None:
        third_max = max(0.25, 2.8 / np.sqrt(coup.N))
    if int(order[0]) not in (mu, successor):
        return False
    if mu not in set(int(index) for index in order[:min(2, coup.P)]):
        return False
    if coup.P >= 2 and int(order[1]) not in (mu, successor):
        if int(order[0]) != mu or abs(coefficients[order[1]]) >= third_max:
            return False
    if coup.P >= 3 and abs(coefficients[order[2]]) > third_max:
        return False
    return True


def solve_memory_node(
    coup,
    mu: int,
    lam: float,
    beta: float,
    *,
    initial_u: Array | None = None,
    tolerance: float = 1e-10,
) -> tuple[Array, bool, float]:
    """Solve one memory node with beta annealing and return (u, ok, residual)."""
    from robust_branch import anneal_newton, woodbury_newton

    if not 0 <= mu < coup.P:
        raise ValueError(f"mu must lie in [0,{coup.P})")
    seed = (
        2.0 * coup._xi_np[mu]
        if initial_u is None
        else np.asarray(initial_u, float)
    )
    u, ok = woodbury_newton(
        coup, seed, lam, beta, tol=tolerance, max_iter=80)
    residual = float(
        np.linalg.norm(coup.field_F(u, lam, beta)) / np.sqrt(coup.N))
    identity_ok = memory_branch_identity(coup, u, mu)
    if ok and identity_ok and residual < tolerance * 10:
        return u, True, residual

    # Annealing is a fallback only: on small/high-beta systems it can otherwise
    # cross into the trivial basin even when direct Newton from 2*xi_mu works.
    u, ok = anneal_newton(
        coup, seed, lam, beta, tol=tolerance, max_iter=80)
    residual = float(
        np.linalg.norm(coup.field_F(u, lam, beta)) / np.sqrt(coup.N))
    identity_ok = memory_branch_identity(coup, u, mu)
    return u, bool(ok and identity_ok and residual < tolerance * 10), residual


def approach_memory_fold(
    coup,
    mu: int,
    beta: float,
    u_start: Array,
    lam_start: float,
    *,
    lam_limit: float = 0.8,
    step_initial: float = 0.01,
    step_min: float = 2e-4,
    eig_stop: float = -0.05,
) -> tuple[Array, float, str]:
    """Warm-start the stable node toward its fold and retain the last safe point."""
    from robust_branch import eigmax_M, woodbury_newton

    u = np.asarray(u_start, float).copy()
    lam = float(lam_start)
    step = float(step_initial)
    status = "lam_limit"
    while lam < lam_limit - 1e-14:
        candidate_lam = min(lam + step, lam_limit)
        candidate_u, ok = woodbury_newton(
            coup, u, candidate_lam, beta, tol=1e-11, max_iter=80)
        candidate_ok = (
            ok
            and memory_branch_identity(coup, candidate_u, mu)
            and np.linalg.norm(
                coup.field_F(candidate_u, candidate_lam, beta)
            ) / np.sqrt(coup.N) < 1e-9
        )
        if candidate_ok:
            u = candidate_u
            lam = candidate_lam
            eigenvalue = eigmax_M(coup, u, lam, beta)
            if eigenvalue >= eig_stop:
                status = "eig_stop"
                break
            step = min(step * 1.2, step_initial)
        else:
            step *= 0.5
            if step < step_min:
                status = "newton_edge"
                break
    return u, lam, status


def continue_node_fold_saddle(
    coup,
    mu: int,
    beta: float,
    node_at_target: Array,
    lam_target: float,
    near_fold_node: Array,
    near_fold_lam: float,
    *,
    step_initial: float = 0.01,
    step_min: float = 1e-5,
    step_max: float = 0.03,
    max_steps: int = 500,
    progress_callback: Callable[[int, float, float, bool], None] | None = None,
) -> FoldPairResult:
    """Trace a project memory node through its fold and polish the saddle."""
    from robust_branch import eigmax_M, woodbury_newton, woodbury_solve

    Q = gram_matrix(coup._xi_np)

    def residual(state: Array, parameter: float) -> Array:
        return coup.field_F(state, parameter, beta)

    def linear_solve(state: Array, parameter: float, rhs: Array) -> Array:
        gain = coup.compute_gain(state, beta)
        return woodbury_solve(coup, gain, parameter, rhs)

    def parameter_derivative(state: Array, parameter: float) -> Array:
        del parameter
        return coup.dF_dlam(state, beta)

    trace = trace_upper_fold(
        near_fold_node,
        near_fold_lam,
        residual,
        linear_solve,
        parameter_derivative,
        identity_check=lambda state, parameter: memory_branch_identity(
            coup, state, mu, Q=Q),
        stop_after_turn_parameter=lam_target,
        step_initial=step_initial,
        step_min=step_min,
        step_max=step_max,
        max_steps=max_steps,
        progress_callback=progress_callback,
    )
    node_eigenvalue = eigmax_M(coup, node_at_target, lam_target, beta)
    if not trace.turned:
        return FoldPairResult(
            node=node_at_target,
            saddle=None,
            lam_target=lam_target,
            fold_parameter=trace.fold_parameter,
            node_eigmax=node_eigenvalue,
            saddle_eigmax=None,
            trace=trace,
            converged=False,
            reason=f"arclength did not turn ({trace.status})",
        )

    saddle_guess = trace.states[-1]
    saddle, saddle_ok = woodbury_newton(
        coup,
        saddle_guess,
        lam_target,
        beta,
        tol=1e-11,
        max_iter=80,
    )
    saddle_residual = float(
        np.linalg.norm(coup.field_F(saddle, lam_target, beta))
        / np.sqrt(coup.N)
    )
    saddle_identity = memory_branch_identity(coup, saddle, mu, Q=Q)
    saddle_eigenvalue = eigmax_M(coup, saddle, lam_target, beta)
    converged = bool(
        saddle_ok
        and saddle_residual < 1e-9
        and saddle_identity
        and node_eigenvalue < 0
        and saddle_eigenvalue > 0
    )
    return FoldPairResult(
        node=node_at_target,
        saddle=saddle,
        lam_target=lam_target,
        fold_parameter=trace.fold_parameter,
        node_eigmax=node_eigenvalue,
        saddle_eigmax=saddle_eigenvalue,
        trace=trace,
        converged=converged,
        reason=(
            "node/fold/saddle pair obtained"
            if converged
            else (
                "saddle polish or static index check failed: "
                f"ok={saddle_ok}, residual={saddle_residual:.3e}, "
                f"identity={saddle_identity}, eig={saddle_eigenvalue:+.3e}"
            )
        ),
    )


def right_half_rectangle(
    rho: float,
    t0: float,
    *,
    gamma: float = 1e-7,
    eta: float | None = None,
) -> Rectangle:
    """
    Rectangle enclosing the Re(z)>=gamma part of |t0*z+1|<=rho.

    When rho < 1+t0*gamma, the spectral disk itself excludes that half-plane;
    callers should return a zero count without evaluating a contour.
    """
    rho = float(rho)
    t0 = float(t0)
    gamma = float(gamma)
    if rho < 0 or t0 <= 0 or gamma < 0:
        raise ValueError("require rho >= 0, t0 > 0 and gamma >= 0")
    if eta is None:
        eta = max(0.05, 0.02 * rho / t0)
    eta = float(eta)
    if eta <= 0:
        raise ValueError("eta must be positive")
    x_right = max(gamma + eta, (rho - 1.0) / t0 + eta)
    y_radius = rho / t0 + eta
    return gamma, x_right, -y_radius, y_radius


def unstable_argument_count(
    characteristic_matrix: Callable[[complex], Array],
    rho: float,
    t0: float,
    *,
    gamma: float = 1e-7,
    eta: float | None = None,
    **argument_kwargs,
) -> WindingResult:
    """Count roots with Re(z)>=gamma inside the a-priori spectral bound."""
    if rho < 1.0 + t0 * gamma:
        rectangle = right_half_rectangle(rho, t0, gamma=gamma, eta=eta)
        return WindingResult(
            count=0,
            converged=True,
            reason="a-priori disk excludes Re(z)>=gamma",
            rectangle=rectangle,
            iterations=(),
        )
    rectangle = right_half_rectangle(rho, t0, gamma=gamma, eta=eta)
    return argument_count(
        characteristic_matrix,
        rectangle,
        **argument_kwargs,
    )


def characteristic_spectral_bound(GJ: Array, GK: Array, lam: float) -> float:
    """
    Return rho with |t0*z+1| <= rho for every Re(z)>=0 characteristic root.

    Absolute mixing coefficients keep the bound valid outside lam in [0, 1],
    although E34 itself uses that interval.
    """
    GJ = np.asarray(GJ)
    GK = np.asarray(GK)
    if GJ.ndim != 2 or GJ.shape[0] != GJ.shape[1] or GK.shape != GJ.shape:
        raise ValueError("GJ and GK must be square matrices with equal shape")
    return float(
        abs(1.0 - lam) * np.linalg.norm(GJ, 2)
        + abs(lam) * np.linalg.norm(GK, 2)
    )


def count_unstable_reduced_spectrum(
    GJ: Array,
    GK: Array,
    t0: float,
    tau: float,
    lam: float,
    *,
    gamma: float = 1e-7,
    eta: float | None = None,
    **argument_kwargs,
) -> WindingResult:
    """Argument count for the project's reduced characteristic matrix T_P(z)."""
    GJ = np.asarray(GJ)
    GK = np.asarray(GK)
    if GJ.ndim != 2 or GJ.shape[0] != GJ.shape[1] or GK.shape != GJ.shape:
        raise ValueError("GJ and GK must be square matrices with equal shape")
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    identity = np.eye(GJ.shape[0], dtype=np.result_type(GJ, GK, complex))

    def characteristic(z: complex) -> Array:
        return (
            (t0 * z + 1.0) * identity
            - (1.0 - lam) * GJ
            - lam * np.exp(-z * tau) * GK
        )

    rho = characteristic_spectral_bound(GJ, GK, lam)
    return unstable_argument_count(
        characteristic,
        rho,
        t0,
        gamma=gamma,
        eta=eta,
        **argument_kwargs,
    )
