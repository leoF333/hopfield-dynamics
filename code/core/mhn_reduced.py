"""Low-rank Modern-Hopfield fields for the E35 delayed model.

The pattern array convention is ``xi.shape == (P, N)`` and ``X = xi.T``.
No function in this module constructs an ``N x N`` coupling.  The mandatory
architectures form the exact factorial

    J in {HEBB, PINV} x K in {HEBB, PINV}.

The optional softmax storage field is deliberately gated by
``make_soft_architecture(..., enabled=True)`` so that it cannot enter the
nominal factorial by accident.
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
from typing import Dict, Literal, Optional

import numpy as np


FieldKind = Literal["hebb", "pinv", "soft"]


@dataclass(frozen=True)
class GramDiagnostics:
    """Numerical diagnostics for the retained singular subspace of ``G``."""

    n: int
    p: int
    rank: int
    rank_fraction: float
    tolerance: float
    relative_tolerance: float
    singular_max: float
    singular_min_retained: float
    condition_retained: float
    condition_risk: bool
    moore_penrose_residual: float

    def to_dict(self) -> Dict[str, float | int | bool]:
        return asdict(self)


class MhnContext:
    """Patterns, Gram SVD and low-rank field operations.

    Parameters are created with :meth:`from_patterns`; arrays are copied and
    made read-only to prevent a silent mismatch between ``xi`` and its SVD.
    """

    def __init__(
        self,
        *,
        xi: np.ndarray,
        beta: float,
        gram: np.ndarray,
        singular_vectors: np.ndarray,
        singular_values: np.ndarray,
        keep: np.ndarray,
        retained_vectors: np.ndarray,
        retained_values: np.ndarray,
        tolerance: float,
        relative_tolerance: float,
        diagnostics: GramDiagnostics,
    ) -> None:
        self.xi = xi
        self.beta = float(beta)
        self.gram = gram
        self.singular_vectors = singular_vectors
        self.singular_values = singular_values
        self.keep = keep
        self._retained_vectors = retained_vectors
        self._retained_values = retained_values
        self._inverse_retained_values = 1.0 / retained_values
        self.tolerance = float(tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.diagnostics = diagnostics
        self.p, self.n = self.xi.shape

        for array in (
            self.xi,
            self.gram,
            self.singular_vectors,
            self.singular_values,
            self.keep,
            self._retained_vectors,
            self._retained_values,
            self._inverse_retained_values,
        ):
            array.setflags(write=False)

    @classmethod
    def from_patterns(
        cls,
        xi: np.ndarray,
        beta: float,
        *,
        tsvd_rtol: Optional[float] = None,
        require_p_lt_n: bool = True,
    ) -> "MhnContext":
        """Build ``G=X.T@X/N`` and its Moore-Penrose/TSVD representation.

        ``tsvd_rtol=None`` gives the standard numerical Moore-Penrose cutoff
        ``eps * max(N,P) * s_max``.  A positive ``tsvd_rtol`` raises that
        cutoff to ``tsvd_rtol * s_max``.  Choosing a non-default cutoff is an
        experimental decision and must be recorded by the caller.
        """

        patterns = np.array(xi, dtype=np.float64, copy=True, order="C")
        if patterns.ndim != 2:
            raise ValueError(f"xi must have shape (P,N), got {patterns.shape}")
        p, n = patterns.shape
        if p < 1 or n < 1:
            raise ValueError("xi must contain at least one pattern and one neuron")
        if require_p_lt_n and p >= n:
            raise ValueError(f"E35 nominal regime requires P<N, got P={p}, N={n}")
        if not np.isfinite(patterns).all():
            raise ValueError("xi contains NaN or infinity")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if tsvd_rtol is not None and (
            not np.isfinite(tsvd_rtol) or not 0.0 <= tsvd_rtol < 1.0
        ):
            raise ValueError("tsvd_rtol must be in [0,1)")

        gram = (patterns @ patterns.T) / float(n)
        gram = 0.5 * (gram + gram.T)
        u, singular_values, _ = np.linalg.svd(
            gram, full_matrices=False, hermitian=True
        )
        singular_values = np.maximum(singular_values, 0.0)
        singular_max = float(singular_values[0]) if p else 0.0
        if singular_max == 0.0:
            raise ValueError("all patterns have zero norm")

        base_tolerance = (
            np.finfo(np.float64).eps * max(n, p) * singular_max
        )
        relative_tolerance = 0.0 if tsvd_rtol is None else float(tsvd_rtol)
        tolerance = max(base_tolerance, relative_tolerance * singular_max)
        keep = singular_values > tolerance
        rank = int(np.count_nonzero(keep))
        if rank == 0:
            raise ValueError("SVD cutoff removes every Gram mode")

        u_retained = u[:, keep]
        s_retained = singular_values[keep]
        # P x P only: this is a diagnostic, never an N x N coupling.
        gram_pinv = (u_retained / s_retained[None, :]) @ u_retained.T
        denominator = max(float(np.linalg.norm(gram)), np.finfo(float).tiny)
        mp_residual = float(
            np.linalg.norm(gram @ gram_pinv @ gram - gram) / denominator
        )
        singular_min = float(s_retained[-1])
        diagnostics = GramDiagnostics(
            n=n,
            p=p,
            rank=rank,
            rank_fraction=rank / p,
            tolerance=float(tolerance),
            relative_tolerance=relative_tolerance,
            singular_max=singular_max,
            singular_min_retained=singular_min,
            condition_retained=singular_max / singular_min,
            condition_risk=(singular_max / singular_min) > 1.0e10,
            moore_penrose_residual=mp_residual,
        )
        return cls(
            xi=patterns,
            beta=float(beta),
            gram=gram,
            singular_vectors=u,
            singular_values=singular_values,
            keep=keep,
            retained_vectors=np.array(u_retained, copy=True, order="C"),
            retained_values=np.array(s_retained, copy=True),
            tolerance=tolerance,
            relative_tolerance=relative_tolerance,
            diagnostics=diagnostics,
        )

    @property
    def retained_vectors(self) -> np.ndarray:
        return self._retained_vectors

    @property
    def retained_values(self) -> np.ndarray:
        return self._retained_values

    def apply_pinv(self, value: np.ndarray) -> np.ndarray:
        """Apply ``G dagger`` to a vector or a matrix without forming it."""

        array = np.asarray(value, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != self.p:
            raise ValueError(
                f"value must have leading dimension P={self.p}, got {array.shape}"
            )
        u = self.retained_vectors
        projected = u.T @ array
        if array.ndim == 1:
            projected = projected * self._inverse_retained_values
        else:
            projected = projected * self._inverse_retained_values[:, None]
        return u @ projected

    def pinv_matrix(self) -> np.ndarray:
        """Return the diagnostic ``P x P`` pseudoinverse, never an ``N x N`` matrix."""

        return self.apply_pinv(np.eye(self.p, dtype=np.float64))

    def hebb_field(self, a: np.ndarray) -> np.ndarray:
        """``H(a) = xi @ tanh(beta * xi.T @ a) / N``."""

        coeff = self._validate_coefficients(a)
        return self.xi @ np.tanh(self.beta * (self.xi.T @ coeff)) / self.n

    def physical_overlaps_batch(
        self,
        coefficients: np.ndarray,
        *,
        batch_size: Optional[int] = None,
        max_workspace_bytes: int = 64 * 1024 * 1024,
    ) -> np.ndarray:
        """Evaluate physical overlaps ``m(t)`` on a coefficient trajectory.

        For ``A.shape == (samples,P)``, this computes

        ``m = tanh(beta * A @ xi) @ xi.T / N``

        in bounded ``samples x N`` chunks.  It is the physical recall
        observable and must be used for cycle events when patterns are
        correlated; the coordinates ``a`` are retained for reconstruction.
        """

        trajectory = np.asarray(coefficients, dtype=np.float64)
        if trajectory.ndim == 1:
            return self.hebb_field(trajectory)[None, :]
        if trajectory.ndim != 2 or trajectory.shape[1] != self.p:
            raise ValueError(
                f"coefficients must have shape (samples,{self.p}), "
                f"got {trajectory.shape}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError("coefficients contains NaN or infinity")
        samples = len(trajectory)
        if batch_size is None:
            if max_workspace_bytes < self.n * np.dtype(np.float64).itemsize:
                raise ValueError("max_workspace_bytes is too small for one full state")
            batch_size = max(
                1,
                min(
                    samples,
                    max_workspace_bytes
                    // (self.n * np.dtype(np.float64).itemsize),
                ),
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        overlaps = np.empty((samples, self.p), dtype=np.float64)
        for start in range(0, samples, batch_size):
            stop = min(samples, start + batch_size)
            full_state = trajectory[start:stop] @ self.xi
            full_state *= self.beta
            np.tanh(full_state, out=full_state)
            overlaps[start:stop] = full_state @ self.xi.T / self.n
        return overlaps

    def pinv_field(self, a: np.ndarray) -> np.ndarray:
        """``Q(a) = G dagger H(a)``."""

        return self.apply_pinv(self.hebb_field(a))

    def pinv_field_amplification(self, coefficients: np.ndarray) -> float:
        """Maximum ``||Q(a)||/||H(a)||`` on caller-supplied sentinels."""

        sentinels = np.asarray(coefficients, dtype=np.float64)
        if sentinels.ndim == 1:
            sentinels = sentinels[None, :]
        if sentinels.ndim != 2 or sentinels.shape[1] != self.p:
            raise ValueError(
                f"coefficients must have shape (samples,{self.p}), got {sentinels.shape}"
            )
        amplification = 0.0
        for sentinel in sentinels:
            hebb = self.hebb_field(sentinel)
            projected = self.apply_pinv(hebb)
            ratio = np.linalg.norm(projected) / max(
                np.linalg.norm(hebb), np.finfo(float).tiny
            )
            amplification = max(amplification, float(ratio))
        return amplification

    def hebb_jacobian(self, a: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of :meth:`hebb_field`, shape ``(P,P)``."""

        coeff = self._validate_coefficients(a)
        activation = np.tanh(self.beta * (self.xi.T @ coeff))
        gain = self.beta * (1.0 - activation * activation)
        return (self.xi * gain[None, :]) @ self.xi.T / self.n

    def pinv_jacobian(self, a: np.ndarray) -> np.ndarray:
        """Analytic Jacobian ``G dagger D H(a)``."""

        return self.apply_pinv(self.hebb_jacobian(a))

    def soft_field(self, a: np.ndarray, b2: float) -> np.ndarray:
        """Conditional storage field ``softmax(b2 * G @ a)``."""

        coeff = self._validate_coefficients(a)
        scale = _validate_b2(b2)
        return stable_softmax(scale * (self.gram @ coeff))

    def soft_jacobian(self, a: np.ndarray, b2: float) -> np.ndarray:
        """Analytic Jacobian of :meth:`soft_field`, shape ``(P,P)``."""

        scale = _validate_b2(b2)
        probability = self.soft_field(a, scale)
        softmax_jacobian = np.diag(probability) - np.outer(
            probability, probability
        )
        return scale * softmax_jacobian @ self.gram

    def overlap_from_full_state(self, u: np.ndarray) -> np.ndarray:
        """Hebbian overlap field evaluated directly from a full ``N`` state."""

        state = np.asarray(u, dtype=np.float64)
        if state.shape != (self.n,):
            raise ValueError(f"u must have shape ({self.n},), got {state.shape}")
        return self.xi @ np.tanh(self.beta * state) / self.n

    def _validate_coefficients(self, a: np.ndarray) -> np.ndarray:
        coeff = np.asarray(a, dtype=np.float64)
        if coeff.shape != (self.p,):
            raise ValueError(f"a must have shape ({self.p},), got {coeff.shape}")
        if not np.isfinite(coeff).all():
            raise ValueError("a contains NaN or infinity")
        return coeff


@dataclass(frozen=True)
class MhnArchitecture:
    """One precisely named choice of storage ``J`` and sequence ``K`` fields."""

    context: MhnContext
    identifier: str
    j_kind: FieldKind
    k_kind: Literal["hebb", "pinv"]
    b2: Optional[float] = None
    soft_enabled: bool = False

    def __post_init__(self) -> None:
        if self.j_kind == "soft":
            if not self.soft_enabled or self.b2 is None:
                raise ValueError(
                    "soft storage is conditional; use make_soft_architecture(..., enabled=True)"
                )
            _validate_b2(self.b2)
        elif self.j_kind not in ("hebb", "pinv"):
            raise ValueError(f"unknown J field {self.j_kind!r}")
        elif self.b2 is not None:
            raise ValueError("b2 is only valid for soft storage")
        if self.k_kind not in ("hebb", "pinv"):
            raise ValueError(f"unknown K field {self.k_kind!r}")

    def store_field(self, a: np.ndarray) -> np.ndarray:
        return self._field(self.j_kind, a)

    def sequence_field(self, a: np.ndarray) -> np.ndarray:
        """Return the K field before the cyclic shift ``S``."""

        return self._field(self.k_kind, a)

    def delayed_drive(self, a_delayed: np.ndarray) -> np.ndarray:
        """Return ``S f_K(a_delayed)`` with ``(S v)_nu=v_(nu-1)``."""

        return np.roll(self.sequence_field(a_delayed), 1)

    def store_jacobian(self, a: np.ndarray) -> np.ndarray:
        return self._jacobian(self.j_kind, a)

    def sequence_jacobian(self, a: np.ndarray) -> np.ndarray:
        return self._jacobian(self.k_kind, a)

    def delayed_jacobian(self, a_delayed: np.ndarray) -> np.ndarray:
        """Jacobian of :meth:`delayed_drive` (row permutation by ``S``)."""

        return np.roll(self.sequence_jacobian(a_delayed), 1, axis=0)

    def rhs(
        self,
        a_now: np.ndarray,
        a_delayed: np.ndarray,
        lam: float,
        *,
        t0: float = 1.0,
    ) -> np.ndarray:
        """Reduced DDE right-hand side, without performing time integration."""

        lam_value, t0_value = _validate_dynamics(lam, t0)
        now = self.context._validate_coefficients(a_now)
        self.context._validate_coefficients(a_delayed)
        drive = (
            (1.0 - lam_value) * self.store_field(now)
            + lam_value * self.delayed_drive(a_delayed)
        )
        return (-now + drive) / t0_value

    def rhs_jacobians(
        self,
        a_now: np.ndarray,
        a_delayed: np.ndarray,
        lam: float,
        *,
        t0: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Jacobians of ``rhs`` with respect to current and delayed states."""

        lam_value, t0_value = _validate_dynamics(lam, t0)
        current = (
            -np.eye(self.context.p)
            + (1.0 - lam_value) * self.store_jacobian(a_now)
        ) / t0_value
        delayed = (
            lam_value * self.delayed_jacobian(a_delayed) / t0_value
        )
        return current, delayed

    def full_rhs(
        self,
        u_now: np.ndarray,
        u_delayed: np.ndarray,
        lam: float,
        *,
        t0: float = 1.0,
    ) -> np.ndarray:
        """Low-rank full-N RHS used only for algebraic validation.

        The method applies ``X`` and ``X.T`` factors directly and never forms
        either full coupling.  For an in-span history it must equal
        ``X @ rhs(a_now, a_delayed)``.  Off span, the orthogonal component is
        exactly ``-w/t0`` while the parallel component can depend on ``w``.
        """

        lam_value, t0_value = _validate_dynamics(lam, t0)
        now = np.asarray(u_now, dtype=np.float64)
        delayed = np.asarray(u_delayed, dtype=np.float64)
        if now.shape != (self.context.n,) or delayed.shape != (self.context.n,):
            raise ValueError(
                f"full states must both have shape ({self.context.n},)"
            )
        j_coeff = self._full_field(self.j_kind, now)
        k_coeff = self._full_field(self.k_kind, delayed)
        drive_coeff = (
            (1.0 - lam_value) * j_coeff
            + lam_value * np.roll(k_coeff, 1)
        )
        return (-now + self.context.xi.T @ drive_coeff) / t0_value

    def _field(self, kind: FieldKind, a: np.ndarray) -> np.ndarray:
        if kind == "hebb":
            return self.context.hebb_field(a)
        if kind == "pinv":
            return self.context.pinv_field(a)
        if kind == "soft":
            assert self.b2 is not None
            return self.context.soft_field(a, self.b2)
        raise AssertionError(f"unreachable field kind {kind!r}")

    def _jacobian(self, kind: FieldKind, a: np.ndarray) -> np.ndarray:
        if kind == "hebb":
            return self.context.hebb_jacobian(a)
        if kind == "pinv":
            return self.context.pinv_jacobian(a)
        if kind == "soft":
            assert self.b2 is not None
            return self.context.soft_jacobian(a, self.b2)
        raise AssertionError(f"unreachable field kind {kind!r}")

    def _full_field(self, kind: FieldKind, u: np.ndarray) -> np.ndarray:
        if kind == "soft":
            assert self.b2 is not None
            # xi @ u / N == G @ a whenever u == X @ a.
            return stable_softmax(self.b2 * (self.context.xi @ u) / self.context.n)
        overlap = self.context.overlap_from_full_state(u)
        if kind == "hebb":
            return overlap
        if kind == "pinv":
            return self.context.apply_pinv(overlap)
        raise AssertionError(f"unreachable field kind {kind!r}")


def factorial_architectures(
    context: MhnContext,
) -> Dict[str, MhnArchitecture]:
    """Return the mandatory J x K factorial with stable identifiers."""

    definitions = {
        "JH_KH": ("hebb", "hebb"),
        "JP_KH": ("pinv", "hebb"),
        "JH_KP": ("hebb", "pinv"),
        "JP_KP": ("pinv", "pinv"),
    }
    return {
        identifier: MhnArchitecture(
            context=context,
            identifier=identifier,
            j_kind=j_kind,
            k_kind=k_kind,
        )
        for identifier, (j_kind, k_kind) in definitions.items()
    }


def make_soft_architecture(
    context: MhnContext,
    *,
    b2: float,
    k_kind: Literal["hebb", "pinv"],
    enabled: bool = False,
) -> MhnArchitecture:
    """Create the gated optional J-SOFT arm.

    The explicit ``enabled=True`` is intentional: the roadmap requires a GO
    decision before softmax enters an experiment.
    """

    if not enabled:
        raise PermissionError(
            "J-SOFT is conditional; pass enabled=True only after its roadmap GO"
        )
    suffix = "KH" if k_kind == "hebb" else "KP"
    identifier = f"JS_{suffix}_b{float(b2):g}"
    return MhnArchitecture(
        context=context,
        identifier=identifier,
        j_kind="soft",
        k_kind=k_kind,
        b2=float(b2),
        soft_enabled=True,
    )


def stable_softmax(values: np.ndarray) -> np.ndarray:
    """One-dimensional float64 softmax stable for very large finite logits."""

    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("softmax expects a non-empty one-dimensional array")
    if not np.isfinite(logits).all():
        raise ValueError("softmax logits must be finite")
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    normalizer = float(np.sum(weights))
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise FloatingPointError("invalid softmax normalization")
    return weights / normalizer


def _validate_b2(b2: float) -> float:
    value = float(b2)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("b2 must be finite and positive")
    return value


def _validate_dynamics(lam: float, t0: float) -> tuple[float, float]:
    lam_value = float(lam)
    t0_value = float(t0)
    if not np.isfinite(lam_value):
        raise ValueError("lam must be finite")
    if not np.isfinite(t0_value) or t0_value <= 0:
        raise ValueError("t0 must be finite and positive")
    return lam_value, t0_value
