"""Streaming float64 Benettin/QR integrator for the exact reduced DDE.

Unlike the archived E23 implementation, memory is O(k P tau/dt), independent of
the accumulation time.  The tangent histories and their derivatives are
transformed by the same QR change of basis, preserving Hermite interpolation.
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
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from couplings import make_patterns
from v5_paths import RUNS, activate_reference_modules, environment_manifest, write_json

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402


@dataclass
class LyapunovConfig:
    N: int = 2000
    P: int = 100
    seed: int = 42
    beta: float = 20.0
    tau: float = 10.0
    t0: float = 1.0
    lam: float = 0.31
    dt: float = 0.01
    transient: float = 400.0
    accumulation: float = 1000.0
    k: int = 8
    qr_time: float = 2.0
    record_dt: float = 0.2
    tangent_seed: int = 12345


class StreamingBenettin:
    def __init__(self, system: ReducedDDE, dt: float, k: int,
                 qr_steps: int, rng: np.random.Generator):
        self.system = system
        self.dt = float(dt)
        self.k = int(k)
        self.qr_steps = int(qr_steps)
        self.rng = rng
        self.L = int(round(system.tau / dt))
        if abs(self.L * dt - system.tau) > 1e-12:
            raise ValueError("dt must divide tau")
        self.R = self.L + 2
        self.shift = np.roll(np.eye(system.P), 1, axis=0)

    def _dm_batch(self, a: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Apply Dm(a) to k row-vectors without forming the P x P matrix."""
        xi = self.system.xi
        gain = self.system.beta * (
            1.0 - np.tanh(self.system.beta * (xi.T @ a)) ** 2
        )
        neuron = xi.T @ vectors.T
        return (xi @ (gain[:, None] * neuron) / self.system.N).T

    def _var_rhs(self, a_now: np.ndarray, a_del: np.ndarray,
                 v_now: np.ndarray, v_del: np.ndarray) -> np.ndarray:
        return (
            -v_now
            + (1.0 - self.system.lam) * self._dm_batch(a_now, v_now)
            + self.system.lam * np.roll(
                self._dm_batch(a_del, v_del), 1, axis=1
            )
        ) / self.system.t0

    def run(self, history: np.ndarray, derivative_history: np.ndarray,
            accumulation: float, record_every: int = 1) -> dict:
        P, k, L, R, dt = (
            self.system.P, self.k, self.L, self.R, self.dt
        )
        if history.shape != (L + 1, P):
            raise ValueError(f"history must have shape {(L + 1, P)}")
        if derivative_history.shape != history.shape:
            raise ValueError("history and derivative_history shapes differ")
        n_steps = int(round(accumulation / dt))

        A = np.empty((R, P), dtype=np.float64)
        dA = np.empty_like(A)
        for g in range(L + 1):
            A[g % R] = history[g]
            dA[g % R] = derivative_history[g]

        raw = self.rng.standard_normal((k, (L + 1) * P))
        Q, _ = np.linalg.qr(raw.T)
        initial = Q.T.reshape(k, L + 1, P)
        V = np.empty((k, R, P), dtype=np.float64)
        dV = np.empty_like(V)
        for g in range(L + 1):
            V[:, g % R, :] = initial[:, g, :]
        for g in range(L + 1):
            gd = max(0, g - L)
            dV[:, g % R, :] = self._var_rhs(
                A[g % R], A[gd % R], V[:, g % R], V[:, gd % R]
            )

        def interpolate(buffer: np.ndarray, derivative: np.ndarray, x: float):
            i0 = int(np.floor(x))
            s = x - i0
            if s < 1e-14:
                return buffer[..., i0 % R, :] if buffer.ndim == 3 else buffer[i0 % R]
            h00 = (1 + 2*s) * (1-s)**2
            h10 = s * (1-s)**2
            h01 = s*s * (3-2*s)
            h11 = s*s * (s-1)
            if buffer.ndim == 3:
                return (
                    h00 * buffer[:, i0 % R, :]
                    + h10 * dt * derivative[:, i0 % R, :]
                    + h01 * buffer[:, (i0 + 1) % R, :]
                    + h11 * dt * derivative[:, (i0 + 1) % R, :]
                )
            return (
                h00 * buffer[i0 % R] + h10 * dt * derivative[i0 % R]
                + h01 * buffer[(i0 + 1) % R]
                + h11 * dt * derivative[(i0 + 1) % R]
            )

        logsum = np.zeros(k)
        block_time = []
        block_exponents = []
        cumulative = []
        record_t = []
        record_a = []
        started = time.perf_counter()
        n_qr_done = 0
        for n in range(n_steps):
            g = L + n
            a = A[g % R]
            ad1 = interpolate(A, dA, g - L)
            ad2 = interpolate(A, dA, g + 0.5 - L)
            ad4 = interpolate(A, dA, g + 1.0 - L)
            k1 = self.system.rhs(a, ad1)
            a2 = a + 0.5 * dt * k1
            k2 = self.system.rhs(a2, ad2)
            a3 = a + 0.5 * dt * k2
            k3 = self.system.rhs(a3, ad2)
            a4 = a + dt * k3
            k4 = self.system.rhs(a4, ad4)
            anew = a + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

            v = V[:, g % R, :]
            vd1 = interpolate(V, dV, g - L)
            vd2 = interpolate(V, dV, g + 0.5 - L)
            vd4 = interpolate(V, dV, g + 1.0 - L)
            q1 = self._var_rhs(a, ad1, v, vd1)
            q2 = self._var_rhs(a2, ad2, v + 0.5*dt*q1, vd2)
            q3 = self._var_rhs(a3, ad2, v + 0.5*dt*q2, vd2)
            q4 = self._var_rhs(a4, ad4, v + dt*q3, vd4)
            vnew = v + dt / 6.0 * (q1 + 2*q2 + 2*q3 + q4)

            gn = g + 1
            A[gn % R] = anew
            dA[gn % R] = self.system.rhs(anew, ad4)
            V[:, gn % R, :] = vnew
            dV[:, gn % R, :] = self._var_rhs(anew, ad4, vnew, vd4)

            if (n + 1) % self.qr_steps == 0:
                indices = np.arange(gn - L, gn + 1)
                window = np.stack([V[:, idx % R, :] for idx in indices], axis=1)
                W = window.reshape(k, -1)
                Qm, Rm = np.linalg.qr(W.T)
                diagonal = np.abs(np.diag(Rm))
                logs = np.log(np.maximum(diagonal, np.finfo(float).tiny))
                logsum += logs
                n_qr_done += 1
                local_dt = self.qr_steps * dt
                block_time.append((n + 1) * dt)
                block_exponents.append(logs / local_dt)
                cumulative.append(logsum / (n_qr_done * local_dt))
                transform = np.linalg.solve(Rm.T, np.eye(k))
                for idx in indices:
                    slot = idx % R
                    V[:, slot, :] = transform @ V[:, slot, :]
                    dV[:, slot, :] = transform @ dV[:, slot, :]

            if (n + 1) % record_every == 0:
                record_t.append((n + 1) * dt)
                record_a.append(anew.copy())

        elapsed = time.perf_counter() - started
        total_qr_time = n_qr_done * self.qr_steps * dt
        return {
            "lyapunov": logsum / total_qr_time,
            "block_time": np.asarray(block_time),
            "block_exponents": np.asarray(block_exponents),
            "cumulative_exponents": np.asarray(cumulative),
            "record_t": np.asarray(record_t),
            "record_a": np.asarray(record_a),
            "final_history": np.stack(
                [A[g % R] for g in range(L + n_steps - L, L + n_steps + 1)]
            ),
            "final_derivative_history": np.stack(
                [dA[g % R] for g in range(L + n_steps - L, L + n_steps + 1)]
            ),
            "wall_seconds": elapsed,
        }


def run(config: LyapunovConfig, output: Path,
        initial_history: np.ndarray | None = None) -> dict:
    sidecar = output.with_suffix(".json")
    if (
        output.exists() and sidecar.exists()
        and os.environ.get("V5_FORCE", "0") != "1"
    ):
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if saved.get("config") != asdict(config):
            raise RuntimeError(
                f"refusing to reuse {output}: saved Lyapunov configuration "
                "does not match the requested configuration"
            )
        return saved
    xi, _ = make_patterns(config.N, config.P, config.seed)
    system = ReducedDDE(
        xi, config.beta, config.lam, config.tau, config.t0
    )
    if initial_history is None:
        rng_ic = np.random.default_rng(config.seed + 1000)
        a0 = 0.3 * rng_ic.standard_normal(config.P)
        transient = system.integrate(a0, config.transient, config.dt)
    else:
        transient = system.integrate(
            initial_history, config.transient, config.dt
        )
    runner = StreamingBenettin(
        system, config.dt, config.k,
        max(1, round(config.qr_time / config.dt)),
        np.random.default_rng(config.tangent_seed),
    )
    result = runner.run(
        transient["hist"], transient["dhist"], config.accumulation,
        record_every=max(1, round(config.record_dt / config.dt)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{key: value for key, value in result.items() if key != "wall_seconds"},
        wall_seconds=np.array(result["wall_seconds"]),
        **{key: np.array(value) for key, value in asdict(config).items()},
    )
    positive = int(np.sum(result["lyapunov"] > 0))
    summary = {
        "config": asdict(config), "output": str(output),
        "lyapunov": result["lyapunov"].tolist(),
        "positive_raw": positive,
        "wall_seconds": result["wall_seconds"],
        "environment": environment_manifest(),
    }
    write_json(sidecar, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.31)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--transient", type=float, default=400.0)
    parser.add_argument("--accumulation", type=float, default=1000.0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--qr-time", type=float, default=2.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = LyapunovConfig(
        N=args.N, P=args.P, seed=args.seed, lam=args.lam, tau=args.tau,
        dt=args.dt, transient=args.transient, accumulation=args.accumulation,
        k=args.k, qr_time=args.qr_time,
    )
    output = args.out or (
        RUNS / "N6" /
        f"lyap_N{args.N}_P{args.P}_s{args.seed}_lam{args.lam:.4f}_dt{args.dt}.npz"
    )
    print(run(cfg, output))


if __name__ == "__main__":
    main()
