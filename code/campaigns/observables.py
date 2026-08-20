"""Shared trajectory observables for delay, correlation, load and aging scans."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


from pathlib import Path

import numpy as np

from dynamics import periodic_closure, relay_statistics, reduced_fixed_residual
from v5_paths import activate_reference_modules

activate_reference_modules()
from cycle_reduced import ReducedDDE  # noqa: E402


def simulate_observables(
    xi: np.ndarray,
    *,
    beta: float,
    lam: float,
    tau: float,
    t0: float,
    dt: float,
    transient: float,
    observation: float,
    initial_history: np.ndarray | None = None,
    initial_derivative_history: np.ndarray | None = None,
    record_dt: float = 0.1,
) -> tuple[dict, np.ndarray, np.ndarray]:
    P, N = xi.shape
    system = ReducedDDE(xi, beta, lam, tau, t0)
    if initial_history is None:
        a0 = np.zeros(P)
        a0[0] = 0.99
        a0[1 % P] = 0.05
        initial_history = a0
    settled = system.integrate(
        initial_history, transient, dt,
        record_every=max(1, round(record_dt / dt)),
        da_hist0=initial_derivative_history,
    )
    recorded = system.integrate(
        settled["hist"], observation, dt,
        record_every=max(1, round(record_dt / dt)),
        da_hist0=settled["dhist"],
    )
    times = recorded["t"]
    A = recorded["a"]
    winner = np.argmax(A, axis=1)
    relay = relay_statistics(times, winner, P)

    positive = np.clip(A, 0.0, None)
    s2 = np.sum(positive**2, axis=1)
    s4 = np.sum(positive**4, axis=1)
    participation = s2**2 / np.maximum(s4, 1e-30)
    phase = np.exp(2j * np.pi * np.arange(P) / P)
    mass = positive.sum(axis=1)
    coherence = np.abs(positive @ phase) / np.maximum(mass, 1e-30)
    peak = A.max(axis=1)
    width = np.sum(A > 0.5 * peak[:, None], axis=1)

    subsample = np.linspace(0, len(A) - 1, min(len(A), 1000), dtype=int)
    U = xi.T @ A[subsample].T
    sign_u = np.sign(U)
    sign_u[sign_u == 0] = 1.0
    binary = xi @ sign_u / N
    selected_binary = binary[
        winner[subsample], np.arange(len(subsample))
    ]
    closure, candidate_period = periodic_closure(times, A, tau)
    residual = reduced_fixed_residual(system, recorded["hist"])
    if residual < 1e-10:
        classification = "fixed_point"
    elif relay["tours"] >= 2 and relay["forward_fraction"] > 0.99:
        classification = (
            "periodic_sequence_candidate" if closure < 1e-5
            else "moving_requires_lyapunov"
        )
    elif closure < 1e-5:
        classification = "nonmoving_periodic"
    else:
        classification = "other_requires_lyapunov"

    changes = np.flatnonzero(np.diff(winner) != 0) + 1
    keep = np.linspace(0, len(A) - 1, min(1500, len(A)), dtype=int)
    output = {
        "classification": classification,
        "fixed_residual": residual,
        "periodic_closure": closure,
        "candidate_period": candidate_period,
        "relay_count": int(len(changes)),
        "forward_fraction": relay["forward_fraction"],
        "reversal_fraction": relay["reversal_fraction"],
        "tours": relay["tours"],
        "slowest_bond": relay["slowest_bond"],
        "dwell": relay["dwell"],
        "relay_times": times[changes],
        "relay_winner_before": winner[np.maximum(changes - 1, 0)],
        "relay_winner_after": winner[changes],
        "participation_mean": float(np.mean(participation)),
        "participation_std": float(np.std(participation)),
        "coherence_mean": float(np.mean(coherence)),
        "coherence_std": float(np.std(coherence)),
        "front_width_mean": float(np.mean(width)),
        "front_width_std": float(np.std(width)),
        "peak_analog_mean": float(np.mean(peak)),
        "peak_analog_min": float(np.min(peak)),
        "binary_retrieval_mean": float(np.mean(selected_binary)),
        "binary_retrieval_min": float(np.min(selected_binary)),
        "kymograph_t": times[keep],
        "kymograph_a": A[keep],
    }
    return output, recorded["hist"], recorded["dhist"]


def save_observables(path: Path, result: dict,
                     history: np.ndarray, derivative_history: np.ndarray,
                     parameters: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, value in result.items():
        arrays[key] = np.array(value)
    for key, value in parameters.items():
        arrays[key] = np.array(value)
    arrays["final_history"] = history
    arrays["final_derivative_history"] = derivative_history
    np.savez_compressed(path, **arrays)
