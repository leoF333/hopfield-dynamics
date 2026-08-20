from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["V5_BACKEND"] = "cpu"

from couplings import Couplings, make_patterns
from dynamics import (
    DynamicConfig,
    integrate_and_classify,
    relay_statistics,
    relay_statistics_from_events,
)
from fold_refinement import refine_fold_from_full_state
from patterns import adjacent_correlations, gp_arcsine_prediction, gp_circle, markov_flip
from static_thresholds import StaticConfig, run as run_static
from v5_paths import activate_reference_modules

activate_reference_modules()
from cycle_reduced import ReducedDDE, integrate_fullN


def test_cpu_operators_match_dense():
    xi, xis = make_patterns(40, 5, 9)
    coup = Couplings(xi, xis)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(40)
    assert np.allclose(coup.apply_J(x), coup.dense_J() @ x, atol=1e-12)
    assert np.allclose(coup.apply_K(x), coup.dense_K() @ x, atol=1e-12)
    for lam in (0.0, 0.37, 1.0):
        dense = ((1 - lam) * coup.dense_J() + lam * coup.dense_K()) @ x
        assert np.allclose(coup.apply_C(x, lam), dense, atol=1e-12)


def test_exact_reduction_matches_full_n():
    N, P = 80, 4
    xi, _ = make_patterns(N, P, 4)
    sysP = ReducedDDE(xi, beta=8.0, lam=0.4, tau=1.0, t0=1.0)
    rng = np.random.default_rng(3)
    a0 = rng.normal(scale=0.1, size=P)
    dt = 0.02
    reduced = sysP.integrate(a0, 2.0, dt)["a"]
    full = integrate_fullN(xi, 8.0, 0.4, 1.0, 1.0, xi.T @ a0, 2.0, dt)
    assert np.linalg.norm(xi.T @ reduced[-1] - full[-1]) < 1e-10


def test_relay_detector_counts_ordered_tours():
    P = 7
    winner = np.repeat(np.arange(P).tolist() * 6, 4)
    times = np.arange(len(winner), dtype=float) * 0.1
    stats = relay_statistics(times, winner, P)
    assert stats["tours"] >= 5
    assert stats["forward_fraction"] == 1.0
    assert stats["reversal_fraction"] == 0.0


def test_streaming_relay_events_match_full_detector():
    P = 7
    winner = np.repeat(np.arange(P).tolist() * 8, 4)
    times = np.arange(len(winner), dtype=float) * 0.1
    changes = np.flatnonzero(np.diff(winner) != 0) + 1
    full = relay_statistics(times, winner, P)
    streamed = relay_statistics_from_events(
        times[changes], winner[changes - 1], winner[changes], P
    )
    for key in (
        "changes", "advance", "tours", "forward_fraction",
        "slowest_bond", "reversal_fraction",
    ):
        assert streamed[key] == full[key]
    assert np.allclose(streamed["dwell"], full["dwell"], equal_nan=True)
    assert np.array_equal(streamed["relay_times"], full["relay_times"])
    assert np.array_equal(streamed["relay_before"], full["relay_before"])
    assert np.array_equal(streamed["relay_after"], full["relay_after"])


def test_streaming_relay_count_survives_state_tail_pruning():
    class SlowOrderedSystem:
        P = 4
        tau = 10.0

        def __init__(self):
            self.elapsed = 0.0

        def integrate(self, history, duration, dt, record_every, da_hist0=None):
            sample_dt = dt * record_every
            local_t = np.arange(0.0, duration + 0.5 * sample_dt, sample_dt)
            global_t = self.elapsed + local_t
            winner = (global_t // 600.0).astype(int) % self.P
            states = np.zeros((len(local_t), self.P))
            states[np.arange(len(local_t)), winner] = 1.0
            self.elapsed += duration
            return {
                "t": local_t,
                "a": states,
                "hist": states[-2:],
                "dhist": np.zeros((2, self.P)),
            }

        def rhs(self, a, delayed):
            return np.ones(self.P)

    result, _, _ = integrate_and_classify(
        SlowOrderedSystem(),
        np.array([1.0, 0.0, 0.0, 0.0]),
        None,
        DynamicConfig(
            N=4, P=4, dt=0.01, record_dt=0.05, chunk_time=200.0,
            max_time=15_000.0, required_tours=5,
        ),
    )
    assert result["label"] == "ordered_cycle"
    assert result["time"] > 10_000.0
    assert result["tours"] >= 5.0
    assert len(result["relay_times"]) >= 20


def test_pattern_generators_have_expected_correlations():
    N, P = 30_000, 30
    c = 0.6
    markov = markov_flip(N, P, c, 10)
    q = adjacent_correlations(markov)
    assert abs(q[:-1].mean() - c) < 0.02
    assert abs(q[-1]) < 0.03

    kappa = 0.1
    gp = gp_circle(N, P, kappa, 11)
    q_gp = adjacent_correlations(gp).mean()
    assert abs(q_gp - gp_arcsine_prediction(P, kappa)) < 0.03


def test_augmented_fold_refinement_resolves_zero_mode(tmp_path):
    output = tmp_path / "static.npz"
    run_static(
        StaticConfig(
            N=120, P=6, seed=42, beta=12.0,
            initial_step=0.04, bracket_tol=0.002,
            lambda_max=0.75, relaxed_identity=True,
        ),
        output,
        nproc=1,
    )
    data = np.load(output)
    valid = np.flatnonzero(data["resolved"])
    motif = int(valid[np.argmax(data["lambda_low"][valid])])
    guess = 0.5 * (
        float(data["lambda_low"][motif])
        + float(data["lambda_high"][motif])
    )
    xi, _ = make_patterns(120, 6, 42)
    fold = refine_fold_from_full_state(
        xi,
        data["state"][motif],
        guess,
        beta=12.0,
        xtol=1e-10,
    )
    assert fold.converged
    assert fold.fixed_residual < 1e-9
    assert fold.null_residual < 1e-9
    assert data["lambda_low"][motif] - 0.01 < fold.lambda_fold
    assert fold.lambda_fold < data["lambda_high"][motif] + 0.01
