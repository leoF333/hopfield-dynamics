"""Fast numerical regression tests of the existing research implementation."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/core"))
from cycle_reduced import ReducedDDE, integrate_fullN


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.xi = np.random.default_rng(7).choice([-1., 1.], size=(5, 80))
        self.model = ReducedDDE(self.xi, beta=3., lam=.65, tau=1.)
        self.a0 = np.array([.7, .1, -.05, .02, .01])

    def test_reduced_rhs_matches_dense_hebbian_model(self):
        a, b = self.a0, self.a0[::-1]
        x = self.xi; J = x.T @ x / 80
        K = np.roll(x, -1, axis=0).T @ x / 80
        u, delayed = x.T @ a, x.T @ b
        full = -u + .35 * J @ np.tanh(3*u) + .65 * K @ np.tanh(3*delayed)
        np.testing.assert_allclose(x.T @ self.model.rhs(a,b), full, atol=2e-14, rtol=2e-13)

    def test_analytic_jacobian_matches_finite_differences(self):
        h = 1e-6; basis = np.eye(5)
        fd = np.column_stack([(self.model.m(self.a0+h*v)-self.model.m(self.a0-h*v))/(2*h) for v in basis])
        np.testing.assert_allclose(self.model.Dm(self.a0), fd, atol=2e-9, rtol=2e-8)

    def test_reduced_and_full_trajectories_agree_across_delay(self):
        reduced = self.model.integrate(self.a0, 2., .02)
        full = integrate_fullN(self.xi, 3., .65, 1., 1., self.xi.T @ self.a0, 2., .02)
        np.testing.assert_allclose(reduced["a"] @ self.xi, full, atol=1e-12, rtol=1e-10)

    def test_checkpoint_restart_preserves_history_derivatives(self):
        first = self.model.integrate(self.a0, 1., .02, da_hist0=np.zeros(5))
        resumed = self.model.integrate(first["hist"], 1., .02, da_hist0=first["dhist"])
        whole = self.model.integrate(self.a0, 2., .02, da_hist0=np.zeros(5))
        np.testing.assert_allclose(resumed["a"][-1], whole["a"][-1], atol=1e-13)

    def test_timestep_refinement(self):
        finals = [self.model.integrate(self.a0, 3., h, da_hist0=np.zeros(5))["a"][-1] for h in (.1,.05,.025)]
        self.assertLess(np.linalg.norm(finals[1]-finals[2]), np.linalg.norm(finals[0]-finals[1])/4)

    def test_archived_threshold_reference(self):
        with np.load(ROOT / "data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz", allow_pickle=False) as d:
            t = d["lam_c"]
            self.assertEqual(t.shape, (100,))
            self.assertTrue(np.isfinite(t).all())
            self.assertAlmostEqual(float(t.min()), .2195, delta=.001)
            self.assertAlmostEqual(float(t.max()), .3276, delta=.001)
            self.assertAlmostEqual(float(np.median(t)), .2742, delta=.001)


if __name__ == "__main__":
    unittest.main()
