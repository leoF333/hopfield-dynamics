# Numerical methods: a guided reading

## Model and reduction

The continuous network combines symmetric Hebbian storage with a delayed successor coupling. In neuron coordinates, t0 u' = -u + (1-lambda) J tanh(beta u) + lambda K tanh(beta u(t-tau)). Both coupling matrices have image inside the span of the stored patterns. The orthogonal component therefore decays exponentially; the in-span dynamics closes exactly on P pattern coordinates.

Read `code/core/cycle_reduced.py`: `ReducedDDE.m`, `rhs`, `Dm`, and `integrate`. This exact finite-sample reduction preserves the cross-overlaps that disorder averaging would remove.

## Integration and numerical consistency

The existing integrator uses fixed-step RK4 with cubic-Hermite history interpolation. Its delay must be an integer multiple of the step. When supplying an arbitrary constant history, the example supplies its zero derivative explicitly. Checkpoints retain both the history and its derivative.

The fast tests compare the implementation against a separately constructed dense Hebbian right-hand side, finite-difference derivatives and full-neuron trajectories. Step refinement and restart tests address integration accuracy and reproducibility. They are bounded regressions, not a complete convergence study for every regime.

## Continuation and instability

`code/core/robust_branch.py` and `code/experiments/e24_thresholds.py` trace memory-connected fixed points and refine terminal thresholds. The primary estimator is the bisection threshold `lam_c`, not the spectral extrapolation `lam_extrap`. Branch identity, residuals and spectral behaviour matter: an algebraically converged root need not be the intended stable memory.

## Statistics and evidence

The quick-start threshold plot reads the archived N=2000, P=100, seed=42 result without refitting it. It is a reproduction of a figure from stored results, distinct from regeneration of those results. The E-series and independent N-series campaigns provide the deeper research record. In cross-seed uncertainty estimates, disorder realizations rather than individual memories are the relevant resampling units.

`docs/CLAIMS.md` maps statements to experiments and data. `docs/OPEN_ITEMS.md` records known qualifications. In particular, floored Floquet multiplier estimates must be presented as bounds, not direct measurements, and a measured threshold law must not be described as a proved central-limit theorem.

## Questions for further work

- Derive the threshold law and its finite-size variance analytically.
- Test the mechanism beyond the controlled low-load regime.
- Determine whether the implicit-delay architecture reproduces the threshold ensemble, not only the recall clock.
- Complete portability and figure assembly for the historical research workflows.
