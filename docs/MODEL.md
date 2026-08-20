# The model and the exact reduction

Everything in this repository rests on one algebraic fact: the `N`-dimensional delayed
dynamics closes **exactly** on `P` overlap coordinates, without any disorder average. That
is what makes an exhaustive bifurcation study of a disordered network feasible at all.

## Dynamics

`N` continuous neurons `u_i(t)`, firing rates `φ_i = tanh(β u_i)`:

```
t₀ u̇(t) = −u(t) + (1−λ) J φ(t) + λ K φ(t−τ)
```

- `t₀` — single-neuron relaxation time (the unit of time throughout).
- `τ` — transmission delay, acting **only** on the non-reciprocal pathway.
- `λ ∈ [0,1]` — mixes energetic symmetric storage with delayed non-equilibrium recall.
  `λ = 0` recovers Hopfield's graded-response network exactly.

For unbiased binary patterns `ξ^μ ∈ {−1,+1}^N`, with the cyclic convention
`ξ^(P+1) = ξ^1`:

```
J = (1/N) Σ_μ ξ^μ (ξ^μ)ᵀ          symmetric Hebbian: each pattern is a static retrieval state
K = (1/N) Σ_μ ξ^(μ+1) (ξ^μ)ᵀ      non-reciprocal: maps activity at ξ^μ toward ξ^(μ+1)
```

`K ≠ Kᵀ` puts the model outside gradient dynamics: there is no energy landscape to descend.

## Exact reduction to the pattern subspace

Write the patterns as the rows of `X ∈ {−1,+1}^(P×N)` and let `S` be the cyclic shift.
Assuming the Gram matrix `X Xᵀ` is invertible — which holds with probability approaching one
at the loads studied here — we have `J = N⁻¹ Xᵀ X` and `K = N⁻¹ Xᵀ S X`, and the
decomposition `u = Xᵀ a + w` with `w ⊥ range(Xᵀ)` is unique.

Both local fields lie in `range(Xᵀ)`, so the transverse component obeys exactly

```
t₀ ẇ = −w
```

The realized pattern subspace is therefore invariant and globally attracting, and on it the
dynamics closes exactly:

```
t₀ ȧ(t) = −a(t) + (1−λ) m[a(t)] + λ S m[a(t−τ)]

m(a) = (1/N) X tanh(β Xᵀ a)
```

This is a **finite-sample** reduction, not a disorder-averaged closure. Every cross-overlap
of the realized patterns survives inside `m`, and it is exactly that retained quenched
interference which splits the bifurcations. On the stored certification trajectory, full and
reduced solutions agree to `10⁻¹⁵`.

Implementation: `code/core/cycle_reduced.py` (the `ReducedDDE` class, with the V0/V1/V4
validations in `__main__`).

## The characteristic problem

At a fixed point, delayed and instantaneous activities coincide, so the delay **drops out of
the equilibrium equation**:

```
F(u; λ) = −u + C(λ) tanh(βu) = 0,     C(λ) = (1−λ) J + λ K
```

The entire fixed-point landscape — every equilibrium and every fold location — is therefore
delay-independent. Only stability margins and the surrounding dynamics depend on `τ`.

Linearizing the full DDE with the diagonal gain `D = β diag{sech²(β u*)}` and inserting
`δu = e^(zt) v` gives the characteristic matrix

```
Δ(z) = (t₀z + 1) I − (1−λ) J D − λ e^(−zτ) K D
```

Because of `e^(−zτ)` this has countably infinitely many roots. But the couplings have rank
at most `P`, so Sylvester's determinant identity collapses the transcendental `N×N` problem
onto `P×P`:

```
det Δ(z) = (t₀z + 1)^(N−P) · det T_P(z)

T_P(z) = (t₀z + 1) I_P − [ (1−λ) I_P + λ e^(−zτ) S ] · (X D Xᵀ / N)
```

The prefactor is the trivial bulk of transverse roots at `z = −1/t₀`; every nontrivial root
solves `det T_P(z) = 0`. A dense `N×N` cross-check agrees with the Sylvester route to
`2 × 10⁻¹⁶` (experiment E21d).

**The fold discriminator.** Evaluating at `z = 0`, where `e^(−zτ) = 1`, gives the exact
identity

```
Δ(0) = I − C D = −M
```

A zero root of the delay equation *is* a zero eigenvalue of the static continuation problem.
A fold can therefore be located on the equilibrium branch and certified independently of the
delay spectrum, and its position is automatically `τ`-independent. The only way the delay
could change the death of a branch is a complex pair crossing the axis *before* the fold —
a delay-induced Hopf bifurcation — which must be checked separately on the delayed spectrum,
and is not observed in the regime studied (see `docs/CLAIMS.md`, C-09 and C-11).

Implementation: `code/core/reduced_spectrum.py` (exact `T_P(z)`),
`code/core/dde_stability.py` (pseudospectral infinitesimal generator, used for
cross-validation), `code/core/robust_branch.py` (Woodbury branch tracer).

## Observables

```
m_μ(t) = (1/N) (ξ^μ)ᵀ tanh(β u(t))       analog overlap
b_μ(t) = (1/N) (ξ^μ)ᵀ sgn(u(t))          binary readout overlap
```

At `β = 20`, `tanh(βu)` is within `10⁻⁴` of `sgn(u)` whenever `|u| ≳ 0.5`. This matters: the
analog overlap saturates, so a branch can tilt strongly toward its successor in the
subspace coordinates `a` while the overlap readout still reads close to one. The two
observables describe the same smooth distortion through two differently saturating lenses.

## The threshold ensemble

Newton continuation from each stored pattern at `λ = 0` tracks its primary stable retrieval
branch, together with the adjacent index-one saddle, up to the point where the smallest
singular value of `M` vanishes. That defines `λ_c(μ)` for each of the `P` patterns.

Labels are inherited **continuously** from the `λ = 0` retrieval state and checked through
the overlap vector, never read off the state at the fold, where the retrieved memory has
become a distorted object. This is what makes `{λ_c(μ)}` a well-defined quenched observable.

The primary estimator is bisection, bracketed to `±2 × 10⁻⁴`. The spectral extrapolation
`eig_max(M) = −c √(λ_c − λ)` is used as a cross-check, never as the primary value.

Implementation: `code/experiments/e24_thresholds.py`.

## Two order statistics, two transitions

```
f_stat(λ) = (1/P) Σ_μ Θ[λ_c(μ) − λ]        fraction of surviving primary branches
λ*        = max_μ λ_c(μ)                   the last surviving barrier
```

The bulk of `{λ_c(μ)}` ends extensive static recall. Sequential recall is all-or-nothing —
a wave that must visit all `P` memories is held by the last surviving barrier — so its onset
is controlled by `λ*`. At finite `N` the two are separated by an observable window; as
`N → ∞` the ensemble narrows and they merge.
