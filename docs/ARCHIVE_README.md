# Delayed non-reciprocal Hopfield networks — code, data and evidence

Sequential memory recall in a Hopfield network is born from an **ensemble of quenched
bifurcations inside a single disordered sample**. This repository holds the code, the
derived data and the claim-by-claim evidence trail behind that result.

The model is `N` continuous neurons with firing rates `φ = tanh(βu)`, evolving under a
delay differential equation that mixes symmetric Hebbian storage with a delayed
non-reciprocal drive toward the next pattern in a stored cycle:

```
t₀ u̇(t) = −u(t) + (1−λ) J φ(t) + λ K φ(t−τ)

J = (1/N) Σ_μ ξ^μ (ξ^μ)ᵀ            symmetric Hebbian storage
K = (1/N) Σ_μ ξ^(μ+1) (ξ^μ)ᵀ        cyclic sequence drive, delayed by τ
```

As `λ` grows, each stored memory tilts toward its successor and dies at its **own**
saddle-node threshold `λ_c(μ)`. In one finite sample those `P = αN` thresholds form a
near-Gaussian, quasi-i.i.d. ensemble. Two different macroscopic transitions read two
different order statistics of that one law: the **bulk** ends extensive static recall, the
**sample maximum** `λ* = max_μ λ_c(μ)` releases the traveling recall cycle. Between them
lies a stationary hyperchaotic attractor. As `N → ∞` the ensemble narrows, the two
boundaries merge, and the mechanism becomes invisible.

---

## What is in here

| Directory | Contents |
|---|---|
| `code/core/` | The reusable library: couplings, the exact `P`-dimensional reduction, Newton/Woodbury continuation, the delayed characteristic problem `T_P(z)`, Floquet monodromy, the layered-chain model. |
| `code/experiments/` | The historical campaigns **E1–E37**, one script per experiment, named after the worklog entry that documents it. |
| `code/campaigns/` | The **N1–N10** verification campaigns: an independent float64 reimplementation written to re-test the load-bearing claims against a second code path. |
| `code/figures/` | Panel assembly for the published figures. |
| `data/` | Derived data: thresholds, spectra, Lyapunov exponents, campaign reports. Everything a claim rests on. See `docs/DATA.md`. |
| `figures/` | The manuscript figures, the composite panels and their components. |
| `docs/` | The evidence register, the code map, the reproduction runbook, and the primary lab worklogs. |
| `tools/` | Manifest verification and archive hashing. |

## Start here

- **`docs/CLAIMS.md`** — every verifiable statement of the manuscript, linked to the
  experiment, the data file, the script and the figure that support it, with an explicit
  status (established / qualified / limitation / analytic).
- **`docs/CODE_MAP.md`** — what each of the 214 source files does.
- **`docs/REPRODUCING.md`** — how to re-run any experiment.
- **`docs/DATA.md`** — what is shipped here, what stays in the local archive, and why.

## Quick start

```bash
git clone <this repository>
cd <this repository>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The per-pattern threshold ensemble of one disorder realization (the central object).
python code/experiments/e24_thresholds.py --N 2000 --P 100 --seed 42 --store-curves

# Its scaling with system size (uses the shipped E30 arrays).
python code/experiments/e30_analysis.py

# The recall cycle and its Floquet spectrum.
python code/experiments/pacemaker_scan.py --N 2000 --tau 10
```

Scripts resolve their imports through a small path bootstrap inserted at the top of each
file, so they run from anywhere without installing a package.

## Reference realization

Unless a script says otherwise, branch-resolved results use

```
N = 2000    P = 100    α = 0.05    β = 20    τ = 10    seed 42
```

with time in units of the single-neuron relaxation time `t₀`. Robustness was checked up to
`N = 1.4 × 10⁴`. The primary threshold estimator is bisection (`lam_c`), bracketed to
`±2 × 10⁻⁴` — never the spectral extrapolation.

## Two generations of code, deliberately kept apart

`E**` and `N**` are not versions of each other. The `N**` campaigns are an **independent
reimplementation** whose purpose was to re-derive the load-bearing numbers through a
different code path, in float64, with checkpointing and explicit acceptance gates. Where
they disagree with the historical campaigns, `docs/CLAIMS.md` says so and says which one the
manuscript uses.

The `N**` campaigns also carry an incident log
(`data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md`, §10) recording five numerical
artefacts that were caught before they became physical conclusions — a Newton continuation
that silently followed the unstable saddle branch, a conjugate-pair mismatch that faked a
spectral disagreement, a classification label that could be overwritten on merge, and two
others. They are documented on purpose.

## What this repository does not claim

Read `docs/CLAIMS.md` for the per-claim status, and `docs/OPEN_ITEMS.md` for the points
still open. In short:

- No analytical central-limit theorem, Gumbel limit, or SRB measure is proved. The
  near-Gaussian threshold law, its `N^-b` narrowing and the stationarity of the chaotic
  window are **measured**, over stated ranges.
- The `N → ∞` merging of the two transitions is a **conditional extrapolation** from
  `N = 10³` to `1.4 × 10⁴`.
- Material under `data/*/9_obsolete*` does not exist here at all: discrete-time and MCMC
  dynamics, and an invalidated float32 run, are excluded from this repository and must
  never be cited.

## Requirements

Python ≥ 3.10, NumPy, SciPy, Matplotlib. Two optional extras:

- **`mlx`** — Apple-silicon GPU acceleration, used by `couplings.py` and the batched
  integrators. Everything has a NumPy float64 fallback; the `N**` campaigns are float64 CPU
  only, by design.
- **LaTeX** — only to rebuild the manuscript, which is not in this repository.

## License

Code is MIT. Data and figures are CC BY 4.0. See `LICENSE` and `LICENSE-DATA`.

## Citation

See `CITATION.cff`. The manuscript is in preparation; this repository will be tagged at
submission.
