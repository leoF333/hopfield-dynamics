# Finite is different: sequential recall in delayed Hopfield networks

[![tests](https://github.com/leoF333/hopfield-dynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/leoF333/hopfield-dynamics/actions/workflows/tests.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](LICENSE-DATA)

Research code and data from my internship at the University of Chicago (James Franck Institute, March–August 2026), supervised by **Vincenzo Vitelli** and **Doruk Efe Gökmen**.

**[Project overview](https://leof333.github.io/research/hopfield.html)** · **[Internship report (PDF)](https://leof333.github.io/documents/hopfield-report.pdf)** · **[Numerical methods](docs/NUMERICAL_METHODS.md)** · **[Claims and evidence](docs/CLAIMS.md)**

---

## The question

A Hopfield network stores memories as stable fixed points. Add a delayed, non-reciprocal coupling that maps each pattern onto its successor, and the same network can replay its memories in order. In the thermodynamic limit, the switch from static to sequential recall is a single phase transition.

**What happens in a finite network, where every memory feels its own quenched disorder?**

## The answer, in one figure

![Distribution of memory-resolved saddle-node thresholds in one network](figures/quickstart/thresholds.png)

*Each of the P = 100 memories of one network (N = 2000) loses stability at its own saddle-node threshold λ_c(μ). Replotted directly from the archived data by `examples/quickstart.py`.*

In a finite network, the single transition splits in two:

- **Static recall ends in the bulk.** Each memory branch terminates at its own saddle-node. Within one sample, the thresholds form a near-Gaussian, nearly independent ensemble. Its centre is set by the load α, and its width shrinks with size as N^(−0.43) (95% CI [0.40, 0.47]).
- **Sequential recall starts at the extreme.** A travelling wave must pass every memory, so it stays pinned until the *last* barrier disappears. In 19 of 19 admissible disorder realizations, the largest static threshold matches the dynamical onset to within 10⁻³. The terminal bifurcation is a saddle-node on an invariant circle (SNIC).
- **Between the two lies a finite-size phase.** In the reference realizations, the window hosts a stationary hyperchaotic attractor (three positive Lyapunov exponents, Kaplan–Yorke dimension ≈ 6.4). As N → ∞ the window closes and the two transitions merge.

## The model

N rate neurons obey a delay-differential equation:

$$
t_0\,\dot{u}(t) = -u(t) + (1-\lambda)\,J\,\tanh[\beta u(t)] + \lambda\,K\,\tanh[\beta u(t-\tau)]
$$

The two coupling matrices are built from P = αN random binary patterns:

- J = (1/N) Σ ξ^μ (ξ^μ)ᵀ is the symmetric Hebbian term. It stores the patterns.
- K = (1/N) Σ ξ^(μ+1) (ξ^μ)ᵀ is the non-reciprocal term. It maps each pattern onto its successor.

The parameter λ interpolates between static storage (λ = 0) and delayed sequence drive (λ = 1).

**Key numerical idea.** The dynamics reduces *exactly* from N neuron coordinates to P pattern coordinates. This reduction keeps the full disorder of the sample instead of averaging it away. Sylvester's determinant identity then brings the transcendental N × N stability problem down to P × P. With this reduction, all 100 memory branches of a 2000-neuron network can be followed individually, with continuation, delay integration and spectral analysis.

## What I built

| Component | Methods |
|---|---|
| Exact low-rank reduction | Pattern-subspace projection, invariant transverse dynamics, checked against dense matrices to 10⁻¹⁵ |
| Branch-resolved continuation | Newton continuation from every stored pattern, fold detection by smallest singular value, bisection to ±2×10⁻⁴ |
| Delay-differential integration | Method of steps with RK4 and Hermite interpolation of the history, checkpoint/restart |
| Stability of the delayed system | Characteristic roots of the reduced P × P problem, Floquet multipliers of the recall cycle, Lyapunov spectra |
| Finite-size statistics | Scaling over N = 10³–1.4×10⁴, bootstrap over whole disorder realizations, Gaussian extreme-value comparison |
| Controlled variations | Short delays, correlated pattern sequences, a layered architecture where depth replaces the explicit delay |

## Quick start

Requires Python 3.10 or later. No GPU needed. Tested with Python 3.12.

```bash
git clone https://github.com/leoF333/hopfield-dynamics.git
cd hopfield-dynamics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python examples/quickstart.py            # ~1 min: replots the thresholds, simulates a small network
python -m unittest discover -s tests -v  # fast numerical checks
```

The quick start writes to `outputs/quickstart/`:

- the threshold figure above, regenerated from the archived data;
- `sequential-recall.png`, a fresh simulation of a small network (N = 300, P = 9) replaying its memories in order;
- the raw overlaps and parameters (`recall.npz`, `summary.json`).

Useful options: `--data-only` skips the simulation, and `--output <dir>` changes the destination. The small demo illustrates the dynamics. The N = 2000 reference results come from the campaigns in `code/campaigns/`.

![Sequential recall in a small network](figures/quickstart/sequential-recall.png)

## Tests

The test suite checks the parts of the pipeline that everything else depends on:

- the reduced right-hand side against dense Hebbian matrices built independently;
- the analytical Jacobian against finite differences;
- full versus reduced trajectories over a delay interval;
- checkpoint/restart consistency, including the stored history;
- convergence under time-step refinement;
- the archived threshold data against the documented reference values.

## Repository layout

```
examples/          Quick start
tests/             Numerical regression tests
code/core/         Reduced model, continuation, spectra, integrators
code/experiments/  Exploratory experiments
code/campaigns/    Production runs behind the reported results
code/figures/      Figure generation
data/              Derived data and campaign summaries
figures/           Research and quick-start figures
docs/              Methods, claim-by-claim evidence, open questions
tools/             Archive and integrity utilities
```

This repository is a curated public release of the internship codebase. Some exploratory scripts still assume the original archive layout. [docs/VERIFIED_ENTRY_POINTS.md](docs/VERIFIED_ENTRY_POINTS.md) lists what runs as-is from a fresh clone.

## Scope

The threshold law is measured numerically, not derived analytically. The invariant circle is constructed explicitly for the last two stages of the reference realization. Chaos and stationarity are finite-time observations. Open questions and known discrepancies are tracked in [docs/OPEN_ITEMS.md](docs/OPEN_ITEMS.md).

## Related work

- J. J. Hopfield, *PNAS* 79, 2554 (1982); *PNAS* 81, 3088 (1984).
- H. Sompolinsky & I. Kanter, *Phys. Rev. Lett.* 57, 2861 (1986).
- D. Kleinfeld, *PNAS* 83, 9469 (1986).

## Citation and licence

If you use this code or data, please cite the repository ([CITATION.cff](CITATION.cff)).

Code: [MIT](LICENSE). Data and figures: [CC BY 4.0](LICENSE-DATA).

---

**Leo Flack** · University of Cambridge / École Polytechnique · [leof333.github.io](https://leof333.github.io) · [lhf31@cam.ac.uk](mailto:lhf31@cam.ac.uk)
