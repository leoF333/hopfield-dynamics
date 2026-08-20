# Numerical production runbook

All commands below are run from this directory.  They write only below `runs/`,
`figures/`, `reports/`, `logs/`, and `manifests/`.

The historical folder on the Desktop is read-only input.  Its key source hashes are
recorded in `manifests/reference_baseline.json` and checked again by N10.

## Common preflight

```bash
./run.sh smoke
```

On a normal macOS Terminal, MLX may be checked with:

```bash
./run.sh mlx-preflight
```

MLX is used only for compatible repeated pattern projections.  Continuation,
characteristic roots and Lyapunov estimates remain float64.

## N1 — static extreme and independent dynamic onset

One static seed:

```bash
./run.sh N1-static --N 2000 --P 100 --seed 42 --nproc 4
```

One dynamic seed:

```bash
./run.sh N1-dynamic --N 2000 --P 100 --seed 42 --dt 0.01 \
  --lambda-start 0.40 --lambda-stop 0.25 --lambda-step 0.002 \
  --bracket-tol 0.0005 --max-time 20000 --required-tours 5
```

The dynamic driver checkpoints after every completed descent point.  Static
thresholds are never read by the dynamic driver.  Raw relay events, dwell times and
both independent-history controls are stored with the final bracket.

After all seeds:

```bash
conda run -n mcmc_env python src/n1_compare.py
```

Gate: at least 15 admissible seeds, at least 90% overlapping brackets, and motif
agreement except when the first–second static gap is below the combined resolution.

## N2 — two-sided local laws

N2 requires completed N1 static and dynamic files.  The production seed list is seed
42 plus four admissible seeds spanning the first–second extreme-gap distribution.

```bash
./run.sh N2 --stage smoke --seeds 42
./run.sh N2 --stage production --confirm-heavy --seeds 42 48 47 55 44
```

The latter four are the current static gap-spanning candidates; any seed that fails
the N1 dynamic admissibility gate must be replaced by the nearest admissible gap
quantile before production.

The script performs low-rank pseudo-arclength continuation through the selected fold,
computes the physical DDE roots at `M=48` (plus `M=64` controls), repeats the smallest dynamic offsets at
`dt=0.005`, censors unfinished passage times, fits the three dynamic alternatives,
and saves the seed-42 null/stable projections for the normal-form diagnostic.

## N3 — archived E30 reanalysis

```bash
./run.sh N3 --stage production --confirm-heavy
```

This reads the 303 archived E30 files without changing them.  The bootstrap unit is a
disorder seed, never a motif.

## N4 — central publication figures

```bash
./run.sh N4 --stage production --confirm-heavy
```

Figure 3e remains visibly blank until the N1 completion gate passes.

## N5A — short-delay coherence and cycle–chaos boundary

```bash
./run.sh N5A --stage smoke --skip-lyapunov
./run.sh N5A --stage production --confirm-heavy
```

The coherence sweep saves all direct observables.  The boundary sweep descends in
lambda at each delay and computes float64 maximal exponents at the last two moving
and first two post-boundary points.

## N5B — long-delay roots and nonlinear probes

```bash
./run.sh N5B --stage smoke --N 200
./run.sh N5B --stage production --confirm-heavy
```

A Hopf label is emitted only when the `M=48` and `M=64` roots agree, the scaled
singular-value residual is below `1e-10`, a transverse crossing is located, and
eigenmode-aligned nonlinear probes reproduce the early growth rate and frequency.
Root families are tracked continuously, every spectral point is restartable, and
the pinned-front control covers all 20 N1 disorder seeds.

## N6 — partial Lyapunov spectrum

```bash
./run.sh N6 --stage smoke --N 120 --P 6 --seeds 42
./run.sh N6 --stage production --confirm-heavy
```

One `T=4000` run stores cumulative estimates at `T=1000,2000,4000`; it does not repeat
the same trajectory three times.  Seed 42 is repeated at half step for all lambda
values, and every other seed has a half-step control at `lambda=0.31`.

## N7 — correlated patterns

```bash
./run.sh N7 --stage smoke --N 120 --P 6 --seeds 42 --nproc 1
./run.sh N7 --stage production --confirm-heavy
```

Markov and Gaussian-circle patterns use the archived generators exactly.  “No
pinning” is reported only as a lower tested lambda bound.

## N8 — high-load moving states

```bash
./run.sh N8 --stage smoke --N 120 --seeds 42 --skip-lyapunov
./run.sh N8 --stage production --confirm-heavy
```

Moving states are relabelled only after the maximal exponent is available.  A
four-exponent spectrum is added for representative seed-42 cells.

## N9 — float64 stationarity and operational ergodicity

The two non-reference lambda values are intentionally not guessed.  They must be
selected from completed N8 cells with a sufficient chaotic ensemble:

```bash
./run.sh N9 --stage production --confirm-heavy \
  --lambda-a008 VALUE_FROM_N8 --lambda-a010 VALUE_FROM_N8
```

Each DDE restart carries the complete position and derivative history. Candidate
histories are generated in bounded float64 batches until at least 32 wandering,
non-periodic histories are retained per seed. Early/late KS distances and
time-versus-ensemble KS distances are saved separately.

## N10 — release audit

```bash
./run.sh N10 --stage smoke
./run.sh N10 --stage production --confirm-heavy
```

The production audit repeats full/reduced trajectories, three thresholds, one
dynamic point, one Floquet point and one Lyapunov point, checks forbidden sources,
rechecks the historical source hashes, and writes `manifests/figure_provenance.json`.
