# Reproducing the results

> **Historical research runbook.** The commands below preserve the original campaigns and may rely on the original archive layout. For the portable, locally tested public workflow, start with [VERIFIED_ENTRY_POINTS.md](VERIFIED_ENTRY_POINTS.md). The full campaign list has not been rerun for this portfolio release.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10, NumPy, SciPy, Matplotlib. Optional: `mlx` on Apple silicon for the batched
GPU integrators. Every code path has a NumPy float64 fallback, and the **N** campaigns are
float64 CPU only by design.

Scripts resolve their own imports (see `docs/CODE_MAP.md`), so you can run them from
anywhere:

```bash
python code/experiments/e24_thresholds.py --help
```

## Reference parameters

```
N = 2000    P = 100    α = 0.05    β = 20    τ = 10    seed 42
```

Time in units of `t₀`. Change them with the CLI flags exposed by `code/core/config.py`.

## The four load-bearing computations

### 1. The quenched threshold ensemble of one sample

The central object: the `P` saddle-node thresholds that coexist inside a single disorder
realization.

```bash
python code/experiments/e24_thresholds.py \
    --N 2000 --P 100 --seed 42 --store-curves --nproc 6 \
    --out data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz
```

Expected, for seed 42: `min = 0.2195` (pattern 28), `median = 0.2742`,
`max = 0.3276` (pattern 91), each bracketed to `±2 × 10⁻⁴`.

Runtime is dominated by warm-started Newton continuation over 100 branches; `--nproc`
parallelizes over patterns. Cross-checks of the branch geometry:

```bash
python code/experiments/e24a_geometry.py     # tilt onset, universal collapse
python code/experiments/e25_deathshape.py    # terminal tilt ratio vs threshold and bond overlap
python code/experiments/e26_analysis.py      # the law vs load, KS collapse
```

### 2. Finite-size scaling of the threshold law

The shipped E30 arrays (303 runs, 2.3 MB) reproduce the scaling directly:

```bash
python code/experiments/e30_analysis.py
```

For the **seed-aware** uncertainties used in the manuscript — the bootstrap unit is a
disorder seed, never a pattern — use the V5 reanalysis instead:

```bash
python code/campaigns/n3_reanalyse.py
```

Expected: `σ ∝ N^-b` with `b = 0.43078`, bootstrap median `0.43009`, 95% CI
`[0.39638, 0.46520]` at `α = 0.05`. Note that at fixed `P = 100` the exponent is instead
`0.903 [0.886, 0.921]`: the two thermodynamic protocols are different and must not be
conflated.

To regenerate the raw campaign (long):

```bash
python code/experiments/e30_scaling_overnight.py     # resumable, 303 runs
```

### 3. Static extreme vs dynamic onset, seed by seed

The most direct multi-seed test of the mechanism. Static thresholds and the dynamic onset
are computed by **independent** drivers; the dynamic driver never reads a static threshold.

```bash
# one seed, static side
python code/campaigns/run_workplan.py N1-static --N 2000 --P 100 --seed 42 --nproc 4

# one seed, dynamic side (descending scan, checkpointed after every point)
python code/campaigns/run_workplan.py N1-dynamic --N 2000 --P 100 --seed 42 --dt 0.01 \
    --lambda-start 0.40 --lambda-stop 0.25 --lambda-step 0.002 \
    --bracket-tol 0.0005 --max-time 20000 --required-tours 5

# after all seeds
python code/campaigns/n1_compare.py
```

Acceptance gate, as run: at least 15 admissible seeds, at least 90 % overlapping brackets,
and pattern agreement except where the first–second static gap falls below the combined
resolution. Result: pattern identity **20/20**; **19/19** admissible seeds agree at `10⁻³`,
largest retained discrepancy `7.3 × 10⁻⁴`.

Seed 48 is excluded because its first and second static-extreme brackets overlap, so the
unique extreme the test requires is not identifiable — not because it is an outlier. Its
data and the exclusion reason are in the reports.

### 4. The hyperchaotic window

```bash
python code/experiments/e23a_chaos_nature.py                 # reference spectrum, D_KY ≈ 6.4
python code/campaigns/n6_lyapunov_campaign.py --stage production --confirm-heavy
```

Expected from N6 at `λ = 0.31`: the three leading exponents stay positive for all five seeds
42–46, at both `dt = 0.01` and the `dt = 0.005` control; the smallest third exponent over all
controls is `0.01145`.

## The recall cycle

```bash
python code/experiments/pacemaker_scan.py --N 2000 --tau 10   # cycle + Floquet per λ
python code/experiments/e14_contraction_near_death.py         # floor-free contraction per tour
python code/experiments/e29_cycle_vs_ghosts.py                # exact binary recall, ghost dwell
python code/experiments/e34_necklace_wu_n2000.py              # the invariant circle
```

**A Floquet caution.** The `rez` field of `floquet_scan_N2000_tau10.0.npz` is `ln|μ*|/T`
computed from a *floored* `|μ*|`. It is an upper bound on `Re z₁`, not a measurement. The
`N = 2000` Arnoldi scan points sit at the relative floor `2 × 10⁻¹⁶` and are lower bounds
only. The measured contraction values quoted in the manuscript
(`A = 219.7 / 419.5 / 440.1` at `λ = 0.400 / 0.335 / 0.329`) come from the floor-free
Benettin–QR computation in `e14_contraction_near_death.py`.

## Ablations

```bash
python code/experiments/e22_desync.py                  # short-delay crossover, τ_c ≈ 1.31
python code/campaigns/n5a_delay_scan.py                # multi-seed, τ_c ∈ [1.233, 1.338]
python code/experiments/e27_markov_thresholds.py       # correlated patterns, the seam barrier
python code/experiments/e28_gpcircle_thresholds.py     # smooth periodic patterns, no pinning
python code/experiments/e36a2_delay.py                 # implicit delay from layered depth
```

## Figures

```bash
python code/figures/build_panels.py          # composite panels from components
python code/figures/v2/p_chaos.py            # revised chaos panel
python code/figures/v2/compose05.py          # revised SNIC panel
```

`figures/panel_manifest.json` records the source SHA-256 and crop coordinates of every
component, so a panel can be audited without rebuilding it.

Three manuscript figures — `Figure04`, `Figure06`, `Figure08` — are re-layouts whose final
assembly script is not in this repository. Their component provenance is complete; see
`docs/FIGURE_PROVENANCE.md` and `docs/OPEN_ITEMS.md` §1.

## Verifying what you have

```bash
python tools/verify_manifest.py      # checks the shipped tree against data/MANIFEST_included.csv
python tools/hash_archive.py <path>  # fills the sha256 column of MANIFEST_excluded.csv (resumable)
```

## Runtimes, roughly

On a laptop-class CPU, single process unless noted:

| Computation | Order of magnitude |
|---|---|
| One pattern threshold, `N = 2000` | seconds |
| Full 100-pattern threshold table, `N = 2000` | tens of minutes (`--nproc 6`) |
| E30 scaling campaign, 303 runs | overnight, resumable |
| One Floquet spectrum at `N = 2000` | minutes |
| N1 dynamic descent, one seed | hours, checkpointed after every point |
| N6 Lyapunov, `T = 4000`, one seed | hours |

Every long campaign checkpoints. Completed points are never recomputed after an
interruption; see `data/v5_campaigns/RUN_STATUS.md` for the authoritative resume state of
each V5 campaign.
