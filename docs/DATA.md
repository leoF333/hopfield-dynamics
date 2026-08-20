# Data: what is here, what is not, and why

The full research archive is about **21 GB**. This repository ships **~116 MB**: every file
that a manuscript claim actually rests on, plus the campaign reports and the figures. The
rest is raw integrator output that can be regenerated, and it is listed with its size and
path in `data/MANIFEST_excluded.csv`.

## Inclusion rule

A data file is in this repository if:

1. it is **≤ 2 MB**, or
2. it is **explicitly cited** by a claim in `docs/CLAIMS.md` and ≤ 30 MB;

and it is **not** per-chunk raw integrator output (any path containing `chunks/`).

That rule turned out to be generous rather than restrictive, because the claim-backing data
is small. The central scaling campaign E30 — 303 disorder/size runs, the raw material behind
the `N^-b` narrowing of the threshold law — is **2.3 MB** in total. The complete per-pattern
threshold tables for every size and seed are **184 KB**. What is heavy is trajectory storage,
and no claim reads a raw trajectory directly.

Two files were kept above the 2 MB line because a headline claim reads them:

| File | Size | Backs |
|---|---|---|
| `data/4_attractors_basins_chaos/data/E23A_geometry.npz` | 24.5 MB | broadband power spectrum, return map, PCA projection of the attractor (C-46, C-47, C-49) |
| `data/2_recall_cycle_snic/data/E29_cycle_ghosts.npz` | 2.7 MB | exact binary recall, ghost dwell time at the critical bond (C-42, C-43, C-44) |

## Layout

Theme folders mirror the local archive, with the French names translated:

| Repository | Local archive | Subject |
|---|---|---|
| `data/1_static_bifurcation/` | `3_numerics/results/1_bifurcation_statique/` | memory folds, `λ_c(α)`, `τ`-independence, Hopf margin |
| `data/2_recall_cycle_snic/` | `…/2_cycle_rappel_snic/` | pacemaker law, Floquet, the terminal bifurcation, the invariant circle (E34) |
| `data/3_threshold_scaling/` | `…/3_unification_seuils_scaling/` | **the central result**: per-pattern thresholds, the threshold law, finite-size scaling |
| `data/4_attractors_basins_chaos/` | `…/4_attracteurs_bassins_chaos/` | basins, Lyapunov spectra, `D_KY`, stationarity |
| `data/5_short_delay/` | `…/5_petit_tau/` | the short-delay desynchronization crossover |
| `data/6_correlated_patterns/` | `…/6_motifs_correles/` | Markov and Gaussian-process-circle pattern ensembles |
| `data/7_implicit_delay/` | `…/7_tau_implicite/` | the layered-chain effective delay |
| `data/v5_campaigns/` | `4_campaigns/numerical_workplan/` | reports, manifests and run summaries of N1–N10 |

## The files that matter most

If you only look at a handful:

| File | What it is |
|---|---|
| `data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz` | The 100 quenched thresholds of the reference realization. min 0.2195 (pattern 28), median 0.2742, max 0.3276 (pattern 91). |
| `data/3_threshold_scaling/data/E30_scaling/` | 303 runs, `N = 10³` to `1.4 × 10⁴`. The finite-size narrowing of the threshold law. |
| `data/v5_campaigns/runs/N3/E30_seed_aware_analysis.json` | The seed-aware reanalysis of those 303 files: `b = 0.43078`, 95% CI `[0.39638, 0.46520]`. |
| `data/v5_campaigns/reports/N1_static_extreme_vs_dynamic_onset.csv` | 20 realizations: static extreme vs independently measured dynamic onset. The core multi-seed evidence. |
| `data/v5_campaigns/reports/N6_multiseed_lyapunov.csv` | Five realizations, three positive Lyapunov exponents each: the hyperchaos result. |
| `data/2_recall_cycle_snic/data/E34/E34_cycle_uniqueness_MERGED_lam0.330000.json` | 100 stored-pattern starts converging to one orbit: mean period 1869.406 `t₀`, max waveform distance 5.26 × 10⁻³. |
| `data/2_recall_cycle_snic/data/E9_bond_times.npz` | Per-bond relay times near onset: only bond 91 diverges. |

## Excluded material

`data/MANIFEST_excluded.csv` lists every archived file that is not shipped, with its path in
the archive, its size and the reason. There are about 4 800 rows totalling ~12.8 GB, almost
all of it:

- per-chunk integrator output under `chunks/` directories (E36 campaigns),
- raw `N2` / `N5A` / `N6` / `N1` trajectory `.npz` files from the V5 campaigns,
- a few large stored trajectories such as `cycle_lam0.9_tau10.0_N2000.npz` (167 MB).

The `sha256` column is empty by default because hashing 12.8 GB over a network mount is slow.
Run `tools/hash_archive.py` against your local copy of the archive to fill it in; the script
is resumable.

**Nothing excluded is load-bearing.** Every excluded file is either regenerable from the
shipped code and parameters, or a redundant intermediate of a summary that *is* shipped.

## Material excluded on scientific grounds

Not in this repository at any size, and never to be cited:

- **Discrete-time and MCMC dynamics.** Superseded by the continuous-time study.
- **The invalidated float32 `N = 10⁴` run.** Precision artefact.
- **`N2_INVALID_STATIC_BRANCH_20260729`.** A Newton continuation started exactly at the
  fold, which silently followed the unstable saddle branch. Archived locally for audit, but
  physically invalid.

These live in the local archive under `obsolete/05_invalid_or_refuted/` and
`4_campaigns/numerical_workplan/runs/`.

## Precision and backends

- All fold continuations, characteristic roots and Lyapunov estimates in the **N** campaigns
  are **float64 CPU**. MLX/Metal was deliberately not used for production there.
- Some **E** campaigns used the MLX float32 GPU path for batched trajectory work, behind an
  explicit float32-vs-float64 validation gate. Where this matters to a claim, `docs/CLAIMS.md`
  says so — notably **E33** (the no-aging / stationarity result), which used
  `E33_aging_mlx*.npz`. The planned float64 control was campaign N9, withdrawn from scope on
  2026-07-30.
