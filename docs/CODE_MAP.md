# Code map

What each source file does. Generated from the module docstrings; the authoritative narrative for the `E**` scripts is §11 of `docs/worklogs/CYCLE_worklog.md`.

**220 files**: `code/_scratch/` 19, `code/campaigns/` 41, `code/core/` 44, `code/experiments/` 109, `code/figures/` 7.

## How imports resolve

Every file carries a short path bootstrap inserted at packaging time:

```python
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
```

So `code/core/` and `code/experiments/` are always importable, and a script runs from anywhere without installing a package. Nothing else about the historical code was changed, on purpose: the scripts that produced the published numbers are the scripts shipped here.

The dependency structure is shallow. The most-imported modules are `couplings` (61 importers), `robust_branch` (51), `cycle_reduced` (47), `reduced_spectrum` (25) and `config` (15).

---

## `code/core/` — the reusable library

### Model, couplings and patterns

| File | Role |
|---|---|
| `couplings.py` | Pattern generation and matrix-free coupling operators. |
| `config.py` | Configuration and parameter management for the delayed mixed Hopfield analysis. |
| `simulate_dde.py` | Nonlinear DDE integrator with ring-buffer history. |
| `layered_chain.py` | Pure-NumPy implementation of the E36 layered Hopfield chain. |
| `mhn_reduced.py` | Low-rank Modern-Hopfield fields for the E35 delayed model. |
| `discrete_bridge.py` | Bridge to the discrete synchronous map (§3.8 — optional). |

### Exact P-dimensional reduction and integrators

| File | Role |
|---|---|
| `cycle_reduced.py` | Exact P-dimensional reduction of the delayed mixed Hopfield DDE + validations. |
| `cycle_reduced_batch.py` | Batched float64 integrator for the exact P-dim reduced delayed DDE. |
| `cycle_reduced_mlx.py` | MLX float32 GPU batched integrator for the reduced delayed DDE (Apple Metal). |

### Fixed points and branch continuation

| File | Role |
|---|---|
| `fixed_point.py` | JFNK fixed-point solver + pseudo-arclength continuation in lambda. |
| `robust_branch.py` | Robust memory-branch tracer via the Woodbury exact solve. |
| `multiseed_fold.py` | Multi-seed static-fold study via the robust Woodbury tracer (float64, exact). |
| `find_critical_lambdas.py` | — |
| `lambda_c_grid.py` | Critical-lambda grid  lambda_c(alpha, tau)  for the static memory destabilization. |
| `lambda_c_grid_v2.py` | lambda_c(alpha, tau) grid v2 — corrected machinery, artefact-free bifurcation analysis. |
| `run_N10000.py` | Static fixed-point bifurcation at N=10000, averaged over several seeds. |
| `run_grid.py` | — |
| `run_analysis.py` | Full analysis run at N=10000 (or --small for N=2000). |
| `stitch.py` | — |
| `summary_lambdac_alpha.py` | Summary figure: lambda_c(alpha) from the multi-seed N=10000 fold study. |
| `finite_size_scaling.py` | Finite-size scaling of the memory-branch destabilization. |

### Linear stability: instantaneous and delayed spectra

| File | Role |
|---|---|
| `reduced_spectrum.py` | Exact reduced characteristic spectrum of the DDE (no pseudospectral artefacts). |
| `dde_stability.py` | Pseudospectral infinitesimal-generator (IG) method for the DDE stability analysis. |
| `jacobian_spectrum.py` | — |
| `hopf_locus.py` | Sweep over tau values and trace the Hopf locus lambda_c(tau), omega_c(tau). |
| `hopf_crossover_refine.py` | Refined delay-induced Hopf analysis near the fold (alpha=0.07, large tau). |
| `analytic_check.py` | Condensed-subspace analytic estimate of the Hopf locus (§3.7). |
| `audit_static_analysis.py` | Comprehensive numerical audit of the static-bifurcation analysis chain. |
| `run_validation.py` | Quick validation run at small N=2000. |

### The recall cycle and its Floquet spectrum

| File | Role |
|---|---|
| `pacemaker_cycle.py` | Pacemaker cycle construction from the EXACT P-dim reduction (CYCLE_worklog §5.1). |
| `pacemaker_scan.py` | Descending-lambda Floquet scan of the pacemaker cycle (CYCLE_worklog §5.1). |
| `floquet.py` | Floquet multipliers of the pacemaker periodic orbit. |
| `floquet_monodromy.py` | Floquet analysis of the pacemaker cycle via the EXACT P-dim reduction. |

### Plotting and reporting

| File | Role |
|---|---|
| `plots.py` | Figure-generating functions for the delayed mixed Hopfield analysis. |
| `cycle_figures.py` | Presentation-quality figures for the cycle study (E1-E10 measured results). |
| `fig2_physical_roots.py` | Figure 2 (redone): the 10 rightmost PHYSICAL characteristic roots in the complex |
| `figs_wave_phasemap_v2.py` | Two presentation figures (English, dpi 170): |
| `regen_figs_english.py` | Regenerate figE_v2, figF, figG with ENGLISH titles/labels (project convention) |
| `plot_hopf_margin.py` | Hopf margin vs tau: distance of the rightmost COMPLEX root to the imaginary axis, |
| `plot_critical_lambda_vs_tau.py` | — |
| `compare_boundaries.py` | — |
| `phase_diagrams_mcmc.py` | — |
| `build_report_html.py` | Render REPORT_static_bifurcation.md to a self-contained HTML with typeset math. |
| `job_monitor.py` | Read-only health monitor for long E34/E35/E36 jobs. |

---

## `code/experiments/` — historical campaigns E1–E37

One script per worklog entry. The verdicts, dates and protocols live in `docs/worklogs/CYCLE_worklog.md` §6; the claims they support are listed in `docs/CLAIMS.md`.

### E12, E15, E17 — pinned-front spectra and the Hopf window

| File | Role |
|---|---|
| `e15_pinned_hopf_margin.py` | E15: delayed spectrum of the PINNED FRONT vs tau — does its Hopf margin erode |
| `e17_hopf_birth.py` | E17: birth of the slow oscillation of the MEMORY state just past lambda_Hopf |
| `e17b_identify_final_state.py` | E17b: identify the final state of the E17 test run (lam=0.20167, alpha=0.07, |
| `e17c_fig_refit.py` | E17c: honest re-analysis of the E17 runs from the saved npz + E17b findings, |

### E14 — floor-free contraction per tour

| File | Role |
|---|---|
| `e14_contraction_near_death.py` | E14: total contraction per period A(lambda) = -ln|mu_1| NEAR THE CYCLE DEATH, |

### E18 — birth of pinned fronts, coexistence, basins

| File | Role |
|---|---|
| `e18_figures.py` | E18 figures (presentation quality, English). |
| `e18_pilot.py` | E18 pilot: reconstruct the bond-91 pinned front at lambda=0.30, identify active |
| `e18_pilot2.py` | E18 pilot 2: reconstruct the bond-91 pinned front DIRECTLY as a u-space ansatz |
| `e18_pilot3.py` | E18 pilot 3: reproduce E15's exact build of the bond-91 pinned front at lam=0.318, |
| `e18_pilot4.py` | E18 pilot 4: locate a genuine two-consecutive-pattern pinned front. |
| `e18_pilot5.py` | E18 pilot 5: catch the pinned front during the quench settle. |
| `e18_pilot6.py` | E18 pilot 6: diagnose the a (raw reduced overlap) vs m (condensed) mismatch at the |
| `e18a_front_branch.py` | E18a - Downward branch continuation of the bond-91 pinned front. |
| `e18b_delayed_stability.py` | E18b - Delayed stability of the pinned-front branch at tau=10. |
| `e18c_coexistence.py` | E18c - Coexistence check: memory xi^1 AND the pinned front, below lambda_c. |
| `e18d_basins.py` | E18d - Basin-size scan of the pinned front vs lambda (reduced DDE, tau=10). |
| `e18d_dtcheck.py` | E18d dt-robustness: does dt=0.05 classify a borderline escape the same as |
| `e18d_pilot.py` | E18d pilot: time one directional-escape bisection and a handful of random-IC |

### E19 — small-delay regime: how the cycle dies

| File | Role |
|---|---|
| `e19_pilot.py` | E19 pilot: small-delay cycle. Establish that the machinery works at small tau |
| `e19a_cycle_clock.py` | E19a - Small-delay cycle existence and clock. |
| `e19b_death.py` | E19b - Death of the small-delay cycle under decreasing lambda (seeded hysteresis). |
| `e19c_slowing.py` | E19c - Critical-slowing law of the passage time at small tau. |
| `e19d_fixedpoint.py` | E19d - Fixed-point cross-check at small tau. |

### E20 — basin competition

| File | Role |
|---|---|
| `e20_common.py` | E20 common machinery: reduced-DDE basin-classification for the delayed mixed |
| `e20_figures.py` | E20 figures: merge the main basin-fraction sweep (lam<=0.325) with the high-lam |
| `e20_inspect.py` | Inspect settled states: pinned front vs pure memory overlap structure. |
| `e20_pilot.py` | E20 PILOT: validate the classifier on KNOWN initial conditions (front / memory / |
| `e20a_basin_fractions.py` | E20a -- Monte-Carlo basin fractions vs lambda (headline). |
| `e20a_hi.py` | E20a (completion) -- run the HIGH-lambda points that the main sweep missed after |
| `e20b_metastability.py` | E20b -- metastability quantification (front vs memory). |

### E21 — nature and position of front birth vs tau

| File | Role |
|---|---|
| `e21_figures.py` | E21 figures (presentation quality, English). |
| `e21a2_dissolution.py` | E21a-bis - Pin down the DISSOLUTION mechanism of the pinned front at its birth. |
| `e21a_front_birth.py` | E21a - Properly characterize the BIRTH of the pinned front (bottom of the branch). |
| `e21b_tau_independence.py` | E21b - tau-independence of the front's POSITION and z=0 structure (machine precision). |
| `e21c_tau_nature.py` | E21c - tau-dependence of the front's NATURE / stability boundary (medium & large tau). |
| `e21d_memory_fold_check.py` | E21d -- HONEST re-verification of the static memory-fold analysis (user request). |
| `e21e_figS_full.py` | E21e -- figS rev. 2: extend the left panel of figS_front_birth.png to the full |

### E22 — small-delay desynchronization

| File | Role |
|---|---|
| `e22_desync.py` | E22 - Desynchronization of the sequential-recall cycle at small delay. |

### E23 — nature of the chaos, cycle-chaos frontier

| File | Role |
|---|---|
| `e23a_chaos_nature.py` | E23 PART A - Nature of the chaos in the mixture phase lam_c < lam < lam* (tau=10). |
| `e23b_frontier.py` | E23 PART B - The SNIC/chaos paradox at small tau (PI Q6). |

### E24 — per-pattern thresholds lambda_c(mu)  [central]

| File | Role |
|---|---|
| `e24_analysis.py` | E24b/c/d -- finite-size scaling and distribution of the per-pattern thresholds |
| `e24_thresholds.py` | E24 -- per-pattern saddle-node thresholds lambda_c(mu) of the memory->front branches. |
| `e24a_geometry.py` | E24a -- shape evolution of the memory->front branches (all P=100, N=2000, seed 42). |
| `e24e_spotchecks.py` | E24e -- delayed-stability (T_P, tau=10) spot-checks along 5 representative |

### E25 — shape at the fold vs pattern

| File | Role |
|---|---|
| `e25_deathshape.py` | E25 -- is the death shape (a_mu : a_mu+1 at the fold) universal, or does it depend |

### E26 — threshold law vs load

| File | Role |
|---|---|
| `e26_analysis.py` | E26 -- is the threshold law a FIXED distribution independent of P, from which the |

### E27, E28 — correlated 'video' patterns

| File | Role |
|---|---|
| `e27_markov_thresholds.py` | E27 -- correlated patterns I: MARKOV FLIP CHAIN (user task 2, option 1). |
| `e28_gpcircle_thresholds.py` | E28 -- correlated patterns II: CIRCULAR GAUSSIAN LATENT PROCESS (user task 2, option 2). |

### E29 — what the cycle passes through above lambda*

| File | Role |
|---|---|
| `e29_cycle_vs_ghosts.py` | E29 -- what does the limit cycle pass THROUGH just above lambda*? |
| `e29_fig.py` | E29 figure regeneration from E29_cycle_ghosts.npz (corrected titles) + |

### E30 — threshold scaling campaign

| File | Role |
|---|---|
| `e30_analysis.py` | E30 analysis (task 1): finite-size scaling of the per-pattern threshold law |
| `e30_scaling_overnight.py` | E30 -- overnight scaling campaign for the per-pattern threshold law lambda_c(mu). |

### E31, E32, E33 — survival, basin phase diagram, aging

| File | Role |
|---|---|
| `e31_fixedpoint_census.py` | E31 (large-alpha task, Exp C) -- per-pattern memory SURVIVAL census. |
| `e32_alpha_lambda_basins.py` | E32 (large-alpha task, Exp A) -- (alpha, lambda) basin phase diagram. |
| `e32_analysis.py` | E32 analysis -- (alpha, lambda) basin phase diagram from E32_basins_*.npz. |
| `e33_aging.py` | E33 (aging task, Exp D + two-time correlation) -- does the mixture-phase chaos AGE, |
| `e33_analysis.py` | E33 aging analysis: two-time correlation collapse test + quench relaxation. |
| `e33_vs_alpha.py` | E33 Exp E2 summary: two-time correlation collapse spread vs alpha -> does the |

### E34 — the invariant recall circle

| File | Role |
|---|---|
| `e34_ET0_figures.py` | Four-panel summary figure for gate E34-ET0 (edge-tracking, N=400 pilot). |
| `e34_ET0prime_figures.py` | ET0' amendment figure (mu=2 lower-lambda edge-tracking + upward continuation). |
| `e34_ET1pp_figures.py` | Figure for E34 ET1'' -- the necklace brick tested by W^u at N=2000, P=100. |
| `e34_ET1prime_figures.py` | ET1' summary figure (N=2000 delta-sweep GATE-DYN on 3 sentinels). |
| `e34_alive_ladder.py` | Measured survival threshold of every bead, by WARM-START continuation. |
| `e34_bifurcation_figure.py` | Bifurcation diagram of the delayed mixed Hopfield ring, from the measured data. |
| `e34_branch_diagram.py` | Bifurcation diagram of ONE bead: node, twin and their index-1 partners vs lambda. |
| `e34_certif_tune.py` | Tune the argument-principle settings for P=100 saddle certification. |
| `e34_chain_closure.py` | Do ALL the alive saddle-node pairs sit on ONE invariant circle, at a FIXED lambda? |
| `e34_circle_figure.py` | The invariant circle of the last beads, drawn from the recorded trajectories. |
| `e34_circle_trajectories.py` | Record the invariant-circle trajectories (and the twins) for plotting. |
| `e34_cycle_uniqueness.py` | Above lambda*: is there ONE limit cycle, and is it the same for every pattern? |
| `e34_dwell_profile.py` | Speed/distance profile of an edge-tracking A-side trajectory (ET1' seed bug). |
| `e34_dwell_seed_test.py` | Which equilibrium sits on the node_mu | node_mu+1 boundary at N=2000? |
| `e34_et1_boundary_diag.py` | E34-ET1' diagnostic: WHAT lies on the far side of the node_mu basin boundary? |
| `e34_et1pp_followup.py` | ET1'' follow-ups: identify the three non-positive cells of the W^u grid. |
| `e34_first_results.py` | Curate and plot the first E34 N=400 gate results without new simulations. |
| `e34_lib.py` | Numerical foundations shared by the E34 heteroclinic-necklace experiments. |
| `e34_necklace_wu_n2000.py` | ET1'' -- the necklace brick tested DIRECTLY at N=2000 via W^u of the fold partner. |
| `e34_segment_scan.py` | Are the basins of two beads ADJACENT, when the fold partner says otherwise? |
| `e34_snic_closure.py` | Does the last bead's invariant circle CLOSE? -- SNIC vs bistability. |
| `e34_snic_figures.py` | Figure: is the SNIC carried by ONE invariant circle for all the saddle-nodes? |
| `e34_test_deflation.py` | Focused test of the deflated-Newton fix for the ET1' boundary object. |
| `e34_threshold_table.py` | Safe NumPy-only per-pattern fold table for the E34 pilot. |
| `e34_threshold_table_audit.py` | Audit of the E24 per-bead threshold table against the branch itself. |
| `e34_twin_fold.py` | Does the necklace close THROUGH the twins? |
| `e34_uniqueness_merge.py` | Merge the sharded uniqueness runs and decide, over ALL starting patterns. |
| `e34a_census.py` | E34a node/fold/saddle census with explicit safety gates. |
| `e34b_edge_tracking.py` | E34-ET0 edge-tracking gate at fixed lambda (N=400 pilot). |

### E35 — modern Hopfield + video proof of concept (out of manuscript scope)

| File | Role |
|---|---|
| `e35_analysis.py` | Rigorous post-processing of the selected E35 T1 and V0 production runs. |
| `e35_dynamics.py` | E35 DDE adapter, cycle diagnostics, phase decoding and image metrics. |
| `e35_runtime.py` | Safe artifact, heartbeat and run-mode helpers for E35 scripts. |
| `e35_video_tools.py` | Synthetic cyclic-video data and leakage-safe E35 baselines. |
| `e35b_factorial_pilot.py` | E35 T1: guarded J x K factorial pilot. |
| `e35c_synthetic_video_poc.py` | E35 V0: guarded synthetic cyclic-video proof of concept. |
| `e35d0_make_figure.py` | E35-D0 figure: readout diagnostics on the stored JP_KP V0 trajectory. |
| `e35d0_readout_diagnostics.py` | E35-D0 readout diagnostics on the stored JP_KP V0 trajectory (no new dataset). |
| `e35v1_bench.py` | E35-V1 step B: benchmark + smoke test BEFORE any heavy run. |
| `e35v1_common.py` | Shared machinery for the E35-V1 gate (design != evaluation). |
| `e35v1_design.py` | E35-V1 DESIGN phase (design_seeds {1..6} ONLY). |
| `e35v1_eval.py` | E35-V1 EVALUATION phase -- eval_seeds {101..110}, opened ONCE. |
| `e35v1_figures.py` | E35-V1 figures (design characterisation + evaluation verdict). |

### E36 — implicit delay from layered depth

| File | Role |
|---|---|
| `e36_postprocess.py` | Verify and summarize the completed E36 campaigns without simulation. |
| `e36a2_delay.py` | Guarded E36-A2 delay-scaling campaign. |
| `e36a_regimes.py` | E36-A1 adaptive-sentinel campaign driver. |
| `e36bench.py` | Guarded microbenchmark for E36. |

### E37 — modern-Hopfield component in correlated patterns

| File | Role |
|---|---|
| `e37_arrest_in_time.py` | Is the usual system's recall a transient or an attractor? Resolve it in time. |
| `e37_e27_reconciliation.py` | Why E37's baseline forward fraction differs from E27's, at the same c. |
| `e37_figure_doseresponse.py` | E37 figure 1 -- what modernising K buys, as a dose-response in correlation. |
| `e37_figure_traces.py` | E37 figure 2 -- the same contrast, read directly on the trajectories. |
| `e37_modern_k_markov.py` | E37 -- what the Modern-Hopfield component buys on CORRELATED sequences. |

---

## `code/campaigns/` — V5 verification campaigns N1–N10

An independent float64 reimplementation, written to re-test the load-bearing claims through a second code path with checkpointing and explicit acceptance gates. `RUNBOOK.md` in this folder gives the exact production commands; the results and the incident log are in `data/v5_campaigns/`.

| File | Role |
|---|---|
| `arclength.py` | Low-rank pseudo-arclength continuation for one memory-connected branch. |
| `batch_dde.py` | Restart-safe batched float64 exact reduced DDE integrator. |
| `campaign_specs.py` | Machine-readable grids and rationale copied from NUMERICAL_WORKPLAN.md. |
| `couplings.py` | Backend-selectable pattern operators compatible with the historical API. |
| `dynamics.py` | Exact reduced DDE dynamics, event classification, and warm-start descent. |
| `fold_refinement.py` | High-precision reduced augmented solve for a stationary saddle-node fold. |
| `lyapunov_stream.py` | Streaming float64 Benettin/QR integrator for the exact reduced DDE. |
| `make_e34_bifurcation_schematic.py` | Create the publication schematic summarizing the E34 bifurcation architecture. |
| `mlx_preflight.py` | Run in a normal macOS Terminal before selecting V5_BACKEND=mlx. |
| `n10_audit.py` | N10 numerical and provenance audit for the final figure pipeline. |
| `n1_compare.py` | N1 static-extreme versus independently measured dynamic onset. |
| `n1_convergence_audit.py` | N1 branch/fold audit and restart-safe dynamic convergence controls. |
| `n1_history_control.py` | Independent-history control for completed N1 dynamic brackets. |
| `n1_overnight.py` | Conservative restart-safe overnight orchestration for N1 dynamics. |
| `n2_analyse.py` | Publication-grade aggregation, gate and figure for corrected N2 outputs. |
| `n2_local_laws.py` | N2 two-sided local critical laws and fold normal-form diagnostic. |
| `n3_reanalyse.py` | Seed-aware, read-only reanalysis of the archived E30 campaign (task N3). |
| `n4_figures.py` | N4 publication-style Figures 1--4 from archived and newly derived tables. |
| `n5a_analyse.py` | Derived midpoints, jump brackets and boundary classifications for N5A. |
| `n5a_delay_scan.py` | N5A short-delay coherence and seed-resolved cycle/chaos boundary. |
| `n5b_long_delay.py` | N5B resolution-stable long-delay spectral scan and nonlinear eigenmode test. |
| `n5b_pinned_audit.py` | Resolution audit for positive N5B pinned-front complex margins. |
| `n5b_pinned_refine.py` | Refine resolution-stable N5B pinned-front complex-root crossings. |
| `n5b_recovery_supervisor.py` | Restart-safe supervisor for the interrupted N5B production campaign. |
| `n6_lyapunov_campaign.py` | N6 multi-seed float64 Lyapunov spectrum and attractor geometry. |
| `n7_correlations.py` | N7 multi-seed Markov-flip and smooth periodic correlation controls. |
| `n8_analyse.py` | Audit and preliminary analysis of the completed N8 trajectory grid. |
| `n8_highload.py` | N8 certification of high-load moving states by float64 Lyapunov analysis. |
| `n9_stationarity.py` | N9 float64 stationarity and operational ergodicity controls. |
| `observables.py` | Shared trajectory observables for delay, correlation, load and aging scans. |
| `patterns.py` | Pattern ensembles used by N1, N7, N8 and N9. |
| `plot_n1_multiseed_identity.py` | Publication plots for the multi-seed N1 identity results. |
| `plot_n1_static_dynamic_equality.py` | Direct multi-seed equality plot for static extremes and dynamic onsets. |
| `plot_n6_multiseed_lyapunov.py` | Publication plot for the validated N6 multi-seed Lyapunov result. |
| `run.sh` | !/bin/zsh |
| `run_next24h.py` | Restart-safe, resource-bounded execution of the requested 24-hour campaigns. |
| `run_workplan.py` | Preflight and dispatch for the N1--N10 numerical work plan. |
| `static_thresholds.py` | Per-motif static fold thresholds with the N1 publication diagnostics. |
| `tests/smoke_n1.py` | Small-N end-to-end smoke test for both halves of N1. |
| `tests/test_core.py` | — |
| `v5_paths.py` | Paths, immutable-reference checks, and common provenance for V5 campaigns. |

---

## `code/figures/` — figure assembly

Composite panel builders. `manifest.json` in `figures/panel_manifest.json` records the source SHA-256 and crop coordinates of every component.

| File | Role |
|---|---|
| `build_panels.py` | Rebuild the eight requested article panels without modifying source folders. |
| `make_source_contact_sheet.py` | Build a read-only visual inventory of the requested source figures. |
| `v2/common2.py` | — |
| `v2/compose05.py` | — |
| `v2/p_chaos.py` | — |
| `v2/p_floquet.py` | — |
| `v2/p_model.py` | — |

---

## `code/_scratch/` — scratch, probes and superseded drivers

Kept for traceability, excluded from the narrative. Nothing here backs a published number.

| File | Role |
|---|---|
| `cycle_reduced_pre_dhist_backup.py` | Exact P-dimensional reduction of the delayed mixed Hopfield DDE + validations. |
| `debug.py` | — |
| `health_probe.sh` | !/bin/zsh |
| `plot_dmft_overlay.py` | — |
| `plot_final.py` | — |
| `rerun_N10000.sh` | !/bin/bash |
| `run_gpu_gated.sh` | !/bin/bash |
| `run_lowalpha.sh` | !/bin/bash |
| `run_multiseed.sh` | !/bin/bash |
| `scratch_test_jvp.py` | — |
| `scratch_verify_jvp.py` | — |
| `scratch_verify_jvp_2.py` | — |
| `scratch_verify_jvp_3.py` | — |
| `test_eval.py` | — |
| `test_eval2.py` | — |
| `test_logic.py` | — |
| `test_matmul.py` | — |
| `test_repeat.py` | — |
| `test_tile.py` | — |
