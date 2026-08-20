# Claim register — manuscript ↔ numerical evidence

*Generated 2026-08-19. Machine-readable version: `docs/claims.csv`.*


> **Paths.** This copy is path-remapped for this repository. Entries marked
> `[local archive]` live only in the full research archive, not here.
Every verifiable statement of the manuscript is listed with the experiment, the data file, the script and the figure that support it. Paths are relative to the repository root.

**Status vocabulary**

| Status | Meaning |
|---|---|
| Established | The stored numerical output directly supports the statement as written. |
| Qualified | Supported, but only over the tested range / protocol; the caveat in the *Notes* column must survive into the manuscript. |
| Limitation | Not established, refuted, or explicitly withdrawn from scope. Must not be promoted into a conclusion. |
| Analytic | Exact algebraic result, checked numerically where noted. |

**Experiment prefixes** — `E**` are the historical campaigns logged in `docs/worklogs/CYCLE_worklog.md`; `N**` are the V5 verification campaigns in `data/v5_campaigns/`.

---

## 1 Introduction and model

### `C-01` — The N-dimensional delayed dynamics reduces exactly onto the P-dimensional pattern subspace; the transverse component obeys t0 w' = -w and decays.

- **Where** — Sec. 1.3, Eq. (7)-(9)
- **What is asserted** — Full vs reduced trajectory agreement 1e-15 on the certification trajectory (independently recorded discrepancy 5.49e-16).
- **Experiment** — reduction certification (V0/V1/V4)
- **Data** — `-- (validation run, printed by the module self-test)`
- **Code** — `code/core/cycle_reduced.py  (validations in __main__)`
- **Figure** — -
- **Write-up** — `docs/worklogs/CYCLE_worklog.md §3`
- **Status** — **Established**
- **Notes** — Algebraic identity plus a numerical certification. The manuscript quotes 1e-15; the V5 evidence register quotes 5.49e-16 for the same check.

### `C-02` — The reduced system keeps the realized finite-size disorder: no disorder average is taken anywhere in m(a).

- **Where** — Sec. 1.3, Eq. (10)-(11)
- **What is asserted** — m(a) = N^-1 X tanh(beta X^T a) built from the actual pattern matrix X.
- **Experiment** — -- (definition)
- **Data** — `--`
- **Code** — `code/core/cycle_reduced.py`
- **Figure** — -
- **Write-up** — `docs/worklogs/CYCLE_worklog.md §3`
- **Status** — **Analytic**
- **Notes** — Structural property of the reduction, not a measurement.

### `C-03` — Branch-resolved results refer to one reference realization.

- **Where** — Eq. (12) reference parameters
- **What is asserted** — N=2000, P=100, alpha=0.05, beta=20, tau=10, seed 42, time in units of t0.
- **Experiment** — -- (convention)
- **Data** — `--`
- **Code** — `code/core/config.py`
- **Figure** — -
- **Write-up** — `docs/worklogs/CYCLE_worklog.md §1`
- **Status** — **Established**

### `C-04` — Above threshold a single localized activity peak traverses the complete memory ring, with consecutive overlaps displaying the relay clock.

- **Where** — Fig. 1 (Figure00) panels (b),(c)
- **What is asserted** — lambda=0.9, tau=10, 26 relays; measured T1 = tau + t_esc = 10.83.
- **Experiment** — E1/E4 cycle construction
- **Data** — `data/2_recall_cycle_snic/data/cycle_lam0.9_tau10.0_N2000.npz  (field a_grid)`
- **Code** — `code/core/pacemaker_cycle.py ; code/core/cycle_figures.py`
- **Figure** — Figure00_model_architecture.png
- **Write-up** — `code/figures/v2/README.md (Panel 00)`
- **Status** — **Established**

### `C-05` — Below threshold the directed coupling tilts a retrieval state toward its successor without dislodging it.

- **Where** — Fig. 1 (Figure00) panel (d)
- **What is asserted** — lambda=0.10 fixed point, host and successor overlaps only.
- **Experiment** — E24a
- **Data** — `data/3_threshold_scaling/data/E24_curves_N2000_s42.npz`
- **Code** — `code/experiments/e24a_geometry.py`
- **Figure** — Figure00_model_architecture.png
- **Write-up** — `code/figures/v2/README.md (Panel 00)`
- **Status** — **Qualified**
- **Notes** — The panel-00 README notes that the bars beyond index 1 are drawn at the cross-talk scale for context and are NOT individually measured; the caption must say so or the bars must be dropped.

### `C-06` — A phenomenological (alpha, lambda) survey at three delays separates persistent static retrieval, ordered forward relays and failure of both.

- **Where** — Sec. 1.4, Fig. 2 (Figure01)
- **What is asserted** — N=2500, beta=20, 30x30 grid over alpha in [0.01,0.30] and lambda in [0,1], Euler dt=0.05, J_ii=0, tau/t0 in {1,3,10}.
- **Experiment** — continuous-time phase survey (pre-E campaign)
- **Data** — `[local archive] obsolete/03_legacy_code/Projet_Chicago/stabilite/systeme continu/phases & comparaison/phases continu/  (walkthrough.md + arrays)`
- **Code** — `see walkthrough.md in that folder`
- **Figure** — Figure01_phase_diagram.png
- **Write-up** — `[local archive] obsolete/03_legacy_code/Projet_Chicago/stabilité/système continu/phases & comparaison/phases continu/walkthrough.md`
- **Status** — **Qualified**
- **Notes** — ACTION: the panel file shipped with the manuscript is named 'phases MCMC.png' in figures/panels/, yet the manuscript describes Euler integration of the continuous-time model. Confirm the file is the continuous-time survey and rename it before release. This survey uses a different protocol (hollow J, Euler) from every other result in the paper, as the manuscript already states.


## 2 Quenched fold ensemble

### `C-07` — A zero root of the delay characteristic equation is exactly a zero eigenvalue of the static continuation problem: Delta(0) = -M.

- **Where** — Sec. 2.1, Eq. (17)
- **What is asserted** — Exact identity; fold locations are therefore tau-independent.
- **Experiment** — -- (analytic, re-derived and checked numerically)
- **Data** — `--`
- **Code** — `code/core/reduced_spectrum.py ; code/core/audit_static_analysis.py`
- **Figure** — -
- **Write-up** — `docs/worklogs/REPORT_static_bifurcation.md §3.4b`
- **Status** — **Analytic**

### `C-08` — Sylvester's determinant identity collapses the transcendental NxN characteristic problem onto a PxP problem T_P(z).

- **Where** — Sec. 2.1, Eq. (15)-(16)
- **What is asserted** — det Delta(z) = (t0 z + 1)^(N-P) det T_P(z).
- **Experiment** — -- (analytic) + dense NxN cross-check
- **Data** — `data/3_threshold_scaling/data/E21d_memory_fold_check.npz`
- **Code** — `code/core/robust_branch.py ; code/core/reduced_spectrum.py ; code/experiments/e21d_memory_fold_check.py`
- **Figure** — -
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E21d (dense NxN vs Sylvester agreement 2e-16)`
- **Status** — **Established**

### `C-09` — Over the tested (alpha, tau) grid, every terminal point of a memory branch carries a real zero mode: no delay-induced Hopf bifurcation preempts the fold.

- **Where** — Sec. 2.1
- **What is asserted** — alpha in {0.01,0.03,0.05,0.07,0.10} x tau/t0 in {0,0.5,1,2,4,6,8,10,12,15}.
- **Experiment** — lambda_c grid v2
- **Data** — `data/1_static_bifurcation/data/lambda_c_grid_v2_N2000.npz ; data/1_static_bifurcation/data/lambda_c_grid.npz`
- **Code** — `code/core/lambda_c_grid_v2.py ; code/core/lambda_c_grid.py`
- **Figure** — -  (grid figure: figures/experiments/1_static_bifurcation/lambda_c_grid_v2_N2000.png)
- **Write-up** — `docs/worklogs/REPORT_static_bifurcation.md §5.2`
- **Status** — **Established**

### `C-10` — The alpha=0.10 edge of the grid is reported as qualitative only: near capacity the single-realization fold is not a well-defined quantity.

- **Where** — Sec. 2.1 (caveat paragraph)
- **What is asserted** — Fitted fold location drifts from ~0.18 at N=1e3 to ~0.05 at N=1e4; seed-to-seed scatter reaches ~60% of the mean; some realizations produce spurious unstable starting states.
- **Experiment** — multi-seed static folds
- **Data** — `data/1_static_bifurcation/data/fold_N10000_a0.1.npz ; data/1_static_bifurcation/data/fold_N2000_a0.1.npz`
- **Code** — `code/core/multiseed_fold.py ; code/core/run_N10000.py`
- **Figure** — -
- **Write-up** — `docs/worklogs/REPORT_static_bifurcation.md §5.3-5.4`
- **Status** — **Limitation / not established**
- **Notes** — Explicitly flagged in the manuscript as indicative of the qualitative verdict only, not as a quantitative threshold.

### `C-11` — The Hopf margin at the fold decreases with delay roughly as 1/tau but stays clearly positive for tau up to about 1e2 at alpha <= 0.05.

- **Where** — Sec. 2.1 (last paragraph)
- **What is asserted** — Margin erosion ~1/tau; margin x tau reaches a plateau for the pinned branch.
- **Experiment** — E15 (pinned Hopf margin), E17 (nonlinear probe of the Hopf window)
- **Data** — `data/3_threshold_scaling/data/E15_pinned_margins.npz ; data/1_static_bifurcation/data/E17_hopf_birth.npz`
- **Code** — `code/experiments/e15_pinned_hopf_margin.py ; code/experiments/e17_hopf_birth.py`
- **Figure** — -  (figures/experiments/3_threshold_scaling/figG_pinned_hopf_margin.png)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E15/E17 ; docs/worklogs/REPORT_static_bifurcation.md §5.6`
- **Status** — **Qualified**
- **Notes** — The independent V5 campaign N5B found no imaginary-axis crossing over 200/200 spectral points and 5 seeds, but was withdrawn from the manuscript scope on 2026-07-30; the supported statement is spectral softening without a resolved Hopf preemption.

### `C-12` — The P=100 primary branches of the reference sample terminate at distinct quenched saddle-node thresholds spanning a wide interval.

- **Where** — Sec. 2.2, Eq. (18)
- **What is asserted** — min = 0.2195 (pattern 28), median = 0.2742, max = 0.3276 (pattern 91), lambda_c(1) = 0.2822; each bracketed by bisection to +/- 2e-4; interval width 0.11, ~50x the single-threshold uncertainty.
- **Experiment** — E24 (all-motif threshold tracer)
- **Data** — `data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz`
- **Code** — `code/experiments/e24_thresholds.py`
- **Figure** — Figure02_global_bifurcation.png (b,c)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E24 ; docs/worklogs/REPORT_static_bifurcation.md §5.7-5.8`
- **Status** — **Established**
- **Notes** — Primary estimator is bisection (lam_c), not spectral extrapolation.

### `C-13` — Branches tilt toward their successor well before the fold: the deformation is a broad precursor occupying most of the branch.

- **Where** — Sec. 2.3, Eq. (19)
- **What is asserted** — median over motifs of lambda_50/lambda_c = 0.64, IQR [0.61, 0.66].
- **Experiment** — E24a
- **Data** — `data/3_threshold_scaling/data/E24_curves_N2000_s42.npz`
- **Code** — `code/experiments/e24a_geometry.py`
- **Figure** — Figure03_static_folds.png (c)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E24a`
- **Status** — **Established**

### `C-14` — Rescaling lambda by each branch's own threshold collapses the 100 trajectories onto a quasi-single shape.

- **Where** — Sec. 2.3
- **What is asserted** — Universal normalized path to death; disorder only sets when a branch dies.
- **Experiment** — E24a
- **Data** — `data/3_threshold_scaling/data/E24_curves_N2000_s42.npz ; data/3_threshold_scaling/data/E24_curves_N10000_s42.npz`
- **Code** — `code/experiments/e24a_geometry.py`
- **Figure** — Figure03_static_folds.png (b,c)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E24a ; results index figU_branch_geometry.png`
- **Status** — **Qualified**
- **Notes** — Reported by the V5 evidence register as a 'direct empirical observation', not a proven universality.

### `C-15` — The terminal tilt ratio varies across branches and correlates strongly with the threshold itself.

- **Where** — Sec. 2.3
- **What is asserted** — r_c ranges 0.29-0.49; corr(r_c, lambda_c) = +0.75 at N=1e4 (+0.81 at N=2000).
- **Experiment** — E25 (death shape)
- **Data** — `data/3_threshold_scaling/data/E25_deathshape.npz ; data/3_threshold_scaling/data/E25_deathshape_N10000.npz`
- **Code** — `code/experiments/e25_deathshape.py`
- **Figure** — Figure03_static_folds.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E25`
- **Status** — **Established**
- **Notes** — The V5 register phrases the same data as 'fold shape is not strictly universal' (terminal ratio 0.285-0.489).

### `C-16` — At fixed threshold, residual scatter of the terminal tilt is almost entirely explained by the local quenched bond overlap q_mu.

- **Where** — Sec. 2.3
- **What is asserted** — partial corr(q_mu, r_c | lambda_c) = +0.90 at N=1e4 (+0.95 at N=2000).
- **Experiment** — E25
- **Data** — `data/3_threshold_scaling/data/E25_deathshape.npz`
- **Code** — `code/experiments/e25_deathshape.py`
- **Figure** — Figure03_static_folds.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E25`
- **Status** — **Established**
- **Notes** — The results index flags a nuance on the partial correlation; see the E25 entry.

### `C-17` — The saturated overlap readout hides the amplitude tilt until the immediate vicinity of the fold.

- **Where** — Sec. 2.3 (saturation paragraph)
- **What is asserted** — At beta=20, tanh(beta u) is within 1e-4 of sgn(u) whenever |u| >~ 0.5; on a tilted branch a_mu ~ 0.75, a_{mu+1} ~ 0.25 gives |u_i| >= 0.5 on every site.
- **Experiment** — -- (analytic reading of E24a curves)
- **Data** — `data/3_threshold_scaling/data/E24_curves_N2000_s42.npz`
- **Code** — `code/experiments/e24a_geometry.py`
- **Figure** — Figure03_static_folds.png (e)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E24a`
- **Status** — **Analytic**

### `C-18` — Near the fold the geometry is the generic saddle-node one; the saddle-node distance and the rightmost eigenvalue scale as \|lambda - lambda_c\|^(1/2).

- **Where** — Sec. 2.3 (normal form)
- **What is asserted** — Direct continuation at N=1e4; fits on representative branches give R^2 = 0.93-0.98.
- **Experiment** — E24 / E21d
- **Data** — `data/3_threshold_scaling/data/E24_thr_N16000_P100_s42.npz ; data/3_threshold_scaling/data/E21d_memory_fold_check.npz`
- **Code** — `code/experiments/e24_analysis.py ; code/experiments/e21d_memory_fold_check.py`
- **Figure** — Figure03_static_folds.png (d)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E24/E21d`
- **Status** — **Qualified**
- **Notes** — Direct over the tested windows only.

### `C-19` — Memory labels are inherited continuously from the lambda=0 retrieval state and checked through the overlap vector, so {lambda_c(mu)} is a well-defined quenched observable.

- **Where** — Sec. 2.3 (labelling)
- **What is asserted** — Protocol statement.
- **Experiment** — E24
- **Data** — `data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz`
- **Code** — `code/experiments/e24_thresholds.py`
- **Figure** — -
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E24`
- **Status** — **Established**


## 3 Threshold law

### `C-20` — Load sets the centre of the threshold law; size sets its width.

- **Where** — Sec. 3.1
- **What is asserted** — At N=1e4 the median threshold moves 0.38 -> 0.35 -> 0.27 for alpha = 0.01, 0.02, 0.05, while sigma moves only 0.0076 -> 0.0081 -> 0.0094 (+24% for a five-fold load increase, vs a factor ~2.4 between N=2000 and N=1e4).
- **Experiment** — E26 (threshold law vs alpha)
- **Data** — `data/3_threshold_scaling/data/E26_thr_N10000_P{100,200,500,800,1000}_s{42,43}.npz`
- **Code** — `code/experiments/e26_analysis.py`
- **Figure** — Figure04_thresholds.png (a,f,g)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E26`
- **Status** — **Established**

### `C-21` — The standardized threshold laws at different loads are statistically indistinguishable: the ensemble is a position-scale family.

- **Where** — Sec. 3.1
- **What is asserted** — Pairwise two-sample KS tests (0.01 vs 0.02, 0.01 vs 0.05, 0.02 vs 0.05) give p = 0.85, 0.87, 0.25.
- **Experiment** — E26
- **Data** — `data/3_threshold_scaling/data/E26_thr_N10000_P{100,200,500,800,1000}_s{42,43}.npz`
- **Code** — `code/experiments/e26_analysis.py`
- **Figure** — Figure04_thresholds.png (b)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E26`
- **Status** — **Established**
- **Notes** — Valid within the tested load range only (alpha <= 0.05); E26 records a breakdown for alpha >= 0.08.

### `C-22` — The pooled standardized law is near-Gaussian, with a small negative skew and excess tails.

- **Where** — Sec. 3.1
- **What is asserted** — Pooling fixed-load data for N >= 4000 (8764 branch thresholds): skewness -0.45, excess kurtosis 1.0, max CDF distance 0.026 from a fitted normal.
- **Experiment** — E30 / N3
- **Data** — `data/3_threshold_scaling/data/E30_scaling/  (303 runs)`
- **Code** — `code/experiments/e30_analysis.py ; code/campaigns/n3_reanalyse.py`
- **Figure** — Figure04_thresholds.png (c)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E30 ; data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §5`
- **Status** — **Established**

### `C-23` — The thresholds are quasi-i.i.d. around the memory ring.

- **Where** — Sec. 3.1
- **What is asserted** — lag-one autocorrelation of neighbouring thresholds = -0.009 at N=1e4.
- **Experiment** — E25 / E30
- **Data** — `data/3_threshold_scaling/data/E25_deathshape_N10000.npz`
- **Code** — `code/experiments/e25_deathshape.py`
- **Figure** — -
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E25`
- **Status** — **Established**

### `C-24` — The width of the threshold law narrows with system size as a power law, below the CLT value 1/2.

- **Where** — Sec. 3.1, Eq. (22)
- **What is asserted** — sigma ~ N^-b with b = 0.43, seed-aware 95% CI [0.40, 0.47] at alpha=0.05.
- **Experiment** — N3 (seed-aware reanalysis of the 303 E30 files)
- **Data** — `data/v5_campaigns/runs/N3/E30_seed_aware_analysis.json ; data/v5_campaigns/reports/N3_E30_seed_statistics.csv ; source arrays data/3_threshold_scaling/data/E30_scaling/`
- **Code** — `code/campaigns/n3_reanalyse.py`
- **Figure** — Figure04_thresholds.png (d)
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §5 ; data/v5_campaigns/RUN_STATUS.md`
- **Status** — **Established**
- **Notes** — N3 exact values: b = 0.43078, bootstrap median 0.43009, CI [0.39638, 0.46520]. The bootstrap unit is a disorder seed, never a motif. N3 also records that a pure power law is preferred over a power-plus-floor model by only dAICc = 2.11, i.e. weak model selection. At fixed P=100 the exponent is instead 0.903 [0.886, 0.921] - the two protocols must not be conflated.

### `C-25` — The max-min range of the sample narrows too, more slowly, and is fully accounted for by the one-point width and Gaussian order statistics.

- **Where** — Sec. 3.1 (range paragraph)
- **What is asserted** — Range 0.15 (N=1000) -> 0.078, 0.055, 0.050 at N = 8000, 1e4, 1.4e4. Fitting range(N)/sqrt(2 ln(alpha N)) ~ N^-b_gap gives b_gap = 0.42, seed-aware 95% CI [0.37, 0.47], indistinguishable from the width exponent.
- **Experiment** — E30 / N3
- **Data** — `data/v5_campaigns/runs/N3/E30_seed_aware_analysis.json ; data/3_threshold_scaling/data/E30_scaling/`
- **Code** — `code/experiments/e30_analysis.py ; code/campaigns/n3_reanalyse.py`
- **Figure** — Figure04_thresholds.png (d)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E30 ; data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §5`
- **Status** — **Qualified**
- **Notes** — CHECK: the N3 synthesis reports the *raw* range exponent d = 0.27582, CI [0.22538, 0.33040], i.e. without the sqrt(2 ln P) correction. The manuscript's 0.42 [0.37, 0.47] is the log-corrected fit and is not tabulated in the N3 synthesis - locate or re-run that specific fit before submission.

### `C-26` — A parameter-free finite-P Gaussian extreme-value estimate tracks the observed sample maxima at the level of their natural fluctuations.

- **Where** — Sec. 3.2, Eq. (25)
- **What is asserted** — Mean signed error +0.005, maximum absolute error 0.024, vs a sample-to-sample spread of the maxima of order 0.01-0.03.
- **Experiment** — E24 / N3
- **Data** — `data/v5_campaigns/runs/N3/E30_seed_aware_analysis.json`
- **Code** — `code/campaigns/n3_reanalyse.py`
- **Figure** — Figure04_thresholds.png (e)
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §5`
- **Status** — **Qualified**
- **Notes** — DISCREPANCY TO RESOLVE: the N3 synthesis reports mean signed error +0.00734 and maximum absolute error 0.02745 for the Gaussian extreme prediction, and notes that the imposed extreme form is disfavoured by dAICc = 8.37 against a free power law. The manuscript quotes +0.005 / 0.024. Reconcile the subset used (all sizes vs N >= 4000) or update the numbers.

### `C-27` — Because P grows linearly while sigma shrinks approximately as N^-1/2, the two transitions merge in the thermodynamic limit.

- **Where** — Sec. 3.2, Eq. (26)
- **What is asserted** — lambda* - mean(lambda_c) ~ sigma sqrt(2 ln(alpha N)) -> 0.
- **Experiment** — E30 / N3 (extrapolation)
- **Data** — `data/v5_campaigns/runs/N3/E30_seed_aware_analysis.json`
- **Code** — `code/campaigns/n3_reanalyse.py`
- **Figure** — Figure02_global_bifurcation.png (d)
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §5`
- **Status** — **Qualified**
- **Notes** — Conditional extrapolation: measured over N = 1e3 to 1.4e4 only. The V3 audit explicitly requires this to be stated as conditional.


## 4 Extreme selection

### `C-28` — In every disorder realization the independently continued static maximizer, the slowest relay bond and the final arrest memory are the same pattern; and the static extreme equals the dynamical onset at 1e-3 resolution.

- **Where** — Sec. 4.1, Eq. (27)
- **What is asserted** — 20 networks at N=2000, P=100, beta=20, tau=10. Motif identity 20/20. 19 admissible realizations satisfy |lambda*_static - lambda_onset_dyn| < 1e-3, largest retained discrepancy 7.3e-4, against a threshold spread of order 1e-1.
- **Experiment** — N1 (static extreme vs independent dynamic onset)
- **Data** — `data/v5_campaigns/reports/N1_static_extreme_vs_dynamic_onset.csv ; data/v5_campaigns/reports/N1_multiseed_identity_summary.json ; data/v5_campaigns/runs/N1/`
- **Code** — `code/campaigns/n1_overnight.py ; code/campaigns/n1_compare.py ; code/campaigns/n1_convergence_audit.py`
- **Figure** — Figure05_extreme_selection.png (b,c)
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §3 ; data/v5_campaigns/N1_AUDIT_CONCLUSION.md`
- **Status** — **Established**
- **Notes** — Seed 48 is excluded because its first and second static-extreme brackets overlap, so the unique extreme required by the test is not identifiable - not because it is an outlier. The claim must be worded as an equality at resolution 1e-3, never as an exact mathematical identity.

### `C-29` — In the reference sample the maximizer is memory 91 and lambda* = 0.328.

- **Where** — Sec. 4.1
- **What is asserted** — Below this value a moving initial history stops at memory 91; above it the trajectory completes the ring.
- **Experiment** — E6b/E9 + N1
- **Data** — `data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz ; data/2_recall_cycle_snic/data/E9_bond_times.npz`
- **Code** — `code/experiments/e24_thresholds.py ; code/core/pacemaker_scan.py`
- **Figure** — Figure05_extreme_selection.png (a)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E6/E9`
- **Status** — **Established**

### `C-30` — Approaching onset from above, only the critical bond slows; the other 99 relays barely notice.

- **Where** — Sec. 4.2
- **What is asserted** — For lambda = 0.332, 0.330, 0.329, 0.328 the median relay times are 17.8, 18.0, 18.1, 18.3 while the critical 91->92 relay takes 35.6, 41.5, 46.9, 72.9.
- **Experiment** — E9 (per-bond statistics near lambda*)
- **Data** — `data/2_recall_cycle_snic/data/E9_bond_times.npz`
- **Code** — `code/core/pacemaker_scan.py ; code/core/cycle_figures.py`
- **Figure** — Figure06_snic.png (e)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E9`
- **Status** — **Established**

### `C-31` — A direct ghost fit of the critical passage time gives a square-root divergence.

- **Where** — Sec. 4.2, Eq. (28)
- **What is asserted** — t_(91->92) ~ C / sqrt(lambda - lambda_g) with C = 2.3 +/- 0.2 (9% relative) and lambda_g = 0.3262, consistent with the independently bracketed onset lambda in [0.327, 0.328].
- **Experiment** — E9
- **Data** — `data/2_recall_cycle_snic/data/E9_bond_times.npz`
- **Code** — `code/core/pacemaker_scan.py ; code/core/cycle_figures.py`
- **Figure** — Figure06_snic.png (e, inset)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E9`
- **Status** — **Established**

### `C-32` — A logarithmic divergence is rejected by model comparison; the free exponent brackets 1/2 and excludes 0 and 1.

- **Where** — Sec. 4.2
- **What is asserted** — Free exponent between 0.43 and 0.51 at 95% confidence.
- **Experiment** — N2 (two-sided local laws, 5 seeds)
- **Data** — `data/v5_campaigns/runs/N2/summary_seeds_42.json (and 47, 51, 52, 60) ; data/v5_campaigns/reports/N2_summary.json ; data/v5_campaigns/reports/N2_fit_windows.csv`
- **Code** — `code/campaigns/n2_analyse.py ; code/campaigns/n2_local_laws.py`
- **Figure** — Figure06_snic.png (a)
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §4 ; data/v5_campaigns/reports/N2_GATE_STATUS.md`
- **Status** — **Qualified**
- **Notes** — N2 per-seed effective exponents: 42 -> 0.539, 47 -> 0.520, 52 -> 0.410, 60 -> 0.439; pooled 0.477 with bootstrap [0.424, 0.529] and 0.470 [0.435, 0.504] on the near window. Seed 51 is archived but must not be used for a local exponent (its dynamic bracket does not contain the exact static fold). CHECK: the manuscript's [0.43, 0.51] is close to the near-window interval [0.435, 0.504] but is not verbatim any tabulated N2 interval - state which one. The first N2 static sweep followed the unstable saddle branch and is invalid; it is archived under runs/N2_INVALID_STATIC_BRANCH_20260729/.

### `C-33` — From below, the leading real eigenvalue of the pinned terminal state softens with square-root slope and reaches zero at the same fold.

- **Where** — Sec. 4.2, Eq. (29)
- **What is asserted** — zeta_max ~ -2.1 sqrt(lambda* - lambda), lambda*_spectral ~ 0.328.
- **Experiment** — E12 (spectral reading of the pinned-front saddle-node)
- **Data** — `data/3_threshold_scaling/data/E12_pinned_branch.npz`
- **Code** — `code/experiments/e12*  (see docs/worklogs/CYCLE_worklog.md E12) ; code/core/reduced_spectrum.py`
- **Figure** — Figure06_snic.png (d)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E12`
- **Status** — **Established**

### `C-34` — Points closer than delta ~ 1e-4 to the threshold lie below the resolution at which lambda* itself is defined and are shown as illustration only.

- **Where** — Sec. 4.2 (resolution caveat)
- **What is asserted** — Bisection brackets are +/- 2e-4; the quantitative support for the square-root law is the decade delta in [1e-3, 1e-2].
- **Experiment** — E12 / E24
- **Data** — `data/3_threshold_scaling/data/E12_pinned_branch.npz`
- **Code** — `code/experiments/e24_thresholds.py`
- **Figure** — Figure06_snic.png (d)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E12/E24`
- **Status** — **Limitation / not established**

### `C-35` — The measured inter-memory period decomposes exactly into an imposed delay plus a delay-independent nonlinear escape time.

- **Where** — Sec. 4.3, Eq. (30)
- **What is asserted** — T1(lambda, tau) = tau + t_esc(lambda); for tau = 5, 10, 20 the extracted escape times collapse to three significant digits.
- **Experiment** — E1 + E4 (pacemaker law collapse)
- **Data** — `data/2_recall_cycle_snic/data/floquet_scan_N2000_tau10.0.npz ; data/2_recall_cycle_snic/data/cycle_lam0.9_tau10.0_N2000.npz`
- **Code** — `code/core/pacemaker_scan.py ; code/core/cycle_figures.py`
- **Figure** — Figure06_snic.png (f)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E1/E4`
- **Status** — **Established**

### `C-36` — The recall cycle never loses stability: no Floquet multiplier approaches the unit circle. It dies by losing its route, not its attractivity.

- **Where** — Sec. 4.4
- **What is asserted** — Broad scan lambda = 0.35 to 0.95 plus near-threshold controls at 0.335 and 0.329. Contraction per tour A = -ln|mu_1| grows from 220 at lambda=0.400 to 420 at 0.335 and 440 at 0.329, while the contraction rate per unit time stays between about -0.16 and -0.23 t0^-1. Phase multiplier stays neutral (ln|mu_0| = 0.001).
- **Experiment** — E11 (full Floquet spectra), E13, E14 (floor-free Benettin/QR contraction)
- **Data** — `data/2_recall_cycle_snic/data/floquet_scan_N2000_tau10.0.npz ; data/2_recall_cycle_snic/data/E11_floquet_spectra_N500.npz`
- **Code** — `code/core/floquet_monodromy.py ; code/core/pacemaker_scan.py ; code/experiments/e14_contraction_near_death.py`
- **Figure** — Figure06_snic.png (g)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E11/E13/E14 ; 0_worklogs/FLOQUET_methodology.md`
- **Status** — **Established**
- **Notes** — The rez field of floquet_scan_N2000_tau10.0.npz is ln|mu*|/T computed from a floored |mu*|; it is an upper bound on Re z_1, NOT a measurement, and must not be plotted as one (see figures_v2/README.md). The N=2000 Arnoldi scan points sit at the relative floor 2e-16 and are lower bounds only.


## 5 Invariant circle

### `C-37` — Just below lambda*, the unstable branch of the index-one saddle paired with the terminal memory tours the entire ordered sequence and returns to the node it left: the union of node, saddle and connecting branches is a numerically closed invariant circle.

- **Where** — Sec. 5.1
- **What is asserted** — At lambda = 0.323 (lambda* - lambda = 0.005) the tour takes 1991 t0 and closes with Gram-metric distance 3.5e-14; at lambda = 0.318 it takes 2138 t0 and closes within about 1e-13. The state itself is an order-one vector.
- **Experiment** — E34 (invariant circle / necklace)
- **Data** — `data/2_recall_cycle_snic/E34_INVARIANT_CIRCLE_RESULTS.md ; data/2_recall_cycle_snic/data/E34/`
- **Code** — `code/experiments/e34_lib.py ; code/experiments/e34_necklace_wu_n2000.py ; code/experiments/e34_et1pp_followup.py`
- **Figure** — Figure07_invariant_circle.png (a,b,c)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E34 (ET1'' entry, 2026-07-27)`
- **Status** — **Established**
- **Notes** — Established at the terminal and penultimate stages of the reference realization only.

### `C-38` — One step deeper in the removal sequence the construction repeats: two surviving beads form a single closed circle threading the ghosts of the other 98 memories.

- **Where** — Sec. 5.1
- **What is asserted** — At lambda = 0.3076 patterns 43 and 91 survive; the unstable branch at 43 reaches node 91 and that at 91 reaches node 43.
- **Experiment** — E34
- **Data** — `data/2_recall_cycle_snic/E34_INVARIANT_CIRCLE_RESULTS.md ; data/2_recall_cycle_snic/data/E34/`
- **Code** — `code/experiments/e34_necklace_wu_n2000.py`
- **Figure** — Figure07_invariant_circle.png (a)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E34`
- **Status** — **Established**

### `C-39` — Farther below threshold most local links are certified individually, but the complete many-bead object has not been closed end to end.

- **Where** — Sec. 5.1 (limitation)
- **What is asserted** — Circle established as the terminal organizing structure only.
- **Experiment** — E34
- **Data** — `data/2_recall_cycle_snic/E34_INVARIANT_CIRCLE_RESULTS.md (Sec. 10 'Limites')`
- **Code** — `code/experiments/e34_necklace_wu_n2000.py`
- **Figure** — Figure07_invariant_circle.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E34`
- **Status** — **Limitation / not established**
- **Notes** — The V5 evidence register goes further and asks that a rigorous SNIC not be claimed at all (global invariant-circle topology and asymptotic normal-form coefficients not established). The V4 manuscript does claim a SNIC; this is the single largest wording risk in the paper and should be checked against the audit trail in [local archive] 4_campaigns/manuscript_audits/agent_critique/.

### `C-40` — The stable node persists up to lambda = 0.328 and the paired index-one saddle reaches the same parameter, so the last fold sits on the invariant circle.

- **Where** — Sec. 5.1 (equilibrium branches)
- **What is asserted** — Node self-overlap decreases from 0.999 to 0.982; paired saddle overlap 0.982.
- **Experiment** — E34
- **Data** — `data/2_recall_cycle_snic/E34_INVARIANT_CIRCLE_RESULTS.md`
- **Code** — `code/experiments/e34_lib.py`
- **Figure** — Figure07_invariant_circle.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E34`
- **Status** — **Established**

### `C-41` — All 100 stored-pattern initial conditions converge to one and the same periodic orbit at each tested lambda; the recall cycle is a global attractor of the memory sequence.

- **Where** — Sec. 5.2, Table I
- **What is asserted** — lambda=0.330: 100 starts, mean period 1869.4 t0, d_max = 5.3e-3. lambda=0.350: 100 starts, mean period 1633.5 t0, d_max = 9.2e-3.
- **Experiment** — E34 (cycle uniqueness)
- **Data** — `data/2_recall_cycle_snic/data/E34/E34_cycle_uniqueness_MERGED_lam0.330000.json ; data/2_recall_cycle_snic/data/E34/E34_cycle_uniqueness_MERGED_lam0.350000.json`
- **Code** — `code/experiments/e34_necklace_wu_n2000.py`
- **Figure** — -  (Table I)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E34`
- **Status** — **Established**
- **Notes** — VERIFIED in this reorganisation: the merged JSON for lambda=0.330 gives period_mean = 1869.4063881765176 and dist_max = 5.2637e-3 over n_patterns = 100, matching Table I. Not the only attractor in the full state space (see C-45).

### `C-42` — Close to onset the released orbit performs exact binary recall: the sign of every neuron reproduces the stored pattern at that pattern's own peak.

- **Where** — Sec. 5.3, Eq. (31)
- **What is asserted** — At lambda = 0.330 the median analog peak over one tour is 0.992 (smallest 0.978) and b_mu(t_mu^peak) = 1.000 for every mu = 1..P.
- **Experiment** — E29 (cycle vs ghosts)
- **Data** — `data/2_recall_cycle_snic/data/E29_cycle_ghosts.npz`
- **Code** — `code/experiments/e29_cycle_vs_ghosts.py ; code/experiments/e29_fig.py`
- **Figure** — Figure07_invariant_circle.png (d)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E29`
- **Status** — **Established**

### `C-43` — The fraction of a tour spent in an exact stored pattern rises with lambda.

- **Where** — Sec. 5.3
- **What is asserted** — Refined peak-to-peak period 1869 t0 at lambda = 0.330; exact-pattern dwell 47% of the tour at 0.330, rising to 85% at lambda = 0.90 where the tour shortens to about 1088 t0.
- **Experiment** — E29
- **Data** — `data/2_recall_cycle_snic/data/E29_cycle_ghosts.npz`
- **Code** — `code/experiments/e29_cycle_vs_ghosts.py`
- **Figure** — Figure07_invariant_circle.png (d)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E29`
- **Status** — **Established**

### `C-44` — At the critical bond the orbit creeps along the fold ghost, and the near-critical period accumulates there.

- **Where** — Sec. 5.3
- **What is asserted** — At lambda = 0.330 the velocity minimum of the critical relay coincides with the stored fold state to 6e-3 and its dwell time is 2.3 times the median.
- **Experiment** — E29
- **Data** — `data/2_recall_cycle_snic/data/E29_cycle_ghosts.npz`
- **Code** — `code/experiments/e29_cycle_vs_ghosts.py`
- **Figure** — Figure07_invariant_circle.png (e)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E29`
- **Status** — **Established**


## 6 Hyperchaos

### `C-45` — In the window between the two transitions, residual retrieval states capture only a few percent of the sampled initial-condition ensemble.

- **Where** — Sec. 6.1
- **What is asserted** — 250 initial conditions per lambda from three families, classified after 300 t0. At lambda = 0.31: 0.052 stationary retrieval (0.044 on the terminal branch), 0.47 moving, 0.47 irregular. At lambda = 0.325: 0.048 (all terminal), 0.55, 0.40. Above depinning (lambda = 0.40 and 0.90) the ordered-cycle fraction is about 80%.
- **Experiment** — E20 (basin competition), E20a-hi
- **Data** — `data/4_attractors_basins_chaos/data/E20a_basin_fractions.npz ; data/4_attractors_basins_chaos/data/E20a_basin_fractions_hi.npz ; data/4_attractors_basins_chaos/data/E20_RESULTS.md`
- **Code** — `code/experiments/e20_common.py ; code/experiments/e20a_basin_fractions.py ; code/experiments/e20a_hi.py`
- **Figure** — -  (figures/experiments/4_attractors_basins_chaos/figN_basin_competition.png)
- **Write-up** — `docs/worklogs/REPORT_attracteurs.md E20`
- **Status** — **Qualified**
- **Notes** — These fractions are properties of the stated sampling families, NOT invariant phase-space volumes; their role is only to establish coexistence. The 300 t0 classification is finite-time and deliberately separate from the asymptotic Lyapunov analysis.

### `C-46` — The irregular flow is deterministic chaos of moderate dimension with more than one unstable direction.

- **Where** — Sec. 6.2
- **What is asserted** — Largest finite-time exponent positive at all five resolved points of 0.29 <= lambda <= 0.32, between 0.040 and 0.053 t0^-1. At lambda = 0.31 the trajectory remains non-stationary over T = 3000 t0 (about 150 inverse-Lyapunov times) with largest exponent converging to 0.048 t0^-1.
- **Experiment** — E23A (nature of the chaos)
- **Data** — `data/4_attractors_basins_chaos/data/E23A_lyap.npz ; data/4_attractors_basins_chaos/data/E23A_geometry.npz`
- **Code** — `code/experiments/e23a_chaos_nature.py`
- **Figure** — Figure08_hyperchaos.png (b,c)
- **Write-up** — `docs/worklogs/REPORT_attracteurs.md E23A`
- **Status** — **Established**

### `C-47` — An eight-direction tangent computation gives three positive exponents and an effective Kaplan-Yorke dimension of about 6.4.

- **Where** — Sec. 6.2, Eq. (32)
- **What is asserted** — Spectrum (0.035, 0.011, 0.0015, -0.003, -0.008, -0.022, -0.032, -0.042) t0^-1; D_KY ~ 6.4.
- **Experiment** — E23A
- **Data** — `data/4_attractors_basins_chaos/data/E23A_lyap.npz`
- **Code** — `code/experiments/e23a_chaos_nature.py`
- **Figure** — Figure08_hyperchaos.png (b)
- **Write-up** — `docs/worklogs/REPORT_attracteurs.md E23A`
- **Status** — **Qualified**
- **Notes** — Finite-window characterization of one attractor: an order of magnitude for the dimension, not a converged extensive quantity, and not extensivity in N. The near-zero third value is the finite-time estimator of the exact neutral flow direction. NOTE: this spectrum comes from E23A and differs from the independent float64 N6 spectra quoted in C-48 - the two are different measurements of the same attractor and should not be mixed in one sentence.

### `C-48` — The hyperchaos diagnosis replicates across disorder: the attractor at (alpha, lambda, tau) = (0.05, 0.31, 10) is hyperchaotic in all five tested realizations.

- **Where** — Sec. 6.2 (replication)
- **What is asserted** — Five seeds 42-46; the three leading exponents stay positive at both dt = 0.01 and the dt = 0.005 control. Smallest third exponent over all controls: 0.01145.
- **Experiment** — N6 (multi-seed float64 Lyapunov campaign)
- **Data** — `data/v5_campaigns/reports/N6_multiseed_lyapunov.csv ; data/v5_campaigns/reports/N6_multiseed_lyapunov_summary.json ; data/v5_campaigns/runs/N6/`
- **Code** — `code/campaigns/n6_lyapunov_campaign.py ; code/campaigns/plot_n6_multiseed_lyapunov.py`
- **Figure** — Figure08_hyperchaos.png (a)
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §8 ; data/v5_campaigns/reports/N6_GATE_STATUS.md`
- **Status** — **Established**
- **Notes** — Validated at lambda = 0.31 only. Seed-42 points at lambda = 0.29 and 0.32 are sensitive to basin or time step and must not be used to claim a continuous hyperchaotic interval. Eight exponents do not close the partial sum, so the full Kaplan-Yorke dimension and the neutral flow exponent remain unresolved.

### `C-49` — The power spectrum of the leading overlap time series is continuous and broadband, excluding a noisy periodic orbit.

- **Where** — Sec. 6.2 (spectrum panel)
- **What is asserted** — Squared modulus of the DFT after mean subtraction and Hann windowing, for the pattern with the largest time-averaged |a_mu| on the chaotic trajectory at lambda = 0.31.
- **Experiment** — E23A
- **Data** — `data/4_attractors_basins_chaos/data/E23A_geometry.npz`
- **Code** — `code/experiments/e23a_chaos_nature.py`
- **Figure** — Figure08_hyperchaos.png (d)
- **Write-up** — `docs/worklogs/REPORT_attracteurs.md E23A`
- **Status** — **Established**

### `C-50` — The chaotic window is stationary: two-time correlations collapse as functions of the lag with no t/t_w aging collapse.

- **Where** — Sec. 6.3
- **What is asserted** — Waiting times t_w in {0, 50, 200, 800, 3200}. At alpha = 0.05 the total spread at lag 20 t0 is 0.009 on a normalized correlation of order one, the relative long-wait variation is 0.002 and the decorrelation time is about 2 t0. Lag-20 spreads are 0.021 at alpha = 0.08 and 0.025 at alpha = 0.10, all an order of magnitude below the conventional aging threshold 0.1.
- **Experiment** — E33 (aging test)
- **Data** — `data/4_attractors_basins_chaos/data/E33_aging_mlx.npz ; _a0p08.npz ; _a0p10.npz ; _a0p12.npz`
- **Code** — `code/experiments/e33_aging.py ; code/experiments/e33_analysis.py ; code/experiments/e33_vs_alpha.py`
- **Figure** — Figure08_hyperchaos.png (e)
- **Write-up** — `docs/worklogs/REPORT_attracteurs.md E33`
- **Status** — **Qualified**
- **Notes** — Smaller usable ensembles at higher load. The V5 register rates the stationarity/ergodicity claim as an observation, not a proof of an invariant measure; the planned float64 control (N9) was withdrawn from scope on 2026-07-30. E33 used the MLX/float32 path - state this in the methods. The figures_v2 README also records a convention: t_w = 0 is excluded from the spread because it is still inside the settle transient.


## 7 Delay and correlations

### `C-51` — Reducing the delay does not merely accelerate the orbit; it changes its nature, from a broad multi-memory packet to a reproducible one-memory front.

- **Where** — Sec. 7.1
- **What is asserted** — At lambda = 0.90: zero delay gives T1 = 1.9 +/- 1.2 (no clock), front spanning 15 pattern indices, largest analog overlap 0.16, participation number 21. At tau = 1.4 these become 2.15 +/- 0.01 (0.5% jitter), 1.3, 0.67 and 1.5.
- **Experiment** — E22 (desynchronization at small tau)
- **Data** — `data/5_short_delay/data/E22_desync.npz ; data/5_short_delay/data/E22_RESULTS.md`
- **Code** — `code/experiments/e22_desync.py`
- **Figure** — Figure09_short_delay.png (c,d)
- **Write-up** — `docs/worklogs/REPORT_faible_tau.md E22`
- **Status** — **Established**

### `C-52` — The condensation crossover is sharp and robust to disorder.

- **Where** — Sec. 7.1, Eq. (33)
- **What is asserted** — Reference crossover tau_c ~ 1.31; independent sigmoidal fits for seeds 42-46 give midpoints 1.28, 1.24, 1.34, 1.23, 1.23 - a few percent spread around a value of order the neuronal relaxation time t0 = 1.
- **Experiment** — E22 (reference) + N5A (multi-seed)
- **Data** — `data/v5_campaigns/runs/N5A/analysis.json ; data/5_short_delay/data/E22_desync.npz`
- **Code** — `code/campaigns/n5a_delay_scan.py ; code/campaigns/n5a_analyse.py ; code/experiments/e22_desync.py`
- **Figure** — Figure09_short_delay.png (a,b)
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §6 ; docs/worklogs/REPORT_faible_tau.md E22`
- **Status** — **Established**
- **Notes** — N5A exact midpoints: 1.28225, 1.23475, 1.33786, 1.23384, 1.23338. The rest of the N5A gate FAILED (most boundary lines have only two Lyapunov-resolved points instead of four), so N5A supports the coherence crossover but NOT a cycle-chaos boundary; the manuscript correctly claims only the crossover.

### `C-53` — The clock law changes at the same crossover, recovering the additive pacemaker clock above it.

- **Where** — Sec. 7.1, Eq. (34)
- **What is asserted** — T1 ~ 0.52 tau + 1.73 for tau <~ 1, and T1 ~ tau + 0.815 for tau >~ 2. Raising lambda to 0.99 does not restore the narrow front at short delay.
- **Experiment** — E19a (small-tau clock) + E22 lamcheck
- **Data** — `data/5_short_delay/data/E19a_cycle_clock.npz ; data/5_short_delay/data/E22_lamcheck.npz`
- **Code** — `code/experiments/e19a_cycle_clock.py ; code/experiments/e22_desync.py`
- **Figure** — Figure09_short_delay.png (b)
- **Write-up** — `docs/worklogs/REPORT_faible_tau.md E19/E22`
- **Status** — **Established**
- **Notes** — The limiting resource at short delay is propagation time, not directed coupling strength.

### `C-54` — Markov-correlated patterns collapse the bulk of the threshold distribution while the uncorrelated closing seam keeps the largest threshold in the sample: extreme selection survives but the selected barrier becomes an architectural defect rather than a statistical fluctuation.

- **Where** — Sec. 7.2, Eq. (35)
- **What is asserted** — At N = 1e4, P = 500, c = 0.2 the median bulk threshold collapses to 0.062 while the seam threshold stays at 0.169.
- **Experiment** — E27 (Markov video patterns)
- **Data** — `data/6_correlated_patterns/data/E27_markov_N10000.npz ; data/6_correlated_patterns/data/E27_markov.npz`
- **Code** — `code/experiments/e27_markov_thresholds.py`
- **Figure** — Figure10_markov_correlations.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E27 ; 0_worklogs/LITERATURE_correlated_patterns.md`
- **Status** — **Established**
- **Notes** — Direct for the tested seed and ring topology. At stronger correlation the localized states themselves degrade.

### `C-55` — Smooth, seam-free periodic correlations melt the discrete barriers and the depinning transition itself disappears.

- **Where** — Sec. 7.3
- **What is asserted** — At nearest-neighbour overlap q1 ~ 0.54 and N = 1e4, only 30 of 500 initialized branches retain an identifiable localized fixed point; forward fraction 1.000 at every tested smoothness and no pinned state down to lambda = 0.20.
- **Experiment** — E28 (Gaussian-process circle patterns)
- **Data** — `data/6_correlated_patterns/data/E28_gpcircle_N10000.npz ; data/6_correlated_patterns/data/E28_gpcircle.npz`
- **Code** — `code/experiments/e28_gpcircle_thresholds.py`
- **Figure** — Figure11_smooth_correlations.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E28`
- **Status** — **Qualified**
- **Notes** — 'No pinning' is a LOWER TESTED BOUND (lambda >= 0.20), not a proof for all lambda > 0. The V5 register requires this wording.


## 8 Implicit delay

### `C-56` — A layered Markovian ODE architecture, with no delay operator, leaves the observed layer with a distributed history dependence that can play the same clocking role.

- **Where** — Sec. 8.1, Eq. (36)-(37)
- **What is asserted** — Intermediate layers t0 U_k' = -U_k + J g(U_k) + lambda g(U_{k-1}); ring closure t0 U_0' = -U_0 + K_seq g(U_{K-1}).
- **Experiment** — E36 (implicit delay)
- **Data** — `data/7_implicit_delay/E36_RESULTS.md ; data/7_implicit_delay/E36_RESULTS.json`
- **Code** — `code/core/layered_chain.py ; code/experiments/e36a_regimes.py`
- **Figure** — TikZ diagram in the manuscript (Fig. 14)
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E36`
- **Status** — **Established**

### `C-57` — The recall period is linear in the number of intermediate propagation steps, with reproducible slopes across disorder.

- **Where** — Sec. 8.2, Eq. (38)-(39)
- **What is asserted** — N = 500, P = 25, beta = 20, K_lay in {4,6,8,11}, lambda in {1,2,4}, five seeds; T1 = a(lambda) + b(lambda) (K_lay - 1) with R^2 >= 0.999 in every fit. b(1) = 1.67 +/- 0.08, b(2) = 1.17 +/- 0.02, b(4) = 0.94 +/- 0.01 (seed-to-seed spread at most 5% of the slope).
- **Experiment** — E36 multi-seed
- **Data** — `data/7_implicit_delay/data/E36_MULTISEED_RESULTS.md ; data/7_implicit_delay/data/E36_multiseed_computed.json ; data/7_implicit_delay/data/E36A2_delay_seed42..46/`
- **Code** — `code/experiments/e36a2_delay.py ; code/experiments/e36_postprocess.py`
- **Figure** — Figure13_implicit_delay.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E36`
- **Status** — **Established**
- **Notes** — Reliable at N = 500 (single size for the main law); size controls run to N = 2000.

### `C-58` — Two independent estimators of the hidden propagation time reproduce the slopes.

- **Where** — Sec. 8.2
- **What is asserted** — Cross-correlation shift between layers and delay between matched relay events agree with the fitted slopes to within 3.4% in the worst case.
- **Experiment** — E36
- **Data** — `data/7_implicit_delay/data/E36_multiseed_computed.json`
- **Code** — `code/experiments/e36_postprocess.py`
- **Figure** — Figure13_implicit_delay.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E36`
- **Status** — **Established**

### `C-59` — The per-layer delay quantum decreases with the propagation coupling and saturates near 0.75 t0 at strong coupling, refuting the naive value t0.

- **Where** — Sec. 8.2, Eq. (40)
- **What is asserted** — Dense grid lambda in [1,8] (12 values x 4 depths x 5 seeds, all R^2 >= 0.999): b(lambda) ~ 0.75 + 0.91 lambda^-1.1, with per-seed plateaus b_inf in [0.73, 0.79] and exponents p in [1.0, 1.3].
- **Experiment** — E36 dense grid (AICc model comparison)
- **Data** — `data/7_implicit_delay/data/E36_b_lambda_dense_fits.json ; data/7_implicit_delay/data/E36_b_lambda_dense_fits.npz ; data/7_implicit_delay/data/E36_bdense_seed42..46.json`
- **Code** — `code/experiments/e36_postprocess.py`
- **Figure** — Figure13_implicit_delay.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E36 ; data/7_implicit_delay/data/E36_MULTISEED_RESULTS.md`
- **Status** — **Established**
- **Notes** — The results index records the fit as b(lambda) = 0.752 + 0.909 lambda^-1.14 (AICc, 5 seeds); the manuscript rounds to 0.75 + 0.91 lambda^-1.1. No formal lambda -> infinity or N -> infinity statement is demonstrated.

### `C-60` — On the coarse regime grid all five seeds are stationary below and support an ordered wave above a narrow onset.

- **Where** — Sec. 8.2
- **What is asserted** — Stationary for lambda <= 0.5, ordered wave for lambda >= 1, onset refined to the bracket (0.64, 0.70] at K_lay = 6, with a narrow irregular-switching band just below it.
- **Experiment** — E36
- **Data** — `data/7_implicit_delay/data/E36_lambda_c_refine.json ; data/7_implicit_delay/data/E36A_regimes_seed42..46/`
- **Code** — `code/experiments/e36a_regimes.py`
- **Figure** — Figure13_implicit_delay.png
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E36`
- **Status** — **Established**

### `C-61` — A minimal relay argument explains why a sub-unit plateau is expected: a layer crosses zero after t0 ln 2 ~ 0.69 t0 independently of lambda.

- **Where** — Sec. 8.2 (relay argument)
- **What is asserted** — Measured plateau b_inf ~ 0.75 sits close to and slightly above this switching bound.
- **Experiment** — -- (interpretation of E36)
- **Data** — `data/7_implicit_delay/data/E36_b_lambda_dense_fits.json`
- **Code** — `--`
- **Figure** — -
- **Write-up** — `docs/worklogs/CYCLE_worklog.md E36`
- **Status** — **Qualified**
- **Notes** — Explicitly flagged in the manuscript as an interpretation; what is established numerically is the linear depth law and the saturating quantum.


## 9 Conclusion

### `C-62` — Summary claim: the bulk of the threshold law ends extensive static recall while its upper extreme selects the bottleneck memory in all tested realizations and predicts the operational depinning boundary at 1e-3 resolution.

- **Where** — Conclusion
- **What is asserted** — See C-24, C-26, C-28.
- **Experiment** — N1 + N3 + E24/E30
- **Data** — `see C-24, C-26, C-28`
- **Code** — `see C-24, C-26, C-28`
- **Figure** — Figure04, Figure05
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md`
- **Status** — **Established**
- **Notes** — The conclusion must keep the words 'operational' onset and 'leading predictor', per the V2/V3 audit resolution.

### `C-63` — Ablation experiments identify two necessary ingredients: a delay-localized traveling front and discrete heterogeneous barriers.

- **Where** — Conclusion
- **What is asserted** — See C-51 to C-55.
- **Experiment** — E22/N5A + E27 + E28
- **Data** — `see C-51 to C-55`
- **Code** — `see C-51 to C-55`
- **Figure** — Figure09, Figure10, Figure11
- **Write-up** — `docs/worklogs/CYCLE_worklog.md`
- **Status** — **Established**


## EXCLUDED

### `X-01` — Discrete-time / MCMC dynamics and the invalidated float32 N=10000 run.

- **Where** — not in the manuscript
- **What is asserted** — Refuted or invalid.
- **Experiment** — pre-E campaigns
- **Data** — `[local archive] obsolete/05_invalid_or_refuted/9_obsolete_ne_pas_citer/`
- **Code** — `--`
- **Figure** — -
- **Write-up** — `[local archive] obsolete/05_invalid_or_refuted/9_obsolete_ne_pas_citer/SYNTHESIS_WALKTHROUGH.md`
- **Status** — **Limitation / not established**
- **Notes** — NEVER CITE. Kept only for audit traceability.

### `X-02` — Long-delay Hopf instability of the memory branch (N5B).

- **Where** — not in the manuscript
- **What is asserted** — No imaginary-axis crossing over 200/200 spectral points and five seeds; the line was withdrawn from the manuscript scope on 2026-07-30.
- **Experiment** — N5B
- **Data** — `data/v5_campaigns/runs/N5B/`
- **Code** — `code/campaigns/n5b_long_delay.py`
- **Figure** — -
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §7`
- **Status** — **Limitation / not established**
- **Notes** — Do not claim a delay-induced Hopf transition. Archived for traceability.

### `X-03` — High-load moving-state phase map (N8) and the MHN/video proof of concept (E35).

- **Where** — not in the manuscript
- **What is asserted** — N8: 96/100 trajectories strictly sequential in 19/20 cells, but Lyapunov data cover one seed only. E35: NO-GO on interpolation, kept as a POC.
- **Experiment** — N8, E35
- **Data** — `data/v5_campaigns/runs/N8/ ; 3_numerics/results/8_mhn_video/`
- **Code** — `code/campaigns/n8_highload.py ; code/experiments/e35v1_eval.py`
- **Figure** — -
- **Write-up** — `data/v5_campaigns/SYNTHESE_NUMERICAL_WORKPLAN.md §9 ; docs/worklogs/CYCLE_worklog.md E35`
- **Status** — **Limitation / not established**
- **Notes** — N8 must not be presented as a definitive phase map or as basin volumes. E35 is a separate project, out of scope for this manuscript.


---

## Figure provenance

| Figure | Status | Byte-identical twin in the folder | Build script |
|---|---|---|---|
| Figure00 | traceable | `figures/panels/Panel00_model_architecture.png  (identical)` | `code/figures/v2/p_model.py` |
| Figure01 | check filename | `figures/panels/phases MCMC.png  (identical)` | `see walkthrough.md in the source folder` |
| Figure02 | traceable | `figures/panels/Panel02_global_bifurcation.png  (identical)` | `code/figures/build_panels.py  (panel 'Panel02')` |
| Figure03 | traceable | `figures/panels/Panel03_static_folds_memory_distortion.png (identical)` | `code/figures/build_panels.py` |
| Figure04 | partial - rebuild script missing | `NONE - unique in the folder (closest ancestor: figures/panels/Panel04_threshold_distributions.png, 5400x3763)` | `UNKNOWN - final re-layout script not present in the Chicago folder` |
| Figure05 | traceable | `figures/panels/Panel07_extreme_selection.png  (identical)` | `code/figures/build_panels.py ; code/campaigns/plot_n1_static_dynamic_equality.py` |
| Figure06 | partial - rebuild script missing | `NONE - unique in the folder (ancestor: figures/panels/Panel05_snic_characterization.png, 5400x3453)` | `UNKNOWN for the final assembly; panels (a)-(f) come from code/figures/build_panels.py` |
| Figure07 | traceable | `figures/panels/Panel08_invariant_circle.png  (identical)` | `code/figures/build_panels.py` |
| Figure08 | partial - rebuild script missing | `NONE - unique in the folder (ancestors: figures/panels/Panel01_lyapunov_chaos.png, 5400x2869` | `UNKNOWN for the final assembly; components from build_panels.py and figures_v2/p_chaos.py` |
| Figure09 | traceable | `figures/panels/Panel06_short_delay.png  (identical)` | `code/figures/build_panels.py` |
| Figure10 | traceable | `figures/experiments/6_correlated_patterns/figX_markov_video_N10000.png  (identical, plus 9 other copies)` | `code/experiments/e27_markov_thresholds.py` |
| Figure11 | traceable | `figures/experiments/6_correlated_patterns/figY_gpcircle_video_N10000.png  (identical, plus 8 other copies)` | `code/experiments/e28_gpcircle_thresholds.py` |
| Figure13 | traceable | `[local archive] 4_campaigns/manuscript_audits/agent_redacteur/figures/E36_implicit_delay.png (identical)` | `code/experiments/e36_postprocess.py` |

Full per-figure detail: `FIGURE_PROVENANCE.md`.
