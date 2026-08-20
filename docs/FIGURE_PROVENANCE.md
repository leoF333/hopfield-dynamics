# Figure provenance — working draft V4

*Generated 2026-08-19. Companion to `CLAIMS.md`.*


> **Paths.** This copy is path-remapped for this repository. Entries marked
> `[local archive]` live only in the full research archive, not here.
Each of the 13 figures shipped in `figures/manuscript/` was hashed (SHA-256) and compared against every PNG in the folder. A **byte-identical twin** means the build chain is fully recoverable; where no twin exists, the figure is a later re-layout whose assembly script was produced outside this folder, and only component-level provenance is available.

| Figure | Section | Provenance status |
|---|---|---|
| `Figure00` | Sec. 1.2 | traceable |
| `Figure01` | Sec. 1.4 | check filename |
| `Figure02` | Sec. 2 | traceable |
| `Figure03` | Sec. 2 | traceable |
| `Figure04` | Sec. 3 | partial - rebuild script missing |
| `Figure05` | Sec. 4 | traceable |
| `Figure06` | Sec. 4 | partial - rebuild script missing |
| `Figure07` | Sec. 5 | traceable |
| `Figure08` | Sec. 6 | partial - rebuild script missing |
| `Figure09` | Sec. 7.1 | traceable |
| `Figure10` | Sec. 7.2 | traceable |
| `Figure11` | Sec. 7.3 | traceable |
| `Figure13` | Sec. 8 | traceable |

---

## Figure00 — `Figure00_model_architecture.png`

- **Manuscript section** — Sec. 1.2
- **SHA-256** — `e7fb3c5b9d462e438931b1c3cc7b971bed5885de7a747ebda0b67930902b922a`
- **Dimensions** — 5389x3113
- **Identical copy in the folder** — figures/panels/Panel00_model_architecture.png  (identical) ; code/figures/v2/Panel00_model_architecture.png
- **Build script** — `code/figures/v2/p_model.py`
- **Data and component sources** — (b,c) data/2_recall_cycle_snic/data/cycle_lam0.9_tau10.0_N2000.npz (a_grid) ; (d) data/3_threshold_scaling/data/E24_curves_N2000_s42.npz
- **Claims displayed** — C-04, C-05
- **Status** — **traceable**
- **Notes** — Panel (d): bars beyond index 1 are illustrative, not measured.

## Figure01 — `Figure01_phase_diagram.png`

- **Manuscript section** — Sec. 1.4
- **SHA-256** — `3284cb1ad6c27c81f3bad12085a4a06bad2345edd0b0e71b597eb16c6f253965`
- **Dimensions** — 2038x606
- **Identical copy in the folder** — figures/panels/phases MCMC.png  (identical)
- **Build script** — `see walkthrough.md in the source folder`
- **Data and component sources** — [local archive] obsolete/03_legacy_code/Projet_Chicago/stabilité/système continu/phases & comparaison/phases continu/
- **Claims displayed** — C-06
- **Status** — **check filename**
- **Notes** — The identical twin is stored under the name 'phases MCMC.png' although the manuscript describes Euler integration of the continuous-time model. Confirm and rename before release.

## Figure02 — `Figure02_global_bifurcation.png`

- **Manuscript section** — Sec. 2
- **SHA-256** — `de98a4374358034e10e3bec9360081b1779057ec03d11c45ed39c3e5299d51d2`
- **Dimensions** — 5400x2869
- **Identical copy in the folder** — figures/panels/Panel02_global_bifurcation.png  (identical)
- **Build script** — `code/figures/build_panels.py  (panel 'Panel02')`
- **Data and component sources** — (a) figure_sources/diagramme de bifurcation v1.png ; (b,c,d) panels (a),(b),(c) of figure_sources/fig1_mechanism.png ; underlying arrays data/3_threshold_scaling/data/E24_thr_N2000_P100_s42.npz and E30_scaling/
- **Claims displayed** — C-12, C-27
- **Status** — **traceable**
- **Notes** — Crop coordinates and source SHA-256 recorded in figures/panel_manifest.json.

## Figure03 — `Figure03_static_folds.png`

- **Manuscript section** — Sec. 2
- **SHA-256** — `b849956fcd3c52ce41c2118b28044908f9a2e8665567577f9d2d11096dcdf255`
- **Dimensions** — 5400x2769
- **Identical copy in the folder** — figures/panels/Panel03_static_folds_memory_distortion.png (identical)
- **Build script** — `code/figures/build_panels.py`
- **Data and component sources** — (a) left panel of figure_sources/figS_front_birth.png (E21) ; (b) panel (a) of figure_sources/N4_Figure2.png ; (c) upper-left of figure_sources/figU_branch_geometry.png (E24a) ; (d,e) both panels of figure_sources/fold_N10000_a0.05.png
- **Claims displayed** — C-13, C-14, C-15, C-16, C-17, C-18
- **Status** — **traceable**

## Figure04 — `Figure04_thresholds.png`

- **Manuscript section** — Sec. 3
- **SHA-256** — `77f2095c580b8e6305570ef7a460a8430d647e2828f62781a9f4ee9414e8f63d`
- **Dimensions** — 5400x2380
- **Identical copy in the folder** — NONE - unique in the folder (closest ancestor: figures/panels/Panel04_threshold_distributions.png, 5400x3763)
- **Build script** — `UNKNOWN - final re-layout script not present in the Chicago folder`
- **Data and component sources** — Component-level provenance IS recoverable from figures/panel_manifest.json: (a,b) two left panels of figure_sources/figZ_alpha_law.png (E26) ; (c,d,f) three panels of figure_sources/figAE_threshold_scaling.png (E30) ; (e,g) panels (a),(b) of figure_sources/N4_Figure3.png ; plus N3 seed-aware panels from figure_sources/N3_E30_seed_aware.png
- **Claims displayed** — C-20, C-21, C-22, C-24, C-25, C-26
- **Status** — **partial - rebuild script missing**
- **Notes** — ACTION: the shipped figure is a re-layout of Panel04 (7 sub-panels instead of 10, different height). All components trace to E26/E30/N3 arrays, but the assembly script was produced outside this folder. Either recover it or regenerate the figure from build_panels.py components.

## Figure05 — `Figure05_extreme_selection.png`

- **Manuscript section** — Sec. 4
- **SHA-256** — `f2552b846f6b1fba890fb73916846c085ff0add7ea2df0b809956a81d524238b`
- **Dimensions** — 5400x2419
- **Identical copy in the folder** — figures/panels/Panel07_extreme_selection.png  (identical)
- **Build script** — `code/figures/build_panels.py ; code/campaigns/plot_n1_static_dynamic_equality.py`
- **Data and component sources** — (a) panel (c) of figure_sources/N4_Figure4.png ; (b,c) regenerated from data/v5_campaigns/reports/N1_static_extreme_vs_dynamic_onset.csv
- **Claims displayed** — C-28, C-29
- **Status** — **traceable**
- **Notes** — Uses the 19 rows pre-marked admissible_for_equality_test; seed 48 is absent by the documented admissibility rule.

## Figure06 — `Figure06_snic.png`

- **Manuscript section** — Sec. 4
- **SHA-256** — `fc05ad7ab53d5e1ca0692ea1cf50575e0bde751f62ed395bd93e98d4283ddf97`
- **Dimensions** — 5400x5054
- **Identical copy in the folder** — NONE - unique in the folder (ancestor: figures/panels/Panel05_snic_characterization.png, 5400x3453)
- **Build script** — `UNKNOWN for the final assembly; panels (a)-(f) come from code/figures/build_panels.py`
- **Data and component sources** — (a) N2 critical slowing replotted from data/v5_campaigns/runs/N2/*production.json (seeds 42, 47, 52, 60; seed 51 deliberately absent) ; (b) cycle-side slowing from data/2_recall_cycle_snic/data/E29_cycle_ghosts.npz ; (c,e) data panels of figure_sources/figB_depinning_transition.png ; (d) static softening from the N2 production JSON files ; (f) pacemaker decomposition T1 = tau + t_esc ; (g) ADDED: figures/experiments/2_recall_cycle_snic/figE_floquet_spectrum_v2.png (E11/E13)
- **Claims displayed** — C-30, C-31, C-32, C-33, C-34, C-35, C-36
- **Status** — **partial - rebuild script missing**
- **Notes** — Verified visually during this reorganisation: the shipped figure is Panel05 (a)-(f) with a new row (g) carrying the two Floquet panels of figE_floquet_spectrum_v2.png. Note that a DIFFERENT and more careful Floquet component exists - figures_v2/component_floquet_hyperstability.png, which plots contraction per turn instead of the floored multiplier modulus. Consider using it: figures_v2/README.md explains why the floored quantity must not be plotted as a measurement.

## Figure07 — `Figure07_invariant_circle.png`

- **Manuscript section** — Sec. 5
- **SHA-256** — `c29f9e510d43b851a211381c7f5c53a612e9f04dfb60ab86011180b1b0dc99cb`
- **Dimensions** — 5400x2639
- **Identical copy in the folder** — figures/panels/Panel08_invariant_circle.png  (identical)
- **Build script** — `code/figures/build_panels.py`
- **Data and component sources** — (a,b,c) panels (a),(b),(d) of figure_sources/figE34_invariant_circle.png ; (d,e) the two lower panels of figures/experiments/2_recall_cycle_snic/figAA_cycle_vs_ghosts.png (E29) ; arrays data/2_recall_cycle_snic/data/E34/ and data/2_recall_cycle_snic/data/E29_cycle_ghosts.npz
- **Claims displayed** — C-37, C-38, C-40, C-42, C-43, C-44
- **Status** — **traceable**

## Figure08 — `Figure08_hyperchaos.png`

- **Manuscript section** — Sec. 6
- **SHA-256** — `58593da2533c6e5867195a6c032d14eba72ae0fe775d2025685c47535457b453`
- **Dimensions** — 5400x4358
- **Identical copy in the folder** — NONE - unique in the folder (ancestors: figures/panels/Panel01_lyapunov_chaos.png, 5400x2869 ; figures_v2/Panel01_lyapunov_chaos_v2.png, 5630x4047)
- **Build script** — `UNKNOWN for the final assembly; components from build_panels.py and figures_v2/p_chaos.py`
- **Data and component sources** — (a) five-realization Lyapunov spectra regenerated from data/v5_campaigns/runs/N6/*.npz (seeds 42-46) ; (b,c) panels (a),(c) of figure_sources/figQ_chaos_nature.png (E23A) ; (d) power spectrum panel (f) of figQ_chaos_nature.png ; (e) two-time correlations from data/4_attractors_basins_chaos/data/E33_aging_mlx*.npz
- **Claims displayed** — C-46, C-47, C-48, C-49, C-50
- **Status** — **partial - rebuild script missing**
- **Notes** — Verified visually during this reorganisation. figures_v2/p_chaos.py builds a richer v2 panel (adds convergence of Lambda1, return map, PCA projection, aging vs load, basin ownership); the shipped figure keeps five sub-panels.

## Figure09 — `Figure09_short_delay.png`

- **Manuscript section** — Sec. 7.1
- **SHA-256** — `495c3a8cb845552e25c2f4aaa54b108a7d4631ddc672f4ada5375d89c8c3d0c3`
- **Dimensions** — 5400x2569
- **Identical copy in the folder** — figures/panels/Panel06_short_delay.png  (identical)
- **Build script** — `code/figures/build_panels.py`
- **Data and component sources** — (a) left panel of figure_sources/N5A_delay_boundary.png ; (b,c,d) the three panels of figures/experiments/5_short_delay/figP_desync.png (E22) ; arrays data/5_short_delay/data/E22_desync.npz and data/v5_campaigns/runs/N5A/analysis.json
- **Claims displayed** — C-51, C-52, C-53
- **Status** — **traceable**
- **Notes** — The implicit-delay panels are intentionally absent from this figure.

## Figure10 — `Figure10_markov_correlations.png`

- **Manuscript section** — Sec. 7.2
- **SHA-256** — `56e54c1df8aa1a8c6c81a638bfa734cb3627ba4657f8fed91711b1e106e33608`
- **Dimensions** — 2323x1618
- **Identical copy in the folder** — figures/experiments/6_correlated_patterns/figX_markov_video_N10000.png  (identical, plus 9 other copies)
- **Build script** — `code/experiments/e27_markov_thresholds.py`
- **Data and component sources** — data/6_correlated_patterns/data/E27_markov_N10000.npz
- **Claims displayed** — C-54
- **Status** — **traceable**
- **Notes** — Direct experiment output, not a composite: fully reproducible.

## Figure11 — `Figure11_smooth_correlations.png`

- **Manuscript section** — Sec. 7.3
- **SHA-256** — `95eb11518f608c8edc3f58f42777ea40e1c6b09e18cd936c8844368ff333d09f`
- **Dimensions** — 2324x1618
- **Identical copy in the folder** — figures/experiments/6_correlated_patterns/figY_gpcircle_video_N10000.png  (identical, plus 8 other copies)
- **Build script** — `code/experiments/e28_gpcircle_thresholds.py`
- **Data and component sources** — data/6_correlated_patterns/data/E28_gpcircle_N10000.npz
- **Claims displayed** — C-55
- **Status** — **traceable**
- **Notes** — Direct experiment output, not a composite: fully reproducible.

## Figure13 — `Figure13_implicit_delay.png`

- **Manuscript section** — Sec. 8
- **SHA-256** — `2fea51ab1a95dc5da731c86ab6c4ac4777d9dfa126abc7c4d5a2130779f32f8e`
- **Dimensions** — 2105x660
- **Identical copy in the folder** — [local archive] 4_campaigns/manuscript_audits/agent_redacteur/figures/E36_implicit_delay.png (identical) ; 6_presentations/paper_pi_beamer/assets/E36_implicit_delay.png
- **Build script** — `code/experiments/e36_postprocess.py`
- **Data and component sources** — data/7_implicit_delay/data/E36_multiseed_computed.json ; data/7_implicit_delay/data/E36_b_lambda_dense_fits.json ; data/7_implicit_delay/data/E36A2_delay_seed42..46/
- **Claims displayed** — C-57, C-58, C-59
- **Status** — **traceable**
- **Notes** — There is no Figure12 in the manuscript: the numbering skips it because the layered-architecture diagram (Fig. 14 in the text) is drawn in TikZ.
