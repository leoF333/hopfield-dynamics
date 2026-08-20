# Experiment index


> **Paths.** This copy is path-remapped for this repository. Entries marked
> `[local archive]` live only in the full research archive, not here.
Catalogue of every numerical campaign referenced by the manuscript, with where its code,
data, figures and write-up now live. Paths are relative to the repository root.

Two independent generations of work coexist and must not be confused:

- **`E**` — historical campaigns** (April–July 2026). Code in `code/experiments/`, data and
  figures in `3_numerics/results/<theme>/`, written up in
  `docs/worklogs/CYCLE_worklog.md` and the three `REPORT_*.md`.
- **`N**` — V5 verification campaigns** (late July 2026). A separate, checkpointed,
  float64 reimplementation written specifically to re-test the load-bearing claims against
  an independent code path. Everything lives under `data/v5_campaigns/`,
  written up in `SYNTHESE_NUMERICAL_WORKPLAN.md` and `RUN_STATUS.md`.

The V5 campaigns deliberately treat the historical tree as read-only input, and record its
source hashes in `data/v5_campaigns/manifests/reference_baseline.json`.

---

## Historical campaigns (E)

| ID | Subject | Theme folder | Figure | Used by |
|---|---|---|---|---|
| E1 | Descending Floquet scan of the recall cycle | `2_cycle_rappel_snic` | figA, figB | C-35, C-36 |
| E2 | N-dependence of the exponents | `2_cycle_rappel_snic` | — | control |
| E3–E4 | T₁ toward λ_c⁺; collapse T₁ = τ + t_esc | `2_cycle_rappel_snic` | figA | C-35 |
| E5 | Seeded descending sweep (hysteresis) | `2_cycle_rappel_snic` | — | decisive control |
| E6 | Bracketing λ\* + static machinery | `2_cycle_rappel_snic` | — | C-29 |
| E7 | Attractor census / spin-glass test at λ=0.31 | `2_cycle_rappel_snic` | figC | C-45 |
| E8 | Triple discriminant | `2_cycle_rappel_snic` | figC | C-45 |
| E9 | Per-bond statistics near λ\* — **verdict** | `2_cycle_rappel_snic` | figB, figC | C-30, C-31 |
| E10 | Trapping position vs quench depth | `2_cycle_rappel_snic` | — | — |
| E11 | Complete Floquet spectra (complex plane) | `2_cycle_rappel_snic` | figE_v2 | C-36 |
| E12 | Spectral reading of the pinned-front saddle-node | `3_unification_seuils_scaling` | figF | C-33 |
| E13 | Status of "bond 91": universality vs sample | `2_cycle_rappel_snic` | figE_v2 | C-36 |
| E14 | Floor-free contraction per tour A(λ) = −ln\|μ₁\| | `2_cycle_rappel_snic` | figE_v2 | C-36 |
| E15 | Delayed spectrum of the pinned front vs τ | `3_unification_seuils_scaling` | figG | C-11 |
| E17 | Nonlinear probe of the memory Hopf window | `1_bifurcation_statique` | figH_v2 | C-11 |
| E18 | Birth of pinned fronts; basins | `3_unification_seuils_scaling`, `4_attracteurs…` | figJ, figK | C-45 |
| E19 | Small-τ regime: how the cycle dies | `5_petit_tau` | figL, figM | C-53 |
| E20 | Basin competition: memory vs front vs cycle | `4_attracteurs_bassins_chaos` | figN, figO | C-45 |
| E21 | Nature and position of front birth vs τ | `3_unification_seuils_scaling` | figS, figT | C-18 |
| E22 | Small-τ desynchronization (fine quantification) | `5_petit_tau` | figP | C-51, C-52, C-53 |
| E23A | Nature of the chaos (Lyapunov, D_KY, PCA, spectrum) | `4_attracteurs_bassins_chaos` | figQ | C-46, C-47, C-49 |
| E23B | Cycle–chaos frontier, SNIC paradox | `5_petit_tau` | figR | context |
| E24 | **Per-motif thresholds λ_c(μ)**: shape, finite size, extremes | `3_unification_seuils_scaling` | figU, figV | C-12–C-19, C-26 |
| E25 | Does the shape at the fold depend on μ? | `3_unification_seuils_scaling` | figW | C-15, C-16, C-23 |
| E26 | Does the threshold law depend on P/α? | `3_unification_seuils_scaling` | figZ | C-20, C-21 |
| E27 | Correlated patterns I: Markov flip chain | `6_motifs_correles` | figX | C-54 |
| E28 | Correlated patterns II: Gaussian process on the circle | `6_motifs_correles` | figY | C-55 |
| E29 | What the cycle passes through just above λ\* | `2_cycle_rappel_snic` | figAA | C-42, C-43, C-44 |
| E30 | **Threshold scaling campaign** (303 resumable runs) | `3_unification_seuils_scaling` | figAE | C-22, C-24, C-25 |
| E31 | Per-motif memory survival f_mem(α,λ) | `4_attracteurs_bassins_chaos` | — | context |
| E32 | (α,λ) basin phase diagram (GPU) | `4_attracteurs_bassins_chaos` | figAB | context |
| E33 | Aging of the mixture chaos: none — SRB measure | `4_attracteurs_bassins_chaos` | figAC, figAD | C-50 |
| E34 | Invariant circle / necklace (ET0, ET1″) | `2_cycle_rappel_snic` | figE34 | C-37–C-41 |
| E35 | Modern Hopfield + video POC | `8_mhn_video` | E35_* | **excluded** (X-03) |
| E36 | Implicit delay through layered depth | `7_tau_implicite` | E36_* | C-56–C-61 |
| E37 | Correlated patterns III: modern-Hopfield component | `6_motifs_correles` | figE37 | out of scope |

Full protocols, dates, run times and per-experiment verdicts:
`docs/worklogs/CYCLE_worklog.md` §6 and the appendices.

Thematic reports:

- `docs/worklogs/REPORT_static_bifurcation.md` — static folds, λ_c(α),
  τ-independence, Hopf crossover.
- `docs/worklogs/REPORT_attracteurs.md` — basins, nature of the chaos,
  SRB / no aging.
- `docs/worklogs/REPORT_faible_tau.md` — small-delay regime.
- `docs/worklogs/FLOQUET_methodology.md` — Floquet algorithm and its
  validations.

---

## V5 verification campaigns (N)

| ID | Subject | Verdict | Used by |
|---|---|---|---|
| **N1** | Static extreme vs independent dynamic onset, 20 disorder realizations | **Validated** — motif identity 20/20; 19/19 admissible seeds agree at 1e-3 | C-28 |
| **N2** | Two-sided local laws through the fold, 5 seeds | Partial — square-root compatible, logarithm rejected; strict exponent gate not passed | C-32 |
| **N3** | Seed-aware reanalysis of the 303 archived E30 files | **Analysis complete** — b = 0.43078, CI [0.39638, 0.46520] | C-22, C-24, C-25, C-26 |
| N4 | Figure assembly and traceability (no new physics) | Provenance in `manifests/N4_figure_provenance.json` | figures |
| **N5A** | Multi-seed short-delay coherence | Crossover validated (τ ≈ 1.23–1.34); cycle–chaos boundary NOT resolved | C-52 |
| N5B | Long-delay spectrum near the fold | 200/200 points, no axis crossing — **withdrawn from scope 2026-07-30** | X-02 |
| **N6** | Multi-seed float64 Lyapunov spectra | **Validated at λ = 0.31** — ≥3 positive exponents in 5 realizations | C-48 |
| N7 | Correlated patterns (re-run) | Out of scope by explicit decision | — |
| N8 | High-load moving states | Trajectory result robust; dynamical classification provisional | X-03 |
| N9 | Float64 stationarity / operational ergodicity | Out of scope by explicit decision | X-03 |
| N10 | Release audit | Out of scope by explicit decision | — |

Reproduction commands: `data/v5_campaigns/RUNBOOK.md`.
Execution state and incident log: `data/v5_campaigns/RUN_STATUS.md`.

### Numerical incidents detected and corrected

Recorded in `SYNTHESE_NUMERICAL_WORKPLAN.md` §10. They matter because each one would
otherwise have turned a code artefact into a physical conclusion.

| Campaign | Problem | Treatment |
|---|---|---|
| N1 | At 1e-4 the discrepancy depends on bracket resolution, history and sometimes the stationary branch reached | Claim limited to 1e-3; branch and history controls kept |
| N2 | Newton started exactly at the fold could select the unstable saddle | Invalid outputs archived under `runs/N2_INVALID_STATIC_BRANCH_20260729/`; continuation corrected |
| N5B | Smoke-test checkpoint collided with the production file | Separate `_smoke` and production namespaces |
| N5B | Comparing opposite members of a conjugate pair created false M=48/64 disagreements | Pairing made conjugation-invariant |
| N8 | The historical label required two full tours and could be overwritten on merge | Classification redone from trajectories with an explicit independent criterion |

---

## Material that must never be cited

`[local archive] obsolete/05_invalid_or_refuted/9_obsolete_ne_pas_citer/` — discrete-time and MCMC
dynamics, and the invalidated float32 N=10000 run. Kept only for audit traceability.
See `SYNTHESIS_WALKTHROUGH.md` in that folder (marked REFUTED).
