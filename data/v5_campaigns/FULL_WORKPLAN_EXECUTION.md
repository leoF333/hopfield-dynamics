# Full numerical workplan execution and ETA

Updated: 2026-07-30 20:25 CDT.

This file extends `NEXT_24H_PLAN.md` through the numerical items retained by the
user.  On 2026-07-29, the user explicitly removed N7, N9 and N10 from scope.
The historical Desktop directory remains strictly read-only.

## Statistical inclusion policy

A disorder realization is never excluded merely because its measured value is
far from the other seeds.  Exclusion from a fit or headline count requires a
criterion independent of the discrepancy being tested, such as:

- failed convergence or residual/precision gate;
- insufficient observation time or censored dynamics;
- unresolved static-branch identity;
- a demonstrably different attractor or basin when the claim concerns one
  specified attractor class;
- a predeclared degeneracy or near-tie that makes the selected observable
  non-identifiable at the numerical resolution.

Every excluded realization remains in the audit table with its raw result and
reason.  Where useful, the report gives both the full-sample and admissible-set
statistics.  No post-hoc removal is allowed solely to improve a fit.

## Completion state

The retained numerical path is complete.  The placeholder-free package combines
validated historical results with completed N1, N2, N3 and N6 outputs in:

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5/figures_ready`

The manuscript compiles successfully with these figures. Active simulations continue
restart-safely, but no pending campaign blocks the availability of the core figure
set.

1. N2 was refreshed on the stable N1 terminal branch while reusing all dynamic
   trajectories.
2. Its fit/window controls and scientific gate were completed.
3. The retained figure was generated, the manuscript recompiled and all affected
   pages visually inspected.

On 2026-07-30 the user explicitly removed the remaining N8 Lyapunov and
multi-tour calculations from scope.  The already acquired N8 trajectories and
preliminary audit are preserved, but no new N8 simulation is to be launched.

The 360-cell N5B pinned scan and its completed audits remain archived, but the
user explicitly stopped this direction on 2026-07-30 and decided not to discuss
the Hopf margin in the manuscript.  No further N5B calculation is in scope.

Remaining heavy numerical wall time: **zero**.

## Retained conclusions

Final aggregation is available for N1, corrected N2, N3 and N6.  N5A remains
qualified rather than promoted because its predeclared completion gate failed.
The preliminary N8 audit is preserved, but no further N8 gate is pursued.

N4 is not a numerical experiment. It is the publication-production task that
recomposes existing outputs into consistent article figures, derived tables and
captions. Its essential Figures 1--4 are now complete; further regeneration occurs
only when a newly completed numerical result passes its gate.

N7 (additional multi-seed correlation simulations), further N8 Lyapunov and
multi-tour calculations, N9 (stationarity/ergodicity campaign) and N10 (final
reproducibility reruns) are explicitly out of scope.

## Total ETA

- Corrected N2 refresh, analysis and aggregation: **complete**.
- Final figure regeneration, compilation and inspection: **complete**.
- Total retained heavy work: **complete**.

N4 figure production is complete for the core manuscript and is excluded from this
calculation estimate.

## Resource and restart rules

- At most four single-threaded CPU processes.
- Never overlap two heavy scientific campaigns.
- Smoke test and benchmark the relevant high-\(N\) kernel before every new
  production family.
- Preserve atomic per-cell outputs and checkpoints.
- Treat two consecutive hourly checks without progress as the stagnation
  threshold.
- Do not force classifications, exponents, fits or gates.
