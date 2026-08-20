# N8 — preliminary high-load audit

## Scope and integrity

- All 100 expected trajectories are present and readable: 4 loads, 5 values of
  non-reciprocity, and 5 independent disorder seeds.
- Each cell uses one prescribed retrieval-like initial history per disorder
  realization.  Fractions across seeds are therefore **not basin-volume
  estimates**.
- The stored legacy `classification` field is not used.  It requires two
  complete tours, which is impossible in most high-load records of fixed length,
  and the worker summary merge order can overwrite the Lyapunov-aware label.

## Result from the trajectory data

- A strict sequential trajectory is defined independently of the desired
  conclusion by at least 50 winner changes, nearest-neighbour
  forward fraction above 0.99, reversal fraction below
  0.01, and a nonzero fixed-point residual.
- 96/100 trajectories satisfy this strict criterion.  In
  19/20 parameter cells, all
  five disorder realizations show strict sequential retrieval.
- The only heterogeneous cell is $(\alpha,\lambda)=(0.15,0.35)$:
  seed 44 is a strict sequence; seed 42 is a degraded, jumpy forward state;
  seeds 43 and 46 are irregular non-sequential states; seed 45 is a fixed
  point.  No seed is discarded.
- For every strict sequence, the event count is stationary over the four
  quarters of the observation window: the smallest quarter-to-quarter
  count ratio is 0.96.
- Conditional on strict sequential retrieval, the local relay speed is governed
  mainly by $\lambda$ and depends only weakly on load.  Retrieval quality
  decreases with load and improves with $\lambda$.

## Lyapunov evidence currently available

- Lyapunov data exist for 19/96 strict trajectories:
  seed 46 in 19 cells.  There is no Lyapunov run at the heterogeneous corner
  because seed 46 is not a moving sequence there.
- The stored cumulative leading exponents at time 1000 range from
  -0.008271 to -0.002940.  Their mean over the last 250 time
  units ranges from -0.001341 to +0.001202; none exceeds the
  predeclared positive threshold 0.002.
- The cumulative estimates decay toward zero and the late-window means resolve
  no positive exponent.  For this one disorder realization, the moving states
  are therefore nonchaotic within numerical resolution and are compatible with
  the neutral direction of a stable sequential limit cycle.
- This is not yet a multi-seed classification.  The other 77 strict moving
  trajectories still need their leading Lyapunov exponent.  A long record
  spanning multiple full tours is also needed to certify periodic closure at
  $\alpha\geq0.09$.

## Cell-level summary

| $\alpha$ | $\lambda$ | strict sequences | fixed | degraded | irregular | $v$ | binary overlap | Lyapunov |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.35 | 5/5 | 0 | 0 | 0 | 0.06137 | 0.939 | 1/5 |
| 0.05 | 0.40 | 5/5 | 0 | 0 | 0 | 0.07200 | 0.951 | 1/5 |
| 0.05 | 0.50 | 5/5 | 0 | 0 | 0 | 0.08188 | 0.967 | 1/5 |
| 0.05 | 0.70 | 5/5 | 0 | 0 | 0 | 0.08937 | 0.987 | 1/5 |
| 0.05 | 0.90 | 5/5 | 0 | 0 | 0 | 0.09250 | 0.984 | 1/5 |
| 0.09 | 0.35 | 5/5 | 0 | 0 | 0 | 0.06537 | 0.911 | 1/5 |
| 0.09 | 0.40 | 5/5 | 0 | 0 | 0 | 0.07312 | 0.930 | 1/5 |
| 0.09 | 0.50 | 5/5 | 0 | 0 | 0 | 0.08188 | 0.952 | 1/5 |
| 0.09 | 0.70 | 5/5 | 0 | 0 | 0 | 0.08875 | 0.969 | 1/5 |
| 0.09 | 0.90 | 5/5 | 0 | 0 | 0 | 0.09212 | 0.977 | 1/5 |
| 0.13 | 0.35 | 5/5 | 0 | 0 | 0 | 0.06413 | 0.875 | 1/5 |
| 0.13 | 0.40 | 5/5 | 0 | 0 | 0 | 0.07187 | 0.904 | 1/5 |
| 0.13 | 0.50 | 5/5 | 0 | 0 | 0 | 0.08063 | 0.937 | 1/5 |
| 0.13 | 0.70 | 5/5 | 0 | 0 | 0 | 0.08812 | 0.961 | 1/5 |
| 0.13 | 0.90 | 5/5 | 0 | 0 | 0 | 0.09187 | 0.967 | 1/5 |
| 0.15 | 0.35 | 1/5 | 1 | 1 | 2 | 0.06250 | 0.841 | 0/5 |
| 0.15 | 0.40 | 5/5 | 0 | 0 | 0 | 0.07075 | 0.884 | 1/5 |
| 0.15 | 0.50 | 5/5 | 0 | 0 | 0 | 0.08063 | 0.926 | 1/5 |
| 0.15 | 0.70 | 5/5 | 0 | 0 | 0 | 0.08812 | 0.954 | 1/5 |
| 0.15 | 0.90 | 5/5 | 0 | 0 | 0 | 0.09175 | 0.959 | 1/5 |

## Outputs

- Figure: `/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/figures/N8_preliminary_audit.pdf`
- Per-trajectory audit: `/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/reports/N8_trajectory_audit.csv`
- Per-cell summary: `/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/reports/N8_cell_summary.csv`
- Machine-readable summary: `/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/reports/N8_preliminary_summary.json`
