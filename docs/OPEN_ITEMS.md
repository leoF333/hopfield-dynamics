# Open items found while building the evidence map


> **Paths.** This copy is path-remapped for this repository. Entries marked
> `[local archive]` live only in the full research archive, not here.
These are the points where the manuscript, the data and the campaign reports do not line
up exactly, or where a build chain is incomplete. None of them invalidates a result; all of
them are things a referee could ask about. Ordered by how much work they need.

---

## 1. Three figures have no reproducible assembly script

`Figure04_thresholds`, `Figure06_snic` and `Figure08_hyperchaos` are the only files in
`figures/manuscript/` with **no byte-identical copy anywhere in the folder** (all 1142
PNGs were hashed). They are later re-layouts of the panels in
`figures/panels/`:

| Shipped figure | Dimensions | Closest ancestor | Ancestor dimensions | What changed |
|---|---|---|---|---|
| `Figure04_thresholds` | 5400 × 2380 | `Panel04_threshold_distributions.png` | 5400 × 3763 | 7 sub-panels instead of 10 |
| `Figure06_snic` | 5400 × 5054 | `Panel05_snic_characterization.png` | 5400 × 3453 | a Floquet row (g) added |
| `Figure08_hyperchaos` | 5400 × 4358 | `Panel01_lyapunov_chaos.png` | 5400 × 2869 | re-laid out to 5 sub-panels |

The **component-level** provenance is intact — every sub-panel traces back through
`figures/panel_manifest.json` (which records source SHA-256 and crop
coordinates) to E23A / E26 / E30 / N2 / N3 / N6 arrays. Only the final composition step is
missing, and it was produced outside this folder.

**Options.** Either recover the assembly script from wherever it was written, or rebuild
the three figures from `code/figures/build_panels.py` and
`code/figures/v2/*.py`, which are both present and
now path-portable.

---

## 2. `Figure06(g)` may be using the weaker of two Floquet panels

The Floquet row added to `Figure06_snic` is `figE_floquet_spectrum_v2.png`, which plots
|μ_k| on a log axis. A more careful component exists and is **not** used:
`code/figures/v2/component_floquet_hyperstability.png`.

Its README explains why it was built: the modulus falls below e^−440 and cannot be shown on
a sensible axis, so it plots the contraction per turn A = −ln|ϱ₁| instead, and it separates
three classes of point that the older figure mixes:

- the N = 2000 Arnoldi scan points, which sit at the relative floor 2 × 10⁻¹⁶ and are
  **lower bounds only, not measurements**;
- the N = 500 spectra, which are above the floor and therefore measured
  (A = 16.2 / 22.4 at λ = 0.90 / 0.65);
- the floor-free Benettin–QR measurements at N = 2000 near death
  (A = 219.7 / 419.5 / 440.1 at λ = 0.400 / 0.335 / 0.329) — the numbers the manuscript
  actually quotes in Sec. 4.4.

The same README warns that the `rez` field of `floquet_scan_N2000_tau10.0.npz` is computed
from a floored |μ\*| and is an **upper bound on Re z₁**, not a measurement.

**Recommendation.** Swap panel (g) for `component_floquet_hyperstability.png`, or state in
the caption which points are bounds.

---

## 3. Numbers in Sec. 3 that do not match the N3 report verbatim

| Manuscript | Value quoted | N3 synthesis value | Assessment |
|---|---|---|---|
| Eq. (22), width exponent | b = 0.43, CI [0.40, 0.47] | b = 0.43078, CI [0.39638, 0.46520] | consistent (rounding) |
| Gaussian extreme prediction | mean signed error **+0.005**, max abs **0.024** | **+0.00734**, **0.02745** | **discrepancy — resolve** |
| Range exponent | b_gap = 0.42, CI [0.37, 0.47] | raw d = 0.27582, CI [0.22538, 0.33040] | different quantity: the manuscript's is the fit *after* dividing by √(2 ln P). That specific fit is not tabulated in the N3 synthesis. **Locate or re-run it.** |

The extreme-prediction discrepancy is probably a different pooling subset (all sizes vs
N ≥ 4000). Whichever it is, the manuscript should say so, because N3 additionally records
that the *imposed* Gaussian extreme form is disfavoured by ΔAICc = 8.37 against a free power
law — a caveat the manuscript currently does not carry.

---

## 4. The N2 confidence interval quoted in Sec. 4.2

The manuscript says the free exponent is "between 0.43 and 0.51 at 95% confidence". The
tabulated N2 intervals are **0.477 [0.424, 0.529]** (pooled) and **0.470 [0.435, 0.504]**
(near window). Neither rounds exactly to [0.43, 0.51]. State which interval is being used.

Two N2 facts should also survive into the methods section: seed 51 is archived but its
dynamic bracket does not contain the exact static fold, so it is excluded from the local
exponent; and the first N2 static sweep followed the unstable saddle branch and is invalid
(archived under `runs/N2_INVALID_STATIC_BRANCH_20260729/`).

---

## 5. The SNIC claim is stronger in V4 than the V5 evidence register allows

`[local archive] 4_campaigns/manuscript_audits/agent_redacteur/…` and the V5 register
(`[local archive] obsolete/02_superseded_manuscripts/paper_v5/EVIDENCE_REGISTER.md`) both state:

> *The transition is rigorously a SNIC.* — Global invariant-circle topology and asymptotic
> normal-form coefficients have not been established. **Not established.** Do not claim; use
> "localized saddle-node depinning" and "SNIC-like ghost".

The V4 draft does claim a SNIC, on the strength of E34's direct closures — which are
genuinely strong, but established **only at the terminal and penultimate stages of the
reference realization** (see `E34_INVARIANT_CIRCLE_RESULTS.md` §10 "Limites"). The draft
already says this in Sec. 5.1, so the claim is defensible; but it is the single biggest
wording risk in the paper and worth one deliberate decision before submission. The audit
thread is in `[local archive] 4_campaigns/manuscript_audits/agent_critique/`.

---

## 6. Two Lyapunov spectra coexist for the same attractor

Sec. 6.2 quotes the eight-exponent E23A spectrum (0.035, 0.011, 0.0015, …) for D_KY ≈ 6.4,
and separately the multi-seed N6 result. The two are different measurements: N6 at λ = 0.31
gives leading exponents around 0.046–0.055 for seed 42, against E23A's 0.035. Both are
finite-window estimates of the same attractor and the manuscript uses them in different
sentences, which is fine — but the reader should not be able to read them as one series.
Consider labelling the figure panels by campaign.

---

## 7. `Figure01` ships under a misleading filename

The byte-identical twin of `Figure01_phase_diagram.png` is stored as
`figures/panels/phases MCMC.png`, while the manuscript describes Euler
integration of the continuous-time model with hollow J. The historical MCMC work is in
`[local archive] obsolete/05_invalid_or_refuted/` and must never be cited. Confirm the figure is the
continuous-time survey and rename the file before release.

---

## 8. `E33` (no-aging) used the float32 / MLX path

Sec. 6.3's stationarity result rests on `E33_aging_mlx*.npz`. The planned float64 control
was campaign N9, which was withdrawn from scope on 2026-07-30. The precision path should be
stated in the methods. The `figures_v2` README also records a convention that must be
reproduced by anyone re-deriving the numbers: **t_w = 0 is excluded** from the spread
because it is still inside the settle transient.

---

## 9. Minor

- **No `Figure12`.** The numbering skips it; the layered-architecture diagram is TikZ.
  Harmless, but a copy-editor will ask.
- **`Figure00(d)`** draws bars beyond index 1 at the cross-talk scale for context; they are
  not individually measured. The `figures_v2` README says to state this in the caption or
  drop them. The current caption does neither.
- **Sec. 1.3** quotes a full-vs-reduced agreement of 1e-15; the V5 register records
  5.49e-16 for the same check. Pick one.
- **Bibliography** is deliberately absent from the working draft (`main.tex` says so).
