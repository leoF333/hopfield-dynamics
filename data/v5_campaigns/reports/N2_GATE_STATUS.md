# N2 scientific gate

## Scope and integrity

- Five disorder realizations: [42, 47, 51, 52, 60].
- All 50 corrected stationary points remain on the stable N1-connected branch,
  preserve the selected motif identity and pass the fixed-point and dedicated
  real-root residual gates.
- All 80 dynamic measurements are uncensored.
- The 30 half-step controls have a maximum relative difference of
  0.
- The closest dynamic approach occurs at the statically selected terminal bond
  for every one of the 50 base-step trajectories.

## Local dynamic law

The independent N1 onset bracket contains the exact static fold for seeds
[42, 47, 52, 60]. Seed 51 is retained in all files and plots
but excluded from the local exponent aggregation because its entire dynamic
onset bracket lies below the exact static fold by more than the smallest tested
offsets. This criterion was fixed independently of the fitted exponent.

- Full-window disorder-mean free exponent:
  0.477, nested-bootstrap 95% interval
  [0.424,
  0.529].
- Near-window (δ <= 5e-4) mean:
  0.470, 95% interval
  [0.435,
  0.504].
- The logarithmic alternative is disfavored by at least
  ΔAICc=35.1 over the audited full and near windows.
- A fixed one-half law is nevertheless disfavored relative to a free power by
  as much as ΔAICc=24.5; finite-size/sample-dependent
  deviations remain resolved.

## Static law

The dedicated real-axis search repairs candidate omissions in the generic
ARPACK spectrum and gives scaled characteristic residuals below 1e-10 at all
50 points. Close to the fold the rightmost real mode softens with a
square-root-like trend. Progressively wider windows leave this local asymptotic
regime and produce strong sample and window dependence:

- mean free exponent through δ=1e-4:
  0.654, seed-bootstrap 95% interval
  [0.448,
  1.018];
- mean through δ=5e-4:
  0.474, 95% interval
  [0.335,
  0.592].

The static exponent is therefore not stable over the required 1.5 decades.

## Gate conclusion

**Strict measured-half-exponent gate: FAIL.**

The data robustly establish localized critical slowing, reject a logarithmic
dynamic law for the four correctly centered samples, and are compatible with
the saddle-node square-root mechanism. They do not support quoting a universal
measured exponent exactly equal to one half. The permitted manuscript wording
is therefore: **“compatible with a saddle-node square-root law.”**

## Files

- Machine-readable summary: `reports/N2_summary.json`
- Fit-window table: `reports/N2_fit_windows.csv`
- Time-step controls: `reports/N2_timestep_controls.csv`
- Four-panel diagnostic: `figures/N2_critical_laws.pdf`
