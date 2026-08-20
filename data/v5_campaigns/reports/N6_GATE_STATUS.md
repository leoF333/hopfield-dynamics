# N6 gate status — multi-seed Lyapunov spectrum

Updated: 2026-07-29 19:06 CDT.

## Integrity

All 34 requested production spectra reopen:

- seed 42: five values of \(\lambda\), each at \(dt=0.01\) and \(0.005\);
- seeds 43--46: five values of \(\lambda\) at \(dt=0.01\), plus the
  \(dt=0.005\) control at \(\lambda=0.31\).

Every run stores the leading eight float64 exponents, cumulative estimates at
\(T=1000,2000,4000\), blockwise estimates, the final trajectory, PCA spectrum,
power spectrum, Poincare-section data and relay observables.

## Conservative hyperchaos gate

At \(\lambda=0.31\), all five disorder realizations retain at least three
positive exponents separated from zero under both accumulation-window and
time-step changes:

| seed | \(dt\) | \(\Lambda_1\) | \(\Lambda_2\) | \(\Lambda_3\) |
|---:|---:|---:|---:|---:|
| 42 | 0.010 | 0.04620 | 0.02891 | 0.01957 |
| 42 | 0.005 | 0.03807 | 0.02958 | 0.01145 |
| 43 | 0.010 | 0.05369 | 0.03562 | 0.02219 |
| 43 | 0.005 | 0.05154 | 0.03389 | 0.02261 |
| 44 | 0.010 | 0.05112 | 0.02927 | 0.01786 |
| 44 | 0.005 | 0.05532 | 0.03133 | 0.02056 |
| 45 | 0.010 | 0.04667 | 0.02894 | 0.01620 |
| 45 | 0.005 | 0.05237 | 0.03628 | 0.02657 |
| 46 | 0.010 | 0.04956 | 0.03587 | 0.02138 |
| 46 | 0.005 | 0.05026 | 0.03512 | 0.02081 |

The N6 criterion for using “hyperchaos” beyond the reference realization is
therefore passed at this parameter value.  The supported statement is:
**the \(\lambda=0.31\) chaotic attractor has at least three robustly positive
Lyapunov exponents for all five tested disorder realizations.**

## Qualifications

- Seed 42 at \(\lambda=0.29\) and \(0.32\) reaches different long-time
  attractors under \(dt=0.01\) and \(0.005\).  These cells are not validated
  and cannot be used to infer a continuous hyperchaotic interval.
- The present runs initialize the two time steps independently after a finite
  transient.  Ambiguous cells require a shared, independently classified
  chaotic history before attributing their discrepancy purely to numerical
  time-step error.
- The eight computed exponents do not exhaust the positive partial sum, so the
  Kaplan--Yorke dimension is not resolved: the current data give only a
  finite-spectrum lower bound, not an exact dimension.
- The neutral flow exponent and its alignment with the flow direction have not
  yet been isolated.  Near-zero fourth-to-sixth estimates are not counted in
  the conservative hyperchaos result above.

