# E20 -- basin competition of the delayed mixed Hopfield network

Reduced DDE tau=10.0, N=2000, alpha=0.05 (P=100), beta=20.0, seed=42, t_total=300.0, dt=0.05.
Sampling: 1/3 near_pattern (alpha*e_mu+noise), 1/3 mixture (2-4 consecutive patterns), 1/3 random_small (broad small-norm overlap).
Classes: memory (recall on a strong bond) / front (weak bond 91-92, metastable) / cycle (ring advance) / chaos (non-stationary) / other.

## E20a -- Monte-Carlo basin fractions vs lam

K=300 random ICs per lam.

| lam | memory | front | cycle | chaos | other |
|-----|--------|-------|-------|-------|-------|
| 0.150 | 0.280 | 0.000 | 0.000 | 0.343 | 0.377 |
| 0.200 | 0.293 | 0.003 | 0.000 | 0.547 | 0.157 |
| 0.250 | 0.380 | 0.007 | 0.013 | 0.583 | 0.017 |
| 0.280 | 0.383 | 0.000 | 0.043 | 0.573 | 0.000 |
| 0.150 | 0.316 | 0.004 | 0.000 | 0.352 | 0.328 |
| 0.200 | 0.352 | 0.000 | 0.000 | 0.556 | 0.092 |
| 0.250 | 0.264 | 0.004 | 0.012 | 0.704 | 0.016 |
| 0.280 | 0.484 | 0.016 | 0.020 | 0.476 | 0.004 |
| 0.300 | 0.312 | 0.012 | 0.152 | 0.524 | 0.000 |
| 0.310 | 0.008 | 0.044 | 0.472 | 0.472 | 0.004 |
| 0.320 | 0.000 | 0.032 | 0.564 | 0.404 | 0.000 |
| 0.325 | 0.000 | 0.048 | 0.552 | 0.400 | 0.000 |
| 0.328 | 0.000 | 0.000 | 0.608 | 0.392 | 0.000 |
| 0.330 | 0.000 | 0.000 | 0.656 | 0.344 | 0.000 |
| 0.340 | 0.000 | 0.000 | 0.612 | 0.388 | 0.000 |
| 0.400 | 0.000 | 0.000 | 0.836 | 0.164 | 0.000 |
| 0.600 | 0.000 | 0.000 | 0.840 | 0.160 | 0.000 |
| 0.900 | 0.000 | 0.000 | 0.792 | 0.208 | 0.000 |

(high-lam points from e20a_hi.py; front basin = 0 above depinning lam*=0.328.)

## Verdict
- Front basin is a thin sliver (<=0.05) at ALL lam; front/dominant-basin <= 0.1 everywhere -> pinned fronts are METASTABLE, never win the basin competition.
- Memory dominant for lam<lam_c=0.2822, collapses at the fold (0.484@0.28 -> 0.008@0.31 -> 0 above).
- Cycle basin 0 below 0.30, grows to 0.84 at lam=0.4-0.6, 0.79 at lam=0.9.
- Chaos fills the mixture window (up to 0.70 at lam=0.25) and leaves a 16-21% residual basin even at high lam.
- Figures: figN_basin_competition.png, figO_metastability.png.
