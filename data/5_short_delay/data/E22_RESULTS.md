# E22 — Small-tau desynchronization of the recall cycle (RESULTS)

Model: reduced dim-100 DDE, N=2000, alpha=0.05 (P=100), beta=20, t0=1, seed 42, lambda=0.9.
dt = min(0.01, tau/25); >=25 handoffs after 40% warmup. dt-convergence of T1 verified
(halving at tau=0.5: dT1 < 2e-4). Computed by the orchestrator (sub-agent was blocked on Bash permission).

## E22a/b — dense tau sweep (lambda=0.9)
Columns: T1 mean +- std ; W = front width (patterns) ; a_peak = peak recall ;
n_lobes ; N_part = participation ratio (effective lobe count) ; R_kura = ring-Kuramoto ;
R_pr = participation coherence.

| tau | T1 | +- | W | a_peak | n_lobes | N_part | R_kura | R_pr |
|----|------|------|------|-------|------|-------|-------|-------|
| 0.0 | 1.917 | 1.170 | 14.66 | 0.155 | 4.40 | 20.45 | 0.364 | 0.050 |
| 0.1 | 1.965 | 1.199 | 14.38 | 0.184 | 4.52 | 20.19 | 0.343 | 0.051 |
| 0.2 | 1.823 | 1.152 | 14.40 | 0.188 | 4.27 | 19.47 | 0.334 | 0.052 |
| 0.3 | 1.676 | 0.815 | 12.40 | 0.209 | 3.48 | 17.93 | 0.326 | 0.057 |
| 0.4 | 1.829 | 0.964 | 12.88 | 0.212 | 3.40 | 18.65 | 0.267 | 0.056 |
| 0.5 | 1.825 | 0.979 | 12.85 | 0.218 | 3.22 | 18.04 | 0.290 | 0.057 |
| 0.6 | 2.060 | 1.006 | 12.39 | 0.218 | 3.29 | 17.94 | 0.270 | 0.057 |
| 0.7 | 1.902 | 1.023 | 11.50 | 0.219 | 3.30 | 17.63 | 0.288 | 0.061 |
| 0.8 | 1.975 | 1.140 | 12.35 | 0.215 | 3.56 | 17.89 | 0.292 | 0.060 |
| 0.9 | 2.346 | 1.531 | 14.53 | 0.202 | 4.55 | 20.17 | 0.252 | 0.060 |
| 1.0 | 2.524 | 1.403 | 14.16 | 0.209 | 4.18 | 19.68 | 0.243 | 0.065 |
| 1.2 | 2.171 | 0.804 | 7.55 | 0.296 | 2.56 | 11.89 | 0.318 | 0.128 |
| 1.4 | 2.148 | 0.013 | 1.33 | 0.673 | 1.00 | 1.54 | 0.574 | 0.687 |
| 1.6 | 2.365 | 0.012 | 1.30 | 0.700 | 1.00 | 1.46 | 0.572 | 0.723 |
| 1.8 | 2.579 | 0.012 | 1.29 | 0.720 | 1.00 | 1.42 | 0.569 | 0.747 |
| 2.0 | 2.789 | 0.013 | 1.26 | 0.739 | 1.00 | 1.38 | 0.566 | 0.767 |
| 2.5 | 3.307 | 0.011 | 1.22 | 0.777 | 1.00 | 1.32 | 0.562 | 0.801 |
| 3.0 | 3.816 | 0.011 | 1.20 | 0.804 | 1.00 | 1.28 | 0.560 | 0.821 |
| 5.0 | 5.829 | 0.012 | 1.13 | 0.869 | 1.00 | 1.20 | 0.562 | 0.868 |
| 10.0 | 10.832 | 0.011 | 1.07 | 0.930 | 1.00 | 1.12 | 0.559 | 0.913 |

## E22b — desynchronization order parameter & crossover
- Sharp transition between tau=1.2 and tau=1.4 on ALL indicators simultaneously.
- Best order parameter: R_pr / N_part (jump 0.06 -> 0.69 ; N_part 18 -> 1.5).
- R_kura is a poor indicator here (0.3 -> 0.57, no sharp jump).
- Crossover tau_c = 1.31 (R_kura mid-height), corroborated by the collapse of std(T1) (±1.0 -> ±0.01) and of N_part at the same tau.
- Regime below tau_c: delocalized multi-lobe front (N_part~18-20, W~13-15, weak recall ~0.2, irregular handoffs).
- Regime above tau_c: single coherent front (N_part->1, W->1.1, recall 0.67->0.93, clockwork period).

## E22c — period law
- small tau (<=1): T1 = 0.518 tau + 1.727  (confirms E19a slope 0.56)
- large tau (>=2): T1 = tau + 0.815  (pacemaker, slope 1)
- local slope reaches 0.9 at tau ~ 0.5 (clock crosses over earlier than the shape).

## E22c — lambda re-synchronization control (E22_lamcheck.npz)
Raising lambda 0.9 -> 0.95 -> 0.99 at tau=0.25 and tau=1.0 does NOT re-synchronize:
W~13, N_part~18-21, peak~0.2 unchanged. Desynchronization at small tau is intrinsic to
the short delay, not curable by stronger cyclic coupling.

## Files
- results/5_petit_tau/data/E22_desync.npz, results/5_petit_tau/data/E22_lamcheck.npz
- results/5_petit_tau/figures/figP_desync.png
- src/e22_desync.py
