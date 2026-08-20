# E21 — Nature and tau-dependence of the pinned-front BIRTH

**Date:** 2026-07-05 · N=2000, alpha=0.05 (P=100), beta=20, t0=1, seed 42.
Reduced P-dim exact machinery (Woodbury/Sylvester + T_P(z) reduced spectrum).

PI question: does the bifurcation that GIVES BIRTH to the pinned fronts depend on
tau — for its NATURE (saddle-node vs Hopf) or its POSITION (lam_f)? For medium and
large tau.

---

## E21a — the front birth is NOT a saddle-node fold (pseudo-arclength continuation)

**Setup.** E18a followed the bond-91 pinned front DOWNWARD by naive lambda-stepping
(Woodbury-Newton warm-start) and STOPPED at lam=0.1322 with eig_max(M)=-0.998,
labelling it a "proper saddle-node fold". That label is WRONG: a fixed-point fold
requires a REAL eigenvalue of M crossing 0 (eig_max(M) -> 0, identity Delta(0)=-M).
At eig_max(M)=-0.998 the front is DEEPLY stable — the branch cannot be turning
there. The E18a stall was just a continuation failure (warm-start Newton losing the
basin as ds shrank).

**Method.** Pseudo-arclength continuation in (u, lam): tangent predictor + bordered
Newton corrector, the u-block solved with the exact Woodbury identity. Validated
against E18a at lam=0.20 (a91=0.802 vs 0.804, eig_max(M)=-0.893 vs -0.905 — PASS).
`src/e21a_front_birth.py`, `src/e21a2_dissolution.py`.

**Result — the branch sails straight through lam=0.1322 with NO turning point.**
It continues smoothly to lam<0 with eig_max(M) monotonically -> -1 (deeper into the
saturated memory limit, gain -> 0, M -> -I). Along the way a91 GROWS toward 1 and
a92 SHRINKS toward 0:

| lam    | a91    | a92    | eig_max(M) | eig closest to 0 |
|--------|--------|--------|------------|------------------|
| 0.1500 | +0.851 | +0.138 | -0.9948    | -0.99479         |
| 0.1332 | +0.868 | +0.121 | -0.9981    | -0.99809         |
| 0.1000 | +0.901 | +0.088 | -0.9997    | -0.99974         |
| 0.0670 | +0.934 | +0.054 | -1.0000    | -0.99996         |
| 0.0292 | +0.971 | +0.016 | -1.0000    | ~-1.0000         |
| 0.0171 | +0.983 | +0.003 | -1.0000    | ~-1.0000         |
| 0.0051 | +0.995 | -0.009 | -1.0000    | ~-1.0000         |

**NO eigenvalue of M approaches 0 anywhere on the low branch** (eig closest-to-0 is
pinned at ~-1.0). There is no fold. Instead the "front" character DISSOLVES: the
second overlap a92 -> 0 smoothly and crosses through zero (linearly) at
**lam ~ 0.014** (a92 = +0.0034 at 0.0171, -0.0088 at 0.0051), then goes negative.

## E21a-bis — the mechanism: the front IS the xi^91 memory branch (merge, not fold)

Comparing the "front" against the PURE single-pattern xi^91 memory (Newton from
0.99*xi^91) at matched lam:

| lam    | front a91 | front a92 | mem a91 | mem a92 | ‖a_front-a_mem‖ | eigM front | eigM mem |
|--------|-----------|-----------|---------|---------|-----------------|------------|----------|
| 0.1297 | +0.8714   | +0.1175   | +0.8712 | +0.1178 | 5.6e-4          | -0.9985    | -0.9984  |
| 0.0900 | +0.9108   | +0.0772   | +0.9108 | +0.0773 | 2.7e-5          | -0.9999    | -0.9999  |
| 0.0503 | +0.9502   | +0.0370   | +0.9504 | +0.0367 | 5.0e-4          | -1.0000    | -1.0000  |
| 0.0298 | +0.9705   | +0.0162   | +0.9703 | +0.0164 | 4.4e-4          | -1.0000    | -1.0000  |

**The "pinned front" and the "pure xi^91 memory" are the SAME fixed point** at low
lam (identical to ~1e-4, identical eig_max(M) to 4 decimals). The a92 component is
NOT a separate mode: it is the natural next-neighbour dressing that the memory state
xi^91 carries under the asymmetric coupling lam*K. As lam grows, a92 grows
continuously until (near lam*~0.328) it becomes comparable to a91 and the object
LOOKS like a genuine two-pattern front. As lam shrinks, a92 -> 0 and the object
relaxes continuously into the bare xi^91 memory.

**Verdict on the BIRTH:** the pinned front has NO separate birth. It is the
lam-continuation of the xi^91 memory branch — the SAME solution branch, continuous
from lam=0 (bare memory) up to the depinning saddle-node at lam*~0.328 (E12/E18,
where eig_max(M) -> 0, the TRUE and only fold of this branch). There is no lower
saddle-node, no dissolution into symmetric mixtures, no merge with a distinct
branch: the front is simply what the single-pattern memory becomes when the
sequence-drive lam*K tilts it toward its cyclic successor. E18a's "fold at 0.132"
is refuted.

Files: `E21_front_birth.npz`, `E21_dissolution.npz`.

---

## E21b — POSITION is rigorously tau-independent (machine precision)

T_P(0) = I - (1-lam)G_J - lam*G_K = -M_reduced: the e^{-0*tau}=1 factor removes tau.
Verified across tau in {5,10,20,50,100}:

| lam    | max|T_P(0;tau_i)-T_P(0;tau_0)| | max eig diff across tau | eig_max(M) direct vs -eig(T_P(0)) |
|--------|-------------------------------|-------------------------|-----------------------------------|
| 0.1500 | 0.00e+00                      | 0.00e+00                | -0.994794 (diff 4.4e-16)          |
| 0.2822 | 0.00e+00                      | 0.00e+00                | -0.816708 (diff 2.3e-15)          |
| 0.3100 | 0.00e+00                      | 0.00e+00                | -0.234363 (diff 5.0e-16)          |
| 0.3270 | 0.00e+00                      | 0.00e+00                | -0.062973 (diff 2.2e-15)          |

T_P(0) and its full eigenstructure are IDENTICAL across all tau (exactly 0.00e+00).
The instantaneous fold indicator eig_max(M) equals max Re(-eig T_P(0)) to machine
precision. **The front's POSITION — its only fold, at lam*~0.328 — does NOT depend
on tau. Rigorous (Delta(0)=-M) and confirmed numerically.**

File: `E21_tau_independence.npz`.

---

## E21c — NATURE / stability boundary vs tau (medium & large tau)

(Results appended below once the run completes.)
