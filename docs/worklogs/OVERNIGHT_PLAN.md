# Overnight campaign plan (2026-07-08 15:50 → 2026-07-09 noon)

Orchestrated autonomously. Health-checked every 30 min (`src/health_probe.sh`): CPU load,
python-worker CPU%, GPU (idle by design), E30 progress + stall detection. Rule: no two
CPU-heavy jobs run concurrently (they own the same 8 cores → contention wastes wall time),
so phases are **sequential**; only cheap validations run alongside a heavy job.

## Hardware / policy
- 10 logical cores (8 usable for Pools), 24 GB RAM; Apple GPU via MLX (float32, Metal).
  torch is broken in this env (numpy 2.x ABI); MLX is the GPU path.
- **Precision taxonomy (corrected 2026-07-08 — the float32 lesson was LOCALIZED).**
  The documented false-fold failure was float32 in Newton/continuation/fold-detection, whose
  residual tolerances (1e-9..1e-12) sit below the float32 floor (~1e-5). It is NOT a blanket
  property. Split:
  - **float64 CPU (mandatory):** anything doing root-finding / eig-crossing / small Lyapunov
    exponents — E30 tracer, Exp C (survival), Exp B (Lyapunov/D_KY), aging-E1 (marginal mode).
  - **float32 GPU eligible (gate-validated):** coarse, ensemble-averaged dynamics outcomes —
    Exp A (basin classification), possibly aging two-time C / densities. float32 chaos diverges
    pointwise but the *basin reached* and *fractions* are robust; VALIDATE fractions vs float64
    before trusting (the guard missing in the original fiasco).
- **GPU runs in PARALLEL with E30's CPU work (no contention)** → Exp A can run DURING E30.
- Parallelization: CPU multiprocessing + `cycle_reduced_batch` (float64, validated 1e-13);
  GPU `cycle_reduced_mlx.ReducedDDEMLX` (float32, short-time correctness 7.9e-5, ~14 traj/s).
- Before every costly launch: re-read reasoning + code twice; validate; announce runtime.

## Monitoring notes (live)
- 03:36 — **E30 FINISHED** (all 3 reach runs incl. N=14000 completed overnight, 7.72 h, 0
  failed; N=14000 took 4.77 h — it ran to completion, no need to drop it after all). 303 npz.
  **Task 1 DELIVERED (e30_analysis.py, figAE):** fixed-α σ ~ **N^{-0.508}** (≈ CLT N^{-1/2};
  the earlier -0.38 was small-N contamination — confirmed with reach data), gap ~ N^{-0.42}
  **→ gap→0 CONFIRMED** (0.149→0.0498 at N=14000). Fixed-P σ ~ N^{-0.597}. Standardized λ_c
  near-Gaussian (skew -0.45, KS 0.026). → the threshold law IS a CLT of the quenched cross-talk.
  CASCADE started: (a) E30 analysis DONE → (b) Exp C LAUNCHED (e31, ~15 min) → (c) aging
  float64 validate → (d) Exp B Lyapunov reduced. CPU free, GPU free.
- 02:02 — N=10000/P500 s1000 DONE (~89 min/seed, memory-bound, even slower than the revised
  70-80 min). s1001 running (~03:30). **Decision: DROP N=14000** — at N^{3.4} it'd cost ~4.3 h
  for a bonus point and delay the higher-priority Exp B (task 5, CPU-blocked). gap→0 is served
  by the fixed-α series up to N=10000. Plan: when s1001 completes (~03:30), kill E30 and start
  the CPU cascade (E30 analysis with N≤10000 → Exp C → aging-validate → Exp B). Corrected
  overnight ceiling: **N=10000** at full-P α=0.05 (not 14000/20000). E30 healthy, 301/325.
- 00:25 — **E30 reach REVISED (honest correction to task-1 "how far overnight").** N=10000/P500
  no-eig ran 32 min without finishing → measured cost ~N^{3-4} (Woodbury O(P²N)/solve dominates,
  not eig). Real extrapolation: N=14000 ~100 min/seed, N=20000 ~**6 h/seed** — my earlier
  "N=20000 overnight" estimate was far too optimistic. Killed (300 npz preserved), revised REACH
  to just N=14000×1, DROPPED N=20000. 3 runs remain: N=10000×2, N=14000×1 (all no-eig), ~3 h →
  E30 finish ~03:30, then CPU cascade, done ~06-07h. **Corrected deliverable: the realistic
  full-P α=0.05 overnight ceiling is N≈14000, not 20000.** gap→0 trend already clear on the done
  fixed-α series (gap 0.093→0.070 for N=5657→8000); N=10000+14000 extend it.
- 23:44 — **E30 reprioritized** (stall warning: N=10000/P500-with-eig ran 48 min, memory-
  bandwidth-bound, pathologically slow). Killed cleanly (300 npz preserved, resumable), edited
  `e30_scaling_overnight.py`: N≥10000 fixed-α → `--no-eig` (bisection λ_c is primary, eig
  cross-check non-essential & already validated at N≤8000), N=10000 seeds 4→2. 7 runs remain,
  ALL no-eig: N=10000×2, N=14000×3, N=20000×2 — exactly the gap→0 reach points. Relaunched,
  computing (~7.6 cores). Revised E30 finish ~03:00-03:30 → CPU cascade (E30 analysis + Exp C
  + aging-validate + Exp B) fits well before noon. Rationale: protect the gap→0 deliverable
  (task 1) and stop the slow redundant run (N=10000 redundant with N=8000 done + reach).
- 22:38 — **AGING (task 3) COMPLETE** (figAC + figAD). Verdict: the mixture-phase chaos does
  NOT age — stationary SRB attractor. Two-time C collapses across t_w (spread 0.009 at
  α=0.05), τ_r≈2; no aging up to α=0.10 (spread 0.021/0.025, ≪0.1); at α=0.12 the λ=0.31
  chaotic ensemble vanishes (11 kept) → untestable there (itself informative: near capacity
  λ=0.31 is no longer the chaotic sea). float32/GPU, float64 gate deferred to post-E30.
  GPU now idle by choice (no GPU jobs while E30 in its expensive tail). Status: tasks 2,3,4
  done; task 5 Exp A done (Exp C ready, Exp B pending E30); task 1 E30 data ~91%, analysis
  pending. E30 (CPU) 299/329, N=8000 at 5/6 seeds, full 8 cores (GPU freed).
- 22:05 — **Aging Exp E2 preliminary (GPU float32)**: two-time C collapse spread at lag~20 =
  0.009/0.021/0.025 for α=0.05/0.08/0.10 (all ≪ 0.1 aging threshold), τ_r≈2 at all α. →
  **NO aging emerges up to α=0.10**; the mixture chaos stays a stationary SRB attractor even
  as the memory phase breaks down. The mild spread growth tracks the shrinking chaotic
  ensemble (239→112→41 kept trajectories = statistical noise), not aging. CAVEAT: large-α
  statement is preliminary (only 41 chaotic trajectories at α=0.10 → noisy); a clean
  near-capacity test needs more trajectories or a λ deeper in the chaotic region per α.
  α=0.12 running (last). NOTE: e33_analysis.py overwrites figAC — make a combined
  spread-vs-α figure once α=0.12 done. E30 (CPU) 298/329, N=8000/P400 at 4/6 seeds (~16
  min/seed, recovered rate).
- 21:33 — INSIGHT: the MLX GPU job's Python/sync thread takes ~0.5-1 CPU core, which
  DOES slow E30's expensive fixed-α runs ~10-15% ("GPU parallel to CPU is free" holds only
  for cheap CPU work). Rule updated: NO new GPU jobs while E30 is in its expensive fixed-α/
  reach tail; let aging-vs-α finish (~20 min) then E30 recovers 8 cores. E30 fixed-α tail
  slower than estimated (N=8000/P400 ~25 min/seed with eig); revised E30 finish ~04-06h —
  still before noon, but Exp C/B (CPU, post-E30) get a tighter window. If near ~02h E30 has
  not reached the reach points (N=14000/20000, the key gap→0 evidence), consider reducing
  seed counts on N=8000/10000. aging-vs-α: α=0.08 done (fewer chaotic kept, expected).
- 20:58 — **Aging (GPU float32) DONE + analyzed** (figAC). VERDICT at α=0.05, λ=0.31:
  **STATIONARY, NO AGING**. Two-time C(t_w+t,t_w) collapses across t_w∈{50,200,800,3200}
  (normalized-C spread at lag~20 = 0.009; C(t_w,0) rel spread = 0.002); does NOT collapse
  under t/t_w. Quench: τ_r≈2 (the near-marginal λ₄≈−0.003 does NOT create a slow transient —
  design fear lifted). → the mixture-phase chaos is a stationary SRB-measure attractor.
  Signal so clean (0.009) that float32 noise (~1e-5) is negligible; float64 gate will confirm.
  Then started **aging-vs-α on GPU** (α∈{0.08,0.10,0.12}, `e33 --alphas`) — design Exp E2:
  does aging emerge near capacity? ~30 min, parallel to E30. E30 (CPU) at 295/329, in the
  slow N=8000/P400 tail (~13 min/seed); revised finish ~01-02h.
- 20:30 — GPU free after Exp A → started **aging Exp D + two-time C on GPU float32**
  (`e33_aging.py --run --backend mlx`), parallel to E30 (CPU), ~10-20 min. Two-time C logic
  smoke-tested OK (C(0,t) decays cleanly). Results marked **float32, pending float64
  validation** (aging is a SUBTLE t_w-dependence; float32 validity less certain than for
  coarse basins). float64 validation (`--validate`, CPU-costly) deferred to AFTER E30 (no
  contention). Next: when e33 done → analyze two-time C (t_w-collapse=stationary vs
  t/t_w-fan=aging); when E30 done → E30 analysis + Exp C + Exp B, then aging float64 gate.
- 19:49 — **Exp A (GPU float32) DONE** (58 min, parallel to E30, no contention). npz
  `E32_basins_N2000_mlx.npz`, figure `figAB_alpha_lambda_basins.png`. Result: the cycle
  basin **erodes strongly with α** — at λ=0.40 cycle fraction 0.86→0.08 as α 0.05→0.15; the
  low-λ edge of the cycle region retreats from λ≈0.28 (α=0.05) to λ≥0.40 (α=0.15); chaos
  fills the mid-λ region. NOT a clean pinch-off by α=0.15 (cycle still 0.39 at λ=0.50): the
  memory→chaos route is a GRADUAL erosion, full closure likely near/above capacity α_c≈0.138.
  CAVEAT: coarse classifier counts ring-advance as "cycle" — Exp B (Lyapunov on the orbit,
  CPU float64) needed to test if the surviving high-α cycle is genuine or already chaotic.
  GPU now free; used later for aging float32 parts (Phase 3, post-validation). E30 (CPU) at
  289/329, fixed-α series — Exp C/B (CPU) wait for it.
- 17:23 — E30 healthy but large-N fixed-P slower than estimated (N=22627 ~5 min/seed):
  revised total ETA ~5-6 h (done ~22:00-23:00), still well before noon. Exp A (GPU) runs
  in parallel on Metal — CPU+GPU both full, max parallelism. Order in the ladder: fixed-P
  reach (N=22627,32000) runs before fixed-α reach; the physically-crucial gap→0 fixed-α
  points (N=8000..20000) land later — acceptable given the time buffer; revisit only if
  time gets tight near noon.

## Phase 0 — RUNNING: E30 threshold-law scaling (task 1)
`src/e30_scaling_overnight.py` → `results/3_unification_seuils_scaling/data/E30_scaling/`. 329 runs, resumable (skips
existing). Fixed-P (P=100, N≤32000, many seeds) + fixed-α (α=0.05, N≤10000) + reach
(N=14000,20000). ETA ~4–5 h. Deliverable: σ(N) power-law fit, gap→0 test, Gaussianity.
**Analysis `e30_analysis.py` runs when the driver finishes (not CPU-heavy).**

## Phase 1 — Large-α: static→chaos (task 5), cheap decisive experiments
Launch after E30 frees cores. Two independent, cheap, tracer-independent probes first:
- **Exp C — per-pattern memory SURVIVAL census. [READY + VALIDATED]** `e31_fixedpoint_census.py`.
  Method pivot (smoke-tested 2026-07-08): Newton from DIFFUSE random ICs barely converges
  (4/40) — replaced by beta-annealed Newton **from each pattern** (robust: 40/40 at α=0.05).
  Order parameter f_mem(α,λ) = fraction of patterns retaining a macroscopic stable memory.
  Smoke result: f_mem = 1.00 (α=0.05) → 0.28 (α=0.10, 56/200) at λ=0.15; survivors keep
  maxov≈0.85 and stay distinct (P(q)≈0.06–0.09) → memories vanish one-by-one (extreme-value),
  not gradual decay. Spurious/mixture-attractor proliferation folded into Exp A (dynamics).
  Cost ~15 min at nproc=8. **Launch (after E30 frees cores):**
  `python e31_fixedpoint_census.py --alphas 0.05,0.07,0.09,0.11,0.13 --lams 0.0,0.05,0.15,0.25 --seeds 42,43 --nproc 8`
- **Exp A — (α,λ) basin phase diagram.** Batched-integrator basin classification, coarse
  partition {static/cycle/chaos/other} (robust; the memory/front split is seed-specific).
  Grid α∈{0.05,0.07,0.09,0.11,0.13,0.15} × λ∈{0.15…0.90}, N=2000, ≥4 seeds × 200 ICs, t=400.
  Deliverable: cycle-fraction heatmap + its pinch-off α_cycle. Cost ~1.5–3 h wall (batched).
  `e32_alpha_lambda_basins.py`.

## Phase 2 — Large-α: Lyapunov map (task 5), expensive
- **Exp B — λ_L(α,λ) and D_KY.** Benettin/QR (reuse `e23a_chaos_nature.py`). B1 λ_L grid
  (k=1), B2 cycle-orbit λ_L (is the α=0.1 "cycle" chaotic?), B3 D_KY (k=8) at the near-capacity
  high-λ cells. Cost ~10–30 CPU-h → target the cells A/B2 flag. Only after Phase 1 localizes
  the cycle pinch-off.

## Phase 3 — Aging / invariant measure (task 3)
Reuse the batched integrator (ensemble of IC×disorder).
- **Exp D first (confound-killer, sets settle time):** ensemble quench relaxation
  ⟨|a|²⟩(t), fit τ_r; check whether the near-marginal λ₄≈−0.003 gives τ_r~300.
- **Exp A_aging — two-time C(t_w+t,t_w)** at t_w∈{0,50,200,800,3200}: collapse (stationary)
  vs t/t_w fan (aging).
- **Exp B_aging — invariant-measure test:** density convergence (KS vs window & t_w) +
  time-avg vs ensemble-avg (ergodicity). Confound guards: classify final state, keep only
  trajectories that stay chaotic; use gauge-invariant scalars.
  `e33_aging.py`.

## Ordering rationale
E30 (running) → C+A (cheap, decisive, large-α) → aging D+A+B (moderate, reuse batched) →
B Lyapunov (expensive, targeted). Each phase: double-check code, announce estimate, launch,
monitor every 30 min, analyze on completion. Stop by 2026-07-09 noon; leave partial results
checkpointed (one npz per unit) so nothing is lost.

## Monitoring log
See `results/0_worklogs/HEALTH_LOG.txt` (appended each 30-min probe).
