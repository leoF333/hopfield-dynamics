#!/bin/bash

# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO="${CHICAGO_ROOT:-/Users/leoflack/Desktop/Recherche/Chicago}"
# ---------------------------------------------------------------------------

# Corrected N=10000 re-run (CPU float64 + beta-annealed Newton + guards).
# The first run used MLX float32, which corrupted branch tracing (alpha=0.05
# false-folded low; alpha=0.10 initial Newton failed). No GPU needed here.
set -u
SRC="$CHICAGO/3_numerics/src"
OUT="$CHICAGO/3_numerics/results/N=10000"
CONDA="/opt/homebrew/Caskroom/miniforge/base/bin/conda"
LOG="$OUT/rerun.log"
mkdir -p "$OUT"; export TQDM_DISABLE=1; cd "$SRC" || exit 1
: > "$LOG"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

log "=== RERUN float64 — PART A: run_N10000.py (seeds 42 43 44) ==="
tA=$(date +%s)
"$CONDA" run -n mcmc_env python run_N10000.py --seeds 42 43 44 --n-lam 10 >>"$LOG" 2>&1
log "Part A rc=$? ($(( $(date +%s)-tA ))s)"

log "=== PART B: lambda_c_grid.py --Ngrid 10000 ==="
tB=$(date +%s)
"$CONDA" run -n mcmc_env python lambda_c_grid.py \
    --alphas 0.05 0.10 --taus 1 2 5 10 15 20 --n-lam 10 \
    --Ngrid 10000 --outdir "$OUT" >>"$LOG" 2>&1
log "Part B rc=$? ($(( $(date +%s)-tB ))s)"

log "=== RERUN DONE ==="
touch "$OUT/DONE_RERUN"
