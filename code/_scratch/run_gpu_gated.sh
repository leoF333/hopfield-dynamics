#!/bin/bash

# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO="${CHICAGO_ROOT:-/Users/leoflack/Desktop/Recherche/Chicago}"
# ---------------------------------------------------------------------------

# GPU-gated orchestrator: wait until the GPU is free (your other project has
# released it), then run the N=10000 multi-seed study (Part A) followed by the
# alpha-tau critical-lambda grid at N=10000 (Part B).
#
# GPU availability is read from ioreg "Device Utilization %" (no sudo needed).
# Logs go to results/N=10000/orchestrator.log  (tqdm disabled for clean logs).

set -u
SRC="$CHICAGO/3_numerics/src"
OUT="$CHICAGO/3_numerics/results/N=10000"
CONDA="/opt/homebrew/Caskroom/miniforge/base/bin/conda"
LOG="$OUT/orchestrator.log"
mkdir -p "$OUT"
export TQDM_DISABLE=1
cd "$SRC" || exit 1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

gpu_util() {
  ioreg -r -d 1 -c IOAccelerator 2>/dev/null \
    | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$'
}

# ---- 1. wait for the GPU to be free --------------------------------------
THRESH=25          # consider "free" below this utilization %
NEED=3             # consecutive low readings required (~3 min)
INTERVAL=60        # seconds between polls
MAX_ITERS=1440     # safety cap: 24 h

log "=== orchestrator start; waiting for GPU (<${THRESH}% x${NEED}) ==="
free=0; iters=0
while [ "$iters" -lt "$MAX_ITERS" ]; do
  u=$(gpu_util); u=${u:-100}
  if [ "$u" -lt "$THRESH" ]; then free=$((free+1)); else free=0; fi
  log "GPU util=${u}%  (consecutive-low=${free}/${NEED})"
  if [ "$free" -ge "$NEED" ]; then
    log "GPU is free. Proceeding."
    break
  fi
  iters=$((iters+1))
  sleep "$INTERVAL"
done
if [ "$free" -lt "$NEED" ]; then
  log "ERROR: GPU never freed within cap; aborting."
  exit 2
fi

# ---- 2. Part A: N=10000, 3 seeds -----------------------------------------
log "=== PART A: run_N10000.py (seeds 42 43 44) ==="
tA=$(date +%s)
"$CONDA" run -n mcmc_env python run_N10000.py --seeds 42 43 44 --n-lam 10 >>"$LOG" 2>&1
rcA=$?
log "Part A finished rc=$rcA  ($(( $(date +%s) - tA ))s)"

# ---- 3. Part B: alpha-tau grid at N=10000 --------------------------------
log "=== PART B: lambda_c_grid.py --Ngrid 10000 ==="
tB=$(date +%s)
"$CONDA" run -n mcmc_env python lambda_c_grid.py \
    --alphas 0.05 0.10 --taus 1 2 5 10 15 20 --n-lam 10 \
    --Ngrid 10000 --outdir "$OUT" >>"$LOG" 2>&1
rcB=$?
log "Part B finished rc=$rcB  ($(( $(date +%s) - tB ))s)"

log "=== ALL DONE (Part A rc=$rcA, Part B rc=$rcB) ==="
touch "$OUT/DONE"
