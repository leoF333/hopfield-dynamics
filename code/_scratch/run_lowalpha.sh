#!/bin/bash

# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO="${CHICAGO_ROOT:-/Users/leoflack/Desktop/Recherche/Chicago}"
# ---------------------------------------------------------------------------

set -u
C="/opt/homebrew/Caskroom/miniforge/base/bin/conda"
O10="$CHICAGO/3_numerics/results/N=10000"
LOG="$O10/lowalpha.log"; mkdir -p "$O10"; export TQDM_DISABLE=1
cd "$CHICAGO/3_numerics/src"; : > "$LOG"
for a in 0.01 0.03; do
  echo "$(date '+%H:%M:%S') === alpha=$a N=10000 (5 seeds) ===" | tee -a "$LOG"
  "$C" run -n mcmc_env python multiseed_fold.py --N 10000 --alpha $a --seeds 42 43 44 45 46 --outdir "$O10" >>"$LOG" 2>&1
done
echo "$(date '+%H:%M:%S') === LOWALPHA DONE ===" | tee -a "$LOG"; touch "$O10/DONE_LOWALPHA"
