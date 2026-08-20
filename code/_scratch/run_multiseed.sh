#!/bin/bash

# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO="${CHICAGO_ROOT:-/Users/leoflack/Desktop/Recherche/Chicago}"
# ---------------------------------------------------------------------------

set -u
SRC="$CHICAGO/3_numerics/src"
C="/opt/homebrew/Caskroom/miniforge/base/bin/conda"
O10="$CHICAGO/3_numerics/results/N=10000"
O2="$CHICAGO/3_numerics/results/N=2000"
LOG="$O10/multiseed.log"; mkdir -p "$O10" "$O2"; export TQDM_DISABLE=1; cd "$SRC"; : > "$LOG"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
log "=== alpha=0.05 N=10000 (5 seeds) ==="
"$C" run -n mcmc_env python multiseed_fold.py --N 10000 --alpha 0.05 --seeds 42 43 44 45 46 --outdir "$O10" >>"$LOG" 2>&1
log "=== alpha=0.10 N=2000 (8 seeds) ==="
"$C" run -n mcmc_env python multiseed_fold.py --N 2000 --alpha 0.10 --seeds 42 43 44 45 46 47 48 49 --outdir "$O2" >>"$LOG" 2>&1
log "=== alpha=0.10 N=10000 (8 seeds) ==="
"$C" run -n mcmc_env python multiseed_fold.py --N 10000 --alpha 0.10 --seeds 42 43 44 45 46 47 48 49 --outdir "$O10" >>"$LOG" 2>&1
log "=== MULTISEED DONE ==="; touch "$O10/DONE_MULTISEED"
