#!/bin/zsh

# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO="${CHICAGO_ROOT:-/Users/leoflack/Desktop/Recherche/Chicago}"
# ---------------------------------------------------------------------------

# Health probe for the overnight campaign: CPU load, GPU, E30 progress, stall check.
echo "=== HEALTH $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "-- load average / cpu --"
uptime | sed 's/.*load averages*: //' | awk '{print "  load(1/5/15): "$1" "$2" "$3}'
echo "  logical cores: $(sysctl -n hw.ncpu)"
echo "-- python compute procs (e30/tracer) --"
ps -Ao pid,pcpu,pmem,etime,comm | grep -E "python" | grep -v grep | head -12 | awk '{printf "  pid %s cpu %s%% mem %s%% up %s %s\n",$1,$2,$3,$4,$5}'
NP=$(ps -Ao pcpu,comm | grep python | grep -v grep | awk '{s+=$1} END{print s+0}')
echo "  total python CPU%: $NP  (≈$(echo "scale=1; $NP/100" | bc) cores busy)"
echo "-- GPU (MLX float32: Exp A basin sweep when e32 runs) --"
ioreg -r -d1 -c IOAccelerator 2>/dev/null | grep -o '"PerformanceStatistics"={[^}]*}' | grep -o '"Device Utilization %"=[0-9]*' | head -1 | sed 's/^/  /' || echo "  (GPU util not readable without sudo; workload is CPU-bound by design)"
echo "-- E30 scaling campaign --"
D=$CHICAGO/3_numerics/results/E30_scaling
NPZ=$(ls "$D"/*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "  npz done: $NPZ / 325 planned"
NEW=$(ls -t "$D"/*.npz 2>/dev/null | head -1)
if [ -n "$NEW" ]; then
  AGE=$(( $(date +%s) - $(stat -f %m "$NEW") ))
  echo "  newest: $(basename "$NEW")  (${AGE}s ago)"
  if [ "$AGE" -gt 1800 ]; then echo "  *** STALL WARNING: no new npz in >30 min ***"; fi
fi
pgrep -f e30_scaling >/dev/null && echo "  driver: ALIVE" || echo "  driver: STOPPED (finished or crashed?)"
echo "  log tail:"; tail -1 /tmp/e30.log 2>/dev/null | sed 's/^/    /'
echo "-- Exp A basin sweep (e32, GPU float32) --"
if pgrep -f e32_alpha >/dev/null; then
  echo "  e32(GPU): ALIVE"
  echo "  cells done: $(grep -c 'done' /tmp/e32.log 2>/dev/null) / 12"
  A32=$(( $(date +%s) - $(stat -f %m /tmp/e32.log 2>/dev/null || date +%s) ))
  echo "  e32 log last write: ${A32}s ago"
  [ "$A32" -gt 1800 ] && echo "  *** e32 STALL WARNING ***"
else
  echo "  e32(GPU): not running (finished, not launched, or crashed)"
fi
