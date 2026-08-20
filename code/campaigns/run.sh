#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h}"
export MPLCONFIGDIR="$ROOT_DIR/cache/matplotlib"
export V5_BACKEND="${V5_BACKEND:-cpu}"
export PYTHONPATH="$ROOT_DIR/src"

mode="${1:-list}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "$mode" in
  smoke)
    conda run --no-capture-output -n mcmc_env python -m pytest "$ROOT_DIR/tests/test_core.py" -q
    conda run --no-capture-output -n mcmc_env python "$ROOT_DIR/tests/smoke_n1.py"
    ;;
  mlx-preflight)
    V5_BACKEND=mlx conda run --no-capture-output -n mcmc_env python "$ROOT_DIR/src/mlx_preflight.py"
    ;;
  N1-static)
    conda run --no-capture-output -n mcmc_env python "$ROOT_DIR/src/static_thresholds.py" "$@"
    ;;
  N1-dynamic)
    conda run --no-capture-output -n mcmc_env python "$ROOT_DIR/src/dynamics.py" "$@"
    ;;
  list)
    conda run --no-capture-output -n mcmc_env python "$ROOT_DIR/src/run_workplan.py" list
    ;;
  N1|N2|N3|N4|N5A|N5B|N6|N7|N8|N9|N10)
    conda run --no-capture-output -n mcmc_env python "$ROOT_DIR/src/run_workplan.py" "$mode" "$@"
    ;;
  *)
    print -u2 "Unknown mode: $mode"
    exit 2
    ;;
esac
