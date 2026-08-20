
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import mlx.core as mx
import numpy as np

n_lambda = 2
n_seeds = 3

lambda_array = np.array([10, 20])
lambdas = np.repeat(lambda_array, n_seeds)
print("Lambdas:", lambdas)

patterns = mx.array([[0], [1], [2]])
patterns_b = mx.tile(patterns, (n_lambda, 1))
print("Patterns:\n", patterns_b)
