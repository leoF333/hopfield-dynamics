
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import mlx.core as mx

n_lambda = 40
n_seeds = 10
B = n_lambda * n_seeds
p = 3000
N = 10000

# Patterns for each seed
patterns = mx.ones((n_seeds, p, N))

# We want patterns to match flat B
# We can repeat patterns n_lambda times
# e.g., patterns_b = mx.tile(patterns, (n_lambda, 1, 1))

patterns_b = mx.tile(patterns, (n_lambda, 1, 1))
print("Tiled shape:", patterns_b.shape)

sigma = mx.ones((B, N, 1))
out = mx.matmul(patterns_b, sigma)
print(out.shape)
