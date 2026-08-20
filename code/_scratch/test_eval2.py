
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

print("Creating arrays...")
# patterns: (1, n_seeds, p, N)
patterns = mx.ones((1, n_seeds, p, N))

# sigma: (n_lambda, n_seeds, N, 1)
sigma = mx.ones((n_lambda, n_seeds, N, 1))

print("Matmul...")
out = mx.matmul(patterns, sigma)
mx.eval(out)
print(out.shape)
print("Success!")
