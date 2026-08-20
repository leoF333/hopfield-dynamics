
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import mlx.core as mx

B = 400
p = 3000
N = 10000

# Try broadcasting
patterns = mx.ones((1, p, N))
sigma = mx.ones((B, N, 1))

out = mx.matmul(patterns, sigma)
print(out.shape)
