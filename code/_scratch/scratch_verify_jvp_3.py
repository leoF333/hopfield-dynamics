
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import mlx.core as mx
import numpy as np
import time

N = 10000
p = 500
tau = 2
lam_mx = mx.array(0.5, dtype=mx.float32)
N_float = float(N)
p_float = float(p)

patterns = mx.random.normal([p, N]).astype(mx.float32)
patterns_shifted = mx.random.normal([p, N]).astype(mx.float32)

D_list = [mx.random.normal([N]).astype(mx.float32) for _ in range(100)]

@mx.compile
def run_block(history, block_idx):
    for i in range(100):
        D = D_list[i]
        v_curr = history[0]
        v_del = history[tau]
        
        overlap_x = patterns @ v_curr
        term_x = (1.0 - lam_mx) * (patterns.T @ overlap_x / N_float - (p_float / N_float) * v_curr)
        overlap_y = patterns_shifted @ v_del
        term_y = lam_mx * (patterns_shifted.T @ overlap_y / N_float)
        
        out_new = D * (term_x + term_y)
        
        history = [out_new] + history[:-1]
    return history

history = [mx.random.normal([N]) for _ in range(tau+1)]

# Warmup compile
_ = run_block(history, 0)
mx.eval(_)

t0 = time.time()
for _ in range(15):  # 1500 steps
    history = run_block(history, 0)
mx.eval(history)
t1 = time.time()

print(f"1500 steps took {t1-t0:.4f} seconds")
