
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import mlx.core as mx
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import time

N = 10000
p = 500
beta = 20.0
lam = 0.5
tau = 2

N_float = float(N)
p_float = float(p)
lam_mx = mx.array(lam, dtype=mx.float32)

patterns = mx.random.randint(0, 2, [p, N]).astype(mx.float32) * 2.0 - 1.0
patterns_shifted = mx.concatenate([patterns[1:], patterns[:1]], axis=0)

def field_F(x_curr, x_delayed):
    overlap_curr = patterns @ x_curr
    h = ((1.0 - lam) * (patterns.T @ overlap_curr / N)
         - (1.0 - lam) * (p / N) * x_curr)
    overlap_delayed = patterns_shifted @ x_delayed
    h = h + lam * (patterns_shifted.T @ overlap_delayed / N)
    return mx.tanh(beta * h)

D_list = []
period = p * (tau + 1)
print(f"Period = {period}")
for t in range(period):
    mu_curr = (t // (tau + 1)) % p
    mu_del  = ((t - tau) // (tau + 1)) % p
    
    x_curr = patterns[mu_curr]
    x_del  = patterns[mu_del]
    
    f_val = field_F(x_curr, x_del)
    D = beta * (1.0 - f_val**2)
    mx.eval(D)
    D_list.append(D)

def monodromy_matvec(v_np):
    history = [mx.array(v_np[i*N:(i+1)*N], dtype=mx.float32) for i in range(tau+1)]
    for D in D_list:
        v_curr = history[0]
        v_del = history[tau]
        
        overlap_x = patterns @ v_curr
        term_x = (1.0 - lam_mx) * (patterns.T @ overlap_x / N_float - (p_float / N_float) * v_curr)
        overlap_y = patterns_shifted @ v_del
        term_y = lam_mx * (patterns_shifted.T @ overlap_y / N_float)
        
        out_new = D * (term_x + term_y)
        
        history = [out_new] + history[:-1]
        
    out_cat = mx.concatenate(history)
    mx.eval(out_cat)
    return np.array(out_cat, dtype=np.float64)

v_test = np.random.randn(N * (tau + 1))

t0 = time.time()
res = monodromy_matvec(v_test)
t1 = time.time()
print(f"One matvec took {t1-t0:.4f} seconds")

dim = N * (tau + 1)
op = LinearOperator((dim, dim), matvec=monodromy_matvec, dtype=np.float64)
# evals, _ = eigs(op, k=1, which="LM", tol=1e-2, maxiter=10)
# print(evals)

