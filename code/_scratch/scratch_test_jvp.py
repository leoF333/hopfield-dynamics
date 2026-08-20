
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import mlx.core as mx
import numpy as np

N = 5
p = 2
beta = 1.0
lam = 0.5
patterns = mx.random.randint(0, 2, [p, N]).astype(mx.float32) * 2.0 - 1.0
patterns_shifted = mx.concatenate([patterns[1:], patterns[:1]], axis=0)

x_curr = patterns[0]
x_del = patterns[0]

def field_F(x_curr, x_delayed):
    overlap_curr = patterns @ x_curr
    h = ((1.0 - lam) * (patterns.T @ overlap_curr / N)
         - (1.0 - lam) * (p / N) * x_curr)
    overlap_delayed = patterns @ x_delayed
    h = h + lam * (patterns_shifted.T @ overlap_delayed / N)
    return mx.tanh(beta * h)

def _jacobian_vmap_jvp(f, x: mx.array) -> np.ndarray:
    I = mx.eye(N, dtype=mx.float32)
    def jvp_col(v):
        _, tangent = mx.jvp(lambda z: f(z), [x], [v])
        return tangent[0]
    cols = mx.vmap(jvp_col)(I)
    mx.eval(cols)
    return np.array(cols, dtype=np.float32).T

f_curr = lambda x: field_F(x, x_del)
f_del  = lambda y: field_F(x_curr, y)

A = _jacobian_vmap_jvp(f_curr, x_curr)
B = _jacobian_vmap_jvp(f_del, x_del)

v_x = mx.random.normal([N])
v_y = mx.random.normal([N])

out_old = A @ np.array(v_x) + B @ np.array(v_y)

f_val = field_F(x_curr, x_del)
D = beta * (1.0 - f_val**2)
term_x = (1.0 - lam) * (patterns.T @ (patterns @ v_x) / N - (p / N) * v_x)
term_y = lam * (patterns_shifted.T @ (patterns_shifted @ v_y) / N)
out_new = D * (term_x + term_y)

print("A matrix:\n", A)
print("B matrix:\n", B)
print("out_old:\n", out_old)
print("out_new:\n", np.array(out_new))
