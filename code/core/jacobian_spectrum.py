
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import argparse
import os
import time

import mlx.core as mx
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
import matplotlib
matplotlib.use("agg")  # Non-interactive backend to safely save files without blocking
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# 1. NETWORK FIELD
# ─────────────────────────────────────────────────────────────────

def field_F(x_curr, x_delayed, patterns, patterns_shifted,
            N_float: float, p_float: float, beta: float, lam):
    overlap_curr = patterns @ x_curr
    h = ((1.0 - lam) * (patterns.T @ overlap_curr / N_float)
         - (1.0 - lam) * (p_float / N_float) * x_curr)
    overlap_delayed = patterns @ x_delayed
    h = h + lam * (patterns_shifted.T @ overlap_delayed / N_float)
    return mx.tanh(beta * h)


# ─────────────────────────────────────────────────────────────────
# 2. FAST JACOBIAN-VECTOR PRODUCT (ANALYTICAL)
# ─────────────────────────────────────────────────────────────────

def get_dominant_eigenvalue_fixed_point(patterns, patterns_shifted,
                                        N_float, p_float, beta, lam_mx, tau,
                                        n_eig=1):
    N = int(N_float)
    dim = N * (tau + 1)
    
    x_curr = patterns[0]
    x_delayed = patterns[0]
    
    f_val = field_F(x_curr, x_delayed, patterns, patterns_shifted, N_float, p_float, beta, lam_mx)
    D = beta * (1.0 - f_val**2)
    mx.eval(D)
    
    def apply_AB(v_x_np, v_y_np):
        v_x = mx.array(v_x_np, dtype=mx.float32)
        v_y = mx.array(v_y_np, dtype=mx.float32)
        
        overlap_x = patterns @ v_x
        term_x = (1.0 - lam_mx) * (patterns.T @ overlap_x / N_float - (p_float / N_float) * v_x)
        
        overlap_y = patterns_shifted @ v_y
        term_y = lam_mx * (patterns_shifted.T @ overlap_y / N_float)
        
        out = D * (term_x + term_y)
        mx.eval(out)
        out_np = np.array(out, dtype=np.float64)
        
        # ARPACK (DLASCL) crashes if the operator returns exactly 0.0 or NaNs
        if np.any(np.isnan(out_np)):
            return np.ones_like(out_np) * 1e-8
        norm_val = np.linalg.norm(out_np)
        if norm_val < 1e-12:
            return v_x_np * 1e-8
        return out_np

    if tau == 0:
        def matvec(v):
            return apply_AB(v, v)
    else:
        def matvec(v):
            out = np.zeros(dim, dtype=np.float64)
            out[:N] = apply_AB(v[:N], v[tau*N:(tau+1)*N])
            out[N:] = v[:-N]
            return out
            
    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    
    try:
        evals, _ = eigs(op, k=n_eig, which="LM", tol=1e-3, maxiter=100)
        return np.max(np.abs(evals))
    except Exception as e:
        if "No convergence" in str(e) or "Starting vector is zero" in str(e):
            return 0.0
        print(f"  [Fixed Point ARPACK Failed] {e}")
        return np.nan


def get_dominant_eigenvalue_cycle(patterns, patterns_shifted,
                                  N_float, p_float, beta, lam_mx, tau,
                                  n_eig=1):
    N = int(N_float)
    p = int(p_float)
    period = p * (tau + 1)
    
    unique_D = {}
    for t in range(period):
        mu_curr = (t // (tau + 1)) % p
        mu_del  = ((t - tau) // (tau + 1)) % p
        pair = (mu_curr, mu_del)
        if pair not in unique_D:
            x_curr = patterns[mu_curr]
            x_del  = patterns[mu_del]
            
            f_val = field_F(x_curr, x_del, patterns, patterns_shifted, N_float, p_float, beta, lam_mx)
            D = beta * (1.0 - f_val**2)
            mx.eval(D)
            unique_D[pair] = D
            
    D_list = [unique_D[((t // (tau + 1)) % p, ((t - tau) // (tau + 1)) % p)] for t in range(period)]
    dim = N * (tau + 1)
    
    def monodromy_matvec(v_np):
        history = [mx.array(v_np[i*N:(i+1)*N], dtype=mx.float32) for i in range(tau+1)]
        
        for D in D_list:
            v_curr = history[0]
            v_del = history[-1]
            
            overlap_x = patterns @ v_curr
            term_x = (1.0 - lam_mx) * (patterns.T @ overlap_x / N_float - (p_float / N_float) * v_curr)
            
            overlap_y = patterns_shifted @ v_del
            term_y = lam_mx * (patterns_shifted.T @ overlap_y / N_float)
            
            out_new = D * (term_x + term_y)
            
            history = [out_new] + history[:-1]
            
        out_cat = mx.concatenate(history)
        mx.eval(out_cat)
        out_cat_np = np.array(out_cat, dtype=np.float64)
        
        # ARPACK (DLASCL) crashes if the operator returns exactly 0.0 or NaNs
        if np.any(np.isnan(out_cat_np)):
            return np.ones_like(out_cat_np) * 1e-8
        norm_val = np.linalg.norm(out_cat_np)
        if norm_val < 1e-12:
            return v_np * 1e-8
        return out_cat_np

    op = LinearOperator((dim, dim), matvec=monodromy_matvec, dtype=np.float64)
    
    try:
        evals, _ = eigs(op, k=n_eig, which="LM", tol=1e-3, maxiter=100)
        z_max = np.max(np.abs(evals))
        z_avg = z_max ** (1.0 / period)
        return z_avg
    except Exception as e:
        if "No convergence" in str(e) or "Starting vector is zero" in str(e):
            return 0.0
        print(f"  [Cycle ARPACK Failed] {e}")
        return np.nan


# ─────────────────────────────────────────────────────────────────
# 4. LAMBDA SWEEP
# ─────────────────────────────────────────────────────────────────

def sweep_lambda_both(N: int, alpha: float, tau: int,
                      lambda_values: np.ndarray,
                      n_seeds: int = 1,
                      beta: float = 20.0,
                      n_eig: int = 1):
    
    p = max(1, int(alpha * N))
    p_float = float(p)
    N_float = float(N)

    fp_mean, fp_std = [], []
    cyc_mean, cyc_std = [], []

    for lam in tqdm(lambda_values, desc=f"λ sweep  α={alpha:.3f}  τ={tau}"):
        lam_mx = mx.array(float(lam), dtype=mx.float32)
        seed_fp = []
        seed_cyc = []

        for _ in range(n_seeds):
            patterns_int = mx.random.randint(0, 2, [p, N])
            patterns = patterns_int.astype(mx.float32) * 2.0 - 1.0
            patterns_shifted = mx.concatenate([patterns[1:], patterns[:1]], axis=0)
            mx.eval(patterns, patterns_shifted)

            # Fixed point stability
            z_fp = get_dominant_eigenvalue_fixed_point(
                patterns, patterns_shifted, N_float, p_float, beta, lam_mx, tau, n_eig=n_eig)
            seed_fp.append(z_fp)
            
            # Cycle stability
            z_cyc = get_dominant_eigenvalue_cycle(
                patterns, patterns_shifted, N_float, p_float, beta, lam_mx, tau, n_eig=n_eig)
            seed_cyc.append(z_cyc)

        fp_mean.append(float(np.nanmean(seed_fp)))
        fp_std.append(float(np.nanstd(seed_fp)))
        cyc_mean.append(float(np.nanmean(seed_cyc)))
        cyc_std.append(float(np.nanstd(seed_cyc)))

    return np.array(fp_mean), np.array(fp_std), np.array(cyc_mean), np.array(cyc_std)


# ─────────────────────────────────────────────────────────────────
# 5. CLI & PLOTTING
# ─────────────────────────────────────────────────────────────────

def plot_scissors(lambda_values, fp_mean, fp_std, cyc_mean, cyc_std,
                  alpha: float, tau: int, outfile: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(lambda_values, fp_mean, "b-o", lw=2, ms=4, label="Fixed Point (motif 0)")
    ax.fill_between(lambda_values, fp_mean - fp_std, fp_mean + fp_std, alpha=0.2, color="blue")
    
    ax.plot(lambda_values, cyc_mean, "r-s", lw=2, ms=4, label="Floquet Cycle")
    ax.fill_between(lambda_values, cyc_mean - cyc_std, cyc_mean + cyc_std, alpha=0.2, color="red")
    
    ax.axhline(1.0, color="k", linestyle="--", lw=1.5, label="Stability Boundary (|z|=1)")
    
    ax.set_xlabel(r"Asymmetry parameter $\lambda$")
    ax.set_ylabel(r"Dominant modulus $|z_{\max}|$")
    ax.set_title(rf"Stability transition: Fixed Point vs Cycle ($\alpha={alpha:.2f}$, $\tau={tau}$)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {outfile}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Analytical spectral analysis for delayed mixed Hopfield")
    p.add_argument("--tau",     type=int,   default=10)
    p.add_argument("--N",       type=int,   default=300)
    p.add_argument("--alpha",   type=float, default=0.05)
    p.add_argument("--beta",    type=float, default=20.0)
    p.add_argument("--n_lam",   type=int,   default=11)
    p.add_argument("--lam_min", type=float, default=0.0)
    p.add_argument("--lam_max", type=float, default=1.0)
    p.add_argument("--n_seeds", type=int,   default=1)
    p.add_argument("--outdir",  default="numerics/results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("=== Analytical Stability Analysis ===")
    print(f"N={args.N}, α={args.alpha}, τ={args.tau}, seeds={args.n_seeds}")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    lambda_values = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    
    t0 = time.time()
    fp_mean, fp_std, cyc_mean, cyc_std = sweep_lambda_both(
        N=args.N, alpha=args.alpha, tau=args.tau,
        lambda_values=lambda_values,
        n_seeds=args.n_seeds, beta=args.beta, n_eig=1
    )
    t1 = time.time()
    
    print(f"\nTime taken: {t1 - t0:.1f} seconds")
    
    print("\nResults Table:")
    print(f"{'Lambda':>8} | {'FP |z|':>8} | {'Cyc |z|':>8}")
    print("-" * 31)
    for l, fp, cyc in zip(lambda_values, fp_mean, cyc_mean):
        print(f"{l:8.3f} | {fp:8.4f} | {cyc:8.4f}")
        
    outfile = os.path.join(args.outdir, f"stability_scissors_N{args.N}_tau{args.tau}_a{args.alpha:.3f}.png")
    plot_scissors(lambda_values, fp_mean, fp_std, cyc_mean, cyc_std, args.alpha, args.tau, outfile)
