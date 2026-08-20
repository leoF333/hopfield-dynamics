
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
CHICAGO_S = str(CHICAGO)
# ---------------------------------------------------------------------------


# -----------------------------------------------------------------
# MLX Computation Engine
# -----------------------------------------------------------------
@mx.compile
def fused_step_and_track(sigma_curr, sigma_delayed, patterns_b, patterns_shifted_b, 
                         p_float, N_float, beta, lambda_vals, 
                         max_reached_mu, jump_count, last_jump_time, has_collapsed, 
                         t_val, mx_threshold, mx_collapse_thresh):
    
    # --- 1. Compute Overlaps ---
    overlap_curr = mx.matmul(patterns_b, sigma_curr) 
    abs_overlaps = mx.abs(overlap_curr)
    
    # Extract the maximum overlap and its index (dominant pattern)
    max_overlap = mx.max(abs_overlaps, axis=1) / N_float # Shape: (B, 1)
    dominant_mu = mx.argmax(abs_overlaps, axis=1)        # Shape: (B, 1)
    
    # --- 2. Track Sequence Progression ---
    is_valid_state = max_overlap > mx_threshold
    # A true jump only counts if we reach a pattern further along the sequence
    new_jump = (dominant_mu > max_reached_mu) & is_valid_state
    
    new_jump_count = mx.where(new_jump, jump_count + 1.0, jump_count)
    new_last_jump_time = mx.where(new_jump, t_val, last_jump_time)
    new_max_reached_mu = mx.maximum(max_reached_mu, dominant_mu)
    
    # --- 3. Track Chaotic Collapse ---
    # If the overlap drops below 0.20, the sequence is lost forever
    new_has_collapsed = has_collapsed | (max_overlap < mx_collapse_thresh)

    # --- 4. Network Dynamics ---
    patterns_T = mx.transpose(patterns_b, (0, 2, 1)) 
    h_sym = (1.0 - lambda_vals) * (mx.matmul(patterns_T, overlap_curr) / N_float)
    h_sym = h_sym - (1.0 - lambda_vals) * (p_float / N_float) * sigma_curr
    
    overlap_delayed = mx.matmul(patterns_b, sigma_delayed) 
    patterns_shifted_T = mx.transpose(patterns_shifted_b, (0, 2, 1))
    h_asym = lambda_vals * (mx.matmul(patterns_shifted_T, overlap_delayed) / N_float)
    
    h = h_sym + h_asym
    
    prob_plus_one = 0.5 * (1.0 + mx.tanh(beta * h))
    r = mx.random.uniform(0.0, 1.0, sigma_curr.shape)
    sigma_new = mx.where(r < prob_plus_one, mx.array(1.0), mx.array(-1.0))
    
    return sigma_new, new_max_reached_mu, new_jump_count, new_last_jump_time, new_has_collapsed

# ---------------------------------------------------------
# Batched Classifier
# ---------------------------------------------------------
def get_phase_and_time_batched(N, alpha, lambda_array, tau, steps_max, n_seeds=10, threshold=0.4, T=0.05):
    n_lambda = len(lambda_array)
    B = n_lambda * n_seeds
    
    repeated_lambdas = np.repeat(lambda_array, n_seeds)
    lambda_vals = mx.array(repeated_lambdas, dtype=mx.float32).reshape(B, 1, 1) 
    
    p = max(1, int(alpha * N))
    p_float = float(p)
    N_float = float(N)
    beta = 1.0 / T
    
    patterns_int = mx.random.randint(0, 2, [n_seeds, p, N])
    patterns = patterns_int.astype(mx.float32) * 2.0 - 1.0
    patterns_shifted = mx.concatenate([patterns[:, 1:, :], patterns[:, :1, :]], axis=1)
    
    # Tile patterns to match B
    patterns_b = mx.repeat(patterns, n_lambda, axis=0)
    patterns_shifted_b = mx.repeat(patterns_shifted, n_lambda, axis=0)
    
    target = mx.reshape(patterns_b[:, 0, :], (B, N, 1))
    sigma = target
    history = [sigma] * (tau + 1)
    
    # Initialize rigorous tracking tensors
    max_reached_mu = mx.zeros((B, 1), dtype=mx.uint32)
    jump_count = mx.zeros((B, 1), dtype=mx.float32)
    last_jump_time = mx.zeros((B, 1), dtype=mx.float32)
    has_collapsed = mx.zeros((B, 1), dtype=mx.bool_)
    
    mx_threshold = mx.array(threshold, dtype=mx.float32)
    mx_collapse_thresh = mx.array(0.20, dtype=mx.float32)
    
    for t in range(steps_max):
        sigma_curr = history[-1]
        sigma_delayed = history[0]
        t_val = mx.array(float(t), dtype=mx.float32)
        
        sigma_new, max_reached_mu, jump_count, last_jump_time, has_collapsed = fused_step_and_track(
            sigma_curr, sigma_delayed, patterns_b, patterns_shifted_b, 
            p_float, N_float, beta, lambda_vals, 
            max_reached_mu, jump_count, last_jump_time, has_collapsed, 
            t_val, mx_threshold, mx_collapse_thresh
        )
        
        history.append(sigma_new)
        history.pop(0)
        
        if t % 25 == 0:
            mx.eval(sigma_new, max_reached_mu, jump_count, last_jump_time, has_collapsed)
            
    mx.eval(max_reached_mu, jump_count, last_jump_time, has_collapsed)
    
    jump_count_np = np.array(jump_count).flatten()
    last_jump_time_np = np.array(last_jump_time).flatten()
    has_collapsed_np = np.array(has_collapsed).flatten()
    
    # --- Strict Phase Classification ---
    phases = np.zeros(B)
    
    chaos_mask = has_collapsed_np
    phases[chaos_mask] = -1.0
    
    static_mask = (~chaos_mask) & (jump_count_np == 0)
    phases[static_mask] = 0.0
    
    dynamic_mask = (~chaos_mask) & (jump_count_np > 0)
    phases[dynamic_mask] = last_jump_time_np[dynamic_mask] / jump_count_np[dynamic_mask]
    
    # Aggregate over seeds
    phases_2d = phases.reshape(n_lambda, n_seeds)
    
    final_phases = np.zeros(n_lambda)
    for i in range(n_lambda):
        seed_phases = phases_2d[i]
        
        n_chaos = np.sum(seed_phases == -1.0)
        n_static = np.sum(seed_phases == 0.0)
        dyn_mask = seed_phases > 0.0
        n_dyn = np.sum(dyn_mask)
        
        if n_chaos >= max(n_static, n_dyn):
            final_phases[i] = -1.0
        elif n_static >= max(n_chaos, n_dyn):
            final_phases[i] = 0.0
        else:
            if n_dyn > 0:
                final_phases[i] = np.median(seed_phases[dyn_mask])
            else:
                final_phases[i] = 0.0
            
    return final_phases

# ---------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------
if __name__ == "__main__":
    N = 1000
    n_seeds = 10
    tau_values = [1, 10, 20]
    
    threshold = 0.40      
    n_alpha = 40
    n_lambda = 40
    
    alpha_values = np.linspace(0.01, 0.30, n_alpha)
    lambda_values = np.linspace(0.0, 1.0, n_lambda)
    
    os.makedirs(os.path.dirname(CHICAGO_S + "/3_numerics/results/phase_diagrams_mcmc_1x3_T0.png"), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, tau in enumerate(tau_values):
        steps_max = tau * 30  
        phase_matrix = np.zeros((n_lambda, n_alpha))
        
        print(f"Running tau={tau} ({n_alpha}x{n_lambda}) with {n_seeds} seeds at T=0...")
        
        for j, a_val in enumerate(tqdm(alpha_values, desc=f"Sweeping Alpha (tau={tau})", ascii=True, colour="blue")):
            column_phases = get_phase_and_time_batched(N, a_val, lambda_values, tau, steps_max, n_seeds=n_seeds, threshold=threshold, T=0.0001)
            phase_matrix[:, j] = column_phases

        chaos_data = np.ma.masked_where(phase_matrix != -1.0, phase_matrix)
        static_data = np.ma.masked_where(phase_matrix != 0.0, phase_matrix)
        dynamic_data = np.ma.masked_where(phase_matrix <= 0.0, phase_matrix)

        ax = axes[idx]
        A, L = np.meshgrid(alpha_values, lambda_values)

        ax.pcolormesh(A, L, chaos_data, cmap=ListedColormap(['#12012e']), shading='nearest')
        ax.pcolormesh(A, L, static_data, cmap=ListedColormap(['#fcf6bd']), shading='nearest')
        
        mesh_dyn = ax.pcolormesh(A, L, dynamic_data, cmap='plasma_r', 
                                 vmin=tau+1, vmax=max(tau*4, tau+2), shading='nearest')

        cbar = fig.colorbar(mesh_dyn, ax=ax, pad=0.02)
        cbar.set_label(f"Residence Time (Ideal = {tau+1})", fontsize=10)

        legend_elements = [
            Patch(facecolor='#12012e', edgecolor='black', label='Chaotic'),
            Patch(facecolor='#fcf6bd', edgecolor='black', label='Static'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        ax.set_title(rf"$\tau={tau}$", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"$\alpha = p/N$", fontsize=12)
        if idx == 0:
            ax.set_ylabel(r"$\lambda$", fontsize=12)
        
        ax.axvline(0.138, color='cyan', linestyle=':', linewidth=2)
        ax.axvline(0.269, color='lightgreen', linestyle=':', linewidth=2)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(CHICAGO_S + "/3_numerics/results/phase_diagrams_mcmc_1x3_T0.png", dpi=150)
    print("Saved to numerics/results/phase_diagrams_mcmc_1x3_T0.png")
