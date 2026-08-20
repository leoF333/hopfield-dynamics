
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import functions from our existing scripts
from jacobian_spectrum import sweep_lambda_both
from phase_diagrams_mcmc import get_phase_and_time_batched

def find_crossings(lambda_vals, z_vals):
    crossings = []
    for i in range(len(lambda_vals)-1):
        z1, z2 = z_vals[i], z_vals[i+1]
        l1, l2 = lambda_vals[i], lambda_vals[i+1]
        if (z1 - 1.0) * (z2 - 1.0) <= 0 and not np.isnan(z1) and not np.isnan(z2):
            if z2 == z1:
                lc = l1
            else:
                lc = l1 + (1.0 - z1) * (l2 - l1) / (z2 - z1)
            crossings.append(lc)
    return crossings

def get_ana_criticals(alpha, tau, lambda_values, N=10000, beta=20.0):
    fp_mean, _, cyc_mean, _ = sweep_lambda_both(
        N=N, alpha=alpha, tau=tau,
        lambda_values=lambda_values,
        n_seeds=1, beta=beta, n_eig=1
    )
    fp_crossings = find_crossings(lambda_values, fp_mean)
    cyc_crossings = find_crossings(lambda_values, cyc_mean)
    
    fp_c = fp_crossings[0] if fp_crossings else np.nan
    cyc_c = cyc_crossings[-1] if cyc_crossings else np.nan
    return fp_c, cyc_c

def get_mcmc_criticals(alpha, tau, lambda_values, N=10000, n_seeds=10, threshold=0.40, T=0.05):
    steps_max = tau * 30
    phases = get_phase_and_time_batched(N, alpha, lambda_values, tau, steps_max, n_seeds=n_seeds, threshold=threshold, T=T)
    
    non_static_indices = np.where(phases != 0.0)[0]
    if len(non_static_indices) > 0:
        l_c1_mcmc = lambda_values[non_static_indices[0]]
        last_static = non_static_indices[0] - 1
        if last_static >= 0:
            l_c1_mcmc = (lambda_values[last_static] + lambda_values[non_static_indices[0]]) / 2.0
    else:
        l_c1_mcmc = np.nan
        
    perfect_dyn_indices = np.where(np.abs(phases - (tau + 1)) < 0.25)[0]
    if len(perfect_dyn_indices) > 0:
        l_c2_mcmc = lambda_values[perfect_dyn_indices[0]]
        prev = perfect_dyn_indices[0] - 1
        if prev >= 0:
            l_c2_mcmc = (lambda_values[prev] + lambda_values[perfect_dyn_indices[0]]) / 2.0
    else:
        l_c2_mcmc = np.nan
        
    return l_c1_mcmc, l_c2_mcmc

def main():
    alpha_list = [0.05]
    tau_list = [1, 2, 4, 6, 8, 10, 15, 20]
    n_lam = 41
    lambda_values = np.linspace(0.0, 1.0, n_lam)
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    for idx, alpha in enumerate(alpha_list):
        print(f"\n==========================================")
        print(f"=== Sweeping Alpha = {alpha} ===")
        print(f"==========================================")
        
        ana_c1, ana_c2 = [], []
        mcmc_c1, mcmc_c2 = [], []
        
        for tau in tau_list:
            print(f"\n--- Computing for tau={tau} ---")
            
            c1_m, c2_m = get_mcmc_criticals(alpha, tau, lambda_values, T=0.05)
            mcmc_c1.append(c1_m)
            mcmc_c2.append(c2_m)
            
            c1_a, c2_a = get_ana_criticals(alpha, tau, lambda_values)
            ana_c1.append(c1_a)
            ana_c2.append(c2_a)
            
            print(f"MCMC: c1={c1_m:.3f}, c2={c2_m:.3f}" if not np.isnan(c2_m) else f"MCMC: c1={c1_m:.3f}, c2=NaN")
            print(f"ANA : c1={c1_a:.3f}, c2={c2_a:.3f}" if not np.isnan(c2_a) else f"ANA : c1={c1_a:.3f}, c2=NaN")
            
        ax = axes
        ax.plot(tau_list, ana_c1, 'b-o', label=r'$\lambda_c^{(1)}$ (Analytique)')
        ax.plot(tau_list, mcmc_c1, 'b--s', label=r'$\lambda_c^{(1)}$ (MCMC $T=0.05$)')
        
        ax.plot(tau_list, ana_c2, 'r-o', label=r'$\lambda_c^{(2)}$ (Analytique)')
        ax.plot(tau_list, mcmc_c2, 'r--s', label=r'$\lambda_c^{(2)}$ (MCMC $T=0.05$)')
        
        ax.set_title(fr"Valeurs critiques vs $\tau$ ($\alpha={alpha}, N=10000$)", fontsize=14)
        ax.set_xlabel(r"Délai temporel $\tau$", fontsize=12)
        ax.set_ylabel(r"$\lambda$ critique", fontsize=12)
        
        ax.set_ylim(0, 1)
        ax.set_xticks(tau_list)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = "../results/critical_lambda_vs_tau_a0.05_N10000.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
