
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
import time
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# Import the core sweeping function
from jacobian_spectrum import sweep_lambda_both, plot_scissors

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
CHICAGO_S = str(CHICAGO)
# ---------------------------------------------------------------------------


def main():
    N = 1000
    beta = 20.0
    n_seeds = 1
    n_lam = 21
    lam_min = 0.0
    lam_max = 1.0
    
    alpha_list = [0.01, 0.05, 0.1]
    tau_list = [1, 10, 20]
    
    lambda_values = np.linspace(lam_min, lam_max, n_lam)
    
    outdir = CHICAGO_S + "/3_numerics/results"
    os.makedirs(outdir, exist_ok=True)
    
    # To store data for synthesis plot
    results = {}
    
    t0_global = time.time()
    
    for alpha in alpha_list:
        for tau in tau_list:
            print(f"\n========================================")
            print(f" Running alpha={alpha}, tau={tau}")
            print(f"========================================")
            
            t0 = time.time()
            fp_mean, fp_std, cyc_mean, cyc_std = sweep_lambda_both(
                N=N, alpha=alpha, tau=tau,
                lambda_values=lambda_values,
                n_seeds=n_seeds, beta=beta, n_eig=1
            )
            t1 = time.time()
            print(f"  -> Finished in {t1 - t0:.1f} seconds")
            
            # Save individual plot
            outfile = os.path.join(outdir, f"stability_scissors_N{N}_tau{tau}_a{alpha:.3f}.png")
            plot_scissors(lambda_values, fp_mean, fp_std, cyc_mean, cyc_std, alpha, tau, outfile)
            
            # Store results
            results[(alpha, tau)] = (fp_mean, fp_std, cyc_mean, cyc_std)
            
    t1_global = time.time()
    print(f"\nGrid search completed in {(t1_global - t0_global)/60:.1f} minutes.")
    
    # Create synthesis plot
    fig, axes = plt.subplots(len(alpha_list), len(tau_list), figsize=(15, 12), sharex=True, sharey=True)
    
    for i, alpha in enumerate(alpha_list):
        for j, tau in enumerate(tau_list):
            ax = axes[i, j]
            fp_mean, fp_std, cyc_mean, cyc_std = results[(alpha, tau)]
            
            ax.plot(lambda_values, fp_mean, "b-", lw=1.5, label="Fixed Point")
            ax.fill_between(lambda_values, fp_mean - fp_std, fp_mean + fp_std, alpha=0.2, color="blue")
            
            ax.plot(lambda_values, cyc_mean, "r-", lw=1.5, label="Floquet Cycle")
            ax.fill_between(lambda_values, cyc_mean - cyc_std, cyc_mean + cyc_std, alpha=0.2, color="red")
            
            ax.axhline(1.0, color="k", linestyle="--", lw=1.0)
            
            if i == 0:
                ax.set_title(rf"$\tau = {tau}$")
            if j == 0:
                ax.set_ylabel(rf"$\alpha = {alpha}$" + "\nModulus $|z_{\max}|$")
            if i == len(alpha_list) - 1:
                ax.set_xlabel(r"$\lambda$")
                
            ax.grid(True, alpha=0.3)
            
            # Add a single legend to the first subplot
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
                
    plt.tight_layout()
    synth_outfile = os.path.join(outdir, f"synthesis_grid_N{N}.png")
    plt.savefig(synth_outfile, dpi=200, bbox_inches="tight")
    print(f"Synthesis figure saved: {synth_outfile}")

if __name__ == "__main__":
    main()
