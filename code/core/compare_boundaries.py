
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import numpy as np
from phase_diagrams_mcmc import get_phase_and_time_batched
import sys, os

def main():
    N = 1000
    n_seeds = 10
    threshold = 0.40
    
    alpha_list = [0.01, 0.05, 0.1]
    tau_list = [1, 10, 20]
    
    n_lam = 41
    lambda_values = np.linspace(0.0, 1.0, n_lam)
    
    analytical = {
        (0.01, 1): (0.412, 0.546),
        (0.01, 10): (0.403, 0.554),
        (0.01, 20): (0.413, 0.555),
        (0.05, 1): (0.355, 0.593),
        (0.05, 10): (0.277, 0.591),
        (0.05, 20): (0.368, 0.589),
        (0.10, 1): (0.308, 0.623),
        (0.10, 10): (0.246, 0.616),
        (0.10, 20): (0.264, 0.621),
    }
    
    print("| Alpha | Tau | L_c1 (Ana) | L_c1 (MCMC)| Match 1 | L_c2 (Ana) | L_c2 (MCMC) | Match 2 |")
    print("|-------|-----|------------|------------|---------|------------|-------------|---------|")
    
    # Hide tqdm output if any, or general prints
    sys.stdout = sys.__stdout__
    
    for alpha in alpha_list:
        for tau in tau_list:
            steps_max = tau * 30
            # MCMC output is an array of phases for each lambda (using T=0.0001 to simulate T=0)
            phases = get_phase_and_time_batched(N, alpha, lambda_values, tau, steps_max, n_seeds=n_seeds, threshold=threshold, T=0.0001)
            
            # Statique ends when phase is no longer 0.0
            non_static_indices = np.where(phases != 0.0)[0]
            if len(non_static_indices) > 0:
                l_c1_mcmc = lambda_values[non_static_indices[0]]
                # Interpolate if we want to be more exact, but taking the first non-0 is fine.
                # Actually, the boundary is between the last 0.0 and first non-0.0
                last_static = non_static_indices[0] - 1
                if last_static >= 0:
                    l_c1_mcmc = (lambda_values[last_static] + lambda_values[non_static_indices[0]]) / 2.0
            else:
                l_c1_mcmc = np.nan
                
            # Dynamique parfaite begins when phase is VERY CLOSE to tau + 1 (tolerance to account for 1 thermal stutter)
            perfect_dyn_indices = np.where(np.abs(phases - (tau + 1)) < 0.25)[0]
            if len(perfect_dyn_indices) > 0:
                l_c2_mcmc = lambda_values[perfect_dyn_indices[0]]
                # Boundary is between previous and this one
                prev = perfect_dyn_indices[0] - 1
                if prev >= 0:
                    l_c2_mcmc = (lambda_values[prev] + lambda_values[perfect_dyn_indices[0]]) / 2.0
            else:
                l_c2_mcmc = np.nan
                
            l_c1_ana, l_c2_ana = analytical[(alpha, tau)]
            
            diff1 = abs(l_c1_ana - l_c1_mcmc) if not np.isnan(l_c1_mcmc) else np.nan
            diff2 = abs(l_c2_ana - l_c2_mcmc) if not np.isnan(l_c2_mcmc) else np.nan
            
            # Using 0.05 tolerance since MCMC is stochastic and n_lam=41 gives spacing 0.025
            match1 = "YES" if not np.isnan(diff1) and diff1 <= 0.05 else "NO"
            match2 = "YES" if not np.isnan(diff2) and diff2 <= 0.05 else "NO"
            
            print(f"| {alpha:5.2f} | {tau:3d} | {l_c1_ana:10.3f} | {l_c1_mcmc:10.3f} | {match1:7s} | {l_c2_ana:10.3f} | {l_c2_mcmc:11.3f} | {match2:7s} |")

if __name__ == "__main__":
    main()
