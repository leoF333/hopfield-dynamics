
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import numpy as np
from jacobian_spectrum import sweep_lambda_both
import os, sys

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

def main():
    N = 300
    beta = 20.0
    alpha_list = [0.01, 0.05, 0.1]
    tau_list = [1, 10, 20]
    n_lam = 21 
    lambda_values = np.linspace(0.0, 1.0, n_lam)
    
    sys.stdout = open(os.devnull, 'w')
    
    results = []
    for alpha in alpha_list:
        for tau in tau_list:
            fp_mean, _, cyc_mean, _ = sweep_lambda_both(
                N=N, alpha=alpha, tau=tau,
                lambda_values=lambda_values,
                n_seeds=1, beta=beta, n_eig=1
            )
            
            fp_crossings = find_crossings(lambda_values, fp_mean)
            cyc_crossings = find_crossings(lambda_values, cyc_mean)
            
            # Filter crossings
            fp_c = fp_crossings[0] if fp_crossings else None
            # The cycle becomes stable at high lambda, so it crosses 1 from above.
            # We take the last crossing.
            cyc_c = cyc_crossings[-1] if cyc_crossings else None
            
            results.append((alpha, tau, fp_c, cyc_c))
            
    sys.stdout = sys.__stdout__
    
    print("| Alpha | Tau | FP Destabilizes (L_c1) | Cycle Stabilizes (L_c2) |")
    print("|-------|-----|------------------------|-------------------------|")
    for alpha, tau, fp_c, cyc_c in results:
        fp_str = f"{fp_c:.3f}" if fp_c is not None else "None"
        cyc_str = f"{cyc_c:.3f}" if cyc_c is not None else "None"
        print(f"| {alpha:5.2f} | {tau:3d} | {fp_str:22s} | {cyc_str:23s} |")

if __name__ == "__main__":
    main()
