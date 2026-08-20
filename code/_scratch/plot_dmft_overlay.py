
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 (folder reorganisation) ---
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
(CHICAGO / "3_numerics/results/_scratch").mkdir(parents=True, exist_ok=True)


tau_list = [1, 2, 4, 6, 8, 10, 15, 20]
mcmc_c1 = [0.288, 0.288, 0.263, 0.288, 0.237, 0.288, 0.263, 0.288]
mcmc_c2 = [0.613, 0.613, 0.588, 0.588, 0.562, 0.562, 0.538, 0.512]
ana_c1 = [0.274, 0.248, 0.279, 0.296, 0.278, 0.283, 0.289, 0.270]
ana_c2 = [0.607, 0.606, 0.608, 0.605, 0.609, 0.602, 0.606, 0.609]

dmft_c1 = [0.5657] * len(tau_list)

plt.figure(figsize=(10, 6))

plt.plot(tau_list, ana_c1, 'b-o', label=r'$\lambda_c^{(1)}$ (Stabilité Naïve)')
plt.plot(tau_list, mcmc_c1, 'b--s', label=r'$\lambda_c^{(1)}$ (MCMC $T=0.05, N=3000$)')
plt.plot(tau_list, dmft_c1, 'b-.', linewidth=2, label=r'$\lambda_c^{(1)}$ (DMFT Exacte)')

plt.plot(tau_list, ana_c2, 'r-o', label=r'$\lambda_c^{(2)}$ (Stabilité Naïve)')
plt.plot(tau_list, mcmc_c2, 'r--s', label=r'$\lambda_c^{(2)}$ (MCMC $T=0.05, N=3000$)')

plt.title(r"Valeurs critiques vs $\tau$ ($\alpha=0.05$) - Confrontation DMFT", fontsize=14)
plt.xlabel(r"Délai temporel $\tau$", fontsize=12)
plt.ylabel(r"$\lambda$ critique", fontsize=12)

plt.axvspan(0, 20, facecolor='gray', alpha=0.1, label='Zone de Handoff Dynamique')

plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig(str(CHICAGO / "3_numerics/results/_scratch/critical_lambda_vs_tau_dmft_overlay.png"), dpi=150)
print("Plot generated successfully!")
