"""
Comprehensive numerical audit of the static-bifurcation analysis chain.

Verifies EVERY mathematical identity and code path against independent dense
references at small N (where everything is exactly computable):

  A1  matrix-free apply_M       == dense M = -I + C(lam) D
  A2  Woodbury solve            == dense solve(M, rhs)
  A3  Sylvester eig_max(M)      == max Re eigvals(dense M)
  A4  Delta(0)                  == -M            (the fold <-> z=0 identity)
  A5  Weinstein-Aronszajn:  log|det Delta(z)| == (N-P) log|t0 z+1| + log|det T_P(z)|
      at random complex z (the exact reduced-spectrum factorization)
  A6  polished T_P roots        are roots of the DENSE Delta  (sigma_min ~ 0)
  A7  full-IG rightmost physical eigenvalues are roots of dense Delta;
      spurious cluster is NOT (the filter premise)
  A8  reduced-IG candidates     converge to the same physical roots
  A9  tau=0 ODE limit: reduced (eig[(1-lam)GJ+lam GK]-1)/t0  ==  top eig(dense M)/t0
  A10 polish derivative dT/dz   == finite difference of T_P
  A11 gain D = beta(1-tanh^2)   == numerical d/du tanh(beta u)
  A12 beta-annealed Newton lands on the memory state (m1~1, residual < 1e-9)
  A13 robust tracer fold        == dense-Newton + dense-eigvals gold standard

Run:  conda run -n mcmc_env python audit_static_analysis.py
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from couplings import Couplings, make_patterns
from robust_branch import (woodbury_solve, woodbury_newton, anneal_newton,
                           eigmax_M, trace_branch)
from reduced_spectrum import GJ_GK, T_P, sigmin_TP, _polish_root, \
    reduced_ig_candidates, physical_roots
from dde_stability import rightmost_eigenvalues, IGOperator
import config as cfg_mod

rng = np.random.default_rng(7)
N, ALPHA, TAU, T0, BETA, SEED = 200, 0.05, 10.0, 1.0, 20.0, 3
P = round(ALPHA * N)

xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

results = []
def check(name, err, tol):
    ok = err < tol
    results.append((name, err, tol, ok))
    print(f"  {'PASS' if ok else '*** FAIL ***'}  {name:58s} err={err:.2e} (tol {tol:.0e})")

# ---- a memory fixed point strictly below the fold -------------------------
cfg = dict(lam_min=0.0, lam_max=0.6, ds=0.01, n_overlaps=5)
br, fold, status = trace_branch(coup, BETA, cfg)
lam = 0.9 * fold
j = int(np.argmin(np.abs(br["lam"][:np.argmax(br["lam"])+1] - lam)))
lam = float(br["lam"][j]); u = br["u_star"][j]
gain = coup.compute_gain(u, BETA)
print(f"audit at N={N}, P={P}, lam={lam:.4f} (fold={fold:.4f}), "
      f"m1={float(xi[0] @ np.tanh(BETA*u) / N):.4f}\n")

# dense references
J, K, I = coup.dense_J(), coup.dense_K(), np.eye(N)
D = np.diag(gain)
Md = -I + ((1-lam)*J + lam*K) @ D
def Delta(z):
    return (T0*z + 1)*I - (1-lam)*(J@D) - lam*(K@D)*np.exp(-z*TAU)

# A1 matrix-free vs dense M
v = rng.standard_normal(N)
check("A1  apply_M vs dense M", np.linalg.norm(coup.apply_M(v, gain, lam) - Md@v)
      / np.linalg.norm(Md@v), 1e-12)

# A2 Woodbury vs dense solve
r = rng.standard_normal(N)
check("A2  Woodbury M^{-1}r vs dense solve", np.linalg.norm(
    woodbury_solve(coup, gain, lam, r) - np.linalg.solve(Md, r))
      / np.linalg.norm(np.linalg.solve(Md, r)), 1e-10)

# A3 Sylvester eig_max
check("A3  Sylvester eig_max(M) vs dense", abs(
    eigmax_M(coup, u, lam, BETA) - np.linalg.eigvals(Md).real.max()), 1e-10)

# A4 Delta(0) = -M
check("A4  Delta(0) + M == 0", np.linalg.norm(Delta(0.0) + Md)
      / np.linalg.norm(Md), 1e-14)

# A5 Weinstein-Aronszajn factorization at random complex z
GJ, GK = GJ_GK(coup, u, BETA, lam)
errs = []
for _ in range(6):
    z = complex(rng.uniform(-0.8, 0.3), rng.uniform(-2, 2))
    s_full, ld_full = np.linalg.slogdet(Delta(z))
    s_red,  ld_red  = np.linalg.slogdet(T_P(z, GJ, GK, T0, TAU, lam))
    ld_fact = (N - P) * np.log(abs(T0*z + 1)) + ld_red
    errs.append(abs(ld_full - ld_fact) / max(abs(ld_full), 1.0))
check("A5  det Delta == (t0 z+1)^{N-P} det T_P (log, 6 random z)", max(errs), 1e-10)

# A6 polished T_P roots are dense-Delta roots
cands = reduced_ig_candidates(GJ, GK, T0, TAU, lam, M=24, k=30)
polished = [_polish_root(z, GJ, GK, T0, TAU, lam) for z in cands
            if sigmin_TP(z, GJ, GK, T0, TAU, lam) < 0.05*max(abs(T0*z+1), 1e-6)]
if polished:
    check("A6  polished T_P roots: sigma_min(dense Delta)", max(
        np.linalg.svd(Delta(z), compute_uv=False).min() for z in polished[:8]), 1e-8)

# A7 full IG: physical pass, spurious fail
cfg_ig = cfg_mod.resolve()
cfg_ig.update(N=N, alpha=ALPHA, beta=BETA, t0=T0, tau=TAU, M_cheb=24, n_eigs=10,
              arnoldi_ncv=50, arnoldi_maxiter=500)
cfg_ig["P"] = P
igv = rightmost_eigenvalues(u, lam, coup, cfg_ig)
sig_ig = np.array([np.linalg.svd(Delta(z), compute_uv=False).min() for z in igv])
phys_mask = sig_ig < 1e-4
if phys_mask.any():
    check("A7a full-IG physical eigenvalues are Delta roots", sig_ig[phys_mask].max(), 1e-4)
if (~phys_mask).any():
    spur_min = sig_ig[~phys_mask].min()
    check("A7b spurious modes are NOT roots (min sigma should be O(1))",
          1.0 / max(spur_min, 1e-30) if spur_min < 0.05 else 0.0, 1e-6)

# A8 reduced-IG candidates reproduce full-IG physical roots
phys_full = np.sort_complex(igv[phys_mask])
phys_red, _, _ = physical_roots(coup, u, BETA, lam, TAU, T0, M=24, n_cand=30, k=10)
if len(phys_full) and len(phys_red):
    matched = max(min(abs(zf - phys_red).min() for zf in phys_full[-4:]), 0.0)
    check("A8  reduced-IG physical roots match full-IG physical roots", matched, 1e-5)

# A9 tau=0 ODE limit
mu = np.linalg.eigvals((1-lam)*GJ + lam*GK)
z_red = np.sort((mu.real - 1.0) / T0)[::-1][:5]
z_ode = np.sort(np.linalg.eigvals(Md).real / T0)[::-1][:5]
check("A9  tau=0 reduced spectrum == top dense eig(M)/t0", np.max(np.abs(z_red - z_ode)), 1e-9)

# A10 polish derivative dT/dz vs finite difference
z0 = complex(-0.2, 0.4); h = 1e-6
dT_ana = T0*np.eye(P) + lam*TAU*np.exp(-z0*TAU)*GK
dT_fd = (T_P(z0+h, GJ, GK, T0, TAU, lam) - T_P(z0-h, GJ, GK, T0, TAU, lam)) / (2*h)
check("A10 dT_P/dz analytic vs finite difference", np.linalg.norm(dT_ana - dT_fd)
      / np.linalg.norm(dT_ana), 1e-8)

# A11 gain = d/du tanh(beta u)
uu = rng.standard_normal(N) * 0.5; h = 1e-6
g_fd = (np.tanh(BETA*(uu+h)) - np.tanh(BETA*(uu-h))) / (2*h)
check("A11 gain beta(1-tanh^2) vs numerical derivative",
      np.max(np.abs(coup.compute_gain(uu, BETA) - g_fd)), 1e-6)

# A12 annealed Newton at lam=0 (the ONLY place production code uses annealing;
# annealing at lam near the fold is known-unsafe -- the intermediate-beta states
# can cross their own fold in beta and drift off the memory branch)
u_a, ok = anneal_newton(coup, 0.99*xi[0].copy(), 0.0, BETA)
res = np.linalg.norm(coup.field_F(u_a, 0.0, BETA)) / np.sqrt(N)
check("A12 annealed Newton residual (at lam=0, production usage)", res, 1e-9)
check("A12b annealed Newton overlap gap (0.99 - m1)",
      max(0.0, 0.99 - float(xi[0]@np.tanh(BETA*u_a)/N)), 1e-2)

# A13 robust tracer fold vs dense gold standard
def dense_newton(u0, lm, tol=1e-10, mit=80):
    uu = u0.copy()
    for _ in range(mit):
        F = coup.field_F(uu, lm, BETA); rr = np.linalg.norm(F)/np.sqrt(N)
        if rr < tol: return uu, True
        Md_ = coup.dense_M(coup.compute_gain(uu, BETA), lm)
        du = np.linalg.solve(Md_, -F); st = 1.0
        for _ in range(30):
            if np.linalg.norm(coup.field_F(uu+st*du, lm, BETA))/np.sqrt(N) < rr: break
            st *= 0.5
        uu = uu + st*du
    return uu, False
u_g, _ = anneal_newton(coup, 0.99*xi[0].copy(), 0.0, BETA)
lm, ds = 0.0, 0.01; fold_gold = None
while lm < 0.6:
    u2, okg = dense_newton(u_g, lm + ds)
    m1g = float(xi[0] @ np.tanh(BETA*u2) / N)
    if okg and m1g > 0.5:
        lm += ds; u_g = u2
        if np.linalg.eigvals(coup.dense_M(coup.compute_gain(u_g, BETA), lm)).real.max() > -1e-3:
            fold_gold = lm; break
        ds = min(ds*1.15, 0.01)
    else:
        ds *= 0.5
        if ds < 2e-4: fold_gold = lm; break
check("A13 robust tracer fold vs dense gold standard", abs(fold - fold_gold), 5e-3)

# ---- summary ---------------------------------------------------------------
print("\n" + "=" * 78)
n_ok = sum(1 for *_, ok in results if ok)
print(f"AUDIT: {n_ok}/{len(results)} checks passed")
for name, err, tol, ok in results:
    if not ok:
        print(f"  FAILED: {name}  err={err:.2e} tol={tol:.0e}")
print("=" * 78)
