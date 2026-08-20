"""
E32 (large-alpha task, Exp A) -- (alpha, lambda) basin phase diagram.

Coarse, robust partition {static / cycle / chaos / other} from random ICs, over a
grid in (alpha, lambda). Deliverable: cycle-fraction heatmap + its pinch-off
alpha_cycle (where does the recall cycle stop being generically reachable as alpha
grows -> the memory->chaos route).

GPU (MLX float32) is used ONLY after the validation gate: --validate runs one cell
in BOTH float32(GPU) and float64(CPU), classifies the SAME ICs, and reports per-IC
label agreement + basin-fraction agreement. Only if fractions agree within
statistical error do we trust float32 for the sweep (basin outcome is coarse and
ensemble-averaged; the float32 lesson was localized to Newton/fold detection).

CLI:
  e32_alpha_lambda_basins.py --validate            # gate: f32 vs f64 on one cell
  e32_alpha_lambda_basins.py --sweep --backend mlx # full grid on GPU (post-gate)
  e32_alpha_lambda_basins.py --sweep --backend cpu # fallback
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from couplings import make_patterns
from cycle_reduced_batch import ReducedDDEBatch

BETA, TAU, T0, DT = 20.0, 10.0, 1.0, 0.05
DRIFT_STAT = 3e-3        # tail finite-diff drift below this -> stationary
S_MACRO = 0.5            # macroscopic leading overlap
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")


def sample_ics(rng, P, K):
    """K constant-history ICs, three families (near_pattern/mixture/random_small)."""
    A = np.zeros((P, K))
    kinds = rng.integers(0, 3, K)
    for k in range(K):
        if kinds[k] == 0:
            mu = rng.integers(P)
            A[:, k] = rng.uniform(0.02, 0.15) * rng.standard_normal(P)
            A[mu, k] += rng.uniform(0.6, 1.0)
        elif kinds[k] == 1:
            nc = rng.integers(2, 5); mu0 = rng.integers(P)
            w = rng.uniform(0.3, 1.0, nc); w = w / w.sum() * rng.uniform(0.6, 1.2)
            for j in range(nc):
                A[(mu0 + j) % P, k] += w[j]
            A[:, k] += rng.uniform(0.02, 0.10) * rng.standard_normal(P)
        else:
            A[:, k] = rng.uniform(0.05, 0.35) * rng.standard_normal(P)
    return A


def classify_batch(a):
    """a: (n_rec, P, K) recorded trajectory. Return array of labels (K,) in the
    coarse partition {static, cycle, chaos, other} -- robust, seed-independent."""
    n_rec, P, K = a.shape
    aa = np.abs(a)
    last = a[-1]                                     # (P,K)
    s0 = np.max(aa[-1], axis=0)                      # leading overlap (K,)
    # drift: normalized last-step finite difference
    drift = (np.linalg.norm(a[-1] - a[-2], axis=0)
             / (np.linalg.norm(a[-1], axis=0) + 1e-12))     # (K,)
    # ring advance over the tail (last 60%)
    k0 = max(1, int(0.4 * n_rec))
    leads = np.argmax(aa[k0:], axis=1)              # (n_tail, K)
    d = np.diff(leads, axis=0)
    fwd = np.sum((d == 1) | (d == -(P - 1)), axis=0)
    bwd = np.sum((d == -1) | (d == (P - 1)), axis=0)
    moves = fwd + bwd
    n_adv = fwd - bwd
    mono = np.where(moves > 0, fwd / np.maximum(moves, 1), 0.0)

    labels = np.empty(K, dtype=object)
    for k in range(K):
        if drift[k] < DRIFT_STAT:
            labels[k] = "static" if s0[k] > S_MACRO else "other"
        elif n_adv[k] >= 6 and mono[k] > 0.8:
            labels[k] = "cycle"
        elif drift[k] >= DRIFT_STAT:
            labels[k] = "chaos"
        else:
            labels[k] = "other"
    return labels


def fractions(labels):
    K = len(labels)
    return {c: float(np.mean(labels == c)) for c in
            ("static", "cycle", "chaos", "other")}


def integrate_backend(backend, xi, lam, A0, t_total=400.0):
    if backend == "mlx":
        from cycle_reduced_mlx import ReducedDDEMLX
        sysm = ReducedDDEMLX(xi, BETA, lam, TAU, T0)
        return sysm.integrate(A0, t_total, DT, record_every=40)["a"]
    else:
        sysb = ReducedDDEBatch(xi, BETA, lam, TAU, T0)
        return sysb.integrate(A0, t_total, DT, record_every=40)["a"]


def validate(N=2000, alpha=0.05, lams=(0.25, 0.31, 0.40), K=200, seed=42):
    """Gate: same ICs, f32(GPU) vs f64(CPU); compare per-IC labels + fractions."""
    P = round(alpha * N)
    xi, _ = make_patterns(N, P, seed)
    rng = np.random.default_rng(999)
    print(f"[E32 validate] N={N} alpha={alpha} P={P} K={K}, cells lam={lams}")
    all_ok = True
    for lam in lams:
        A0 = sample_ics(rng, P, K)
        t0 = time.time(); aG = integrate_backend("mlx", xi, lam, A0); tg = time.time() - t0
        t0 = time.time(); aC = integrate_backend("cpu", xi, lam, A0); tc = time.time() - t0
        lg = classify_batch(aG); lc = classify_batch(aC)
        agree = float(np.mean(lg == lc))
        fg, fc = fractions(lg), fractions(lc)
        dmax = max(abs(fg[c] - fc[c]) for c in fg)
        binerr = (np.sqrt(0.25 / K))                # ~1-sigma binomial half-width
        ok = dmax < 3 * binerr
        all_ok &= ok
        print(f"  lam={lam:.3f}: per-IC agree {agree:.3f} | frac f32 "
              f"{ {c:round(fg[c],3) for c in fg} } | f64 { {c:round(fc[c],3) for c in fc} } "
              f"| max|df|={dmax:.3f} (3sig binom={3*binerr:.3f}) -> "
              f"{'OK' if ok else 'MISMATCH'} | GPU {tg:.1f}s CPU {tc:.1f}s")
    print(f"[E32 validate] VERDICT: {'float32/GPU TRUSTED for the sweep' if all_ok else 'FALL BACK to CPU float64'}")
    return all_ok


def sweep(backend, N=2000, alphas=(0.05, 0.07, 0.09, 0.11, 0.13, 0.15),
          lams=(0.15, 0.20, 0.25, 0.28, 0.31, 0.35, 0.40, 0.50, 0.70, 0.90),
          K=200, seeds=(42, 43), out=None):
    out = out or os.path.join(OUT, f"E32_basins_N{N}_{backend}.npz")
    grid = np.zeros((len(alphas), len(lams), len(seeds), 4))   # static,cycle,chaos,other
    order = ("static", "cycle", "chaos", "other")
    t_all = time.time()
    for ia, alpha in enumerate(alphas):
        P = round(alpha * N)
        for isd, sd in enumerate(seeds):
            xi, _ = make_patterns(N, P, sd)
            rng = np.random.default_rng(7000 + sd)
            for il, lam in enumerate(lams):
                A0 = sample_ics(rng, P, K)
                a = integrate_backend(backend, xi, lam, A0)
                fr = fractions(classify_batch(a))
                grid[ia, il, isd] = [fr[c] for c in order]
            print(f"[E32 sweep] alpha={alpha:.3f} P={P} seed={sd} done "
                  f"({time.time()-t_all:.0f}s)"); sys.stdout.flush()
    np.savez(out, grid=grid, alphas=np.array(alphas), lams=np.array(lams),
             seeds=np.array(seeds), order=np.array(order), K=K, N=N)
    # cycle-fraction heatmap summary
    cyc = grid[:, :, :, 1].mean(axis=2)              # (alpha,lam)
    print(f"\n[E32] cycle fraction (seed-avg), rows=alpha {alphas}, cols=lam {lams}:")
    for ia, alpha in enumerate(alphas):
        print(f"  a={alpha:.3f}: " + " ".join(f"{cyc[ia,il]:.2f}" for il in range(len(lams))))
    print(f"[E32] -> {out}  ({(time.time()-t_all)/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "cpu"])
    ap.add_argument("--K", type=int, default=200)
    args = ap.parse_args()
    if args.validate:
        validate(K=args.K)
    if args.sweep:
        sweep(args.backend, K=args.K)


if __name__ == "__main__":
    main()
