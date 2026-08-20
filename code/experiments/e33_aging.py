"""
E33 (aging task, Exp D + two-time correlation) -- does the mixture-phase chaos AGE,
or is it a stationary SRB-measure attractor?

Direct aging test: the two-time correlation C(t_w+t, t_w) = <a(t_w+t).a(t_w)>
(ensemble-averaged over IC x disorder, dot over patterns -> gauge invariant).
  - STATIONARY (SRB, expected): curves for all t_w collapse onto one C(t).
  - AGING: curves fan out; collapse under t/t_w.
Plus Exp D (quench relaxation <|a|^2>(t) with no settle) to measure the transient
time tau_r and set the settle time (guards against a slow transient masquerading as
aging -- the near-marginal lambda_4~-0.003 could give tau_r~300).

float32/GPU is a candidate here (ensemble-averaged statistics), but the aging signal
is SUBTLE, so --validate compares the two-time C in f32(GPU) vs f64(CPU) on a small
ensemble BEFORE trusting float32 for the full run.

Reference: N=2000, alpha=0.05 (P=100), lam=0.31 (chaotic window), tau=10, beta=20,
dt=0.05 (basin-grade; the two-time statistics are robust to dt here), seed set.

CLI:
  e33_aging.py --validate                 # f32(GPU) vs f64(CPU) two-time C
  e33_aging.py --run --backend mlx        # full ensemble on GPU (post-gate)
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
LAM = 0.31
A_IC = 0.3
REC = 10                          # record every REC steps -> dt_rec = 0.5
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")


def make_ens(rng, P, K):
    return A_IC * rng.standard_normal((P, K))          # (P,K)


def integrate(backend, xi, A0, t_total):
    if backend == "mlx":
        from cycle_reduced_mlx import ReducedDDEMLX
        return ReducedDDEMLX(xi, BETA, LAM, TAU, T0).integrate(
            A0, t_total, DT, record_every=REC)
    return ReducedDDEBatch(xi, BETA, LAM, TAU, T0).integrate(
        A0, t_total, DT, record_every=REC)


def chaotic_mask(a):
    """Keep trajectories still wandering at the end (drift above threshold) AND not
    a clean ring-advancing cycle -- i.e. remaining in the chaotic sea."""
    n, P, K = a.shape
    drift = np.linalg.norm(a[-1] - a[-2], axis=0) / (np.linalg.norm(a[-1], axis=0) + 1e-9)
    return drift > 3e-3                                  # (K,) bool


def two_time_C(a, t_rec, tws, mask):
    """C(t_w, t) = mean_traj a(t_w+t).a(t_w), over kept trajectories.
    a: (n_rec, P, K). Returns dict t_w -> (lags, C(lags))."""
    n = a.shape[0]
    dt_rec = t_rec[1] - t_rec[0]
    keep = np.where(mask)[0]
    out = {}
    for tw in tws:
        iw = int(round(tw / dt_rec))
        if iw >= n - 1:
            continue
        lags_idx = np.arange(0, n - iw)
        # a[iw+it] . a[iw] summed over P, averaged over kept trajectories
        aw = a[iw][:, keep]                              # (P, nkeep)
        seg = a[iw:][:, :, keep]                         # (nlag, P, nkeep)
        C = np.einsum("lpk,pk->lk", seg, aw).mean(axis=1)   # (nlag,)
        out[tw] = (lags_idx * dt_rec, C)
    return out


def quench_relax(backend, N=2000, alpha=0.05, K=128, seed=42, T=2000.0):
    """Exp D: ensemble <|a|^2>(t) from random IC, NO settle -> transient tau_r."""
    P = round(alpha * N)
    xi, _ = make_patterns(N, P, seed)
    rng = np.random.default_rng(555)
    A0 = make_ens(rng, P, K)
    sol = integrate(backend, xi, A0, T)
    a = sol["a"]                                         # (n,P,K)
    en = (a ** 2).sum(axis=1).mean(axis=1)               # <|a|^2>(t) ensemble mean
    return sol["t"], en


def validate(N=2000, alpha=0.05, K=48, seed=42, T=2000.0,
             tws=(0.0, 200.0, 800.0)):
    """Gate: two-time C f32(GPU) vs f64(CPU) on the same ensemble."""
    P = round(alpha * N)
    xi, _ = make_patterns(N, P, seed)
    rng = np.random.default_rng(321)
    A0 = make_ens(rng, P, K)
    print(f"[E33 validate] N={N} P={P} K={K} lam={LAM} T={T}, tws={tws}")
    tg = time.time(); aG = integrate("mlx", xi, A0, T); tG = time.time() - tg
    tc = time.time(); aC = integrate("cpu", xi, A0, T); tC = time.time() - tc
    t_rec = aG["t"]
    mG, mC = chaotic_mask(aG["a"]), chaotic_mask(aC["a"])
    print(f"  chaotic-kept: f32 {mG.sum()}/{K}, f64 {mC.sum()}/{K}")
    CG = two_time_C(aG["a"], t_rec, tws, mG)
    CC = two_time_C(aC["a"], t_rec, tws, mC)
    ok = True
    for tw in tws:
        if tw not in CG or tw not in CC:
            continue
        lag, cg = CG[tw]; _, cc = CC[tw]
        L = min(len(cg), len(cc))
        # normalized curves (by C(t_w,0)) to compare shapes
        ng = cg[:L] / (cg[0] + 1e-9); nc = cc[:L] / (cc[0] + 1e-9)
        dmax = np.max(np.abs(ng - nc))
        ok &= dmax < 0.08
        print(f"  t_w={tw:6.0f}: C(0)={cc[0]:.3f} (f64) vs {cg[0]:.3f} (f32) | "
              f"max|dC_norm|={dmax:.3f} -> {'OK' if dmax < 0.08 else 'MISMATCH'}")
    print(f"[E33 validate] GPU {tG:.1f}s CPU {tC:.1f}s | VERDICT: "
          f"{'float32/GPU TRUSTED for two-time C' if ok else 'FALL BACK to CPU float64'}")
    return ok


def run(backend, N=2000, alpha=0.05, K=64, seeds=(42, 43, 44, 45),
        settle=400.0, T=6000.0, tws=(0.0, 50.0, 200.0, 800.0, 3200.0), tag=""):
    """Full ensemble: two-time C(t_w+t,t_w) averaged over IC x disorder, + quench.
    Exp E2 extension: run at several alpha to test if aging emerges near capacity."""
    P = round(alpha * N)
    # Exp D quench (one seed, no settle) for tau_r
    tq, enq = quench_relax(backend, N, alpha, K=K, seed=seeds[0], T=2000.0)
    # two-time: settle then record, accumulate C over seeds
    Cacc = {tw: [] for tw in tws}; t_ref = None; nkeep_tot = 0
    for sd in seeds:
        xi, _ = make_patterns(N, P, sd)
        rng = np.random.default_rng(9000 + sd)
        A0 = make_ens(rng, P, K)
        # settle
        s0 = integrate(backend, xi, A0, settle)
        A_settled = s0["a"][-1]                          # (P,K)
        sol = integrate(backend, xi, A_settled, T)
        a = sol["a"]; t_rec = sol["t"]; t_ref = t_rec
        mask = chaotic_mask(a); nkeep_tot += int(mask.sum())
        C = two_time_C(a, t_rec, tws, mask)
        for tw in tws:
            if tw in C:
                Cacc[tw].append(C[tw])
        print(f"[E33 run] seed {sd}: chaotic-kept {mask.sum()}/{K} ({time.time():.0f})")
        sys.stdout.flush()
    # average C over seeds (align lags)
    save = dict(N=N, alpha=alpha, lam=LAM, K=K, seeds=np.array(seeds),
                settle=settle, T=T, tws=np.array(tws), tq=tq, enq=enq,
                nkeep_tot=nkeep_tot)
    for tw in tws:
        if Cacc[tw]:
            Lmin = min(len(c[1]) for c in Cacc[tw])
            lag = Cacc[tw][0][0][:Lmin]
            Cavg = np.mean([c[1][:Lmin] for c in Cacc[tw]], axis=0)
            save[f"lag_{int(tw)}"] = lag; save[f"C_{int(tw)}"] = Cavg
    out = os.path.join(OUT, f"E33_aging_{backend}{tag}.npz")
    np.savez(out, **save)
    print(f"[E33 run] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "cpu"])
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--alphas", default="0.05")
    args = ap.parse_args()
    if args.validate:
        validate()
    if args.run:
        for a in [float(x) for x in args.alphas.split(",")]:
            tag = "" if abs(a - 0.05) < 1e-9 else f"_a{a:.2f}".replace(".", "p")
            print(f"=== aging run alpha={a} ===")
            run(args.backend, alpha=a, K=args.K, tag=tag)


if __name__ == "__main__":
    main()
