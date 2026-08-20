"""
E23 PART A - Nature of the chaos in the mixture phase lam_c < lam < lam* (tau=10).

Question (PI Q5): in the mixture window at tau=10, is the generic attractor a
GENUINE low-dimensional strange attractor (Lorenz-like), or something else
(high-dimensional/extensive chaos, or a chaotic transient/saddle)?

Diagnostics implemented here:
  A1  Largest Lyapunov exponent lam_L(lam) via Benettin/QR on the reduced
      variational DDE, long trajectory, transient discarded; check the
      finite-time exponent CONVERGES to a positive constant (attractor) and that
      the trajectory does NOT fall onto the pinned front / memory (else it is a
      chaotic SADDLE/transient).  Scanned over lam in the window.
  A2  Partial Lyapunov SPECTRUM (k-vector QR, k~8) at lam=0.31 -> ordered
      exponents + Kaplan-Yorke dimension D_KY.  Low D_KY (a few) => Lorenz-like;
      large => spatiotemporal/extensive chaos.
  A3  Attractor geometry at lam=0.31: 2-3 leading-overlap projection (+PCA),
      Poincare section + first-return map of one coordinate, power spectrum.

Method: EXACT reduced DDE (dim P=100).  Reference trajectory a(t) and k tangent
vectors are advanced in LOCKSTEP with the SAME RK4 + cubic-Hermite history scheme
as cycle_reduced.ReducedDDE, so the tangent flow uses the linearization along the
recorded reference.  Autodiff is not needed: the reduced Jacobian is the exact
analytic Dm(a) (sys.Dm), matrix-free through the P-dim reduction.

Variational DDE (cycle_reduced docstring):
    t0 dda' = -dda + (1-lam) Dm(a(t)) dda(t) + lam S Dm(a(t-tau)) dda(t-tau),
with S = roll(+1) matching rhs's np.roll(m(a_del), 1).

Outputs: results/cycle/E23A_lyap.npz, E23A_geometry.npz, and
figures/figQ_chaos_nature.png.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np

from couplings import make_patterns
from cycle_reduced import ReducedDDE

N, ALPHA, BETA, TAU, T0, SEED = 2000, 0.05, 20.0, 10.0, 1.0, 42
P = round(ALPHA * N)
xi, _ = make_patterns(N, P, SEED)
DT = 0.02                       # dt divides tau=10 (L=500). float64 CPU.
S_SHIFT = 1                     # roll(+1): (S v)_nu = v_{nu-1}, matches rhs

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIGDIR = os.path.join(OUT, "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Coupled reference + tangent integrator (method of steps, RK4 + Hermite hist).
# k tangent vectors carried; QR every n_qr steps; log-norms accumulated.
# ---------------------------------------------------------------------------
def benettin(sys, a_hist0, k, t_total, dt, t_transient, n_qr,
             rng, record_ref=False, record_every=10, verbose_every=None,
             front_thresh=0.55):
    """Return finite-time Lyapunov exponents (k of them) + diagnostics.

    sys        : ReducedDDE
    a_hist0    : (L+1,P) or (P,) initial history for the reference
    k          : number of tangent vectors (>=1)
    QR every n_qr steps; discard the first t_transient of BOTH ref settle and
    Lyapunov accumulation.
    Detects collapse onto a fixed point (front/memory) -> chaotic-saddle flag.
    """
    lam, tau, t0 = sys.lam, sys.tau, sys.t0
    L = int(round(tau / dt))
    assert abs(L * dt - tau) < 1e-12
    Pdim = sys.P

    # --- settle the reference alone through the transient (fixed buffers) ---
    n_settle = int(round(t_transient / dt))
    ref = sys.integrate(a_hist0, t_transient, dt, record_every=1)
    hist = ref["hist"]                          # (L+1,P) history at t=0 (post-settle)

    # ring buffers for reference position + derivative (Hermite)
    n_steps = int(round(t_total / dt))
    buf = np.empty((L + n_steps + 1, Pdim))
    dbuf = np.empty_like(buf)
    buf[:L + 1] = hist
    for i in range(L + 1):
        dbuf[i] = sys.rhs(buf[i], buf[max(i - L, 0)])

    def delayed_ref(idx_float):
        i0 = int(np.floor(idx_float)); s = idx_float - i0
        if s < 1e-14:
            return buf[i0]
        h00 = (1 + 2 * s) * (1 - s) ** 2; h10 = s * (1 - s) ** 2
        h01 = s * s * (3 - 2 * s); h11 = s * s * (s - 1)
        return (h00 * buf[i0] + h10 * dt * dbuf[i0]
                + h01 * buf[i0 + 1] + h11 * dt * dbuf[i0 + 1])

    # tangent history buffers: k vectors, each a (L+n_steps+1, P) buffer + deriv
    V = rng.standard_normal((k, (L + 1) * Pdim))
    Q, _ = np.linalg.qr(V.T); V = Q.T
    tbuf = [np.empty((L + n_steps + 1, Pdim)) for _ in range(k)]
    tdbuf = [np.empty_like(tbuf[0]) for _ in range(k)]

    # precompute reference Dm on the initial history window (for tangent deriv)
    # variational rhs for tangent j at reference indices:
    #   t0 dv' = -dv + (1-lam) Dm(a_i) dv_now + lam S Dm(a_{i-L}) dv_del
    Ssh = np.roll(np.eye(Pdim), S_SHIFT, axis=0)   # (S v)_nu = v_{nu-1}

    def Dm_at(idx):
        return sys.Dm(buf[idx])

    # cache Dm along the whole reference AS WE GO (computed lazily per step)
    Dm_cache = {}
    def Dm_idx(idx):
        d = Dm_cache.get(idx)
        if d is None:
            d = sys.Dm(buf[idx]); Dm_cache[idx] = d
        return d

    def var_rhs(dv_now, dv_del, i_now, i_del):
        return (-dv_now + (1 - lam) * (Dm_idx(i_now) @ dv_now)
                + lam * (Ssh @ (Dm_idx(i_del) @ dv_del))) / t0

    # init tangent history and its derivative
    for j in range(k):
        tbuf[j][:L + 1] = V[j].reshape(L + 1, Pdim)
        for i in range(L + 1):
            i_del = max(i - L, 0)
            tdbuf[j][i] = var_rhs(tbuf[j][i], tbuf[j][i_del], i, i_del)

    def delayed_tan(j, idx_float):
        b = tbuf[j]; db = tdbuf[j]
        i0 = int(np.floor(idx_float)); s = idx_float - i0
        if s < 1e-14:
            return b[i0]
        h00 = (1 + 2 * s) * (1 - s) ** 2; h10 = s * (1 - s) ** 2
        h01 = s * s * (3 - 2 * s); h11 = s * s * (s - 1)
        return (h00 * b[i0] + h10 * dt * db[i0]
                + h01 * b[i0 + 1] + h11 * dt * db[i0 + 1])

    logsum = np.zeros(k)
    n_accum = 0
    ft_hist = []                    # finite-time largest exponent trace
    ref_rec = ([], []) if record_ref else None
    front_flag = False
    adot_tail = []

    t_wall = time.time()
    for n in range(n_steps):
        i = L + n
        # --- advance reference (RK4 + Hermite) ---
        a = buf[i]
        k1 = sys.rhs(a, delayed_ref(i - L))
        k2 = sys.rhs(a + 0.5 * dt * k1, delayed_ref(i + 0.5 - L))
        k3 = sys.rhs(a + 0.5 * dt * k2, delayed_ref(i + 0.5 - L))
        k4 = sys.rhs(a + dt * k3, delayed_ref(i + 1.0 - L))
        buf[i + 1] = a + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        dbuf[i + 1] = sys.rhs(buf[i + 1], delayed_ref(i + 1.0 - L))

        # --- advance each tangent along the reference (frozen Dm at ref pts) ---
        # RK stages need Dm at i, i+0.5(-> use midpoint ref), i+1. We use the
        # reference index for now/del; for half steps reuse i (Dm smooth, dt small)
        for j in range(k):
            dv = tbuf[j][i]
            d1 = var_rhs(dv, delayed_tan(j, i - L), i, i - L)
            d2 = var_rhs(dv + 0.5 * dt * d1, delayed_tan(j, i + 0.5 - L), i, i - L)
            d3 = var_rhs(dv + 0.5 * dt * d2, delayed_tan(j, i + 0.5 - L), i, i - L)
            d4 = var_rhs(dv + dt * d3, delayed_tan(j, i + 1.0 - L), i + 1, i + 1 - L)
            tbuf[j][i + 1] = dv + dt / 6.0 * (d1 + 2 * d2 + 2 * d3 + d4)
            tdbuf[j][i + 1] = var_rhs(tbuf[j][i + 1],
                                      delayed_tan(j, i + 1.0 - L), i + 1, i + 1 - L)

        # --- QR reorthonormalization on the current history WINDOW [n+1,..+L] ---
        if (n + 1) % n_qr == 0:
            i1 = i + 1
            W = np.stack([tbuf[j][i1 - L:i1 + 1].reshape(-1) for j in range(k)])
            Qm, Rm = np.linalg.qr(W.T)
            diagR = np.abs(np.diag(Rm))
            logsum += np.log(diagR)
            n_accum += n_qr
            ft_hist.append((n_accum * dt, logsum[0] / (n_accum * dt)))
            Wn = Qm.T
            for j in range(k):
                seg = Wn[j].reshape(L + 1, Pdim)
                tbuf[j][i1 - L:i1 + 1] = seg
                for m in range(L + 1):
                    ii = i1 - L + m; i_del = ii - L
                    tdbuf[j][ii] = var_rhs(tbuf[j][ii],
                                           tbuf[j][max(i_del, i1 - L)],
                                           ii, max(i_del, i1 - L))
            # prune Dm cache far behind to bound memory
            if len(Dm_cache) > 3 * L:
                lo = i1 - 2 * L
                for key in [kk for kk in Dm_cache if kk < lo]:
                    del Dm_cache[key]

        if record_ref and (n + 1) % record_every == 0:
            ref_rec[0].append((n + 1) * dt)
            ref_rec[1].append(buf[i + 1].copy())

        if verbose_every and (n + 1) % verbose_every == 0:
            lam0 = logsum[0] / max(n_accum * dt, 1e-9)
            print(f"    t={(n+1)*dt:8.1f}  ft_lyap0={lam0:+.5f}  "
                  f"|a|max={np.abs(buf[i+1]).max():.3f}  "
                  f"[{time.time()-t_wall:.0f}s]", flush=True)

    # per-unit-time exponents
    T_accum = n_accum * dt
    lyaps = logsum / T_accum

    # --- attractor vs transient: is the tail chaotic or a fixed point? ---
    tail = buf[L + int(0.7 * n_steps):L + n_steps + 1]
    adot = np.array([np.linalg.norm(sys.rhs(tail[m], tail[m]))
                     for m in range(0, len(tail), max(1, len(tail) // 200))])
    # a fixed point would have rhs(a,a)->0; a cycle/chaos keeps |adot| finite
    dnorm_tail = float(adot.mean())
    a_last = buf[L + n_steps]
    order = np.argsort(np.abs(a_last))[::-1]
    is_fixed = dnorm_tail < 1e-3
    label_final = ("FIXED-POINT COLLAPSE (front/memory) -> chaotic SADDLE/transient"
                   if is_fixed else "SUSTAINED (non-fixed tail)")

    ft = np.array(ft_hist)
    out = dict(lyaps=lyaps, T_accum=T_accum, ft_hist=ft,
               dnorm_tail=dnorm_tail, is_fixed=is_fixed,
               label_final=label_final,
               a_last=a_last, top=[int(x) for x in order[:4]],
               vals=[float(np.abs(a_last)[x]) for x in order[:4]])
    if record_ref:
        out["ref_t"] = np.array(ref_rec[0])
        out["ref_a"] = np.array(ref_rec[1])
    return out


# ===========================================================================
# A1 - largest Lyapunov exponent across the window
# ===========================================================================
def run_A1():
    print("=" * 74)
    print("A1  Largest Lyapunov exponent lam_L(lam), tau=10 mixture window")
    print("=" * 74, flush=True)
    LAMS = [0.29, 0.30, 0.31, 0.32, 0.325]
    rng0 = np.random.default_rng(7)
    # generic chaotic IC: random in-span, moderate amplitude (avoid pinned basin)
    a0 = 0.3 * rng0.standard_normal(P)
    rows = []
    for lam in LAMS:
        sys = ReducedDDE(xi, BETA, lam, TAU, T0)
        rng = np.random.default_rng(101)
        t0w = time.time()
        r = benettin(sys, a0, k=1, t_total=T_A1, dt=DT, t_transient=T_TRANS,
                     n_qr=N_QR, rng=rng, verbose_every=None)
        wall = time.time() - t0w
        rows.append((lam, r))
        print(f"  lam={lam:.3f}  lam_L={r['lyaps'][0]:+.5f}  "
              f"tail|adot|={r['dnorm_tail']:.3e}  {r['label_final'][:30]}  "
              f"[{wall:.0f}s]", flush=True)
    return LAMS, rows


# ===========================================================================
# A1b - CONVERGENCE / attractor-vs-transient at lam=0.31 (very long run)
# ===========================================================================
def run_A1b():
    print("\n" + "=" * 74)
    print("A1b Sustained check at lam=0.31 (long run, finite-time convergence)")
    print("=" * 74, flush=True)
    sys = ReducedDDE(xi, BETA, 0.31, TAU, T0)
    rng = np.random.default_rng(202)
    rng0 = np.random.default_rng(9)
    a0 = 0.3 * rng0.standard_normal(P)
    t0w = time.time()
    r = benettin(sys, a0, k=1, t_total=T_A1B, dt=DT, t_transient=T_TRANS,
                 n_qr=N_QR, rng=rng, verbose_every=int(500 / DT))
    print(f"  lam=0.31  lam_L={r['lyaps'][0]:+.5f} over T={r['T_accum']:.0f}  "
          f"tail|adot|={r['dnorm_tail']:.3e}  {r['label_final']}  "
          f"[{time.time()-t0w:.0f}s]", flush=True)
    return r


# ===========================================================================
# A2 - partial Lyapunov spectrum + Kaplan-Yorke at lam=0.31
# ===========================================================================
def run_A2():
    print("\n" + "=" * 74)
    print(f"A2  Partial Lyapunov spectrum (k={K_SPEC}) + D_KY at lam=0.31")
    print("=" * 74, flush=True)
    sys = ReducedDDE(xi, BETA, 0.31, TAU, T0)
    rng = np.random.default_rng(303)
    rng0 = np.random.default_rng(11)
    a0 = 0.3 * rng0.standard_normal(P)
    t0w = time.time()
    r = benettin(sys, a0, k=K_SPEC, t_total=T_A2, dt=DT, t_transient=T_TRANS,
                 n_qr=N_QR, rng=rng, verbose_every=int(500 / DT))
    ls = r["lyaps"]
    # Kaplan-Yorke: j = largest index with sum_{1..j} >=0; D_KY = j + S_j/|l_{j+1}|
    cum = np.cumsum(ls)
    j = int(np.sum(cum >= 0))
    if 0 < j < len(ls):
        D_KY = j + cum[j - 1] / abs(ls[j])
    elif j >= len(ls):
        D_KY = float(len(ls))   # spectrum not fully resolved to the negative tail
    else:
        D_KY = 0.0
    print(f"  ordered exponents (k={K_SPEC}):")
    for m, l in enumerate(ls):
        print(f"    lam_{m+1:<2d} = {l:+.5f}", flush=True)
    print(f"  cumulative sums: {np.array2string(cum, precision=4)}")
    print(f"  Kaplan-Yorke dimension D_KY = {D_KY:.3f}  "
          f"(j={j}; {'RESOLVED' if 0<j<len(ls) else 'k too small -> lower bound'})",
          flush=True)
    print(f"  [{time.time()-t0w:.0f}s]", flush=True)
    return dict(lyaps=ls, D_KY=D_KY, j=j, cum=cum, dnorm_tail=r["dnorm_tail"],
                is_fixed=r["is_fixed"])


# ===========================================================================
# A3 - attractor geometry at lam=0.31 (projection, Poincare/return map, spectrum)
# ===========================================================================
def run_A3():
    print("\n" + "=" * 74)
    print("A3  Attractor geometry at lam=0.31 (projection, Poincare, spectrum)")
    print("=" * 74, flush=True)
    sys = ReducedDDE(xi, BETA, 0.31, TAU, T0)
    rng0 = np.random.default_rng(13)
    a0 = 0.3 * rng0.standard_normal(P)
    # long settle then a long recorded run
    settle = sys.integrate(a0, T_TRANS, DT)
    sol = sys.integrate(settle["hist"], T_A3, DT, record_every=A3_REC)
    t = sol["t"]; A = sol["a"]              # (M,P)
    print(f"  recorded {A.shape[0]} samples over t={T_A3} (dt_rec={A3_REC*DT})",
          flush=True)
    # leading 3 overlaps: pick the 3 patterns with largest time-averaged |a|
    var_rank = np.argsort(np.mean(np.abs(A), axis=0))[::-1]
    lead3 = var_rank[:3]
    # PCA of a(t)
    Ac = A - A.mean(axis=0, keepdims=True)
    U, sig, Vt = np.linalg.svd(Ac, full_matrices=False)
    pcs = Ac @ Vt[:3].T                     # first 3 principal components
    var_explained = (sig[:6] ** 2) / np.sum(sig ** 2)
    print(f"  PCA variance explained (first 6): "
          f"{np.array2string(var_explained, precision=3)}", flush=True)
    # Poincare section: crossings of PC1 = mean (upward) -> record PC2
    x = pcs[:, 0]; y = pcs[:, 1]
    xm = 0.0                                 # PC1 is mean-centred
    cross = np.nonzero((x[:-1] < xm) & (x[1:] >= xm))[0]
    # linear-interpolated PC2 at crossing
    poinc = []
    for c in cross:
        f = (xm - x[c]) / (x[c + 1] - x[c] + 1e-30)
        poinc.append(y[c] + f * (y[c + 1] - y[c]))
    poinc = np.array(poinc)
    # first-return map of a single coordinate: successive maxima of PC1
    dxp = np.diff(x)
    maxima_idx = np.nonzero((dxp[:-1] > 0) & (dxp[1:] <= 0))[0] + 1
    peaks = x[maxima_idx]
    # power spectrum of leading overlap (mean-subtracted), Welch-ish single FFT
    sig_ts = A[:, lead3[0]] - A[:, lead3[0]].mean()
    fs = 1.0 / (A3_REC * DT)
    freqs = np.fft.rfftfreq(len(sig_ts), d=A3_REC * DT)
    psd = np.abs(np.fft.rfft(sig_ts * np.hanning(len(sig_ts)))) ** 2
    print(f"  Poincare crossings: {len(poinc)}; PC1 maxima (return map): "
          f"{len(peaks)}", flush=True)
    return dict(t=t, A=A, lead3=[int(x) for x in lead3], pcs=pcs,
                var_explained=var_explained, poinc=poinc, peaks=peaks,
                freqs=freqs, psd=psd, x=x, y=y, pc3=pcs[:, 2])


# ---------------------------------------------------------------------------
# RUN PARAMETERS (set after pilot).  Overridden by --pilot.
# ---------------------------------------------------------------------------
T_TRANS = 400.0
N_QR = 100          # QR every 100 steps = every 2.0 time units
T_A1 = 2000.0
T_A1B = 3000.0
K_SPEC = 8
T_A2 = 2000.0
T_A3 = 3000.0
A3_REC = 5          # record every 5 steps (dt_rec=0.1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()

    if args.pilot:
        T_TRANS = 100.0; T_A1 = 300.0; T_A1B = 400.0; T_A2 = 300.0; T_A3 = 400.0
        K_SPEC = 4
        print("[E23A] PILOT MODE (short runs, timing)\n", flush=True)

    t_all = time.time()
    results = {}
    if args.stage in ("all", "A1"):
        results["A1_lams"], results["A1_rows"] = run_A1()
    if args.stage in ("all", "A1b"):
        results["A1b"] = run_A1b()
    if args.stage in ("all", "A2"):
        results["A2"] = run_A2()
    if args.stage in ("all", "A3"):
        results["A3"] = run_A3()

    # ---- save npz (Part A) ----
    if not args.pilot and args.stage == "all":
        save = dict(
            A1_lams=np.array(results["A1_lams"]),
            A1_lyaps=np.array([r["lyaps"][0] for _, r in results["A1_rows"]]),
            A1_dnorm=np.array([r["dnorm_tail"] for _, r in results["A1_rows"]]),
            A1_isfixed=np.array([r["is_fixed"] for _, r in results["A1_rows"]]),
            A1b_lyap=results["A1b"]["lyaps"][0],
            A1b_T=results["A1b"]["T_accum"],
            A1b_ft=results["A1b"]["ft_hist"],
            A1b_dnorm=results["A1b"]["dnorm_tail"],
            A2_lyaps=results["A2"]["lyaps"],
            A2_DKY=results["A2"]["D_KY"],
            A2_cum=results["A2"]["cum"],
            A3_lead3=np.array(results["A3"]["lead3"]),
            A3_var=results["A3"]["var_explained"],
            A3_poinc=results["A3"]["poinc"],
            A3_peaks=results["A3"]["peaks"],
            A3_freqs=results["A3"]["freqs"],
            A3_psd=results["A3"]["psd"],
        )
        np.savez(os.path.join(OUT, "E23A_lyap.npz"), **save)
        np.savez(os.path.join(OUT, "E23A_geometry.npz"),
                 pcs=results["A3"]["pcs"], A=results["A3"]["A"],
                 t=results["A3"]["t"], x=results["A3"]["x"],
                 y=results["A3"]["y"], pc3=results["A3"]["pc3"])
        print(f"\n[E23A] npz saved to {OUT}/E23A_lyap.npz, E23A_geometry.npz")

        # ---- figure Q ----
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R = results
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3)

        # (a) Lyapunov spectrum bar + D_KY
        axa = fig.add_subplot(gs[0, 0])
        ls = R["A2"]["lyaps"]
        axa.bar(np.arange(1, len(ls) + 1), ls,
                color=["crimson" if l > 0 else "steelblue" for l in ls])
        axa.axhline(0, color="k", lw=0.8)
        axa.set_xlabel("index $i$"); axa.set_ylabel(r"$\lambda_i$")
        axa.set_title(f"(a) Lyapunov spectrum @ $\\lambda$=0.31\n"
                      f"$D_{{KY}}$={R['A2']['D_KY']:.2f}")
        axa.grid(alpha=.3)

        # (b) finite-time largest exponent convergence (A1b)
        axb = fig.add_subplot(gs[0, 1])
        ft = R["A1b"]["ft_hist"]
        axb.plot(ft[:, 0], ft[:, 1], "-", color="darkgreen")
        axb.axhline(R["A1b"]["lyaps"][0], color="crimson", ls="--",
                    label=f"$\\lambda_L$={R['A1b']['lyaps'][0]:+.4f}")
        axb.axhline(0, color="k", lw=0.8)
        axb.set_xlabel("t"); axb.set_ylabel(r"finite-time $\lambda_L(t)$")
        axb.set_title("(b) Convergence of largest exponent\n(sustained attractor)")
        axb.legend(fontsize=9); axb.grid(alpha=.3)

        # (c) lam_L(lam) across window
        axc = fig.add_subplot(gs[0, 2])
        lam_arr = np.array(R["A1_lams"])
        ll = np.array([r["lyaps"][0] for _, r in R["A1_rows"]])
        axc.plot(lam_arr, ll, "o-", color="navy")
        axc.axhline(0, color="k", lw=0.8)
        axc.axvline(0.2822, color="gray", ls=":", label=r"$\lambda_c$")
        axc.axvline(0.328, color="crimson", ls=":", label=r"$\lambda^*$")
        axc.set_xlabel(r"$\lambda$"); axc.set_ylabel(r"$\lambda_L$")
        axc.set_title("(c) Largest exponent across mixture window")
        axc.legend(fontsize=8); axc.grid(alpha=.3)

        # (d) attractor projection PC1-PC2
        axd = fig.add_subplot(gs[1, 0])
        pcs = R["A3"]["pcs"]
        axd.plot(pcs[:, 0], pcs[:, 1], "-", lw=0.3, color="purple", alpha=0.6)
        axd.set_xlabel("PC1"); axd.set_ylabel("PC2")
        axd.set_title(f"(d) Attractor projection (PCA)\n"
                      f"var expl. PC1-3="
                      f"{np.round(R['A3']['var_explained'][:3],2)}")
        axd.grid(alpha=.3)

        # (e) Poincare first-return map
        axe = fig.add_subplot(gs[1, 1])
        pk = R["A3"]["peaks"]
        if len(pk) > 3:
            axe.plot(pk[:-1], pk[1:], ".", ms=3, color="darkorange")
            lo = min(pk.min(), pk.min()); hi = max(pk.max(), pk.max())
            axe.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        axe.set_xlabel(r"PC1 max $n$"); axe.set_ylabel(r"PC1 max $n{+}1$")
        axe.set_title("(e) First-return map (PC1 maxima)")
        axe.grid(alpha=.3)

        # (f) power spectrum (log)
        axf = fig.add_subplot(gs[1, 2])
        fr = R["A3"]["freqs"]; ps = R["A3"]["psd"]
        axf.semilogy(fr[1:], ps[1:], "-", color="teal", lw=0.7)
        axf.set_xlabel("frequency"); axf.set_ylabel("power")
        axf.set_xlim(0, min(1.0, fr[-1]))
        axf.set_title("(f) Power spectrum of leading overlap\n(broadband?)")
        axf.grid(alpha=.3)

        fig.suptitle(f"E23 Part A - Nature of the mixture-phase chaos "
                     f"($N={N}$, $\\alpha={ALPHA}$, $\\tau={TAU}$, seed {SEED})",
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fp = os.path.join(FIGDIR, "figQ_chaos_nature.png")
        fig.savefig(fp, dpi=175, bbox_inches="tight")
        print(f"[E23A] figure -> {fp}")

    print(f"\n[E23A] total wall {time.time()-t_all:.0f}s")
