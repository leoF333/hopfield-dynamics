"""
E23 PART B - The SNIC/chaos paradox at small tau (PI Q6).

Paradox: the pinned FRONTS are tau-independent fixed points with a saddle-node at
lam*_front ~ 0.328 (tau-indep, Delta(0)=-M).  At tau=10 the recall cycle dies BY
reaching that pinning (a clean SNIC, sqrt slowing).  At small tau (E19) the cycle
dies into "chaos/drift" with NO sqrt law.  Resolution proposed: the front's
saddle-node is still there, but at small tau the delocalized cycle collides with
the chaotic set (boundary crisis) at lam_crisis(tau) > lam*_front, so it never
gets to ride the front-saddle-node ghost.

  B1  Fine (tau,lam) frontier map, tau in {0.25,0.5,1,2,5}, lam descending in
      fine steps around [0.325,0.40].  Seeded descent from the cycle continuation;
      classify each attractor: cycle / chaos / pinned-front / memory.  Produce the
      cycle-death curve lam_death(tau) and the post-death state; overlay the
      tau-independent front saddle-node lam*_front.  Show whether a CHAOS BAND sits
      between the cycle-death and the front pinning at small tau.
  B2  Nature of "dying on chaos": largest Lyapunov exponent just ABOVE death (on
      the cycle, expect ~0) and just BELOW (expect jump to >0) = boundary crisis /
      collision with a chaotic set.  Contrast tau=10 (lam_L<=0 until pinning).
  B3  Coexistence: is the pinned front STILL an attractor in that same lam range
      at small tau (it should be, tau-indep)?  -> chaos and pinned front COEXIST;
      the cycle collapses onto the chaos, not the front, by basin geometry.

Reuses: cycle_reduced.ReducedDDE, robust_branch (front saddle-node & stability),
and e23a_chaos_nature.benettin (largest Lyapunov).

Outputs: results/cycle/E23B_frontier.npz + figures/figR_cycle_chaos_frontier.png.
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

from couplings import make_patterns, Couplings
from cycle_reduced import ReducedDDE
from robust_branch import woodbury_newton, eigmax_M
from reduced_spectrum import physical_roots
from e23a_chaos_nature import benettin

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
FIGDIR = os.path.join(OUT, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def dt_for(tau):
    if tau <= 0.0:
        return 0.01
    return min(0.01, tau / 25.0)


# ---------------------------------------------------------------------------
# Attractor classifier from a trajectory tail (extends E19b classify_state).
# labels: cycle / chaos / pinned-front / memory / fixed(other)
# ---------------------------------------------------------------------------
def classify(A, dt, sys, tau):
    a = A[-1]
    adot = sys.rhs(a, a)
    dnorm = float(np.linalg.norm(adot))
    tail = A[int(0.6 * len(A)):]
    var = float(np.mean(np.std(tail, axis=0)))
    # front handoff count over the tail (advancing sequence = cycle)
    lead = np.argmax(A, axis=1)
    i0 = int(0.5 * len(A))
    chg = np.nonzero(np.diff(lead[i0:]) != 0)[0]
    n_hand = len(chg)
    order = np.argsort(np.abs(a))[::-1]
    top = order[:4]; vals = np.abs(a[top])
    d01 = min(abs(int(top[0]) - int(top[1])), P - abs(int(top[0]) - int(top[1])))
    consecutive = (d01 == 1)
    if dnorm < 2e-3 and var < 2e-3:
        # a fixed point
        if vals[0] > 0.9 and vals[1] < 0.15:
            label = "memory"
        elif vals[1] > 0.12 and consecutive:
            label = "pinned-front"
        else:
            label = "fixed-other"
    else:
        # non-fixed: cycle (advancing front) vs chaos (irregular)
        if n_hand >= 3:
            label = "cycle"
        else:
            label = "chaos"
    return dict(label=label, dnorm=dnorm, var=var, n_hand=n_hand,
                top=[int(k) for k in top], vals=[float(v) for v in vals])


def make_cycle_seed(tau, lam_hi, dt):
    """Build a travelling cycle history at lam_hi from a memory IC + bias."""
    sys = ReducedDDE(xi, BETA, lam_hi, tau, T0)
    a0 = np.zeros(P); a0[0] = 0.99; a0[1] = 0.05
    sol = sys.integrate(a0, max(200.0, 80 * (tau + 1)), dt)
    return sol["hist"]


# ===========================================================================
# B1 - fine (tau,lam) frontier map by seeded descent
# ===========================================================================
def run_B1(taus, lam_hi, lam_lo, step, t_seed):
    print("=" * 74)
    print(f"B1  (tau,lam) frontier map; lam {lam_hi}->{lam_lo} step {step}")
    print("=" * 74, flush=True)
    lam_grid = np.round(np.arange(lam_hi, lam_lo - 1e-9, -step), 4)
    grid = {}       # tau -> list of (lam,label,...)
    death = {}      # tau -> lam of last surviving cycle
    for tau in taus:
        dt = dt_for(tau)
        t0w = time.time()
        hist = make_cycle_seed(tau, lam_hi, dt)
        rows = []
        last_cycle = np.nan
        for lam in lam_grid:
            sys = ReducedDDE(xi, BETA, lam, tau, T0)
            sol = sys.integrate(hist, t_seed, dt)
            A = sol["a"]
            st = classify(A, dt, sys, tau)
            rows.append(dict(lam=float(lam), **st))
            if st["label"] == "cycle":
                last_cycle = float(lam)
                hist = sol["hist"]       # seed next lower lam only while alive
            else:
                hist = sol["hist"]       # keep descending through the dead phase
        grid[tau] = rows
        death[tau] = last_cycle
        labs = "".join({"cycle": "C", "chaos": "X", "pinned-front": "F",
                        "memory": "M", "fixed-other": "o"}[r["label"]]
                       for r in rows)
        print(f"  tau={tau:>4}: death@lam={last_cycle}  "
              f"[{lam_hi}->{lam_lo}]  {labs}  [{time.time()-t0w:.0f}s]", flush=True)
    return lam_grid, grid, death


# ===========================================================================
# front saddle-node lam*_front (tau-independent) via eigmax_M crossing
# ===========================================================================
def front_saddle_node():
    print("\n" + "=" * 74)
    print("front saddle-node lam*_front (tau-indep) via eigmax_M(front) -> 0")
    print("=" * 74, flush=True)
    # weak-bond 91/92 front: build near lam=0.30 then continue up until eig->0
    lam = 0.30
    a = np.zeros(P); a[91] = 0.70; a[92] = 0.30
    u = xi.T @ a
    u, ok = woodbury_newton(coup, u, lam, BETA)
    lams, ems = [], []
    step = 0.002
    lam_prev, e_prev = None, None
    lam_star = np.nan
    while lam < 0.345:
        u2, ok = woodbury_newton(coup, u, lam, BETA)
        if not ok:
            # Newton stalls at the saddle-node (M singular)
            lam_star = lam; break
        e = eigmax_M(coup, u2, lam, BETA)
        lams.append(lam); ems.append(e); u = u2
        if e >= 0 and e_prev is not None and e_prev < 0:
            lam_star = lam_prev + (0 - e_prev) / (e - e_prev) * (lam - lam_prev)
            break
        lam_prev, e_prev = lam, e
        lam += step
    print(f"  lam*_front = {lam_star:.4f}  (eig_max(M) crossing / Newton stall)",
          flush=True)
    return lam_star, np.array(lams), np.array(ems)


# ===========================================================================
# B2 - Lyapunov just above / below death; contrast tau=10
# ===========================================================================
def run_B2(taus_small, death, lam_star_front):
    print("\n" + "=" * 74)
    print("B2  Largest Lyapunov just ABOVE vs BELOW cycle-death (boundary crisis)")
    print("=" * 74, flush=True)
    rows = []
    for tau in taus_small:
        ld = death[tau]
        if not np.isfinite(ld):
            continue
        dt = dt_for(tau)
        # ABOVE death: on the cycle (lam = ld) ; BELOW: just under (ld - 0.006)
        for tag, lam in [("above(cycle)", ld), ("below(post)", round(ld - 0.006, 4))]:
            sys = ReducedDDE(xi, BETA, lam, tau, T0)
            rng = np.random.default_rng(555)
            # seed from a travelling cycle so we sit on the relevant set
            hist = make_cycle_seed(tau, 0.42, dt)
            settle = sys.integrate(hist, 150.0, dt)
            r = benettin(sys, settle["hist"], k=1, t_total=600.0, dt=dt,
                         t_transient=100.0, n_qr=max(20, int(2.0 / dt)), rng=rng)
            rows.append((tau, tag, lam, r["lyaps"][0], r["dnorm_tail"],
                         r["label_final"]))
            print(f"  tau={tau:>4} {tag:>13} lam={lam:.4f}: "
                  f"lam_L={r['lyaps'][0]:+.5f}  tail|adot|={r['dnorm_tail']:.2e}",
                  flush=True)
    # tau=10 contrast: on the cycle just above 0.328 (expect lam_L<=0)
    print("  --- tau=10 contrast (expect lam_L<=0 on cycle until pinning) ---",
          flush=True)
    for lam in [0.335, 0.330]:
        sys = ReducedDDE(xi, BETA, lam, 10.0, T0)
        rng = np.random.default_rng(556)
        hist = make_cycle_seed(10.0, 0.45, 0.02)
        settle = sys.integrate(hist, 200.0, 0.02)
        r = benettin(sys, settle["hist"], k=1, t_total=800.0, dt=0.02,
                     t_transient=200.0, n_qr=100, rng=rng)
        rows.append((10.0, "cycle", lam, r["lyaps"][0], r["dnorm_tail"],
                     r["label_final"]))
        print(f"  tau=10.0 {'cycle':>13} lam={lam:.4f}: "
              f"lam_L={r['lyaps'][0]:+.5f}  tail|adot|={r['dnorm_tail']:.2e}",
              flush=True)
    return rows


# ===========================================================================
# B3 - coexistence: is the pinned front an attractor in the small-tau death band?
# ===========================================================================
def run_B3(taus_small, death):
    print("\n" + "=" * 74)
    print("B3  Coexistence: pinned front stability in the death band (tau-indep)")
    print("=" * 74, flush=True)
    rows = []
    for tau in taus_small:
        ld = death[tau]
        if not np.isfinite(ld):
            continue
        dt = dt_for(tau)
        # front fixed point at lam just below death band (tau-indep existence)
        for lam in [round(ld - 0.003, 4), round(ld - 0.010, 4)]:
            a = np.zeros(P); a[91] = 0.70; a[92] = 0.30
            u = xi.T @ a
            u, ok = woodbury_newton(coup, u, lam, BETA)
            em = eigmax_M(coup, u, lam, BETA)
            # delayed spectrum at this small tau: count unstable roots of T_P
            u_star = u
            try:
                ph, _, _ = physical_roots(coup, u_star, BETA, lam, tau, T0,
                                          M=24, n_cand=30, k=6)
                n_unstable = int(np.sum(np.real(ph) > 1e-6)) if len(ph) else 0
                re_dom = float(np.max(np.real(ph))) if len(ph) else np.nan
            except Exception as e:
                n_unstable, re_dom = -1, np.nan
            # dynamical basin test: does a small perturbation of the front stay?
            sysP = ReducedDDE(xi, BETA, lam, tau, T0)
            a_pin = np.linalg.lstsq(xi.T, u, rcond=None)[0]
            rng = np.random.default_rng(77)
            a_pert = a_pin + 0.02 * rng.standard_normal(P)
            sol = sysP.integrate(a_pert, 200.0, dt)
            a_end = sol["a"][-1]
            stayed = float(np.linalg.norm(a_end - a_pin)) < 0.05
            rows.append((tau, lam, ok, em, n_unstable, re_dom, stayed))
            print(f"  tau={tau:>4} lam={lam:.4f}: front ok={ok} "
                  f"eig_max(M)={em:+.4f} n_unstable(T_P)={n_unstable} "
                  f"Re_dom={re_dom:+.4f} basin_stayed={stayed}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# RUN PARAMETERS (set after pilot)
# ---------------------------------------------------------------------------
TAUS = [0.25, 0.5, 1.0, 2.0, 5.0]
LAM_HI = 0.40
LAM_LO = 0.325
STEP = 0.005
T_SEED = 200.0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    if args.pilot:
        TAUS = [0.25, 2.0]; STEP = 0.01; T_SEED = 120.0
        print("[E23B] PILOT MODE\n", flush=True)

    t_all = time.time()
    lam_grid, grid, death = run_B1(TAUS, LAM_HI, LAM_LO, STEP, T_SEED)
    lam_star_front, fl_lams, fl_ems = front_saddle_node()

    if not args.pilot:
        B2 = run_B2(TAUS, death, lam_star_front)
        B3 = run_B3(TAUS, death)

        # ---- summary tables ----
        print("\n" + "=" * 74)
        print("SUMMARY: cycle-death vs front saddle-node")
        print("=" * 74)
        print(f"  lam*_front (tau-indep) = {lam_star_front:.4f}")
        print(f"{'tau':>6} {'lam_death':>10} {'gap=death-front':>16} {'post-death':>14}")
        for tau in TAUS:
            ld = death[tau]
            # post-death label = first non-cycle label below death
            post = "n/a"
            for r in grid[tau]:
                if r["lam"] < ld - 1e-9 and r["label"] != "cycle":
                    post = r["label"]; break
            gap = ld - lam_star_front if np.isfinite(ld) else np.nan
            print(f"{tau:>6} {ld:>10.4f} {gap:>16.4f} {post:>14}")

        # ---- save npz ----
        np.savez(os.path.join(OUT, "E23B_frontier.npz"),
                 taus=np.array(TAUS), lam_grid=lam_grid,
                 lam_star_front=lam_star_front,
                 fl_lams=fl_lams, fl_ems=fl_ems,
                 death=np.array([death[t] for t in TAUS]),
                 **{f"labels_tau{t}": np.array([r["label"] for r in grid[t]])
                    for t in TAUS},
                 **{f"nhand_tau{t}": np.array([r["n_hand"] for r in grid[t]])
                    for t in TAUS},
                 **{f"dnorm_tau{t}": np.array([r["dnorm"] for r in grid[t]])
                    for t in TAUS},
                 B2=np.array([[str(x) for x in row] for row in B2], dtype=object),
                 B3=np.array([[str(x) for x in row] for row in B3], dtype=object))
        print(f"\n[E23B] npz saved -> {OUT}/E23B_frontier.npz")

        # ---- figure R ----
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = {"cycle": 0, "chaos": 1, "pinned-front": 2, "memory": 3,
                "fixed-other": 4}
        colors = ["#2166ac", "#b2182b", "#f4a582", "#4d9221", "#999999"]
        names = ["cycle", "chaos", "pinned-front", "memory", "fixed-other"]

        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])

        # (a) phase diagram tau vs lam
        axa = fig.add_subplot(gs[:, 0])
        tau_ax = np.array(TAUS)
        for ti, tau in enumerate(TAUS):
            for r in grid[tau]:
                axa.add_patch(plt.Rectangle(
                    (r["lam"] - STEP / 2, ti - 0.42), STEP, 0.84,
                    color=colors[cmap[r["label"]]], ec="none"))
        axa.axvline(lam_star_front, color="k", ls="--", lw=1.6,
                    label=rf"$\lambda^*_{{\rm front}}$={lam_star_front:.3f} (SN, $\tau$-indep)")
        dvals = np.array([death[t] for t in TAUS])
        axa.plot(dvals, np.arange(len(TAUS)), "k*-", ms=13,
                 label=r"cycle-death $\lambda_{\rm death}(\tau)$")
        axa.set_yticks(range(len(TAUS)))
        axa.set_yticklabels([str(t) for t in TAUS])
        axa.set_xlabel(r"$\lambda$"); axa.set_ylabel(r"$\tau$")
        axa.set_xlim(LAM_LO - STEP, LAM_HI + STEP)
        axa.set_ylim(-0.6, len(TAUS) - 0.4)
        axa.set_title("(a) $(\\tau,\\lambda)$ attractor map (seeded cycle descent)")
        # legend patches
        from matplotlib.patches import Patch
        handles = [Patch(color=colors[i], label=names[i]) for i in range(5)]
        handles += [plt.Line2D([], [], color="k", ls="--", label="front SN"),
                    plt.Line2D([], [], color="k", marker="*", ls="-",
                               label="cycle-death")]
        axa.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)

        # (b) gap = lam_death - lam*_front vs tau (chaos band width)
        axb = fig.add_subplot(gs[0, 1])
        gaps = dvals - lam_star_front
        axb.plot(tau_ax, gaps, "o-", color="crimson")
        axb.axhline(0, color="k", lw=0.8, label="cycle-death = front SN")
        axb.set_xlabel(r"$\tau$")
        axb.set_ylabel(r"$\lambda_{\rm death}-\lambda^*_{\rm front}$")
        axb.set_title("(b) Chaos-band width (death above front SN)")
        axb.legend(fontsize=8); axb.grid(alpha=.3)

        # (c) lam_L jump at death (B2)
        axc = fig.add_subplot(gs[1, 1])
        # plot above/below pairs for small tau
        sm = [r for r in B2 if r[0] in TAUS]
        taus_b2 = sorted(set(r[0] for r in sm))
        for tau in taus_b2:
            ab = [r for r in sm if r[0] == tau and r[1].startswith("above")]
            be = [r for r in sm if r[0] == tau and r[1].startswith("below")]
            if ab and be:
                axc.plot([0, 1], [ab[0][3], be[0][3]], "o-",
                         label=rf"$\tau$={tau}")
        axc.axhline(0, color="k", lw=0.8)
        axc.set_xticks([0, 1]); axc.set_xticklabels(["above\n(cycle)", "below\n(post)"])
        axc.set_ylabel(r"$\lambda_L$")
        axc.set_title("(c) Lyapunov jump at cycle-death")
        axc.legend(fontsize=8); axc.grid(alpha=.3)

        fig.suptitle(f"E23 Part B - Cycle/chaos frontier at small $\\tau$ "
                     f"($N={N}$, $\\alpha={ALPHA}$, seed {SEED})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fp = os.path.join(FIGDIR, "figR_cycle_chaos_frontier.png")
        fig.savefig(fp, dpi=175, bbox_inches="tight")
        print(f"[E23B] figure -> {fp}")

    print(f"\n[E23B] total wall {time.time()-t_all:.0f}s")
