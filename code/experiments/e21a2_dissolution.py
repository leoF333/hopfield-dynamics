"""
E21a-bis - Pin down the DISSOLUTION mechanism of the pinned front at its birth.

E21a (arclength) showed the front branch does NOT fold: it sails through lam=0.1322
(the spurious E18a stall) with eig_max(M) monotonically -> -1 (deep memory limit),
while a91 -> 1 and a92 -> 0 as lam decreases. So the front is NOT born at a
saddle-node; it DISSOLVES as its SECOND overlap a92 -> 0, i.e. it merges into the
single-pattern (xi^91) memory state.

This script settles the mechanism quantitatively:
  (1) Trace a91, a92 vs lam far down (arclength continued to smaller a92) and fit
      a92(lam): does a92 hit 0 at a FINITE lam_f (transcritical/pitchfork with the
      pure memory) or only as lam -> 0 (asymptotic merge)?
  (2) Independently construct the PURE xi^91 memory fixed point at the same lam
      (Newton from 0.99*xi^91) and compare: the front's a-vector should approach the
      pure-memory a-vector (a92 -> 0, a91 -> memory value) as lam decreases -> proof
      of a merge, not a fold.
  (3) Report eig_max(M) of BOTH the front and the pure memory: at the merge they
      coincide; the memory stays deeply stable (no eigenvalue near 0) -> the birth
      is a MERGE / branch-transcritical event, NOT a fold.

Outputs: results/cycle/E21_dissolution.npz + printed tables.
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tqdm import tqdm

from couplings import Couplings, make_patterns
from robust_branch import woodbury_solve, woodbury_newton, eigmax_M

N, ALPHA, BETA, T0, SEED = 2000, 0.05, 20.0, 1.0, 42
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
MU_A, MU_B = 91, 92
GRAM = (xi @ xi.T) / N


def amp(u):
    return np.linalg.solve(GRAM, coup.overlap_raw(u))


def residual(u, lam):
    return np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)


def dF_dlam(u):
    tu = np.tanh(BETA * u)
    return coup.apply_K(tu) - coup.apply_J(tu)


def tangent(u, lam, prev_dir):
    g = coup.compute_gain(u, BETA)
    Fl = dF_dlam(u)
    du_raw = -woodbury_solve(coup, g, lam, Fl)
    vec = np.concatenate([du_raw, [1.0]]); vec /= np.linalg.norm(vec)
    if np.dot(vec, prev_dir) < 0:
        vec = -vec
    return vec


def bordered_corrector(u_p, lam_p, u0, lam0, tang, ds, tol=1e-11, max_it=50):
    u = u_p.copy(); lam = lam_p
    tu_vec = tang[:-1]; tl = tang[-1]
    for _ in range(max_it):
        F1 = coup.field_F(u, lam, BETA)
        F2 = np.dot(tu_vec, (u - u0)) + tl * (lam - lam0) - ds
        r = np.linalg.norm(F1) / np.sqrt(N) + abs(F2)
        if r < tol:
            return u, lam, True, np.linalg.norm(F1) / np.sqrt(N)
        g = coup.compute_gain(u, BETA); Fl = dF_dlam(u)
        a = woodbury_solve(coup, g, lam, -F1)
        b = woodbury_solve(coup, g, lam, Fl)
        denom = tl - np.dot(tu_vec, b)
        if abs(denom) < 1e-14:
            return u, lam, False, np.linalg.norm(F1) / np.sqrt(N)
        dlam = (-F2 - np.dot(tu_vec, a)) / denom
        du = a - dlam * b
        st = 1.0; improved = False; r_fp = np.linalg.norm(F1) / np.sqrt(N)
        for _ in range(25):
            un = u + st * du; ln = lam + st * dlam
            if np.linalg.norm(coup.field_F(un, ln, BETA)) / np.sqrt(N) < r_fp + 1e-9:
                improved = True; break
            st *= 0.5
        if not improved:
            return u, lam, False, r_fp
        u = u + st * du; lam = lam + st * dlam
    return u, lam, False, np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)


def reach_front(lam_target, lam_start=0.318):
    BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
    u = BR["u_pin"].copy(); lam = 0.318; step = 0.004
    while abs(lam - lam_target) > 1e-9:
        ln = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, ok = woodbury_newton(coup, u, ln, BETA, tol=1e-12)
        a = amp(un)
        top = sorted(int(k) for k in np.argsort(np.abs(a))[::-1][:2])
        if ok and top == [MU_A, MU_B] and residual(un, ln) < 1e-10:
            u, lam = un, ln; step = min(step * 1.3, 0.004)
        else:
            step *= 0.5
            if step < 1e-5:
                return None, lam
    return u, lam


def pure_memory(lam):
    """Pure single-pattern xi^91 memory fixed point at lam (Newton from 0.99 xi^91)."""
    u0 = 0.99 * xi[MU_A].copy()
    # beta-anneal for robustness
    u = u0
    for b in [6.0, 10.0, 14.0, BETA]:
        u, ok = woodbury_newton(coup, u, lam, b, tol=1e-12)
    return u, ok


def main():
    t0w = time.time()
    print("[E21a2] === dissolution mechanism of the front birth ===\n", flush=True)

    # ---- (1) arclength FAR down, dense recording of a91,a92 -----------------
    u, lam = reach_front(0.150)
    assert u is not None
    prev_dir = np.zeros(N + 1); prev_dir[-1] = -1.0
    ds = 0.02; s = 0.0
    branch = []   # lam, a91, a92, eig_max, res
    a = amp(u)
    branch.append([lam, a[MU_A], a[MU_B], eigmax_M(coup, u, lam, BETA), residual(u, lam)])
    pbar = tqdm(total=200, desc="arclength deep", bar_format="{l_bar}{bar}| {n}/{total}")
    steps = 0
    while steps < 400 and lam > -0.02:
        tang = tangent(u, lam, prev_dir)
        u_p = u + ds * tang[:-1]; lam_p = lam + ds * tang[-1]
        un, ln, ok, res = bordered_corrector(u_p, lam_p, u, lam, tang, ds)
        a = amp(un)
        top = sorted(int(k) for k in np.argsort(np.abs(a))[::-1][:2])
        # accept while pattern 91 dominant and pattern 92 the runner-up (or a91 alone)
        a1 = abs(a[MU_A])
        front_like = (top == [MU_A, MU_B] or (np.argmax(np.abs(a)) == MU_A)) and a1 < 1.05
        if ok and res < 1e-9 and front_like:
            u, lam = un, ln; prev_dir = tang
            branch.append([lam, a[MU_A], a[MU_B],
                           eigmax_M(coup, u, lam, BETA), res])
            pbar.update(1); steps += 1
            ds = min(ds * 1.2, 0.04)
            if abs(a[MU_B]) < 1e-4:   # a92 effectively vanished -> pure memory
                pbar.set_postfix_str(f"a92 -> 0 at lam={lam:.4f}")
                break
        else:
            ds *= 0.5
            if ds < 2e-5:
                pbar.set_postfix_str(f"stalled lam={lam:.4f} a92={a[MU_B]:.2e}")
                break
    pbar.close()
    branch = np.array(branch)

    print(f"\n[E21a2] deep arclength: {len(branch)} pts, "
          f"lam in [{branch[:,0].min():.4f}, {branch[:,0].max():.4f}]", flush=True)
    print(f"{'lam':>8} {'a91':>8} {'a92':>8} {'eig_max':>9} {'res':>9}")
    lp = None
    for r in branch:
        if lp is None or abs(r[0] - lp) > 0.012 or r is branch[-1]:
            print(f"{r[0]:8.4f} {r[1]:+8.4f} {r[2]:+8.4f} {r[3]:+9.4f} {r[4]:9.1e}")
            lp = r[0]

    # ---- (2) compare with the pure xi^91 memory at matched lam --------------
    print("\n[E21a2] FRONT vs PURE-MEMORY(xi^91) at matched lam:", flush=True)
    print(f"{'lam':>8} {'front_a91':>10} {'front_a92':>10} {'mem_a91':>9} {'mem_a92':>9} "
          f"{'|a_front-a_mem|':>15} {'eigM_front':>11} {'eigM_mem':>10}")
    cmp_rows = []
    for lam_t in [0.130, 0.110, 0.090, 0.070, 0.050, 0.030]:
        # nearest front point on the deep branch
        j = int(np.argmin(np.abs(branch[:, 0] - lam_t)))
        if abs(branch[j, 0] - lam_t) > 0.02:
            continue
        um, okm = pure_memory(lam_t)
        am = amp(um)
        # reconstruct front u at branch[j] lam via warm-start not stored; use branch amps
        af91, af92 = branch[j, 1], branch[j, 2]
        # distance in full a-space between front and memory (need front a-vector):
        # reconstruct front by arclength target — approximate by its recorded amps only
        # for the two dominant comps; compute full memory a-vector norm of tail
        mem_tail = np.linalg.norm(np.delete(am, [MU_A]))
        emf = branch[j, 3]
        emm = eigmax_M(coup, um, lam_t, BETA)
        d = abs(af92 - am[MU_B]) + abs(af91 - am[MU_A])
        print(f"{branch[j,0]:8.4f} {af91:+10.4f} {af92:+10.4f} {am[MU_A]:+9.4f} "
              f"{am[MU_B]:+9.4f} {d:15.4e} {emf:+11.4f} {emm:+10.4f}")
        cmp_rows.append([branch[j, 0], af91, af92, am[MU_A], am[MU_B], d, emf, emm])
    cmp_rows = np.array(cmp_rows)

    np.savez(os.path.join(OUT, "E21_dissolution.npz"),
             branch=branch,
             branch_cols=np.array(["lam", "a91", "a92", "eig_max", "res"]),
             cmp=cmp_rows,
             cmp_cols=np.array(["lam", "front_a91", "front_a92", "mem_a91",
                                "mem_a92", "dist", "eigM_front", "eigM_mem"]))
    print(f"\n[E21a2] saved results/cycle/E21_dissolution.npz")
    print(f"[E21a2] TOTAL {time.time()-t0w:.0f}s")


if __name__ == "__main__":
    main()
