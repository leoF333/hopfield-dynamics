"""
E18d - Basin-size scan of the pinned front vs lambda (reduced DDE, tau=10).

For ~6 lambda spanning the front's stable existence range [~0.14 ... 0.327]:
  (i)  DIRECTIONAL escape radius (bisection of the critical amplitude at which the
       trajectory leaves the front basin) along
         - the direction toward the memory xi^1  (amplitude e_1), and
         - the weakest STABLE eigendirection of the instantaneous Jacobian M
           (the P x P reduced Jacobian Dm-based operator; slowest-decaying real
           eigenvector), which is the "softest" escape route.
  (ii) RANDOM-IC sampling: n_rand random histories on spheres of a few radii around
       the front; integrate t=200; classify the final state (front / memory / other)
       by the amplitude signature; report the captured fraction vs radius.

The reduced state is the amplitude vector a (u = xi^T a). The front is the exact
fixed point a*(lam) cached by E18a.

Outputs: results/cycle/E18_basins.npz + printed tables.
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
from cycle_reduced import ReducedDDE
from robust_branch import woodbury_newton

N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.05, 42, 10.0
P = round(ALPHA * N)
T_INT = 200.0
N_RAND = 150
RADII = [0.15, 0.30, 0.50]
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
GRAM = (xi @ xi.T) / N
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
u_pin = BR["u_pin"]

def amp(u): return np.linalg.solve(GRAM, coup.overlap_raw(u))

def reach_front(lam_target):
    u = u_pin.copy(); lam = 0.318; step = 0.005
    while abs(lam - lam_target) > 1e-9:
        lam_next = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, okn = woodbury_newton(coup, u, lam_next, BETA, tol=1e-12)
        res = np.linalg.norm(coup.field_F(un, lam_next, BETA)) / np.sqrt(N)
        if okn and res < 1e-10:
            u, lam = un, lam_next; step = min(step * 1.3, 0.005)
        else:
            step *= 0.5
            if step < 2e-5: return None
    return un

def reduced_jac(a, lam):
    """In-span reduced Jacobian: M_red = (-I + (1-lam) Dm + lam S Dm)/t0 at a
    (tau=0 part; the instantaneous operator whose slowest real eigenvector is the
    softest escape direction)."""
    sysP = ReducedDDE(xi, BETA, lam, TAU, T0)
    Dm = sysP.Dm(a)
    S = np.roll(np.eye(P), 1, axis=0)
    return (-np.eye(P) + (1 - lam) * Dm + lam * (S @ Dm)) / T0

def weakest_eigdir(a, lam):
    M = reduced_jac(a, lam)
    w, V = np.linalg.eig(M)
    # weakest STABLE = largest Re(w) among Re(w) < 0
    idx = np.where(w.real < 0)[0]
    j = idx[np.argmax(w.real[idx])]
    v = np.real(V[:, j]); v /= np.linalg.norm(v)
    return v, float(w[j].real)

def classify(a_final, a_front):
    d = np.linalg.norm(a_final - a_front)
    if d < 0.05:
        return "front", d
    top = sorted(int(k) for k in np.argsort(np.abs(a_final))[::-1][:2])
    s = np.sort(np.abs(a_final))[::-1]
    if s[0] > 0.7 and (top[1]-top[0] != 1 or s[1] < 0.15):
        return "memory", d
    if s[0] < 0.6:
        return "chaos/mix", d
    return "other", d

def escaped(sysP, a_front, a0):
    sol = sysP.integrate(a0, T_INT, DT, record_every=400)
    return np.linalg.norm(sol["a"][-1] - a_front) > 0.05, sol["a"][-1]

def bisect_radius(sysP, a_front, direction, r_hi=2.0, iters=12):
    """Critical escape amplitude along a unit direction."""
    # first ensure r_hi escapes; expand if not
    for _ in range(4):
        esc, _ = escaped(sysP, a_front, a_front + r_hi * direction)
        if esc: break
        r_hi *= 1.6
    lo, hi = 0.0, r_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        esc, _ = escaped(sysP, a_front, a_front + mid * direction)
        if esc: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

LAMS = [0.150, 0.200, 0.250, 0.290, 0.310, 0.325]

print(f"[E18d] basin scan of pinned front (reduced DDE, tau={TAU}, t_int={T_INT}, "
      f"dt={DT}, n_rand={N_RAND}/radius)\n")
rng = np.random.default_rng(2025)
rows_dir = []; rows_rand = []
t0w = time.time()
for lam in LAMS:
    uf = reach_front(lam)
    if uf is None:
        print(f"lam={lam}: front unreachable"); continue
    a_front = amp(uf)
    sysP = ReducedDDE(xi, BETA, lam, TAU, T0)

    # directional escape radii
    e1 = np.zeros(P); e1[0] = 1.0
    r_mem = bisect_radius(sysP, a_front, e1)
    vw, lw = weakest_eigdir(a_front, lam)
    # test both +/- along the weakest direction, take the smaller escape radius
    r_wp = bisect_radius(sysP, a_front, vw)
    r_wm = bisect_radius(sysP, a_front, -vw)
    r_weak = min(r_wp, r_wm)
    rows_dir.append([lam, r_mem, r_weak, lw])
    print(f"[E18d] lam={lam:.3f}: r_esc(->xi^1)={r_mem:.3f}  "
          f"r_esc(weak eigdir, decay {lw:+.3f})={r_weak:.3f}", flush=True)

    # random-IC captured fraction vs radius
    for r in RADII:
        n_front = 0; classes = {}
        for _ in tqdm(range(N_RAND), desc=f"  lam={lam:.3f} r={r}", leave=False):
            d = rng.standard_normal(P); d *= r / np.linalg.norm(d)
            esc, a_fin = escaped(sysP, a_front, a_front + d)
            cls, _ = classify(a_fin, a_front)
            classes[cls] = classes.get(cls, 0) + 1
            if cls == "front": n_front += 1
        frac = n_front / N_RAND
        rows_rand.append([lam, r, frac,
                          classes.get("memory", 0)/N_RAND,
                          classes.get("chaos/mix", 0)/N_RAND,
                          classes.get("other", 0)/N_RAND])
        print(f"          random r={r}: captured(front)={frac:.2f}  "
              f"mem={classes.get('memory',0)/N_RAND:.2f} "
              f"chaos={classes.get('chaos/mix',0)/N_RAND:.2f} "
              f"other={classes.get('other',0)/N_RAND:.2f}", flush=True)

rows_dir = np.array(rows_dir); rows_rand = np.array(rows_rand)
np.savez(os.path.join(OUT, "E18_basins.npz"),
         rows_dir=rows_dir, rows_rand=rows_rand,
         dir_cols=np.array(["lam", "r_esc_xi1", "r_esc_weak", "decay_weak"]),
         rand_cols=np.array(["lam", "radius", "frac_front", "frac_mem",
                             "frac_chaos", "frac_other"]),
         radii=np.array(RADII), n_rand=N_RAND, t_int=T_INT, dt=DT)

print("\n[E18d] DIRECTIONAL escape radii:")
print(f"{'lam':>7} {'r->xi^1':>9} {'r weak-dir':>11} {'weak decay':>11}")
for r in rows_dir:
    print(f"{r[0]:7.3f} {r[1]:9.3f} {r[2]:11.3f} {r[3]:+11.4f}")
print("\n[E18d] RANDOM-IC captured fraction:")
print(f"{'lam':>7} {'radius':>7} {'front':>7} {'mem':>6} {'chaos':>6} {'other':>6}")
for r in rows_rand:
    print(f"{r[0]:7.3f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.2f}")
print(f"\n[E18d] TOTAL {time.time()-t0w:.0f}s")
