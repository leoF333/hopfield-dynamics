"""
E18a - Downward branch continuation of the bond-91 pinned front.

PI question addressed: how are the pinned-front spurious attractors born?
  * Reconstruct the bond-91 pinned front (0-indexed patterns 91/92, a~[0.68,0.32])
    by quenching a travelling cycle at lam=0.34 to lam=0.318; the front travels
    around the ring and STOPS DEAD (||a'||=0 exactly) at bond 91/92 -> the exact
    pinned fixed point (matches E12: a91=0.68, a92=0.31, res~6e-15).
  * From that fixed point, continue the branch DOWNWARD in lambda with
    Woodbury-Newton warm-starts (exact P x P algebra, no basin jump for small
    steps). Adaptive step; halve on Newton failure / basin jump (top-2 no longer
    consecutive with a1 in [0.5,0.85]); stop at a terminating fold or lam=0.
  * Record along the branch: lam, residual, a_91, a_92, secondary overlaps,
    eig_max(M) (the exact rightmost real eigenvalue of the instantaneous Jacobian;
    fold = eig_max crosses 0, tau-independent).

Also continues UPWARD to reconnect with the E12 depinning fold (~0.328) for a
complete branch picture.

Outputs: results/cycle/E18_front_branch.npz (rows + u_star states at selected lam),
         printed table.
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
from pacemaker_cycle import build_cycle
from cycle_reduced import ReducedDDE
from robust_branch import woodbury_newton, eigmax_M

N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.01, 42, 10.0
P = round(ALPHA * N)
xi, xis = make_patterns(N, P, SEED)
coup = Couplings(xi, xis); coup._use_np = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")

MU_A, MU_B = 91, 92          # 0-indexed active patterns of the bond-91 front
GRAM = (xi @ xi.T) / N        # P x P Gram; a = GRAM^{-1} (1/N) xi u  (u = xi^T a)


def amp(u):
    """Recover the reduced amplitude vector a from u = xi^T a (a = Gram^{-1} m_raw).
    This is the physically meaningful two-pattern coordinate of the front
    (matches E12 a91/a92); the CONDENSED overlap (1/N)xi.tanh(beta u) saturates
    to ~1 on one pattern at beta=20 and is NOT the right front coordinate."""
    return np.linalg.solve(GRAM, coup.overlap_raw(u))


def diag(u, lam):
    res = np.linalg.norm(coup.field_F(u, lam, BETA)) / np.sqrt(N)
    return res, amp(u)


def is_front(a):
    """True iff the amplitude vector a is still the bond-91 two-consecutive front."""
    top = np.argsort(np.abs(a))[::-1][:2]
    a1, a2 = abs(a[top[0]]), abs(a[top[1]])
    consec = abs(int(top[0]) - int(top[1])) == 1
    return consec and (0.45 < a1 < 0.90) and (0.12 < a2 < 0.52), \
        sorted(int(k) for k in top)


# ---- 1) reconstruct the pinned front by quench + chunked settle --------------
# The travelling front slows around the ring and STOPS DEAD (||a'||->0) at the
# weakest bond (91/92 for seed 42). We integrate in chunks and STOP as soon as
# the state is pinned at bond 91/92 (||a'||~0 and top-2 = consecutive front),
# BEFORE any residual drift can dislodge it into a memory basin.
t0w = time.time()
print("[E18a] reconstructing bond-91 pinned front (quench 0.34 -> chunked settle 0.318)...",
      flush=True)
cyc = build_cycle(xi, BETA, 0.34, TAU, dt=DT, verbose=False, settle_turns=3.0)
h = cyc["a_grid"][:cyc["L"] + 1].copy()
sysP = ReducedDDE(xi, BETA, 0.318, TAU, T0)
a_pin = None
for chunk in range(40):
    sol = sysP.integrate(h, 100.0, DT)
    h = sol["hist"]; a = sol["a"][-1]
    drift = np.linalg.norm(sol["a"][-1] - sol["a"][-2]) / DT
    top = np.argsort(np.abs(a))[::-1][:2]
    if sorted(int(k) for k in top) == [MU_A, MU_B] and drift < 1e-8:
        a_pin = a.copy()
        print(f"[E18a] pinned at bond 91/92, t={(chunk+1)*100:.0f}, ||a'||={drift:.1e}",
              flush=True)
        break
assert a_pin is not None, "front never pinned at bond 91/92"
u_pin = xi.T @ a_pin
res0, a0 = diag(u_pin, 0.318)
ok0, top0 = is_front(a0)
print(f"[E18a] pinned front @0.318: res={res0:.1e} top={top0} "
      f"a=({a0[MU_A]:+.3f},{a0[MU_B]:+.3f}) front={ok0} ({time.time()-t0w:.0f}s)",
      flush=True)
assert top0 == [MU_A, MU_B] and ok0, "did not pin at bond 91/92"

# small Newton polish (should already be a fixed point at ~1e-14)
u_pin, _ = woodbury_newton(coup, u_pin, 0.318, BETA, tol=1e-13)
res0, a0 = diag(u_pin, 0.318)
print(f"[E18a] polished: res={res0:.1e} a=({a0[MU_A]:+.3f},{a0[MU_B]:+.3f})",
      flush=True)


def continue_branch(u_start, lam_start, direction, lam_stop,
                    ds0=0.005, ds_min=2e-5, n_sec=6):
    """Woodbury-Newton warm-start continuation. Returns list of rows + states."""
    rows = []; states = {}
    u = u_start.copy(); lam = lam_start
    # record the start
    res, a = diag(u, lam)
    top = np.argsort(np.abs(a))[::-1][:n_sec]
    rows.append([lam, res, a[MU_A], a[MU_B], eigmax_M(coup, u, lam, BETA)]
                + [a[k] for k in top])
    ds = ds0
    pbar = tqdm(total=abs(lam_stop - lam_start), desc=f"cont {'down' if direction<0 else 'up'}",
                bar_format="{l_bar}{bar}| {n:.3f}/{total:.3f}")
    while (direction < 0 and lam > lam_stop + 1e-9) or \
          (direction > 0 and lam < lam_stop - 1e-9):
        lam_new = lam + direction * ds
        if direction < 0:
            lam_new = max(lam_new, lam_stop)
        else:
            lam_new = min(lam_new, lam_stop)
        u_new, okn = woodbury_newton(coup, u, lam_new, BETA, tol=1e-12)
        res, a = diag(u_new, lam_new)
        okf, top2 = is_front(a)
        if okn and okf and res < 1e-10:
            top = np.argsort(np.abs(a))[::-1][:n_sec]
            em = eigmax_M(coup, u_new, lam_new, BETA)
            rows.append([lam_new, res, a[MU_A], a[MU_B], em] + [a[k] for k in top])
            pbar.update(abs(lam_new - lam))
            lam, u = lam_new, u_new
            ds = min(ds * 1.3, ds0)
        else:
            ds *= 0.5
            if ds < ds_min:
                pbar.set_postfix_str(f"TERMINATED at lam={lam:.5f} "
                                     f"(fold/basin-jump, next res={res:.1e})")
                break
    pbar.close()
    return rows, lam


# ---- 2) continue DOWNWARD to lam=0 (or terminating fold) ---------------------
print("\n[E18a] continuing DOWNWARD in lambda ...", flush=True)
rows_down, lam_min_reached = continue_branch(u_pin, 0.318, -1, 0.0)
print(f"[E18a] downward branch: reached lam={lam_min_reached:.5f}, "
      f"{len(rows_down)} points", flush=True)

# ---- 3) continue UPWARD to the depinning fold (~0.33) ------------------------
print("\n[E18a] continuing UPWARD in lambda (to depinning fold) ...", flush=True)
rows_up, lam_max_reached = continue_branch(u_pin, 0.318, +1, 0.34)
print(f"[E18a] upward branch: reached lam={lam_max_reached:.5f}, "
      f"{len(rows_up)} points", flush=True)

# ---- 4) assemble + save ------------------------------------------------------
# merge: reverse down (drop duplicate start), then up
rows = list(reversed(rows_down)) + rows_up[1:]
rows = np.array(rows)          # cols: lam,res,a91,a92,eigmaxM, then n_sec overlaps
order = np.argsort(rows[:, 0])
rows = rows[order]

# save states at a grid of lambda for E18b/E18c/E18d (rebuild by warm-start)
print("\n[E18a] caching u_star at target lambdas for downstream experiments ...",
      flush=True)
LAM_CACHE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.2822, 0.30, 0.31, 0.32, 0.327]
lam_lo = rows[:, 0].min(); lam_hi = rows[:, 0].max()
u_cache = {}
# march from the pinned front to each cached lambda by warm-start chain
def reach(lam_target):
    u = u_pin.copy(); lam = 0.318
    step = 0.005
    while abs(lam - lam_target) > 1e-9:
        lam_next = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, okn = woodbury_newton(coup, u, lam_next, BETA, tol=1e-12)
        res, a = diag(un, lam_next)
        okf, _ = is_front(a)
        if okn and okf and res < 1e-10:
            u, lam = un, lam_next; step = min(step * 1.3, 0.005)
        else:
            step *= 0.5
            if step < 2e-5:
                return None, lam
    return u, lam
for lt in LAM_CACHE:
    if lt < lam_lo - 1e-6 or lt > lam_hi + 1e-6:
        continue
    u, lr = reach(lt)
    if u is not None:
        u_cache[f"u_{lt:.4f}"] = u
        u_cache[f"a_{lt:.4f}"] = amp(u)    # reduced amplitudes for the DDE

np.savez(os.path.join(OUT, "E18_front_branch.npz"),
         rows=rows, ncols_meta=np.array([5]),  # first 5 cols fixed, rest amplitudes
         lam_min=lam_min_reached, lam_max=lam_max_reached,
         mu_a=MU_A, mu_b=MU_B, u_pin=u_pin, **u_cache)

# ---- 5) print table ----------------------------------------------------------
print("\n[E18a] BRANCH TABLE (bond-91 pinned front):")
print(f"{'lam':>8} {'res':>9} {'a91':>8} {'a92':>8} {'eigmax_M':>10} {'a1+a2':>8}")
for r in rows:
    if (abs(r[0]*1000 - round(r[0]*1000)) < 5) or r[0] in (rows[0,0], rows[-1,0]):
        print(f"{r[0]:8.4f} {r[1]:9.1e} {r[2]:+8.3f} {r[3]:+8.3f} {r[4]:+10.4f} "
              f"{abs(r[2])+abs(r[3]):8.3f}")
print(f"\n[E18a] branch exists over lam in [{lam_min_reached:.4f}, "
      f"{lam_max_reached:.4f}]")
print(f"[E18a] eigmax_M range: [{rows[:,4].min():+.4f}, {rows[:,4].max():+.4f}] "
      f"(all < 0 => linearly stable throughout)")
print(f"[E18a] cached states at: {sorted(u_cache.keys())}")
print(f"[E18a] TOTAL {time.time()-t0w:.0f}s")
