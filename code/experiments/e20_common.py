"""
E20 common machinery: reduced-DDE basin-classification for the delayed mixed
Hopfield network (tau=10). Shared by e20a (basin fractions), e20b (metastability)
and e20c (cycle basin). All in ENGLISH.

Model (reduced, exact P-dim, certified 5.5e-16):
    t0 a' = -a + (1-lam) m(a(t)) + lam S m(a(t-tau)),  m(a) = (1/N) xi tanh(beta xi^T a).

State a in R^P is the vector of pattern overlaps (u = xi^T a). Attractors:
  - memory : a ~ alpha * e_mu (one dominant overlap, others small, NOT consecutive
             pair) -> a memory pattern xi^mu.
  - pinned-front : a ~ [a1, a2, ...] with two DOMINANT CONSECUTIVE overlaps
             (nu, nu+1), stationary (the frozen 2-pattern front, E18).
  - cycle : argmax of a advances monotonically around the ring, sustained
             (the sequential-recall pacemaker; period ~ T1 ~ tau+t_escape).
  - chaos : aperiodic, never settles (sustained non-zero drift, no ring advance
             or irregular advance), the itinerant chaotic sea (E7/E8a).
  - other : anything not matched (small-norm/no macroscopic overlap, etc.).

Classification uses the recorded trajectory tail: drift ||a'||, the sorted
overlaps, the identity/advance of argmax over the tail, and distance to the
cached pinned-front a_front(lam).
"""
from __future__ import annotations

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
from cycle_reduced import ReducedDDE
from robust_branch import woodbury_newton, eigmax_M

# ---- reference parameters (spec) ------------------------------------------
N, ALPHA, BETA, T0, DT, SEED, TAU = 2000, 0.05, 20.0, 1.0, 0.05, 42, 10.0
P = round(ALPHA * N)

xi, xis = make_patterns(N, P, SEED)
COUP = Couplings(xi, xis); COUP._use_np = True
GRAM = (xi @ xi.T) / N                      # ~ I (P x P), overlaps <-> amplitudes

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cycle")
_BR = np.load(os.path.join(OUT, "E18_front_branch.npz"))
U_PIN = _BR["u_pin"]


def amp(u):
    """Amplitude vector a from a full-N state u (solve Gram a = xi u / N)."""
    return np.linalg.solve(GRAM, COUP.overlap_raw(u))


# ---- pinned front for arbitrary lam (E18d continuation) -------------------
def reach_front(lam_target):
    """Continue the pinned front (weak bond 91/92, seed 42) to lam_target by
    warm-started Woodbury-Newton in small lam steps from the cached u_pin@0.318.
    Returns the front amplitude vector a_front, or None if unreachable."""
    u = U_PIN.copy(); lam = 0.318; step = 0.005
    while abs(lam - lam_target) > 1e-9:
        lam_next = lam + np.sign(lam_target - lam) * min(step, abs(lam_target - lam))
        un, okn = woodbury_newton(COUP, u, lam_next, BETA, tol=1e-12)
        res = np.linalg.norm(COUP.field_F(un, lam_next, BETA)) / np.sqrt(N)
        if okn and res < 1e-10:
            u, lam = un, lam_next; step = min(step * 1.3, 0.005)
        else:
            step *= 0.5
            if step < 2e-5:
                return None
    return amp(un)


# ---- memory fixed point (pure pattern xi^mu) for arbitrary lam ------------
def reach_memory(lam, mu=0):
    """Newton-polish the pure-memory fixed point on pattern mu (default xi^1).
    Returns amplitude a_mem or None (above the pattern fold it ceases to exist)."""
    u0 = 0.99 * xi[mu].copy()
    u, ok = woodbury_newton(COUP, u0, lam, BETA, tol=1e-12)
    res = np.linalg.norm(COUP.field_F(u, lam, BETA)) / np.sqrt(N)
    if ok and res < 1e-9:
        a = amp(u)
        # sanity: dominant overlap must be on mu and macroscopic
        if np.argmax(np.abs(a)) == mu and abs(a[mu]) > 0.7:
            return a
    return None


# ---- instantaneous rightmost eigenvalue of M at an amplitude state --------
def eigmax_at_amp(a, lam):
    """eig_max(M) of the FULL-N Jacobian at u = xi^T a (via P x P Sylvester)."""
    u = xi.T @ a
    return eigmax_M(COUP, u, lam, BETA)


# =====================================================================
#  CLASSIFIER
# =====================================================================
# Thresholds (documented; validated on known ICs in the pilot).
FRONT_TOL = 0.06       # ||a_final - a_front|| below this -> pinned-front
DRIFT_STAT = 3e-3      # tail drift ||a'|| below this -> stationary (fixed point)
MEM_DOM = 0.70         # dominant overlap magnitude for a memory-like state
PAIR_SECOND = 0.15     # second overlap magnitude to call a 2-pattern front


def _tail_stats(sol_a, sysP, lam, tail_frac=0.25):
    """Compute tail diagnostics from a recorded trajectory sol_a (n_rec, P)."""
    n = sol_a.shape[0]
    k0 = max(1, int((1 - tail_frac) * n))
    tail = sol_a[k0:]
    a_last = sol_a[-1]
    # drift: RHS norm at the last recorded state (true instantaneous velocity
    # needs the delayed value; approximate by finite diff of the recorded tail)
    if tail.shape[0] >= 2:
        # recorded samples are 'record_every*dt' apart; use normalized diff
        drift = np.linalg.norm(tail[-1] - tail[-2]) / (np.linalg.norm(tail[-1]) + 1e-12)
    else:
        drift = 0.0
    # argmax trajectory over the tail (ring advance detector)
    leads = np.argmax(sol_a, axis=1)
    return a_last, drift, leads


def _ring_advances(leads, P):
    """Count net forward advances of the lead index around the P-ring over a
    sequence of per-sample lead indices. Returns (n_advance, monotone_frac)."""
    d = np.diff(leads.astype(int))
    # forward step on the ring: +1 (mod P), allow wrap -(P-1)
    fwd = np.sum((d == 1) | (d == -(P - 1)))
    bwd = np.sum((d == -1) | (d == (P - 1)))
    moves = fwd + bwd
    n_adv = fwd - bwd
    mono = fwd / moves if moves > 0 else 0.0
    return int(n_adv), float(mono)


def _front_shape_ratio(lam):
    """s1/s0 of the E18 weak-bond pinned front at this lam (the balanced-mixture
    signature). Grows 0.16->0.49 as lam: 0.15->0.328. Used as the memory/front
    ratio threshold (mid-point between memory tail ratio and front ratio)."""
    a_front = reach_front(lam)
    if a_front is None:
        return None
    s = np.sort(np.abs(a_front))[::-1]
    return float(s[1] / s[0])


WEAK_BOND = (91, 92)           # E18 weak (depinning) bond, seed 42


def classify(sol, sysP, lam, a_front, r_front=None):
    """Classify a recorded reduced-DDE trajectory dict {t,a} into
    memory / pinned-front / cycle / chaos / other.

    KEY PHYSICS (E18 + this experiment): every stationary recalled state is a
    dominant pattern xi^mu plus its K-induced cyclic successor xi^{mu+1} -- the
    memory xi^1 and the "pinned front" are the SAME kind of 2-pattern object, with
    a balance ratio s1/s0 fixed by lam (not by the bond). They are NOT separable by
    shape. What makes a front METASTABLE is that it sits on the WEAKEST bond
    (91/92, seed 42), the one that depins at lam*. We therefore classify a
    stationary 2-pattern state by WHICH bond it settles on:
      - dominant bond == weak bond (91,92), i.e. near the cached a_front
                                                    -> 'front' (metastable front)
      - dominant bond == any other bond            -> 'memory' (robust recall of
                                                       pattern mu, its own front)
    This is the honest, unambiguous separatrix given that both are the same object
    on different bonds.

    Logic order: exact-front distance / weak-bond -> cycle (ring advance) ->
    stationary (memory vs front by BOND) -> chaos -> other.
    Returns (label, info_dict).
    """
    a = sol["a"]
    a_last, drift, leads = _tail_stats(a, sysP, lam)
    P = a.shape[1]
    s = np.sort(np.abs(a_last))[::-1]
    top = np.argsort(np.abs(a_last))[::-1]
    b0, b1 = int(top[0]), int(top[1])
    consecutive = (abs(b0 - b1) == 1) or (abs(b0 - b1) == P - 1)
    balance = float(s[1] / s[0]) if s[0] > 1e-9 else 0.0
    on_weak_bond = ({b0, b1} == set(WEAK_BOND))

    d_front = (float(np.linalg.norm(a_last - a_front))
               if a_front is not None else np.inf)
    info = dict(d_front=d_front, drift=float(drift), s0=float(s[0]),
                s1=float(s[1]), consecutive=bool(consecutive), balance=balance,
                bond=(b0, b1))

    # 1. exact weak-bond pinned front (metastable front)
    if d_front < FRONT_TOL or (on_weak_bond and s[0] > 0.45):
        if drift < DRIFT_STAT:
            return "front", info

    # ring-advance over the trajectory tail (last 60%)
    k0 = max(1, int(0.4 * len(leads)))
    n_adv, mono = _ring_advances(leads[k0:], P)
    info["n_adv"] = n_adv; info["mono"] = mono

    # 2. cycle: many sustained forward advances, highly monotone
    if n_adv >= 6 and mono > 0.8:
        return "cycle", info

    # 3. stationary 2-pattern state -> memory (any non-weak bond) vs front (weak)
    if drift < DRIFT_STAT:
        if s[0] < 0.45:
            return "other", info          # no macroscopic retrieval
        if on_weak_bond:
            return "front", info
        if s[0] > MEM_DOM or (consecutive and s[0] > 0.5):
            return "memory", info         # robust recall on its own (strong) bond
        return "other", info

    # 4. non-stationary and not a clean cycle -> chaos
    if drift >= DRIFT_STAT:
        return "chaos", info

    return "other", info


def integrate_classify(sysP, a0, a_front, lam, t_total=400.0, dt=DT, rec_every=40,
                       r_front=None):
    """Integrate the reduced DDE from constant history a0 and classify."""
    sol = sysP.integrate(a0, t_total, dt, record_every=rec_every)
    label, info = classify(sol, sysP, lam, a_front, r_front=r_front)
    return label, info, sol


# =====================================================================
#  SAMPLING of random initial histories (constant histories a0 in R^P)
# =====================================================================
def sample_ic(rng, kind, scale=None):
    """Draw one constant-history initial overlap vector a0 in R^P.

    Three families (documented sampling measure):
      'near_pattern' : alpha*e_mu + noise, mu uniform, alpha in [0.6,1.0],
                       noise ~ N(0, sigma^2 I), sigma in [0.02,0.15].
      'mixture'      : sum of 2-4 CONSECUTIVE patterns with random positive
                       weights (normalized so leading overlap ~ [0.4,0.9]) + noise
                       -> covers the front / sequence-compatible region.
      'random_small' : broadly random small-norm overlap vector, all P comps
                       ~ N(0, s^2), s in [0.05, 0.35] -> covers the diffuse
                       multi-pattern / chaotic sector.
    """
    a0 = np.zeros(P)
    if kind == "near_pattern":
        mu = rng.integers(P)
        alpha = rng.uniform(0.6, 1.0)
        sigma = rng.uniform(0.02, 0.15)
        a0 = sigma * rng.standard_normal(P)
        a0[mu] += alpha
    elif kind == "mixture":
        nc = rng.integers(2, 5)
        mu0 = rng.integers(P)
        w = rng.uniform(0.3, 1.0, size=nc)
        w = w / w.sum() * rng.uniform(0.6, 1.2)
        for j in range(nc):
            a0[(mu0 + j) % P] += w[j]
        a0 += rng.uniform(0.02, 0.10) * rng.standard_normal(P)
    elif kind == "random_small":
        s = rng.uniform(0.05, 0.35)
        a0 = s * rng.standard_normal(P)
    else:
        raise ValueError(kind)
    return a0


IC_KINDS = ("near_pattern", "mixture", "random_small")


def sample_batch(rng, K, weights=(0.34, 0.33, 0.33)):
    """Draw K initial histories, mixing the three families by `weights`."""
    kinds = rng.choice(IC_KINDS, size=K, p=np.array(weights) / np.sum(weights))
    return [sample_ic(rng, k) for k in kinds], kinds
