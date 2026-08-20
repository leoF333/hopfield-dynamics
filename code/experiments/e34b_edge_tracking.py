"""
E34-ET0 edge-tracking gate at fixed lambda (N=400 pilot).

Conditional exploratory gate specified by R2.12 (rev. 3) of
ROADMAP_E34_snic_collier.md and the binding protocol
roadmaps/E34_ET0_PROTOCOL.md. For each sentinel bond mu at its local lambda
lam_local(mu) = lam_c(mu) - 0.020 (bisection lam_c read from the pilot E34
threshold table), this script:

  (a) identifies the two adjacent basins (node_mu and the neighbouring
      attractor) by constant-history integration and Gram-metric classification;
  (b) locates the basin-boundary object by bisection edge-tracking along the
      straight segment between the two nodes (constant history);
  (c) requires convergence under 2 time refinements (dt=0.01, 0.005) and 2
      distinct bracketing segments;
  (d) polishes the boundary object with Newton (residual < 1e-10) and certifies
      it as an equilibrium with EXACTLY one unstable characteristic root via the
      argument principle (count_unstable_reduced_spectrum / localize_roots),
      with rightmost-real-root and eigmax_M cross-checks;
  (e) continues the certified saddle in lambda in BOTH directions, counting
      turning points and checking branch identity, and tests whether it rejoins
      the local fold lam_c(mu);
  (f) shoots both branches of W^u with exponential_history + the corrected
      hist/dhist integrator API (eps in {1e-4, 1e-3}, both signs, dt cross-check)
      and classifies omega-limits against the census.

All arithmetic is float64 on CPU. No dense N x N coupling matrix is ever formed
(low-rank NumpyLowRankCouplings + P x P solves only). Distances use the physical
Gram metric d_G (R2.2). The unstable direction used for W^u shooting is the null
mode of the VARIATIONAL characteristic matrix T_red (using S.Dm), which shares
the roots of T_P (used for counting) but carries the correct flow eigenvector
(worklog V0 index subtlety).

CLI
---
--smoke        sentinel mu=14 only, shortened integrations (timing probe).
--production   full ET0 on --links (default 14,8,2) with the anomaly probes.
--links        comma-separated bonds (default "14,8,2").
--output-dir   defaults to results/2_cycle_rappel_snic/data/E34/.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from e34_lib import (
    NumpyLowRankCouplings,
    approach_memory_fold,
    characteristic_spectral_bound,
    continue_node_fold_saddle,
    count_unstable_reduced_spectrum,
    exponential_history,
    extract_null_mode,
    gram_distance,
    gram_matrix,
    localize_roots,
    make_iid_patterns,
    memory_branch_identity,
    reduced_field_coefficients,
    solve_memory_node,
)
from reduced_spectrum import GJ_GK, T_P, rightmost_real_root
from robust_branch import eigmax_M, woodbury_newton, woodbury_solve
from cycle_reduced import ReducedDDE


DEFAULT_OUTPUT_DIR = (
    REPO_DIR / "results" / "2_cycle_rappel_snic" / "data" / "E34")
DEFAULT_THRESHOLDS = (
    DEFAULT_OUTPUT_DIR / "E34_thresholds_N400_P20_s42_20260723T191104_587960.npz")

# Binding thresholds from roadmaps/E34_ET0_PROTOCOL.md.
LAM_OFFSET = 0.020            # lam_local = lam_c - LAM_OFFSET
BASIN_SEP_MIN = 0.05          # d_G(omega_A, omega_B) admissibility (a)
CLASSIFY_UNAMBIG = 0.05       # second-nearest node must be this far
NODE_TOL = 1e-3              # d_G to a node to call it converged (f)
STAT_TOL = 1e-7              # max ||a(t)-a_end||_G over last 2 tau -> stationary
BISECT_S_TOL = 1e-8          # bisection stopping width in s
DT_AGREE_S = 1e-3           # |s* (dt=.01) - s* (dt=.005)| tolerance (c)
BOUNDARY_AGREE = 1e-4        # d_G between boundary objects from refinements (c)
NEWTON_RES_TOL = 1e-10       # boundary object residual (d)
DEFLATION_ROUNDS_MAX = 2     # deflation retries when Newton lands on a node (d)
Z_U_CROSSCHECK = 1e-6        # |z_u_argument - z_u_realroot| tolerance (d)
FOLD_CROSSCHECK = 2e-3       # |fold_cont - lam_c(mu)| base tolerance (e)
LINEAR_GATE_TOL = 0.02       # relative error of W^u growth rate vs z_u (f)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# omega-limit classification helpers (Gram metric)

def _speed_gram(a_series: np.ndarray, dt_sample: float, Q: np.ndarray) -> np.ndarray:
    """Centered Gram-metric speed ||a'||_G along a decimated series."""
    if len(a_series) < 3:
        return np.zeros(len(a_series))
    diff = a_series[2:] - a_series[:-2]
    speed = np.array([
        float(gram_distance(d, np.zeros_like(d), Q)) for d in diff
    ]) / (2.0 * dt_sample)
    return np.concatenate(([speed[0]], speed, [speed[-1]]))


def integrate_to_settle(
    sysP: ReducedDDE,
    a_hist,
    Q: np.ndarray,
    *,
    dt: float,
    tau: float,
    t_chunk: float = 50.0,
    t_max: float = 600.0,
    record_every: int = 25,
    da_hist0=None,
):
    """Integrate a constant (or exponential) history in chunks until stationary.

    Returns dict(a, t, stationary, hist, dhist). Stationarity is measured over
    the last 2 tau window in the Gram metric.
    """
    ts = []
    traj = []
    elapsed = 0.0
    hist = a_hist
    dhist = da_hist0
    stationary = False
    window = 2.0 * tau
    while elapsed < t_max - 1e-9:
        step = min(t_chunk, t_max - elapsed)
        out = sysP.integrate(
            hist, step, dt, record_every=record_every, da_hist0=dhist)
        chunk_t = out["t"][1:] + elapsed
        chunk_a = out["a"][1:]
        ts.append(chunk_t)
        traj.append(chunk_a)
        elapsed += step
        hist = out["hist"]
        dhist = out["dhist"]
        a_all = np.concatenate(traj)
        t_all = np.concatenate(ts)
        final = a_all[-1]
        mask = t_all >= (elapsed - window)
        if np.count_nonzero(mask) >= 2:
            dev = max(
                float(gram_distance(a, final, Q)) for a in a_all[mask])
            if dev < STAT_TOL:
                stationary = True
                break
    a_all = np.concatenate(traj)
    t_all = np.concatenate(ts)
    return dict(
        a=a_all, t=t_all, stationary=stationary, hist=hist, dhist=dhist,
        elapsed=elapsed)


def classify_omega(final_a, a_nodes: dict, Q: np.ndarray):
    """Return (label, nearest_dist, second_dist). label = node index or 'other'."""
    if not a_nodes:
        return "other", np.inf, np.inf
    keys = list(a_nodes)
    dists = np.array([
        float(gram_distance(final_a, a_nodes[k], Q)) for k in keys])
    order = np.argsort(dists)
    nearest = keys[order[0]]
    d0 = float(dists[order[0]])
    d1 = float(dists[order[1]]) if len(dists) > 1 else np.inf
    if d0 < NODE_TOL and (d1 - d0) > CLASSIFY_UNAMBIG:
        return int(nearest), d0, d1
    if d0 < NODE_TOL:
        return int(nearest), d0, d1        # ambiguous but converged to a node
    return "other", d0, d1


# ---------------------------------------------------------------------------
# edge-tracking bisection

def _converges_to_A(
    sysP: ReducedDDE,
    a_seg,
    a_node_mu,
    Q: np.ndarray,
    *,
    dt: float,
    tau: float,
    t_max: float,
):
    """Binary side test: True if the constant-history run settles onto node_mu.

    The boundary of node_mu's basin is the stable manifold of the target saddle,
    so 'converged to node_mu' vs 'not' is the clean bisection discriminant. Also
    returns the recorded trajectory for dwell extraction.
    """
    out = integrate_to_settle(
        sysP, a_seg, Q, dt=dt, tau=tau, t_max=t_max,
        record_every=int(round(0.5 / dt)))
    final = out["a"][-1]
    d_to_A = float(gram_distance(final, a_node_mu, Q))
    is_A = bool(out["stationary"] and d_to_A < NODE_TOL)
    return is_A, out, d_to_A


TERMINAL_TOL = 0.02          # d_G plateau around the terminal attractor to skip


def _update_dwell(out, a_node_mu, Q, tau, best, record_dt):
    """Track the saddle dwell from an A-side (returns-to-node_mu) trajectory.

    Such a trajectory shadows the saddle's stable manifold (a deep transient
    speed dip near the saddle), then peels off along the unstable direction and
    finally settles back onto node_mu. Excluding only the terminal plateau near
    node_mu (not a fixed radius around every attractor, which would swallow a
    saddle that lies close to its node) isolates the transient saddle dwell as
    the minimum-speed point of the excursion.
    """
    a = out["a"]
    t = out["t"]
    speed = _speed_gram(a, record_dt, Q)
    d_to_mu = np.array([float(gram_distance(x, a_node_mu, Q)) for x in a])
    # last contiguous terminal plateau within TERMINAL_TOL of node_mu
    terminal_start = len(a)
    for i in range(len(a) - 1, -1, -1):
        if d_to_mu[i] < TERMINAL_TOL:
            terminal_start = i
        else:
            break
    lo = int(np.searchsorted(t, 2.0 * tau))
    window = range(lo, terminal_start)
    for j in window:
        if speed[j] < best[0]:
            best = (float(speed[j]), a[j].copy())
    return best


def refine_saddle_seed(
    sysP: ReducedDDE,
    dwell: np.ndarray,
    a_node_mu: np.ndarray,
    Q: np.ndarray,
    *,
    dt: float,
    tau: float,
    t_max: float,
    rounds: int = 3,
):
    """Local iterative edge-tracking restart to tighten the saddle seed.

    The straight-segment dwell lies on the saddle's stable manifold but, in high
    dimension (N=2000/P=100), generically too far from the saddle POINT for
    Newton to converge. This slides the seed closer: integrate the dwell to its
    closest approach, estimate the unstable direction from the subsequent
    divergence, bracket the dwell +/- eps along it, and re-bisect onto the
    boundary. A couple of rounds land inside the Newton basin. Returns the
    refined seed (a-coefficients). Never seeds from the arclength saddle.
    """
    record_every = int(round(0.5 / dt))

    def converges_A(ic):
        out = integrate_to_settle(
            sysP, ic, Q, dt=dt, tau=tau, t_max=t_max, record_every=record_every)
        is_A = bool(out["stationary"]
                    and float(gram_distance(out["a"][-1], a_node_mu, Q)) < NODE_TOL)
        return is_A, out

    cur = np.asarray(dwell, float).copy()
    for _ in range(rounds):
        out = sysP.integrate(cur, 60.0, dt, record_every=5)
        aa = out["a"]
        tt = out["t"]
        sp = _speed_gram(aa, dt * 5, Q)
        d2mu = np.array([float(gram_distance(x, a_node_mu, Q)) for x in aa])
        term = len(aa)
        for i in range(len(aa) - 1, -1, -1):
            if d2mu[i] < TERMINAL_TOL:
                term = i
            else:
                break
        lo_i = int(np.searchsorted(tt, 2.0 * tau))
        win = list(range(lo_i, max(term, lo_i + 1)))
        if not win:
            break
        j = min(win, key=lambda i: sp[i])
        center = aa[j]
        k = min(j + 6, len(aa) - 1)
        udir = aa[k] - aa[j]
        nrm = np.sqrt(max(float(udir @ Q @ udir), 0.0))
        if nrm < 1e-14:
            break
        udir = udir / nrm
        straddled = False
        eps = 0.02
        is_A_m = False
        for eps in (0.02, 0.05, 0.1, 0.2):
            ap, _ = converges_A(center + eps * udir)
            am, _ = converges_A(center - eps * udir)
            if ap != am:
                straddled = True
                is_A_m = am
                break
        if not straddled:
            break
        loP = center + (eps * udir if is_A_m else -eps * udir)   # A side
        hiP = center + (-eps * udir if is_A_m else eps * udir)
        best = (np.inf, None)
        for _ in range(28):
            mid = 0.5 * (loP + hiP)
            is_A, out2 = converges_A(mid)
            best = _update_dwell(out2, a_node_mu, Q, tau, best, 0.5)
            if is_A:
                loP = mid
            else:
                hiP = mid
        if best[1] is None:
            break
        cur = best[1]
    return cur


def edge_track_segment(
    sysP: ReducedDDE,
    a_node_mu: np.ndarray,
    a_far: np.ndarray,
    Q: np.ndarray,
    a_nodes: dict,
    *,
    dt: float,
    tau: float,
    t_max: float,
):
    """Bisect s on a(s) = (1-s) node_mu + s far between basin-A and not-A.

    Returns dict with s_star, dwell state, bisection history and endpoint labels.
    """
    a0 = a_node_mu
    a1 = a_far

    def seg(s):
        return (1.0 - s) * a0 + s * a1

    lo_is_A, _, _ = _converges_to_A(
        sysP, seg(0.0), a_node_mu, Q, dt=dt, tau=tau, t_max=t_max)
    hi_is_A, _, _ = _converges_to_A(
        sysP, seg(1.0), a_node_mu, Q, dt=dt, tau=tau, t_max=t_max)
    if not lo_is_A or hi_is_A:
        return dict(
            ok=False,
            reason=f"bracket_invalid(lo_A={lo_is_A}, hi_A={hi_is_A})",
            s_lo=0.0, s_hi=1.0, s_star=np.nan,
            history=np.empty((0, 2)), dwell=None, dwell_speed=np.nan,
        )
    s_lo, s_hi = 0.0, 1.0
    history = [(0.0, 1.0), (1.0, 0.0)]
    best = (np.inf, None)
    record_dt = 0.5
    while (s_hi - s_lo) > BISECT_S_TOL:
        s_mid = 0.5 * (s_lo + s_hi)
        is_A, out, _ = _converges_to_A(
            sysP, seg(s_mid), a_node_mu, Q, dt=dt, tau=tau, t_max=t_max)
        history.append((s_mid, 1.0 if is_A else 0.0))
        if is_A:
            # A-side runs return to node_mu: clean terminal plateau for dwell.
            best = _update_dwell(out, a_node_mu, Q, tau, best, record_dt)
            s_lo = s_mid
        else:
            s_hi = s_mid
    return dict(
        ok=True,
        reason="converged",
        s_lo=s_lo, s_hi=s_hi, s_star=0.5 * (s_lo + s_hi),
        history=np.asarray(history), dwell=best[1],
        dwell_speed=best[0],
    )


def deflated_newton(coup, u0, lam, beta, deflate_us, *, power=2.0, shift=1.0,
                    tol=1e-12, max_iter=160, floor=1e-8):
    """Newton on the deflated field M(u) F(u), M = prod_k (d_k^-power + shift).

    Near a fold the saddle sits close to its node, and ordinary Newton from an
    edge-tracking dwell converges to the (attracting) node instead. Deflating
    the already-found equilibria removes them as roots, pushing Newton onto the
    remaining one. M is scalar, so the deflated step is the ordinary Woodbury
    step rescaled: with y = J^-1 F and w = grad log M, delta = -y / (1 + w.y),
    i.e. delta = du / (1 - w.du) with du = -y the undeflated step. Distances are
    scaled by sqrt(N) to match the residual convention.
    """
    N = coup.N
    root = np.sqrt(N)
    pts = [np.asarray(x, float) for x in deflate_us]
    fnorm = lambda uu: float(np.linalg.norm(coup.field_F(uu, lam, beta)) / root)

    def log_grad_M(u):
        w = np.zeros_like(u)
        for uk in pts:
            diff = u - uk
            d = max(float(np.linalg.norm(diff)) / root, floor)
            mk = d ** (-power) + shift
            w += (-power * d ** (-power - 2.0)) * diff / (N * mk)
        return w

    def merit(u):
        m = 1.0
        for uk in pts:
            d = max(float(np.linalg.norm(u - uk)) / root, floor)
            m *= d ** (-power) + shift
        return m * fnorm(u)

    u = np.asarray(u0, float).copy()
    r = fnorm(u)
    g = merit(u)
    for _ in range(max_iter):
        if r < tol:
            return u, True
        F = coup.field_F(u, lam, beta)
        du = woodbury_solve(coup, coup.compute_gain(u, beta), lam, -F)
        denom = 1.0 - float(log_grad_M(u) @ du)
        step = du / denom if abs(denom) > 1e-10 else du
        st, improved = 1.0, False
        for _ in range(40):
            trial = u + st * step
            gt = merit(trial)
            if np.isfinite(gt) and gt < g:
                u, g, r, improved = trial, gt, fnorm(trial), True
                break
            st *= 0.5
        if not improved:
            break
    return u, bool(r < tol)


# ---------------------------------------------------------------------------
# spectral certification (d)

def certify_equilibrium(coup, u, beta, lam, tau, t0, Q, *,
                        samples_per_edge=16, max_refinements=7):
    """Argument-principle unstable count + real-root and eigmax_M cross-checks."""
    GJ, GK = GJ_GK(coup, u, beta, lam)
    rho = characteristic_spectral_bound(GJ, GK, lam)
    eta = max(0.05, 0.02 * rho / t0)
    # P=100 needs a finer contour than P=20: with (16, 7) the winding count does
    # not stabilise at N=2000 (measured), while (32, 9) does. Callers escalate.
    base = count_unstable_reduced_spectrum(
        GJ, GK, t0, tau, lam,
        samples_per_edge=samples_per_edge, max_refinements=max_refinements)
    shifted = count_unstable_reduced_spectrum(
        GJ, GK, t0, tau, lam, eta=1.5 * eta,
        samples_per_edge=samples_per_edge, max_refinements=max_refinements)
    contour_stable = bool(
        base.converged and shifted.converged
        and base.count == shifted.count)
    result = dict(
        n_unstable=base.count if base.count is not None else -1,
        contour_stable=contour_stable,
        base_reason=base.reason,
        shifted_reason=shifted.reason,
        eigmax=float(eigmax_M(coup, u, lam, beta)),
        rho=float(rho),
        z_u=np.nan, z_u_imag=np.nan, root_residual=np.nan,
        mode=None, mode_residual=np.nan, mode_realif=np.nan,
        real_root=np.nan, z_u_crosscheck_ok=False, localize_ok=False,
    )
    rr = rightmost_real_root(GJ, GK, t0, tau, lam, x_hi=rho / t0, x_lo=-0.5, n=400)
    result["real_root"] = float(rr.real) if rr is not None else np.nan
    if not contour_stable or base.count != 1:
        return result, GJ, GK

    ident = np.eye(GJ.shape[0], dtype=complex)

    def char_tp(z):
        return T_P(z, GJ, GK, t0, tau, lam)

    def deriv_tp(z):
        return t0 * ident + lam * tau * np.exp(-z * tau) * GK

    loc = localize_roots(
        char_tp, base.rectangle, derivative=deriv_tp, parent_result=base,
        target_width=5e-4, target_height=5e-4,
        argument_kwargs=dict(samples_per_edge=12, max_refinements=7),
        polish_residual_tol=1e-9)
    result["localize_ok"] = bool(loc.converged and len(loc.roots) == 1)
    if not result["localize_ok"]:
        return result, GJ, GK
    root = complex(loc.roots[0].value)
    result["z_u"] = float(root.real)
    result["z_u_imag"] = float(root.imag)
    result["root_residual"] = float(loc.roots[0].residual)
    result["z_u_crosscheck_ok"] = bool(
        rr is not None and abs(root.real - rr.real) < Z_U_CROSSCHECK)

    # Flow eigenvector: null mode of the VARIATIONAL T_red (uses S.Dm).
    P = GJ.shape[0]
    S = np.roll(np.eye(P), 1, axis=0)

    def char_red(z):
        return (t0 * z + 1.0) * np.eye(P) - (1.0 - lam) * GJ \
            - lam * np.exp(-z * tau) * (S @ GJ)

    mode = extract_null_mode(char_red, root.real, Q=Q, expect_real=True)
    result["mode"] = np.asarray(mode.vector, float)
    result["mode_residual"] = float(mode.residual)
    result["mode_realif"] = float(mode.realification_error)
    result["mode_ok"] = bool(mode.converged)
    return result, GJ, GK


# ---------------------------------------------------------------------------
# saddle continuation in lambda, both directions (e)


def _count_turns(parameters, tol=1e-8):
    diffs = np.diff(np.asarray(parameters, float))
    signs = np.sign(diffs[np.abs(diffs) > tol])
    if not len(signs):
        return 0
    compressed = signs[np.r_[True, signs[1:] != signs[:-1]]]
    return int(np.sum(compressed[1:] != compressed[:-1]))


def continue_saddle_down(coup, mu, beta, saddle, lam0, lam_stop):
    """Warm-start Newton walk of the saddle to decreasing lambda (identity +
    eigmax_M>0 guard). Returns (lams, turns, identity_ok, reached)."""
    Q = gram_matrix(coup._xi_np)
    u = np.asarray(saddle, float).copy()
    lam = float(lam0)
    lams = [lam]
    step = 2e-3
    identity_ok = True
    while lam > lam_stop + 1e-12:
        trial = max(lam - step, lam_stop)
        cand, ok = woodbury_newton(coup, u, trial, beta, tol=1e-11, max_iter=80)
        res = float(np.linalg.norm(coup.field_F(cand, trial, beta)) / np.sqrt(coup.N))
        ident = memory_branch_identity(coup, cand, mu, Q=Q)
        eig = eigmax_M(coup, cand, trial, beta)
        if ok and res < 1e-9 and ident and eig > 0:
            u = cand
            lam = trial
            lams.append(lam)
        else:
            step *= 0.5
            if step < 1e-5:
                identity_ok = identity_ok and ident
                break
    reached = lam <= lam_stop + 1e-6
    return np.asarray(lams), _count_turns(lams), bool(identity_ok), bool(reached)


# ---------------------------------------------------------------------------
# W^u shooting (f)


def linear_gate(sysP, a_star, z_u, v, Q, tau, dt=0.01, eps=1e-4):
    """Return relative error between the fitted early growth rate and z_u."""
    _, hist, dhist = exponential_history(a_star, z_u, v, eps, tau, dt, Q=Q)
    out = sysP.integrate(hist, 1.5, dt, record_every=1, da_hist0=dhist)
    dd = np.array([float(gram_distance(a, a_star, Q)) for a in out["a"]])
    tt = out["t"]
    mask = (tt >= 0.3) & (tt <= 1.5) & (dd > 0)
    if np.count_nonzero(mask) < 5:
        return np.inf
    slope = np.polyfit(tt[mask], np.log(dd[mask]), 1)[0]
    return abs(slope - z_u) / abs(z_u)


def shoot_branch(
    sysP, a_star, z_u, v, sign, eps, Q, a_nodes, *,
    tau, dt, t_max, t_chunk=50.0,
):
    """Integrate one W^u branch and classify its omega-limit."""
    _, hist, dhist = exponential_history(
        a_star, z_u, v, sign * eps, tau, dt, Q=Q)
    out = integrate_to_settle(
        sysP, hist, Q, dt=dt, tau=tau, t_chunk=t_chunk, t_max=t_max,
        record_every=int(round(1.0 / dt)), da_hist0=dhist)
    final = out["a"][-1]
    label, d0, d1 = classify_omega(final, a_nodes, Q)
    if out["stationary"] and isinstance(label, int):
        omega = label
    elif out["stationary"]:
        omega = -4                     # settled on a non-census fixed point
    else:
        # not stationary at t_max: distinguish transit/cycle from escape/chaos
        dmin = min(
            float(gram_distance(final, a_nodes[k], Q)) for k in a_nodes
        ) if a_nodes else np.inf
        omega = -3 if dmin < 0.2 else -1
    return dict(
        omega=omega, nearest_dist=d0, second_dist=d1,
        stationary=bool(out["stationary"]), elapsed=float(out["elapsed"]),
        final=final,
    )


# ---------------------------------------------------------------------------
# per-sentinel driver

def solve_alive_nodes(coup, lam, beta, lam_c, margin=3e-3):
    """Solve all census nodes alive at lambda (lam_c[k] > lam + margin)."""
    nodes = {}
    for k in range(coup.P):
        if np.isfinite(lam_c[k]) and lam_c[k] > lam + margin:
            u, ok, res = solve_memory_node(coup, k, lam, beta)
            if ok and res < 1e-9:
                Q = gram_matrix(coup._xi_np)
                nodes[k] = reduced_field_coefficients(coup, u, Q)
    return nodes


def run_sentinel(coup, Q, mu, lam_c_mu, bracket_w, beta, tau, t0, cfg, log,
                 lam_override=None):
    # lam_override lets ET0' bracket mu at a lower lambda (where a neighbour is
    # alive) while still cross-checking the continuation against its OWN fold
    # lam_c_mu. The fold reached upward is lam_c_mu regardless of lam_override.
    lam = float(lam_override) if lam_override is not None else float(lam_c_mu - LAM_OFFSET)
    rec = dict(mu=int(mu), lam_local=lam, lam_c_table=float(lam_c_mu),
               status="running")
    log(f"[mu={mu}] lam_local={lam:.6f} lam_c={lam_c_mu:.6f}")
    sysP = ReducedDDE(coup._xi_np, beta, lam, tau, t0)
    t_max_settle = cfg["t_settle_smoke"] if cfg["smoke"] else 800.0

    # -- (a) node_mu, adjacent basins --
    node_u, ok, res = solve_memory_node(coup, mu, lam, beta)
    if not ok:
        rec.update(status="node_failed", node_residual=res)
        return rec
    a_node_mu = reduced_field_coefficients(coup, node_u, Q)
    rec["node_residual"] = float(res)
    rec["node_eigmax"] = float(eigmax_M(coup, node_u, lam, beta))
    rec["a_node"] = a_node_mu
    rec["m_node"] = coup.overlap_raw(np.tanh(beta * node_u))

    a_nodes = solve_alive_nodes(coup, lam, beta, cfg["lam_c"])
    a_nodes[int(mu)] = a_node_mu
    rec["alive_nodes"] = sorted(int(k) for k in a_nodes)
    log(f"[mu={mu}] alive nodes at lam_local: {rec['alive_nodes']}")

    succ = (mu + 1) % coup.P
    pred = (mu - 1) % coup.P
    if cfg.get("neighbor_rule"):
        # ET1': bracket against a LIVE adjacent node (successor preferred). If
        # neither neighbour is alive, the cell is inadmissible (expected near
        # lambda*) and is recorded as data, not forced.
        far_idx = succ if succ in a_nodes else (pred if pred in a_nodes else None)
        if far_idx is None:
            rec.update(far_source="no_alive_neighbor", basin_admissible=False,
                       basinB_label="none", basin_separation=np.nan,
                       status="inadmissible_no_neighbor")
            log(f"[mu={mu}] no adjacent node alive at lam={lam:.4f} "
                f"-> inadmissible (expected near lambda*)")
            return rec
        a_far = a_nodes[far_idx]
        far_source = f"node_{far_idx}"
        ref_idx = far_idx
    else:
        far_idx = succ if succ in a_nodes else None
        if succ in a_nodes:
            a_far = a_nodes[succ]
            far_source = f"node_{succ}"
        else:
            a_far = reduced_field_coefficients(coup, 2.0 * coup._xi_np[succ], Q)
            far_source = f"probe_2xi_{succ}"
        ref_idx = succ
    rec["far_source"] = far_source
    rec["far_idx"] = int(far_idx) if far_idx is not None else -1
    rec["ref_idx"] = int(ref_idx)

    # classify the far endpoint's own omega-limit -> attractor B
    outB = integrate_to_settle(
        sysP, a_far, Q, dt=0.01, tau=tau, t_max=t_max_settle,
        record_every=50)
    labelB, dB0, dB1 = classify_omega(outB["a"][-1], a_nodes, Q)
    d_AB = float(gram_distance(a_node_mu, outB["a"][-1], Q))
    rec.update(basinB_label=str(labelB), basinB_dist=dB0,
               basin_separation=d_AB,
               basinB_stationary=bool(outB["stationary"]))
    admissible = bool(d_AB > BASIN_SEP_MIN and (labelB != int(mu)))
    rec["basin_admissible"] = admissible
    log(f"[mu={mu}] basin B={labelB} sep_d_G={d_AB:.4f} admissible={admissible}")
    if not admissible:
        rec["status"] = "basin_ambiguous"
        return rec

    # -- (b,c) edge-tracking: 2 dt x 2 segments --
    # segment 2: node_mu -> perturbed point pulled toward B until it crosses
    direction = a_far - a_node_mu
    a_far2 = a_node_mu + 1.4 * direction
    two_segments = cfg.get("et1_two_segments", True)
    segments = {"S1": a_far, "S2": a_far2} if two_segments else {"S1": a_far}
    dts = cfg.get("et1_dts", None)
    if dts is None:
        dts = [0.01] if cfg["smoke"] else [0.01, 0.005]
    et_results = {}
    for seg_name, seg_far in segments.items():
        for dt in dts:
            key = f"{seg_name}_dt{dt}"
            t0w = time.perf_counter()
            et = edge_track_segment(
                sysP, a_node_mu, seg_far, Q, a_nodes, dt=dt, tau=tau,
                t_max=t_max_settle)
            et["wall"] = time.perf_counter() - t0w
            et_results[key] = et
            log(f"[mu={mu}] edge {key}: ok={et['ok']} s*={et['s_star']:.3e} "
                f"reason={et['reason']} ({et['wall']:.1f}s)")
    rec["edge_results"] = {
        k: dict(ok=v["ok"], s_star=float(v["s_star"]), reason=v["reason"],
                dwell_speed=float(v["dwell_speed"]))
        for k, v in et_results.items()
    }
    rec["edge_history_S1_dt0.01"] = et_results["S1_dt0.01"]["history"]

    primary = et_results["S1_dt0.01"]
    if not primary["ok"] or primary["dwell"] is None:
        rec["status"] = "edge_track_failed"
        return rec

    # -- (d) polish + certify boundary object --
    from robust_branch import anneal_newton

    def _polish(seed_a):
        ue = coup._xi_np.T @ seed_a
        us, ok = woodbury_newton(coup, ue, lam, beta, tol=1e-12, max_iter=160)
        rr = float(np.linalg.norm(coup.field_F(us, lam, beta)) / np.sqrt(coup.N))
        return us, rr, ok

    seed = primary["dwell"]
    u_star, saddle_res, pok = _polish(seed)
    n_refine = 0
    # In high dimension the straight-segment dwell may sit outside the Newton
    # basin; tighten it by local iterative edge-tracking (never from arclength).
    if not (pok and saddle_res < NEWTON_RES_TOL):
        seed = refine_saddle_seed(
            sysP, primary["dwell"], a_node_mu, Q,
            dt=0.01, tau=tau, t_max=t_max_settle, rounds=3)
        u_star, saddle_res, pok = _polish(seed)
        n_refine = 1
        if not (pok and saddle_res < NEWTON_RES_TOL):
            u_star2, ok2 = anneal_newton(
                coup, coup._xi_np.T @ seed, lam, beta, tol=1e-12, max_iter=160)
            r2 = float(np.linalg.norm(coup.field_F(u_star2, lam, beta)) / np.sqrt(coup.N))
            if r2 < saddle_res:
                u_star, saddle_res, pok = u_star2, r2, ok2
            n_refine = 2
    ident = memory_branch_identity(coup, u_star, mu, Q=Q)
    a_star = reduced_field_coefficients(coup, u_star, Q)
    rec.update(saddle_residual=saddle_res, saddle_identity=bool(ident),
               n_refine_rounds=int(n_refine), a_saddle=a_star,
               m_saddle=coup.overlap_raw(np.tanh(beta * u_star)))
    log(f"[mu={mu}] boundary Newton res={saddle_res:.2e} identity={ident} "
        f"(refine={n_refine})")
    if not (pok and saddle_res < NEWTON_RES_TOL and ident):
        rec["status"] = "boundary_polish_failed"
        return rec

    cert, GJ, GK = certify_equilibrium(coup, u_star, beta, lam, tau, t0, Q)

    # Deflation, triggered by a threshold-free criterion: an index-0 polished
    # object IS a node (near the fold the saddle is close to its node and the
    # attracting one wins Newton). Deflate what was found and re-solve from the
    # same edge seed; never seed from the arclength saddle (independence).
    n_deflate = 0
    deflated_pts = []
    while cert["n_unstable"] == 0 and n_deflate < DEFLATION_ROUNDS_MAX:
        deflated_pts.append(u_star.copy())
        if not any(np.allclose(node_u, p) for p in deflated_pts):
            deflated_pts.append(node_u.copy())
        u_d, ok_d = deflated_newton(
            coup, coup._xi_np.T @ seed, lam, beta, deflated_pts, tol=1e-12)
        r_d = float(np.linalg.norm(coup.field_F(u_d, lam, beta)) / np.sqrt(coup.N))
        n_deflate += 1
        log(f"[mu={mu}] deflation round {n_deflate}: res={r_d:.2e} ok={ok_d} "
            f"(deflated {len(deflated_pts)} roots)")
        if not (ok_d and r_d < NEWTON_RES_TOL):
            break
        u_star, saddle_res, pok = u_d, r_d, True
        ident = memory_branch_identity(coup, u_star, mu, Q=Q)
        a_star = reduced_field_coefficients(coup, u_star, Q)
        cert, GJ, GK = certify_equilibrium(coup, u_star, beta, lam, tau, t0, Q)
    if n_deflate:
        rec.update(saddle_residual=saddle_res, saddle_identity=bool(ident),
                   a_saddle=a_star,
                   m_saddle=coup.overlap_raw(np.tanh(beta * u_star)))
    rec["n_deflation_rounds"] = int(n_deflate)

    rec.update(
        n_unstable_saddle=int(cert["n_unstable"]),
        contour_stable=bool(cert["contour_stable"]),
        saddle_eigmax=float(cert["eigmax"]),
        z_u=float(cert["z_u"]),
        z_u_imag=float(cert["z_u_imag"]),
        root_residual=float(cert["root_residual"]),
        real_root=float(cert["real_root"]),
        z_u_crosscheck_ok=bool(cert["z_u_crosscheck_ok"]),
        localize_ok=bool(cert["localize_ok"]),
        mode_residual=float(cert["mode_residual"]),
    )
    log(f"[mu={mu}] n_unstable={cert['n_unstable']} z_u={cert['z_u']:.5f} "
        f"eigmax={cert['eigmax']:+.4f} real_root={cert['real_root']:.5f} "
        f"crosscheck={cert['z_u_crosscheck_ok']}")
    index_one = bool(
        cert["contour_stable"] and cert["n_unstable"] == 1
        and cert["localize_ok"] and abs(cert["z_u_imag"]) < 1e-8
        and cert["eigmax"] > 0 and cert["z_u_crosscheck_ok"])
    rec["index_one_certified"] = index_one

    # node re-certification
    node_cert, _, _ = certify_equilibrium(coup, node_u, beta, lam, tau, t0, Q)
    rec["n_unstable_node"] = int(node_cert["n_unstable"])

    # consistency: edge boundary object vs arclength-through-fold saddle
    near_node, near_lam, _ = approach_memory_fold(
        coup, mu, beta, node_u, lam, lam_limit=float(lam_c_mu - 4e-3),
        eig_stop=1e9)
    pair = continue_node_fold_saddle(
        coup, mu, beta, node_u, lam, near_node, near_lam)
    rec["arclength_saddle_ok"] = bool(pair.converged and pair.saddle is not None)
    if pair.converged and pair.saddle is not None:
        a_arc = reduced_field_coefficients(coup, pair.saddle, Q)
        rec["edge_vs_arclength_dG"] = float(gram_distance(a_star, a_arc, Q))
    else:
        rec["edge_vs_arclength_dG"] = np.nan

    if not index_one:
        rec["status"] = "index_certification_failed"
        return rec

    # -- convergence check (c): agreement across dt and segments --
    s_ref = primary["s_star"]
    dt_ok = True
    if not cfg["smoke"] and et_results.get("S1_dt0.005", {}).get("ok"):
        dt_ok = abs(et_results["S1_dt0.005"]["s_star"] - s_ref) < DT_AGREE_S
    seg_ok = True
    boundary_agree = np.nan
    s2 = et_results.get("S2_dt0.01")
    if s2 and s2["ok"] and s2["dwell"] is not None:
        u_s2, r_s2, ok_s2 = _polish(s2["dwell"])
        if not (ok_s2 and r_s2 < NEWTON_RES_TOL):
            seed2 = refine_saddle_seed(
                sysP, s2["dwell"], a_node_mu, Q,
                dt=0.01, tau=tau, t_max=t_max_settle, rounds=3)
            u_s2, r_s2, ok_s2 = _polish(seed2)
        a_s2 = reduced_field_coefficients(coup, u_s2, Q)
        boundary_agree = float(gram_distance(a_star, a_s2, Q))
        seg_ok = bool(r_s2 < NEWTON_RES_TOL and boundary_agree < BOUNDARY_AGREE)
    rec.update(dt_refine_ok=bool(dt_ok), segment_agree_ok=bool(seg_ok),
               boundary_agree_dG=float(boundary_agree))
    edge_converged = bool(dt_ok and seg_ok)
    rec["edge_converged"] = edge_converged

    # -- (e) continuation both directions --
    fold_cont = float(pair.fold_parameter) if pair.converged else np.nan
    turns_up = _count_turns(pair.trace.parameters) if pair.converged else -1
    fold_tol = max(FOLD_CROSSCHECK, 2.5 * bracket_w) if np.isfinite(bracket_w) \
        else FOLD_CROSSCHECK
    fold_mismatch = abs(fold_cont - lam_c_mu) if np.isfinite(fold_cont) else np.nan
    rejoins_fold = bool(
        pair.converged and turns_up == 1
        and np.isfinite(fold_mismatch) and fold_mismatch <= fold_tol)
    lams_down, turns_down, ident_down, reached_down = continue_saddle_down(
        coup, mu, beta, u_star, lam, lam - 0.02)
    rec.update(
        fold_cont=fold_cont, fold_mismatch=float(fold_mismatch),
        fold_tol=float(fold_tol), turns_up=int(turns_up),
        rejoins_fold=rejoins_fold, turns_down=int(turns_down),
        identity_down_ok=bool(ident_down), reached_down=bool(reached_down),
        lam_down_min=float(lams_down.min()),
    )
    identity_ok = bool(rejoins_fold and ident_down and turns_down == 0)
    rec["continuation_identity_ok"] = identity_ok
    log(f"[mu={mu}] fold_cont={fold_cont:.6f} mismatch={fold_mismatch:.2e} "
        f"turns_up={turns_up} turns_down={turns_down} rejoins={rejoins_fold}")

    # -- (f) W^u shooting --
    v = np.asarray(cert["mode"], float)
    if v[ref_idx] < 0:              # "+" branch points toward the bracket node
        v = -v
    lg = linear_gate(sysP, a_star, cert["z_u"], v, Q, tau)
    rec["linear_gate_relerr"] = float(lg)
    log(f"[mu={mu}] W^u linear gate rel_err={lg:.4f}")

    t_max_wu = cfg["t_wu_smoke"] if cfg["smoke"] else (
        20000.0 if lam >= 0.31 else 4000.0)
    eps_list = [1e-3] if cfg["smoke"] else [1e-4, 1e-3]
    dt_list_wu = [0.01] if cfg["smoke"] else [0.01, 0.005]
    branch_out = {}
    for sign, name in ((-1.0, "minus"), (1.0, "plus")):
        settings = {}
        for eps in eps_list:
            for dtw in dt_list_wu:
                b = shoot_branch(
                    sysP, a_star, cert["z_u"], v, sign, eps, Q, a_nodes,
                    tau=tau, dt=dtw, t_max=t_max_wu)
                settings[f"eps{eps}_dt{dtw}"] = b["omega"]
                log(f"[mu={mu}] W^u {name} eps={eps} dt={dtw} -> omega={b['omega']}")
        omegas = set(settings.values())
        robust = (len(omegas) == 1)
        branch_out[name] = dict(
            omega=(settings[f"eps{eps_list[0]}_dt{dt_list_wu[0]}"]),
            robust=robust, settings=settings)
    rec["wu_minus"] = branch_out["minus"]
    rec["wu_plus"] = branch_out["plus"]
    wu_robust = bool(branch_out["minus"]["robust"] and branch_out["plus"]["robust"])
    landed_on_nodes = bool(
        isinstance(branch_out["minus"]["omega"], int) and branch_out["minus"]["omega"] >= 0
        and isinstance(branch_out["plus"]["omega"], int) and branch_out["plus"]["omega"] >= 0)
    rec["wu_robust"] = wu_robust
    rec["wu_landed_on_nodes"] = landed_on_nodes
    wu_adjacent = bool(
        landed_on_nodes
        and {branch_out["minus"]["omega"], branch_out["plus"]["omega"]}
        == {mu % coup.P, ref_idx})
    rec["wu_adjacent"] = wu_adjacent
    rec["linear_gate_ok"] = bool(lg < LINEAR_GATE_TOL)
    # GATE-DYN (fixed-lambda dynamical topology) vs GATE-BRANCH (branch geometry)
    rec["gate_dyn"] = bool(
        admissible and edge_converged and index_one
        and rec["linear_gate_ok"] and wu_robust and wu_adjacent
        and (rec.get("edge_vs_arclength_dG", 1.0) < 1e-4))
    rec["gate_branch"] = bool(identity_ok)

    # -- verdict (g) --
    positive = bool(
        admissible and edge_converged and index_one
        and rec["linear_gate_ok"] and identity_ok
        and wu_robust and landed_on_nodes)
    if positive:
        verdict = "positive"
    elif index_one and admissible and edge_converged and (
            (not identity_ok) or (not landed_on_nodes and wu_robust)):
        verdict = "limited"
    elif not index_one:
        verdict = "negative"
    else:
        verdict = "indeterminate"
    rec["verdict"] = verdict
    rec["status"] = "done"
    log(f"[mu={mu}] VERDICT = {verdict}")
    return rec


# ---------------------------------------------------------------------------
# anomaly probes (secondary)

def probe_attractors(coup, mu, lam, beta, tau, t0, a_nodes, log, n_ic=6):
    """Exploratory: which attractors does the flow provide near node_mu at lam.

    Returns the list of distinct settled attractors (census node index or a
    machine label) reached from diffuse/perturbed initial histories. Labelled
    exploratory; used only for the ET0' secondary bracket note.
    """
    Q = gram_matrix(coup._xi_np)
    sysP = ReducedDDE(coup._xi_np, beta, lam, tau, t0)
    u_node, ok, _ = solve_memory_node(coup, mu, lam, beta)
    a_node = reduced_field_coefficients(coup, u_node, Q) if ok else None
    rng = np.random.default_rng(0)
    found = []
    for i in range(n_ic):
        if a_node is not None:
            a0 = a_node + 0.6 * rng.standard_normal(coup.P)
        else:
            a0 = 0.5 * rng.standard_normal(coup.P)
        out = integrate_to_settle(
            sysP, a0, Q, dt=0.01, tau=tau, t_max=600.0, record_every=50)
        label, d0, d1 = classify_omega(out["a"][-1], a_nodes, Q)
        found.append(dict(label=str(label), stationary=bool(out["stationary"]),
                          d_to_node_mu=float(gram_distance(out["a"][-1], a_node, Q))
                          if a_node is not None else None))
    labels = sorted(set(f["label"] for f in found))
    distinct_non_mu = [l for l in labels if l != str(mu)]
    log(f"[probe mu={mu} lam={lam:.4f}] labels reached: {labels}; "
        f"distinct non-node_{mu}: {distinct_non_mu}")
    return dict(labels=labels, distinct_non_mu=distinct_non_mu, detail=found)


def anomaly_mu17(coup, beta, tau, t0, log):
    """Retry mu=17 fold with tighter arclength (roadmap section 9)."""
    from e34_threshold_table import trace_threshold
    log("[anomaly mu=17] retry with lam_start seed near fold, tight arclength")
    out = dict(mu=17)
    try:
        res = trace_threshold(
            coup, 17, beta=beta, lam_start=0.05, lam_limit=0.8,
            step_initial=0.01, bisect_tol=1e-5)
        out.update(status=str(res["threshold_status"]),
                   lam_c=float(res["lam_c"]) if np.isfinite(res["lam_c"]) else None,
                   detail=str(res["fail_detail"]))
    except Exception as exc:                              # noqa: BLE001
        out.update(status="exception", detail=repr(exc))
    log(f"[anomaly mu=17] status={out.get('status')} lam_c={out.get('lam_c')}")
    return out


def anomaly_mu3(coup, beta, tau, t0, log):
    """Count turning points on mu=3 component (table 0.2429 vs census 0.2622)."""
    log("[anomaly mu=3] counting turning points on the branch component")
    out = dict(mu=3)
    Q = gram_matrix(coup._xi_np)
    lam_target = 0.2429096 - 0.02
    node_u, ok, res = solve_memory_node(coup, 3, lam_target, beta)
    if not ok:
        out.update(status="node_failed")
        return out
    near_node, near_lam, _ = approach_memory_fold(
        coup, 3, beta, node_u, lam_target, lam_limit=0.2429096 - 4e-3,
        eig_stop=1e9)
    pair = continue_node_fold_saddle(
        coup, 3, beta, node_u, lam_target, near_node, near_lam)
    turns = _count_turns(pair.trace.parameters)
    out.update(status="multifold_indeterminate" if turns != 1 else "single_fold",
               turning_count=int(turns),
               fold_cont=float(pair.fold_parameter),
               fold_table=0.2429096,
               fold_mismatch=abs(float(pair.fold_parameter) - 0.2429096))
    log(f"[anomaly mu=3] turns={turns} fold_cont={pair.fold_parameter:.6f}")
    return out


# ---------------------------------------------------------------------------
# main

def _et1_cell(payload):
    """Worker: run the full GATE-DYN battery for one ET1' (mu, delta) cell.

    Rebuilds the low-rank couplings inside the worker (cheap, avoids pickling)
    and returns (record, log_lines). OMP_NUM_THREADS is pinned to 1 at import.
    """
    (N, P, seed, mu, delta, beta, tau, t0, lam_c_arr, lam_c_mu, bw,
     do_branch, dts, segments_two) = payload
    xi, xis = make_iid_patterns(N, P, seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)
    logs = []

    def log(msg):
        logs.append(f"{_utc_now()} {msg}")

    cfg = dict(
        smoke=False, lam_c=np.asarray(lam_c_arr, float),
        t_settle_smoke=200.0, t_wu_smoke=400.0,
        neighbor_rule=True, delta=delta,
        do_branch=do_branch, et1_dts=dts, et1_two_segments=segments_two)
    lam = float(lam_c_mu - delta)
    t_cell = time.perf_counter()
    rec = run_sentinel(
        coup, Q, mu, float(lam_c_mu), float(bw), beta, tau, t0, cfg, log,
        lam_override=lam)
    rec["delta"] = float(delta)
    rec["wall_seconds"] = time.perf_counter() - t_cell
    return rec, logs


def _load_thresholds(path):
    with np.load(path) as data:
        lam_c = np.asarray(data["lam_c"], float)
        status = np.asarray(data["threshold_status"], dtype=str)
        bw = np.asarray(data["fold_bracket_width"], float)
        N = int(data["N"]); P = int(data["P"]); seed = int(data["seed"])
    return lam_c, status, bw, N, P, seed


def _to_jsonable(rec):
    out = {}
    for k, v in rec.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = _to_jsonable(v)
        else:
            out[k] = v
    return out


def _run_et1prime(args):
    """ET1': N=2000 delta-sweep GATE-DYN on 3 sentinels, up to 3 workers."""
    import multiprocessing as mp

    with np.load(args.et1_thresholds) as data:
        lam_c = np.asarray(data["lam_c"], float)
        N = int(data["N"]); P = int(data["P"]); seed = int(data["seed"])
    if seed != args.seed:
        raise ValueError(f"E24 seed {seed} != requested {args.seed}")
    bw_default = 2e-4                       # E24 bisection resolution (§6.3)

    links = [int(x) for x in args.et1_links.split(",") if x.strip()]
    deltas = [float(x) for x in args.et1_deltas.split(",") if x.strip()]
    if args.et1_smoke:
        links, deltas = [0], [0.020]

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"E34_ET1prime_log_{run_id}.txt"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg):
        line = f"{_utc_now()} {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"E34-ET1prime run {run_id}  N={N} P={P} seed={seed} beta={args.beta} "
        f"tau={args.tau}  nproc={args.nproc}")
    log(f"E24 thresholds: {args.et1_thresholds}")
    log(f"sentinels(links)={links}  deltas={deltas}  "
        f"one_segment={args.et1_one_segment}")
    for mu in links:
        log(f"  lam_c[{mu}]={lam_c[mu]:.6f}")

    cells = [(mu, d) for mu in links for d in deltas]
    dts = [0.01, 0.005]
    two_segments = not args.et1_one_segment
    payloads = [
        (N, P, seed, mu, d, args.beta, args.tau, args.t0, lam_c,
         float(lam_c[mu]), bw_default, True, dts, two_segments)
        for (mu, d) in cells]

    start = time.perf_counter()
    results = []
    nproc = max(1, min(args.nproc, 3, len(payloads)))
    if nproc == 1:
        for pl in payloads:
            rec, logs = _et1_cell(pl)
            for line in logs:
                log_handle.write(line + "\n")
            results.append(rec)
            log(f"[mu={rec['mu']} d={rec['delta']}] done "
                f"{rec.get('wall_seconds', 0):.1f}s status={rec['status']} "
                f"verdict={rec.get('verdict', '-')}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nproc) as pool:
            for rec, logs in pool.imap_unordered(_et1_cell, payloads):
                for line in logs:
                    log_handle.write(line + "\n")
                log_handle.flush()
                results.append(rec)
                log(f"[mu={rec['mu']} d={rec['delta']}] done "
                    f"{rec.get('wall_seconds', 0):.1f}s status={rec['status']} "
                    f"verdict={rec.get('verdict', '-')}")
    elapsed = time.perf_counter() - start

    results.sort(key=lambda r: (r["mu"], r["delta"]))
    n_dyn = sum(1 for r in results if r.get("gate_dyn"))
    n_admis = sum(1 for r in results if r.get("basin_admissible"))
    log(f"ET1prime done: {len(results)} cells, {elapsed:.1f}s; "
        f"admissible={n_admis}, GATE-DYN pass={n_dyn}")

    # npz
    npz_path = args.output_dir / f"E34_ET1prime_results_{run_id}.npz"
    payload = dict(run_id=run_id, N=N, P=P, seed=seed, beta=args.beta,
                   tau=args.tau, t0=args.t0, elapsed_seconds=elapsed,
                   links=np.asarray(links), deltas=np.asarray(deltas),
                   lam_c_links=np.asarray([lam_c[m] for m in links]))
    for r in results:
        pre = f"mu{r['mu']}_d{r['delta']:.3f}_"
        for key in ("a_node", "a_saddle", "edge_history_S1_dt0.01"):
            if key in r:
                nm = "edge_history" if "edge_history" in key else key
                payload[pre + nm] = r[key]
        for key in ("lam_local", "delta", "basin_separation", "far_idx",
                    "ref_idx", "edge_converged", "saddle_residual",
                    "n_unstable_saddle", "z_u", "real_root", "saddle_eigmax",
                    "edge_vs_arclength_dG", "fold_cont", "fold_mismatch",
                    "turns_up", "linear_gate_relerr", "gate_dyn", "gate_branch",
                    "wu_adjacent"):
            if key in r:
                payload[pre + key] = r[key]
    np.savez_compressed(npz_path, **payload)

    summary = dict(run_id=run_id, mode="et1prime", N=N, P=P, seed=seed,
                   elapsed_seconds=elapsed, gate_dyn_pass=n_dyn,
                   n_admissible=n_admis, n_cells=len(results),
                   deltas=deltas, links=links,
                   one_segment=bool(args.et1_one_segment),
                   npz=str(npz_path), log=str(log_path),
                   records=[_to_jsonable(r) for r in results])
    json_path = args.output_dir / f"E34_ET1prime_summary_{run_id}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    log(f"npz={npz_path}")
    log(f"summary={json_path}")
    log_handle.close()
    print(json.dumps(dict(mode="et1prime", n_cells=len(results),
                          n_admissible=n_admis, gate_dyn_pass=n_dyn,
                          elapsed_seconds=elapsed, npz=str(npz_path),
                          summary=str(json_path)), indent=2, default=str))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--production", action="store_true")
    modes.add_argument("--et0prime", action="store_true",
                       help="ET0' amendment: mu=2 at a lower lambda + probe.")
    modes.add_argument("--et1prime", action="store_true",
                       help="ET1': N=2000 delta-sweep GATE-DYN on 3 sentinels.")
    parser.add_argument("--links", default="14,8,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-anomaly", action="store_true")
    parser.add_argument("--et1-thresholds", type=Path,
                        default=REPO_DIR / "results" / "3_unification_seuils_scaling"
                        / "data" / "E24_thr_N2000_P100_s42.npz")
    parser.add_argument("--et1-links", default="28,0,91")
    parser.add_argument("--et1-deltas", default="0.005,0.010,0.020")
    parser.add_argument("--et1-smoke", action="store_true",
                        help="Run only the (mu=0, delta=0.020) timing cell.")
    parser.add_argument("--nproc", type=int, default=3)
    parser.add_argument("--et1-one-segment", action="store_true",
                        help="Budget: edge-track one segment only (still 2 dt).")
    args = parser.parse_args(argv)

    if args.et1prime:
        return _run_et1prime(args)

    lam_c, status, bw, N, P, seed = _load_thresholds(args.thresholds)
    if seed != args.seed:
        raise ValueError(f"threshold seed {seed} != requested {args.seed}")
    links = [int(x) for x in args.links.split(",") if x.strip()]
    if args.smoke:
        links = links[:1]

    xi, xis = make_iid_patterns(N, P, args.seed)
    coup = NumpyLowRankCouplings(xi, xis)
    Q = gram_matrix(xi)

    tag = "ET0prime" if args.et0prime else "ET0"
    mode = "et0prime" if args.et0prime else ("smoke" if args.smoke else "production")
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"E34_{tag}_log_{run_id}.txt"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg):
        line = f"{_utc_now()} {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    cfg = dict(
        smoke=args.smoke, lam_c=lam_c,
        t_settle_smoke=200.0, t_wu_smoke=400.0)
    log(f"E34-{tag} run {run_id}  mode={mode}"
        f"  N={N} P={P} seed={seed} beta={args.beta}")
    log(f"threshold table: {args.thresholds}")

    start = time.perf_counter()
    records = []
    anomalies = {}

    if args.et0prime:
        # mu=2 at lam' = lam_c(succ=3) - 0.02, bracketed against node_3, then
        # continued upward to its own fold lam_c(2).
        mu = 2
        succ = (mu + 1) % P
        lam_prime = float(lam_c[succ] - LAM_OFFSET)
        log(f"ET0' primary: mu={mu} at lam'={lam_prime:.7f} "
            f"(= lam_c({succ})-0.02); node_{mu} and node_{succ} both alive; "
            f"upward continuation target lam_c({mu})={lam_c[mu]:.7f}")
        t_mu = time.perf_counter()
        rec = run_sentinel(
            coup, Q, mu, float(lam_c[mu]), float(bw[mu]),
            args.beta, args.tau, args.t0, cfg, log, lam_override=lam_prime)
        rec["et0prime_lam"] = lam_prime
        rec["et0prime_succ"] = int(succ)
        rec["wall_seconds"] = time.perf_counter() - t_mu
        records.append(rec)
        log(f"[mu={mu}] ET0' done in {rec['wall_seconds']:.1f}s status={rec['status']}")

        # secondary exploratory probe at the original lam_local(2) = 0.3299
        lam_orig = float(lam_c[mu] - LAM_OFFSET)
        a_nodes_orig = solve_alive_nodes(coup, lam_orig, args.beta, lam_c)
        u2, ok2, _ = solve_memory_node(coup, mu, lam_orig, args.beta)
        if ok2:
            a_nodes_orig[mu] = reduced_field_coefficients(coup, u2, Q)
        anomalies["probe_0.3299"] = probe_attractors(
            coup, mu, lam_orig, args.beta, args.tau, args.t0, a_nodes_orig, log)
        links = [mu]
    else:
        links_iter = links[:1] if args.smoke else links
        links = links_iter
        for mu in links:
            if not np.isfinite(lam_c[mu]) or status[mu] != "spectral_fold":
                records.append(dict(mu=int(mu), status="threshold_indeterminate"))
                log(f"[mu={mu}] threshold indeterminate ({status[mu]}) - skipped")
                continue
            t_mu = time.perf_counter()
            rec = run_sentinel(
                coup, Q, mu, float(lam_c[mu]), float(bw[mu]),
                args.beta, args.tau, args.t0, cfg, log)
            rec["wall_seconds"] = time.perf_counter() - t_mu
            records.append(rec)
            log(f"[mu={mu}] done in {rec['wall_seconds']:.1f}s status={rec['status']}")

        if args.production and not args.no_anomaly:
            t_an = time.perf_counter()
            anomalies["mu17"] = anomaly_mu17(coup, args.beta, args.tau, args.t0, log)
            anomalies["mu3"] = anomaly_mu3(coup, args.beta, args.tau, args.t0, log)
            log(f"anomaly probes done in {time.perf_counter() - t_an:.1f}s")

    elapsed = time.perf_counter() - start

    # gate verdict
    verdicts = {r["mu"]: r.get("verdict", r.get("status"))
                for r in records if "mu" in r}
    gate_go = all(
        r.get("verdict") == "positive" for r in records if "verdict" in r) \
        and len([r for r in records if "verdict" in r]) == len(links)
    gate = "GO" if gate_go else "NO-GO"

    # save npz
    npz_path = args.output_dir / f"E34_{tag}_results_{run_id}.npz"
    payload = dict(
        run_id=run_id, N=N, P=P, seed=seed, beta=args.beta, tau=args.tau,
        t0=args.t0, links=np.asarray(links),
        lam_c_table=np.asarray([lam_c[m] for m in links]),
        gate=gate, elapsed_seconds=elapsed,
        mode=mode,
    )
    for r in records:
        mu = r.get("mu")
        prefix = f"mu{mu}_"
        for key in ("a_node", "a_saddle", "m_node", "m_saddle"):
            if key in r:
                payload[prefix + key] = r[key]
        if "edge_history_S1_dt0.01" in r:
            payload[prefix + "edge_history"] = r["edge_history_S1_dt0.01"]
        for key in ("lam_local", "lam_c_table", "z_u", "z_u_imag", "real_root",
                    "saddle_residual", "node_residual", "saddle_eigmax",
                    "node_eigmax", "n_unstable_saddle", "n_unstable_node",
                    "fold_cont", "fold_mismatch", "turns_up", "turns_down",
                    "linear_gate_relerr", "basin_separation",
                    "edge_vs_arclength_dG", "boundary_agree_dG",
                    "et0prime_lam", "et0prime_succ", "gate_dyn", "gate_branch",
                    "wu_adjacent"):
            if key in r:
                payload[prefix + key] = r[key]
    np.savez_compressed(npz_path, **payload)

    summary = dict(
        run_id=run_id, mode=payload["mode"], gate=gate,
        verdicts={str(k): v for k, v in verdicts.items()},
        elapsed_seconds=elapsed, npz=str(npz_path), log=str(log_path),
        anomalies=anomalies,
        records=[_to_jsonable(r) for r in records],
    )
    json_path = args.output_dir / f"E34_{tag}_summary_{run_id}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    log(f"GATE {tag} = {gate}  verdicts={verdicts}")
    log(f"elapsed={elapsed:.1f}s  npz={npz_path}")
    log(f"summary={json_path}")
    log_handle.close()
    print(json.dumps(dict(gate=gate, verdicts={str(k): v for k, v in verdicts.items()},
                          elapsed_seconds=elapsed, npz=str(npz_path),
                          summary=str(json_path)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
