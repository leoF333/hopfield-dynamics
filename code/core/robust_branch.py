"""
Robust memory-branch tracer via the Woodbury exact solve.

Why this exists
---------------
The pseudo-arclength continuation in fixed_point.py solves M du = -F with GMRES.
Near the Hopfield capacity (e.g. alpha=0.10, alpha_c~0.138) M is ill-conditioned,
GMRES stalls, Newton leaves the memory basin, and the fold-detector fires a FALSE
fold (observed: alpha=0.10 folding at 0.05-0.11 instead of ~0.18).

M has EXACT low-rank structure though:
    M(lam) = -I + C(lam) D ,   C(lam) D = (1/N) U V  with rank <= P
    U = ((1-lam) xi + lam xi_shift)^T   (N x P),   V = xi diag(gain)   (P x N).
So M du = -F is solved EXACTLY by the Woodbury identity at cost O(N P^2 + P^3):
    M^{-1} r = -r - U'(I_P - V U')^{-1}(V r),   U' = U/N.
No iterative convergence to fail.

The rightmost eigenvalue of M is also exact and cheap via Sylvester:
    nonzero eig(C D) = eig((1/N) V U)  (P x P), so
    eig_max(M) = -1 + max Re eig((1/N) V U).
The static fold (saddle-node) is exactly where eig_max(M) crosses 0 (identity
Delta(0) = -M, tau-independent), so we locate lambda_fold by that crossing rather
than by the fragile tangent-reversal heuristic.

Tracking uses a TANGENT predictor  u_pred = u + dlam * du/dlam  (du/dlam via the
same Woodbury solve), arclength-limited so it does not overshoot as du/dlam blows
up near the fold — this is what lets it follow the tiny near-capacity basin far
enough for eig_max(M) to approach 0.
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import numpy as np


def woodbury_solve(coup, gain, lam, rhs):
    """Exact M(lam)^{-1} rhs, M = -I + C(lam) diag(gain)."""
    xi = coup._xi_np; xis = coup._xis_np; N = coup.N; P = coup.P
    Ut = (1.0 - lam) * xi + lam * xis      # (P,N) = U^T   (U = Ut.T is N x P)
    V  = xi * gain[None, :]                # (P,N) = xi diag(gain)
    x  = np.linalg.solve(np.eye(P) - (V @ Ut.T) / N, V @ rhs)
    return -rhs - (Ut.T @ x) / N


def eigmax_M(coup, u, lam, beta):
    """Exact rightmost eigenvalue of M(lam) at u via the P x P Sylvester matrix."""
    xi = coup._xi_np; xis = coup._xis_np; N = coup.N
    g  = coup.compute_gain(u, beta)
    Ut = (1.0 - lam) * xi + lam * xis
    V  = xi * g[None, :]
    return -1.0 + float(np.linalg.eigvals((V @ Ut.T) / N).real.max())


def woodbury_newton(coup, u, lam, beta, tol=1e-10, max_iter=60):
    """Newton with exact Woodbury step + backtracking line search."""
    N = coup.N; u = u.copy()
    fnorm = lambda uu: np.linalg.norm(coup.field_F(uu, lam, beta)) / np.sqrt(N)
    r = fnorm(u)
    for _ in range(max_iter):
        if r < tol:
            return u, True
        F = coup.field_F(u, lam, beta)
        du = woodbury_solve(coup, coup.compute_gain(u, beta), lam, -F)
        st = 1.0; improved = False
        for _ in range(30):
            if fnorm(u + st * du) < r:
                improved = True; break
            st *= 0.5
        if not improved:
            break
        u = u + st * du; r = fnorm(u)
    return u, bool(r < tol * 10)


def anneal_newton(coup, u0, lam, beta, betas=(5.0, 8.0, 12.0, 16.0), **kw):
    """Robust initial solve via beta-annealing (tanh near-singular at large beta)."""
    u = u0.copy(); ok = False
    for b in [b for b in betas if b < beta] + [beta]:
        u, ok = woodbury_newton(coup, u, lam, float(b), **kw)
    return u, ok


def trace_branch(coup, beta, cfg, ds0=None, m1_min=0.5, verbose=False):
    """
    Trace the xi^1-connected memory branch from lam_min up to its saddle-node fold,
    returning a branch dict and (lambda_fold, status).

    Method: warm-start (previous solution) + DIRECT beta Newton with exact Woodbury
    step + backtracking, small adaptive ds. The fold is detected two equivalent ways
    (identity Delta(0) = -M): either eig_max(M) crosses 0, OR Newton stalls because M
    becomes singular AT the fold. Both give lambda_fold; the stall IS the saddle-node.

    A tangent predictor was tried and REMOVED: near the Hopfield capacity it drifts
    onto spurious nearby m1~1 solutions. Plain warm-start stays on the branch.

    status: 'fold'  -> eig_max(M) crossed 0 (bracketed & interpolated)
            'stall' -> Newton stalled at the fold (M singular); lambda_fold = stall lam
            'lam_max'-> reached lam_max with eig_max < 0 (no fold below lam_max)
    """
    lam_min = cfg["lam_min"]; lam_max = cfg["lam_max"]
    n_ov = cfg["n_overlaps"]
    ds0 = ds0 if ds0 is not None else cfg.get("ds", 0.01)
    xi0 = coup._xi_np[0]; N = coup.N

    u, ok = anneal_newton(coup, 0.99 * xi0.copy(), lam_min, beta)
    lam = lam_min
    lams = [lam]; us = [u.copy()]
    ems  = [eigmax_M(coup, u, lam, beta)]
    ms   = [coup.condensed_overlaps(u, beta, n_ov)]
    gs   = [coup.compute_gain(u, beta)]

    ds = ds0
    status = "lam_max"; lam_fold = lam_max
    while lam < lam_max - 1e-12:
        lam_new = min(lam + ds, lam_max)
        u_new, okn = woodbury_newton(coup, u, lam_new, beta)   # warm-start from u
        m1 = float(xi0 @ np.tanh(beta * u_new) / N)
        if okn and m1 > m1_min:
            e_new = eigmax_M(coup, u_new, lam_new, beta)
            lams.append(lam_new); us.append(u_new.copy()); ems.append(e_new)
            ms.append(coup.condensed_overlaps(u_new, beta, n_ov))
            gs.append(coup.compute_gain(u_new, beta))
            lam, u = lam_new, u_new
            if verbose:
                print(f"    lam={lam:.4f} m1={m1:.4f} eig_max(M)={e_new:+.4f} ds={ds:.4f}",
                      flush=True)
            if e_new >= 0.0:                       # fold bracketed by eig crossing
                e0, e1 = ems[-2], ems[-1]
                lam_fold = lams[-2] + (0.0 - e0) / (e1 - e0 + 1e-30) * (lams[-1] - lams[-2])
                status = "fold"; break
            ds = min(ds * 1.15, ds0)
        else:
            ds *= 0.5
            if ds < 1e-4:                          # Newton stalls -> M singular = fold
                lam_fold = lam
                status = "stall"; break

    branch = dict(lam=np.array(lams), u_star=np.array(us),
                  m_nu=np.array(ms), gain=np.array(gs), eig_max=np.array(ems))
    return branch, float(lam_fold), status
