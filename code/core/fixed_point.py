"""
JFNK fixed-point solver + pseudo-arclength continuation in lambda.

Fixed-point equation (tau-independent):
    F(u*, lambda) = -u* + C(lambda) tanh(beta u*) = 0

The branch is computed ONCE and reused for all tau values.

Pseudo-arclength continuation
------------------------------
Parameterize by arclength s. At each step:
  - Predictor: (u_p, lam_p) = (u + ds * u_dot, lam + ds * lam_dot)
  - Corrector: Newton on augmented system
        F(u, lam)                          = 0   (N equations)
        u_dot^T (u - u_p) + lam_dot*(lam - lam_p) = 0   (arclength constraint)
    Solved via the bordered system (Keller bordering):
        M z1 = -F             (GMRES)
        M z2 = -dF/dlam       (GMRES)
        dlam = (-g - u_dot.T z1) / (lam_dot + u_dot.T z2)
        du   = z1 + dlam * z2

Output NPZ keys
---------------
  lam      : (n_pts,) lambda values along the branch
  u_star   : (n_pts, N) fixed-point states
  m_nu     : (n_pts, n_ov) overlaps m_nu = xi^nu . tanh(beta u*) / N
  gain     : (n_pts, N) diagonal of D = beta*(1-tanh^2(beta u*))
  converged: (n_pts,) bool, whether Newton converged at each point
  params   : dict of config parameters used
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
from scipy.sparse.linalg import LinearOperator, gmres
from tqdm import tqdm

from couplings import Couplings
import config as cfg_mod


# ----------------------------------------------------------------- Newton / JFNK

def _make_linop(coup: Couplings, gain: np.ndarray, lam: float, N: int):
    """LinearOperator wrapping M(lam) = -I + C(lam) diag(gain)."""
    def mv(v):
        return coup.apply_M(v, gain, lam)
    return LinearOperator((N, N), matvec=mv, dtype=np.float64)


def newton_solve(u0: np.ndarray, lam: float, coup: Couplings,
                 beta: float,
                 tol: float = 1e-8,
                 max_iter: int = 20,
                 gmres_tol: float = 1e-6,
                 gmres_restart: int = 30) -> tuple[np.ndarray, bool]:
    """
    Newton–GMRES solve for F(u, lam) = 0 starting from u0.
    Returns (u_star, converged).
    """
    N  = u0.size
    u  = u0.copy()

    for _ in range(max_iter):
        F   = coup.field_F(u, lam, beta)
        res = np.linalg.norm(F) / np.sqrt(N)
        if res < tol:
            return u, True

        gain = coup.compute_gain(u, beta)
        A    = _make_linop(coup, gain, lam, N)
        du, info = gmres(A, -F,
                         rtol=gmres_tol, restart=gmres_restart,
                         maxiter=max(20, N // max(1, gmres_restart)))
        if info != 0:
            # GMRES failed; take a partial step anyway
            pass
        u = u + du

    F   = coup.field_F(u, lam, beta)
    res = np.linalg.norm(F) / np.sqrt(N)
    return u, bool(res < tol * 10)


# ------------------------------------------------------------ continuation tangent

def _compute_tangent(u: np.ndarray, lam: float, coup: Couplings,
                     beta: float, sign: float,
                     gmres_tol: float, gmres_restart: int) -> tuple[np.ndarray, float]:
    """
    Compute the unit tangent (u_dot, lam_dot) satisfying
        M(lam) u_dot + F_lam lam_dot = 0,   ||u_dot||^2 + lam_dot^2 = 1.
    sign: +1 for increasing lam direction, -1 for decreasing.
    """
    N     = u.size
    gain  = coup.compute_gain(u, beta)
    A     = _make_linop(coup, gain, lam, N)
    f_lam = coup.dF_dlam(u, beta)     # dF/dlam, shape (N,)

    # M w = -f_lam  =>  u_dot = w * lam_dot  (unnormalized)
    w, info = gmres(A, -f_lam,
                    rtol=gmres_tol, restart=gmres_restart,
                    maxiter=max(20, N // max(1, gmres_restart)))
    norm     = np.sqrt(np.dot(w, w) + 1.0)
    lam_dot  = sign / norm
    u_dot    = w / norm
    return u_dot, lam_dot


def _corrector(u: np.ndarray, lam: float,
               u_pred: np.ndarray, lam_pred: float,
               u_dot: np.ndarray, lam_dot: float,
               coup: Couplings, beta: float,
               newton_tol: float, gmres_tol: float, gmres_restart: int,
               max_newton: int = 10) -> tuple[np.ndarray, float, bool]:
    """
    Newton corrector for pseudo-arclength augmented system.
    Returns (u_new, lam_new, converged).
    """
    N = u.size
    for _ in range(max_newton):
        F     = coup.field_F(u, lam, beta)
        g     = float(np.dot(u_dot, u - u_pred) + lam_dot * (lam - lam_pred))
        res   = np.sqrt(np.dot(F, F) / N + g ** 2)
        if res < newton_tol:
            return u, lam, True

        gain   = coup.compute_gain(u, beta)
        A      = _make_linop(coup, gain, lam, N)
        f_lam  = coup.dF_dlam(u, beta)

        z1, _  = gmres(A, -F,       rtol=gmres_tol, restart=gmres_restart, maxiter=max(20, N // max(1, gmres_restart)))
        z2, _  = gmres(A, -f_lam,   rtol=gmres_tol, restart=gmres_restart, maxiter=max(20, N // max(1, gmres_restart)))

        denom  = lam_dot + float(np.dot(u_dot, z2))
        if abs(denom) < 1e-14:
            return u, lam, False
        dlam   = (-g - float(np.dot(u_dot, z1))) / denom
        du     = z1 + dlam * z2

        u   = u + du
        lam = lam + dlam

    F   = coup.field_F(u, lam, beta)
    res = np.linalg.norm(F) / np.sqrt(N)
    return u, lam, bool(res < newton_tol * 10)


# ---------------------------------------------------------------- main routine

def continuation(coup: Couplings, beta: float, cfg: dict,
                 u0: np.ndarray | None = None) -> dict:
    """
    Pseudo-arclength continuation of F(u, lam)=0 from lam_min to lam_max.

    Parameters
    ----------
    coup : Couplings object
    beta : inverse temperature
    cfg  : config dict from config.resolve()
    u0   : optional starting guess at lam_min; if None, uses xi^1 (pattern 0)

    Returns
    -------
    branch : dict with keys lam, u_star, m_nu, gain, converged
    """
    N         = coup.N
    n_ov      = cfg["n_overlaps"]
    lam_min   = cfg["lam_min"]
    lam_max   = cfg["lam_max"]
    ds        = cfg["ds"]
    nt        = cfg.get("newton_tol",   1e-8)
    gt        = cfg.get("gmres_tol",    1e-6)
    gr        = cfg.get("gmres_restart", 30)

    # ---- initial fixed point at lam_min --------------------------------
    if u0 is None:
        # Warm start: first pattern (nearly saturated at beta=20)
        u0 = 0.99 * coup._xi_np[0].copy()

    # beta-annealing for a robust initial fixed point: at large beta the tanh is
    # nearly a step function and direct Newton from the raw pattern can stall
    # (seen at alpha=0.10, N=10000). Warm-start up through a schedule of smaller
    # beta, where the fixed-point map is smooth. Cheap: each low-beta solve is easy.
    print(f"[continuation] Initial Newton solve at lam={lam_min:.3f} "
          f"(beta-annealed) ...", flush=True)
    u_star = u0.copy()
    beta_sched = [b for b in (5.0, 8.0, 12.0, 16.0) if b < beta] + [beta]
    for b in beta_sched:
        u_star, ok = newton_solve(u_star, lam_min, coup, b,
                                  tol=nt, gmres_tol=gt, gmres_restart=gr)
    if not ok:
        print("  WARNING: initial Newton did not fully converge (even annealed).")

    # ---- initial tangent (increasing direction) -------------------------
    u_dot, lam_dot = _compute_tangent(u_star, lam_min, coup, beta, +1.0, gt, gr)

    # ---- storage --------------------------------------------------------
    lam_list  = [lam_min]
    u_list    = [u_star.copy()]
    m_list    = [coup.condensed_overlaps(u_star, beta, n_ov)]
    gain_list = [coup.compute_gain(u_star, beta)]
    conv_list = [ok]

    # estimated number of steps
    n_est = int((lam_max - lam_min) / (ds * lam_dot + 1e-12))
    n_max = max(n_est * 3, 200)

    u_prev   = u_star.copy()
    lam_prev = lam_min
    _n_neg_ldot = 0  # consecutive steps with lam_dot < 0 (fold-back counter)

    pbar = tqdm(total=n_max, desc="continuation", unit="step")

    for step in range(n_max):
        # predictor
        u_pred   = u_star + ds * u_dot
        lam_pred = float(np.clip(lam_prev + ds * lam_dot, -0.1, lam_max + 0.1))

        # corrector
        u_new, lam_new, conv = _corrector(
            u_pred, lam_pred, u_pred, lam_pred,
            u_dot, lam_dot, coup, beta, nt, gt, gr)

        # step-size control
        if not conv:
            ds *= 0.5
            pbar.set_postfix(ds=f"{ds:.4f}", status="halved")
            if ds < 1e-5:
                print("\n[continuation] Step size too small; stopping.")
                break
            # Early fold stop: repeated corrector failures with tiny ds
            # and branch has already made significant progress.
            lam_cur_max = max(lam_list) if lam_list else lam_min
            if ds < 5e-4 and step > 10 and lam_cur_max > lam_min + 0.05:
                print(f"\n[continuation] Fold at lam≈{lam_cur_max:.4f}; "
                      f"ds={ds:.2e}. Stopping.")
                break
            continue

        # update tangent (sign: consistency with previous)
        u_dot_new, lam_dot_new = _compute_tangent(u_new, lam_new, coup, beta, +1.0, gt, gr)
        if (u_dot @ u_dot_new + lam_dot * lam_dot_new) < 0:
            u_dot_new  = -u_dot_new
            lam_dot_new = -lam_dot_new

        u_prev, lam_prev = u_star.copy(), float(lam_new)
        u_star = u_new
        u_dot, lam_dot = u_dot_new, lam_dot_new

        # Fold detection: sustained negative lam_dot means the branch is
        # traversing back (lower branch of saddle-node).  Stop early.
        if lam_dot < 0:
            _n_neg_ldot += 1
        else:
            _n_neg_ldot = 0
        lam_max_seen_now = max(lam_list) if lam_list else lam_min
        if (_n_neg_ldot >= 5 and step > 10
                and lam_max_seen_now > lam_min + 0.05):
            print(f"\n[continuation] Fold at lam≈{lam_max_seen_now:.4f}; "
                  f"branch returning (lam_dot<0 for {_n_neg_ldot} steps). Stopping.")
            break

        lam_list.append(lam_new)
        u_list.append(u_star.copy())
        m_list.append(coup.condensed_overlaps(u_star, beta, n_ov))
        gain_list.append(coup.compute_gain(u_star, beta))
        conv_list.append(conv)

        pbar.update(1)
        pbar.set_postfix(lam=f"{lam_new:.4f}", m1=f"{m_list[-1][0]:.3f}",
                         ldot=f"{lam_dot:.3f}")

        # mild increase of step size on easy steps
        ds = min(ds * 1.05, cfg["ds"] * 2.0)

        if lam_new >= lam_max:
            break

        # Fold detection: stop if branch has folded back past lam_min
        # (allows plotting the fold but avoids infinite wandering)
        lam_max_seen = max(lam_list) if lam_list else lam_min
        if step > 20 and lam_new < lam_min and lam_max_seen > lam_min + 3 * cfg["ds"]:
            print(f"\n[continuation] Fold detected; branch returned to lam={lam_new:.4f}. Stopping.")
            break

    pbar.close()

    branch = dict(
        lam      = np.array(lam_list),
        u_star   = np.array(u_list),          # (n_pts, N)
        m_nu     = np.array(m_list),          # (n_pts, n_ov)
        gain     = np.array(gain_list),        # (n_pts, N)
        converged= np.array(conv_list),
    )
    return branch


def save_branch(branch: dict, coup: Couplings, cfg: dict):
    """Save the fixed-point branch to NPZ."""
    tag   = cfg_mod.param_tag(cfg)
    path  = cfg_mod.out_path(cfg, "fixed_point", f"branch_{tag}.npz")
    np.savez(path,
             lam       = branch["lam"],
             u_star    = branch["u_star"],
             m_nu      = branch["m_nu"],
             gain      = branch["gain"],
             converged = branch["converged"],
             N    = cfg["N"],
             P    = cfg["P"],
             alpha= cfg["alpha"],
             beta = cfg["beta"],
             seed = cfg["seed"])
    print(f"[fixed_point] Branch saved → {path}")
    return path


def load_branch(path: str) -> dict:
    """Load branch from NPZ."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}
