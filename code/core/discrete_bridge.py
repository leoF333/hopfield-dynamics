"""
Bridge to the discrete synchronous map (§3.8 — optional).

The discrete soft-spin update:
    x(t+1) = tanh(beta * [(1-lam)*J*x(t) + lam*K*x(t-tau)])

Fixed point: x* = tanh(beta * C x*)  (differs from the continuous-time fixed point).

Augmented state: X(t) = (x(t), x(t-1), ..., x(t-tau)) in R^{N*(tau+1)}.
The Jacobian of the synchronous map at x* has block companion form:
    Row 0: [A_disc, 0, ..., 0, B_disc]
    Row 1: [I,       0, ..., 0, 0    ]
    ...
    Row tau: [0, ..., I, 0       ]

where:
    A_disc = diag(D_disc) * (1-lam) * J   (matrix-free)
    B_disc = diag(D_disc) * lam * K       (matrix-free)
    D_disc = beta * diag(1 - (x*)^2)      [sech^2 at the discrete fixed point]

Neimark-Sacker instability: a complex-conjugate pair of eigenvalues leaves the
unit circle |mu| = 1.  This corresponds (via tau->0 limit) to the continuous Hopf.
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
from scipy.sparse.linalg import LinearOperator, eigs

from couplings import Couplings
import config as cfg_mod


def solve_discrete_fp(coup: Couplings, lam: float, beta: float,
                      tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """
    Solve x* = tanh(beta * C(lam) x*) by fixed-point iteration.
    Initialized on pattern 1 (saturated).
    """
    u = 0.99 * coup._xi_np[0].copy()
    for _ in range(max_iter):
        h   = coup.apply_C(u, lam)
        u_new = np.tanh(beta * h)
        if np.linalg.norm(u_new - u) / np.sqrt(coup.N) < tol:
            return u_new
        u = u_new
    return u


def discrete_companion_mv(v_flat: np.ndarray, x_star: np.ndarray,
                          lam: float, coup: Couplings,
                          beta: float, tau: int) -> np.ndarray:
    """
    Apply the block-companion Jacobian to v_flat in R^{N*(tau+1)}.

    v_flat = [v_0, v_1, ..., v_tau] stacked (each v_j in R^N).
    Row 0 output: A_disc v_0 + B_disc v_tau
    Row j>0 output: v_{j-1}
    """
    N    = coup.N
    blks = tau + 1
    V    = v_flat.reshape(blks, N)

    D_disc = beta * (1.0 - x_star ** 2)    # (N,) gain at discrete fixed point

    # Row 0: A v_0 + B v_tau
    Av0  = D_disc * coup.apply_J(V[0])     # diag(D) * J v_0  -- wrong order
    # Correct: A = diag(D) * (1-lam)*J  means A v = D * (J v)
    Av0  = D_disc * ((1.0 - lam) * coup.apply_J(V[0]))
    BvT  = D_disc * (lam          * coup.apply_K(V[tau]))
    out  = np.empty_like(V)
    out[0] = Av0 + BvT

    # Rows 1,...,tau: companion shift
    for j in range(1, blks):
        out[j] = V[j - 1]

    return out.reshape(-1)


def neimark_sacker_eigenvalues(x_star: np.ndarray, lam: float,
                                coup: Couplings, cfg: dict,
                                tau: int | None = None,
                                k: int = 20) -> np.ndarray:
    """
    Leading eigenvalues of the discrete companion Jacobian.
    Neimark-Sacker: |mu| = 1 with complex pair.

    Returns (k,) complex array sorted by |mu| descending.
    """
    tau   = tau if tau is not None else cfg["tau"]
    beta  = cfg["beta"]
    N     = coup.N
    dim   = N * (tau + 1)

    def mv(v):
        if np.iscomplexobj(v):
            return (discrete_companion_mv(v.real, x_star, lam, coup, beta, tau)
                    + 1j * discrete_companion_mv(v.imag, x_star, lam, coup, beta, tau))
        return discrete_companion_mv(v, x_star, lam, coup, beta, tau)

    A_op = LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)
    ncv  = min(max(3 * k, 40), dim - 1)

    try:
        vals, _ = eigs(A_op, k=k, which='LM', ncv=ncv, maxiter=300,
                       tol=1e-8, return_eigenvectors=True)
    except Exception as e:
        print(f"[discrete_bridge] eigs failed: {e}")
        return np.array([])

    idx = np.argsort(-np.abs(vals))
    return vals[idx]


def find_discrete_lc(coup: Couplings, cfg: dict,
                      lam_values: np.ndarray | None = None,
                      n_eig: int = 6) -> dict:
    """
    Sweep lambda and find the Neimark-Sacker crossing |mu|=1.

    Returns dict: lam, max_abs_mu, lam_c_disc, omega_c_disc.
    """
    beta  = cfg["beta"]
    tau   = cfg["tau"]

    if lam_values is None:
        lam_values = np.linspace(0.1, 0.9, 20)

    max_abs = []
    for lam in lam_values:
        xst = solve_discrete_fp(coup, lam, beta)
        ev  = neimark_sacker_eigenvalues(xst, lam, coup, cfg, tau=tau, k=n_eig)
        max_abs.append(np.abs(ev).max() if len(ev) else np.nan)

    max_abs = np.array(max_abs)

    # Find where |mu| crosses 1
    lam_c_disc = omega_c_disc = None
    for i in range(1, len(lam_values)):
        if (max_abs[i - 1] < 1 <= max_abs[i] and
                np.isfinite(max_abs[i - 1]) and np.isfinite(max_abs[i])):
            t = (1.0 - max_abs[i - 1]) / (max_abs[i] - max_abs[i - 1] + 1e-14)
            lam_c_disc = float(lam_values[i - 1] + t * (lam_values[i] - lam_values[i - 1]))
            # Compute eigenvalue at crossing for omega
            xst = solve_discrete_fp(coup, lam_c_disc, beta)
            ev  = neimark_sacker_eigenvalues(xst, lam_c_disc, coup, cfg, tau=tau, k=n_eig)
            # Eigenvalue on unit circle: mu = exp(i*theta)
            lead_mu = ev[0]
            omega_c_disc = float(np.abs(np.angle(lead_mu)) / cfg["t0"])
            break

    return dict(
        lam       = lam_values,
        max_abs_mu= max_abs,
        lam_c_disc= lam_c_disc,
        omega_c_disc = omega_c_disc,
    )
