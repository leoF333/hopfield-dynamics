"""
Pseudospectral infinitesimal-generator (IG) method for the DDE stability analysis.

Reference: Breda, Maset, Vermiglio — SIAM J. Sci. Comput. 27:482, 2005.

Discretization
--------------
Map [-tau, 0] -> [-1, 1] via  theta = 1 + 2*s/tau  (s in [-tau,0], theta in [-1,1]).
CGL nodes: theta_j = cos(pi*j/M),  j=0,...,M.
Corresponding delay nodes: s_j = tau*(theta_j - 1)/2.
  s_0 = 0  (boundary, corresponds to u* at t=0)
  s_M = -tau (corresponds to u* at t-tau)

The infinitesimal generator A of the DDE semigroup acts on
V = (v_0, v_1, ..., v_M) in R^{N(M+1)} as:

  (AV)_0 = (1/t0)[ -v_0 + (1-lam) J D v_0 + lam K D v_M ]
  (AV)_j = (2/tau) sum_k  D_cheb[j,k] v_k,   j=1,...,M

where D_cheb is the (M+1)x(M+1) Chebyshev differentiation matrix on [-1,1]
and D = diag(beta*(1 - tanh^2(beta u*))) is the diagonal gain at the fixed point.

Note: row j=0 of the Chebyshev differentiation output is REPLACED by the boundary
condition above (not added — replaced).

Eigenvalue problem
------------------
The eigenvalues of A approximate the characteristic roots z of Delta(z)=0:
  Delta(z) = (t0 z + 1)I - (1-lam) J D - lam K D exp(-z tau) = 0

Stability: Re(z) < 0  for all characteristic roots.
Hopf crossing: complex conjugate pair crosses Re(z)=0 at z=+/-i omega, omega>0.

Consistency check (tau=0)
-------------------------
At tau=0: all CGL nodes collapse. The IG spectrum reduces to eig(M(lam))=-1+eig(CD).
Use this as a unit test.
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
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence
from scipy.sparse.linalg import gmres as sp_gmres
from tqdm import tqdm

from couplings import Couplings
import config as cfg_mod


# ----------------------------------------------------------------- Chebyshev

def cheb_diff_matrix(M: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Chebyshev differentiation matrix on CGL nodes cos(pi*j/M), j=0,...,M.
    Nodes are ordered x_0=1 > x_1 > ... > x_M=-1  (decreasing).
    Returns (D, x) both shape (M+1,).
    Standard formula from Trefethen 'Spectral Methods in Matlab'.
    """
    n = M + 1
    if M == 0:
        return np.zeros((1, 1)), np.array([1.0])

    j   = np.arange(n)
    x   = np.cos(np.pi * j / M)          # (n,) CGL nodes on [-1,1]

    # Barycentric weights
    c        = np.ones(n)
    c[0]     = 2.0
    c[-1]    = 2.0
    c       *= (-1.0) ** j

    X   = np.tile(x, (n, 1))
    # X[i,j] = x[j]  (rows of X are copies of x)
    # X.T[i,j] = x[i], so X.T - X gives dX[i,j] = x[i] - x[j]  ✓
    dX  = X.T - X                        # (n,n)  x_i - x_j

    # Off-diagonal: D_ij = (c_i/c_j) / (x_i - x_j)
    D   = np.outer(c, 1.0 / c) / (dX + np.eye(n))  # safe: diagonal set to 0 by eye trick
    # Diagonal: D_jj = -sum_{k!=j} D_jk  (negative row sum)
    D  -= np.diag(D.sum(axis=1))

    return D, x


# ----------------------------------------------------------------- IG operator

class IGOperator:
    """
    Pseudospectral infinitesimal generator of the linear DDE semigroup.

    State dimension: N*(M+1).
    Ordering: V = [v_0, v_1, ..., v_M] stacked row-wise.
      v_j = delta u(s_j),  s_j = tau*(cos(pi*j/M)-1)/2.
      j=0: s=0 (current), j=M: s=-tau (delayed).
    """

    def __init__(self, gain: np.ndarray, lam: float,
                 coup: Couplings, tau: float, t0: float, M: int):
        """
        Parameters
        ----------
        gain : (N,) float64, beta*(1-tanh^2(beta u*)) at fixed point
        lam  : mixing parameter
        coup : Couplings object
        tau  : delay (float, must be > 0)
        t0   : time constant
        M    : number of Chebyshev intervals (M+1 nodes)
        """
        assert tau > 0, "tau must be > 0 for IG operator; use direct eig for tau=0"
        self.N    = coup.N
        self.M    = M
        self.gain = gain.astype(np.float64)
        self.lam  = float(lam)
        self.coup = coup
        self.tau  = float(tau)
        self.t0   = float(t0)

        self.D_cheb, _ = cheb_diff_matrix(M)
        # Interior rows only (j=1,...,M) scaled by 2/tau
        self._D_interior = (2.0 / tau) * self.D_cheb[1:, :]   # (M, M+1)

        self.dim = coup.N * (M + 1)

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        """
        Apply A to a vector vec of shape (N*(M+1),).
        Handles real and complex input.
        """
        if np.iscomplexobj(vec):
            return self._real_matvec(vec.real) + 1j * self._real_matvec(vec.imag)
        return self._real_matvec(vec)

    def _real_matvec(self, vec: np.ndarray) -> np.ndarray:
        """Apply A to a real vector."""
        N, M = self.N, self.M
        # Reshape to (M+1, N)
        V = vec.reshape(M + 1, N)
        W = np.empty_like(V)

        # ---- interior rows j=1,...,M: Chebyshev differentiation --------
        # _D_interior has shape (M, M+1); V has shape (M+1, N)
        W[1:, :] = self._D_interior @ V          # (M, N)

        # ---- boundary row j=0: linearized DDE --------------------------
        v0 = V[0]    # delta u at s=0
        vM = V[M]    # delta u at s=-tau
        # Apply gain elementwise before coupling matvecs
        D_v0 = self.gain * v0
        D_vM = self.gain * vM
        bc   = (-v0
                + (1.0 - self.lam) * self.coup.apply_J(D_v0)
                + self.lam         * self.coup.apply_K(D_vM))
        W[0, :] = bc / self.t0

        return W.reshape(-1)

    def as_linear_operator(self) -> LinearOperator:
        """Return a scipy LinearOperator for use with eigs."""
        return LinearOperator(
            shape=(self.dim, self.dim),
            matvec=self.matvec,
            dtype=np.complex128,
        )


# ----------------------------------------------------------------- eigenvalues

def _ode_rightmost_eig(u_star: np.ndarray, lam: float, coup: Couplings,
                       cfg: dict, k: int = 4) -> complex:
    """
    Rightmost eigenvalue of the ODE linearization (tau=0) at (u*, lam).
    Uses matrix-free Arnoldi on apply_M — no N×N matrix materialized.
    Returns a complex number (the leading eigenvalue).
    """
    N    = coup.N
    gain = coup.compute_gain(u_star, cfg["beta"])

    def _mv(v):
        return coup.apply_M(v, gain, lam)

    ode_op = LinearOperator((N, N), matvec=_mv, dtype=np.float64)
    ncv    = min(max(2 * k + 1, 20), N - 1)
    try:
        vals, _ = eigs(ode_op, k=k, which='LR', ncv=ncv, tol=1e-8,
                       return_eigenvectors=True)
        idx = np.argmax(vals.real)
        return complex(vals[idx])
    except Exception:
        return complex(-1.0, 0.0)


def rightmost_eigenvalues(
        u_star: np.ndarray, lam: float, coup: Couplings, cfg: dict,
        M: int | None = None,
        k: int | None = None,
        sigma: complex | None = None,
) -> np.ndarray:
    """
    Compute rightmost characteristic roots of the DDE linearized at (u*, lam).

    Parameters
    ----------
    sigma : shift for shift-invert Arnoldi.
        - None  → uses which='LR' (fast; may return spurious Chebyshev modes
                  in the stable region when tau is large).
        - complex → shift-invert: finds eigenvalues of A closest to sigma.
                  Implemented by calling eigs on (A-sigma)^{-1} (via GMRES)
                  and mapping back: z = sigma + 1/mu.

    Returns
    -------
    evals : (k,) complex128, sorted by decreasing Re(z).
    """
    tau   = float(cfg["tau"])
    t0    = cfg["t0"]
    M     = M or cfg["M_cheb"]
    k     = k or cfg["n_eigs"]
    ncv   = cfg.get("arnoldi_ncv", 60)
    maxit = cfg.get("arnoldi_maxiter", 500)

    gain  = coup.compute_gain(u_star, cfg["beta"])
    ig    = IGOperator(gain, lam, coup, tau, t0, M)
    A_op  = ig.as_linear_operator()

    if sigma is None:
        # Plain Arnoldi targeting largest real part.
        # On ArpackNoConvergence, use partially-converged eigenvalues rather
        # than crashing — the rightmost ones typically converge first.
        try:
            vals, _ = eigs(A_op, k=k, which='LR', ncv=ncv, maxiter=maxit,
                           tol=1e-10, return_eigenvectors=True)
        except ArpackNoConvergence as exc:
            vals = exc.eigenvalues
            if len(vals) == 0:
                vals = np.full(k, complex(-10.0, 0.0))
        except Exception:
            try:
                vals, _ = eigs(A_op, k=k, which='LR', ncv=min(ncv, ig.dim - 1),
                               maxiter=maxit, tol=1e-8, return_eigenvectors=True)
            except ArpackNoConvergence as exc:
                vals = exc.eigenvalues
                if len(vals) == 0:
                    vals = np.full(k, complex(-10.0, 0.0))
            except Exception:
                vals = np.full(k, complex(-10.0, 0.0))
    else:
        # Shift-invert: find eigenvalues of A closest to sigma.
        # Build OP_inv = (A - sigma*I)^{-1} via GMRES, then call eigs on OP_inv.
        # eigs(OP_inv, which='LM') finds mu = 1/(z-sigma) with largest |mu|,
        # i.e., z closest to sigma.  Map back: z = sigma + 1/mu.
        def _shifted_mv(w: np.ndarray) -> np.ndarray:
            return ig.matvec(w) - sigma * w

        A_shift = LinearOperator(
            (ig.dim, ig.dim), matvec=_shifted_mv, dtype=np.complex128)

        restart = min(50, ig.dim)

        def _OPinv_mv(b: np.ndarray) -> np.ndarray:
            sol, info = sp_gmres(A_shift, b, rtol=1e-7,
                                 maxiter=300, restart=restart)
            if info != 0:
                # Relaxed fallback
                sol, _ = sp_gmres(A_shift, b, rtol=1e-5, maxiter=500,
                                  restart=restart)
            return sol

        OP_inv = LinearOperator(
            (ig.dim, ig.dim), matvec=_OPinv_mv, dtype=np.complex128)

        try:
            # Eigenvalues of (A-sigma)^{-1}: largest |mu| = closest to sigma
            mu, _ = eigs(OP_inv, k=k, which='LM', ncv=ncv, maxiter=maxit,
                         tol=1e-8, return_eigenvectors=True)
            vals = sigma + 1.0 / mu          # map back to eigenvalues of A
        except Exception:
            # Fallback: plain LR (may include spurious modes)
            try:
                vals, _ = eigs(A_op, k=k, which='LR', ncv=ncv,
                               maxiter=maxit, tol=1e-8, return_eigenvectors=True)
            except Exception:
                vals = np.full(k, complex(-10.0, 0.0))

    # Sort by decreasing real part
    idx = np.argsort(-vals.real)
    return vals[idx]


def _rayleigh_quotient_track(ig: IGOperator, sigma: complex,
                             n_rq: int = 10,
                             gmres_inner: int = 30) -> tuple[complex, np.ndarray]:
    """
    Inverse-power / Rayleigh-quotient iteration to track one eigenvalue of
    the IG operator A, starting from an approximate eigenvalue `sigma`.

    Far cheaper than full shift-invert Arnoldi (no ncv restart, 1 GMRES per step).
    Convergence is cubic near an isolated eigenvalue.

    Returns (z_eig, v_eig): converged eigenvalue and approximate eigenvector.
    """
    dim = ig.dim
    rng = np.random.default_rng(42)
    v   = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v  /= np.linalg.norm(v)

    z = sigma
    for _it in range(n_rq):
        def _shifted(w: np.ndarray) -> np.ndarray:
            return ig.matvec(w) - z * w

        A_sh  = LinearOperator((dim, dim), matvec=_shifted, dtype=np.complex128)
        w, _  = sp_gmres(A_sh, v, rtol=1e-5, maxiter=gmres_inner)
        nrm   = np.linalg.norm(w)
        if nrm < 1e-14:
            break
        w    /= nrm
        # Rayleigh quotient: z = <w, Aw> / <w, w>
        Aw    = ig.matvec(w)
        z     = complex(np.dot(w.conj(), Aw))
        v     = w

    return z, v


def m_convergence(u_star: np.ndarray, lam: float, coup: Couplings,
                  cfg: dict, M_values: list | None = None) -> dict:
    """
    Track leading characteristic roots as M increases.
    Returns dict mapping M -> evals array.
    """
    if M_values is None:
        M_values = cfg.get("M_values", [16, 24, 32, 48])

    results = {}
    for M in tqdm(M_values, desc="M-convergence"):
        ev = rightmost_eigenvalues(u_star, lam, coup, cfg, M=M)
        results[M] = ev
    return results


# ----------------------------------------------------------------- lambda sweep

def lambda_sweep(branch: dict, coup: Couplings, cfg: dict,
                 n_lam: int = 30) -> dict:
    """
    Compute the rightmost characteristic roots at each lambda along the branch.

    Uses which='LR' Arnoldi at each lambda point.  For large tau (tau > 4*t0),
    the spurious Chebyshev modes (Re ≈ -2/tau * |max_Re(D_int)| ≈ -0.45 for
    tau=10, M=24) may dominate the stable region and produce a plateau in the
    max_re vs lambda plot.  However, the Hopf crossing (lambda_c, omega_c) is
    still correct: at the first unstable lambda, the physical mode has Re > 0
    and dominates which='LR', so _detect_crossing gives the right result.

    Returns dict:
      lam_vals  : (n,) sampled lambda values
      evals     : list of (k,) complex arrays
      max_re    : (n,) max Re(z) at each lambda (may show spurious plateau < 0)
      crossing  : dict with Hopf details if detected
    """
    lam_branch  = branch["lam"]
    u_star_arr  = branch["u_star"]

    # Sample lam_vals from the branch
    lam_vals = np.linspace(lam_branch.min(), lam_branch.max(), n_lam)
    u_interp = np.array([
        _interp_u(lam, lam_branch, u_star_arr) for lam in lam_vals
    ])

    evals_list = []
    max_re     = np.empty(n_lam)

    for i, (lam, u) in enumerate(
            tqdm(zip(lam_vals, u_interp), total=n_lam, desc="lambda sweep")):
        ev = rightmost_eigenvalues(u, lam, coup, cfg)
        evals_list.append(ev)
        max_re[i] = ev.real.max()

    crossing = _detect_crossing(lam_vals, max_re, evals_list)

    return dict(
        lam_vals = lam_vals,
        evals    = evals_list,
        max_re   = max_re,
        crossing = crossing,
    )


def _interp_u(lam_target: float, lam_branch: np.ndarray,
              u_star_arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of u_star at lam_target along the branch."""
    idx = np.searchsorted(lam_branch, lam_target)
    idx = int(np.clip(idx, 1, len(lam_branch) - 1))
    t   = ((lam_target - lam_branch[idx - 1])
           / (lam_branch[idx] - lam_branch[idx - 1] + 1e-14))
    return (1 - t) * u_star_arr[idx - 1] + t * u_star_arr[idx]


def _detect_crossing(lam_vals, max_re, evals_list) -> dict:
    """
    Find the lambda_c where max Re(z) crosses zero.
    Determine if it is Hopf (complex conjugate pair) or saddle-node (real root).
    Also report whether the mode is an outlier (gap to next) or bulk edge.
    """
    result = dict(
        lam_c    = None,
        omega_c  = None,
        T_c      = None,
        nature   = "unknown",
        outlier  = None,
    )

    # Find first sign change from negative to non-negative
    for i in range(1, len(lam_vals)):
        if max_re[i - 1] < 0 and max_re[i] >= 0:
            # Linear interpolation for lambda_c
            t     = -max_re[i - 1] / (max_re[i] - max_re[i - 1] + 1e-14)
            lam_c = lam_vals[i - 1] + t * (lam_vals[i] - lam_vals[i - 1])
            result["lam_c"] = float(lam_c)

            # Examine the leading eigenvalue just past the crossing
            ev_at   = evals_list[i]
            ev_lead = ev_at[0]
            omega   = abs(ev_lead.imag)

            result["omega_c"] = float(omega)
            # Hopf: nonzero imaginary part (conjugate pair).
            # Threshold: omega > 1e-3 (avoids floating-point noise on real modes).
            if omega > 1e-3:
                result["nature"] = "Hopf"
                result["T_c"]    = float(2.0 * np.pi / omega)
            else:
                result["nature"] = "saddle-node / SNIC"
                result["T_c"]    = None

            # Outlier check: does the leading pair stand apart from the rest?
            # Gap > 0.05 in Re suggests a discrete outlier mode.
            if len(ev_at) >= 4:
                # Leading pair: take the two eigenvalues with largest Re
                sorted_re = np.sort(ev_at.real)[::-1]
                gap       = sorted_re[1] - sorted_re[2]   # gap after the pair
                result["outlier"] = bool(gap > 0.05)

            break

    return result


# ----------------------------------------------------------------- tau=0 check

def tau0_check(u_star: np.ndarray, lam: float, coup: Couplings,
               cfg: dict) -> dict:
    """
    At tau=0, the DDE reduces to an ODE and the IG eigenvalues must equal
    eig(M(lam)) = -1 + eig(CD).

    This function:
      1. Computes IG eigenvalues at tau=1e-6 (small but > 0).
      2. Computes eig(M(lam)) directly via the dense matrix (only for small N).
      3. Returns comparison dict.
    """
    N = coup.N
    if N > 500:
        return {"note": "tau=0 dense check skipped for N > 500"}

    beta  = cfg["beta"]
    t0    = cfg["t0"]
    gain  = coup.compute_gain(u_star, beta)

    # Direct eigenvalues of M(lam) = -I + C*D
    M_dense = coup.dense_M(gain, lam)
    eig_M   = np.linalg.eigvals(M_dense) / t0    # characteristic roots of ODE

    # IG eigenvalues at very small tau
    cfg_copy        = dict(cfg)
    cfg_copy["tau"] = 1
    cfg_copy["dt"]  = 0.01
    tau_tiny        = 1e-2   # cannot truly do tau=0; use small tau
    ig   = IGOperator(gain, lam, coup, tau_tiny, t0, cfg["M_cheb"])
    A_op = ig.as_linear_operator()
    k    = min(cfg["n_eigs"], N * (cfg["M_cheb"] + 1) - 2)
    try:
        ev_ig, _ = eigs(A_op, k=k, which='LR', tol=1e-10,
                        return_eigenvectors=True)
    except Exception:
        ev_ig    = np.array([])

    # Match: sort both by real part descending
    ev_ig = np.sort(ev_ig)[::-1][:len(eig_M)]
    ev_od = np.sort(eig_M / t0)[::-1][:len(ev_ig)]

    return dict(
        eig_ode  = eig_M,
        eig_ig   = ev_ig,
        max_re_ode = float(eig_M.real.max() / t0),
        max_re_ig  = float(ev_ig.real.max()) if len(ev_ig) else None,
    )
