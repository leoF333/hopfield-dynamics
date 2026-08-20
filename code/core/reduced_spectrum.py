"""
Exact reduced characteristic spectrum of the DDE (no pseudospectral artefacts).

Because J D and K D have rank <= P, the characteristic determinant factorizes
(Weinstein-Aronszajn / Sylvester):
    det Delta(z) = (t0 z + 1)^{N-P} * det T_P(z),
    T_P(z) = (t0 z + 1) I_P - (1-lam) G_J - lam e^{-z tau} G_K,
    G_J = (1/N) xi D xi^T,   G_K = (1/N) xi D xi_shift^T   (P x P),
where D = diag(gain) at the fixed point. The (t0 z+1)^{N-P} factor is the trivial
bulk root z = -1/t0 (deep stable). All PHYSICAL characteristic roots are the
solutions of the P x P NONLINEAR eigenvalue problem det T_P(z) = 0. This has NO
spurious Chebyshev modes (there is no discretization), so it shows only the true
physics-carrying eigenvalues.

Roots are found by Beyn's contour-integral method (2012): the eigenvalues of the
analytic matrix T_P inside a contour Gamma are recovered from
    A_0 = 1/(2 pi i) oint T_P(z)^{-1} Vhat dz,   A_1 = 1/(2 pi i) oint z T_P(z)^{-1} Vhat dz,
via an SVD of A_0. Circular contour + trapezoidal rule = spectral accuracy.
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


def GJ_GK(coup, u, beta, lam):
    """P x P reduced coupling matrices G_J, G_K at the fixed point u."""
    xi = coup._xi_np; xis = coup._xis_np; N = coup.N
    gain = coup.compute_gain(u, beta)          # (N,)
    xg = xi * gain[None, :]                     # (P,N) = xi D
    GJ = (xg @ xi.T) / N                        # (P,P)
    GK = (xg @ xis.T) / N                       # (P,P)
    return GJ, GK


def T_P(z, GJ, GK, t0, tau, lam):
    """Reduced characteristic matrix T_P(z) (P x P)."""
    P = GJ.shape[0]
    return (t0 * z + 1.0) * np.eye(P) - (1.0 - lam) * GJ - lam * np.exp(-z * tau) * GK


def beyn_roots(GJ, GK, t0, tau, lam, center=0.0 + 0j, R=2.0, Nq=200, L=50,
               svd_tol=1e-10, seed=0):
    """
    Physical characteristic roots inside the circle |z-center|=R, via Beyn.
    Returns them sorted by decreasing Re(z). L must exceed the number of roots in
    the disk (a warning-free run has rank k < L).
    """
    P = GJ.shape[0]
    rng = np.random.default_rng(seed)
    Vhat = rng.standard_normal((P, L)) + 1j * rng.standard_normal((P, L))

    theta = 2.0 * np.pi * np.arange(Nq) / Nq
    z = center + R * np.exp(1j * theta)
    dz = 1j * R * np.exp(1j * theta) * (2.0 * np.pi / Nq)

    A0 = np.zeros((P, L), complex)
    A1 = np.zeros((P, L), complex)
    for zq, dzq in zip(z, dz):
        Y = np.linalg.solve(T_P(zq, GJ, GK, t0, tau, lam), Vhat)   # T^{-1} Vhat
        A0 += Y * dzq
        A1 += zq * Y * dzq
    A0 /= (2j * np.pi); A1 /= (2j * np.pi)

    W, s, Sh = np.linalg.svd(A0, full_matrices=False)
    k = int(np.sum(s > svd_tol * max(s[0], 1e-300)))
    if k == 0:
        return np.array([], complex), False
    saturated = (k >= L)                        # too many roots for this L
    Wk = W[:, :k]; sk = s[:k]; Sk = Sh[:k, :].conj().T
    B = Wk.conj().T @ A1 @ Sk @ np.diag(1.0 / sk)
    ev = np.linalg.eigvals(B)
    ev = ev[np.argsort(-ev.real)]
    return ev, saturated


def rightmost_physical_roots(coup, u, beta, lam, tau, t0, k=8,
                             center=-0.4 + 0j, R=1.6, Nq=200, L=60):
    """Convenience: the k rightmost physical roots at (u, lam, tau)."""
    GJ, GK = GJ_GK(coup, u, beta, lam)
    ev, sat = beyn_roots(GJ, GK, t0, tau, lam, center=center, R=R, Nq=Nq, L=L)
    return ev[:k], sat


def sigmin_TP(z, GJ, GK, t0, tau, lam):
    """Smallest singular value of the reduced characteristic matrix T_P(z)."""
    return float(np.linalg.svd(T_P(z, GJ, GK, t0, tau, lam), compute_uv=False).min())


def _polish_root(z, GJ, GK, t0, tau, lam, iters=6):
    """Refine a candidate root by residual inverse iteration on T_P(z)v=0.
    Uses the smallest-singular-vector as the (left/right) null estimate and a
    secant/Newton step on the smallest singular value along the real+imag plane.
    Cheap and robust for isolated roots; returns the polished z."""
    zc = complex(z)
    for _ in range(iters):
        T = T_P(zc, GJ, GK, t0, tau, lam)
        U, s, Vh = np.linalg.svd(T)
        vr = Vh[-1].conj()                 # right null vector
        ul = U[:, -1]                       # left null vector
        # dT/dz = t0 I + lam*tau*e^{-z tau} GK
        dT = t0 * np.eye(GJ.shape[0]) + lam * tau * np.exp(-zc * tau) * GK
        num = ul.conj() @ (T @ vr)          # ~ s_min (residual)
        den = ul.conj() @ (dT @ vr)         # d(residual)/dz
        if abs(den) < 1e-30:
            break
        step = num / den
        zc = zc - step
        if abs(step) < 1e-12:
            break
    return zc


def reduced_ig_candidates(GJ, GK, t0, tau, lam, M=24, k=40):
    """
    Rightmost eigenvalues of the REDUCED P-dim DDE whose characteristic matrix is
    exactly T_P:   t0 y' = -y + (1-lam) G_J y(t) + lam G_K y(t-tau).
    Pseudospectral IG operator of dimension P(M+1) (small, since it uses the P x P
    G-matrices, not the full N). Returns candidate roots (physical + reduced-model
    spurious); the spurious are filtered by sigma_min(T_P) in physical_roots().
    """
    from dde_stability import cheb_diff_matrix
    from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence
    P = GJ.shape[0]
    Dch, _ = cheb_diff_matrix(M)
    Dint = (2.0 / tau) * Dch[1:, :]           # (M, M+1)
    dim = P * (M + 1)

    def mv(vec):
        V = vec.reshape(M + 1, P)
        W = np.empty((M + 1, P), dtype=complex)
        W[1:, :] = Dint @ V
        v0 = V[0]; vM = V[M]
        W[0, :] = (-v0 + (1.0 - lam) * (GJ @ v0) + lam * (GK @ vM)) / t0
        return W.reshape(-1)

    op = LinearOperator((dim, dim), matvec=mv, dtype=complex)
    ncv = min(max(2 * k + 4, 60), dim - 1)
    try:
        vals, _ = eigs(op, k=k, which="LR", ncv=ncv, maxiter=800, tol=1e-7,
                       return_eigenvectors=True)
    except ArpackNoConvergence as e:
        vals = e.eigenvalues if len(e.eigenvalues) else np.array([-10.0 + 0j])
    except Exception:
        vals = np.array([-10.0 + 0j])
    return vals[np.argsort(-vals.real)]


def physical_roots(coup, u, beta, lam, tau, t0, M=24, n_cand=40,
                   thresh=0.05, k=10, polish=True):
    """
    The k rightmost PHYSICAL characteristic roots (exact, no artefacts). Candidates
    from the reduced P-dim IG operator; kept iff sigma_min(T_P(z)) is a genuine root
    (small vs the scale |t0 z+1|); spurious (sigma_min ~ O(1)) rejected; kept roots
    polished on T_P. Returns (physical_roots_sorted, candidates, sigmin/scale).
    """
    GJ, GK = GJ_GK(coup, u, beta, lam)
    cand = reduced_ig_candidates(GJ, GK, t0, tau, lam, M=M, k=n_cand)
    phys = []; sigs = []
    for z in cand:
        scale = max(abs(t0 * z + 1.0), 1e-6)
        s = sigmin_TP(z, GJ, GK, t0, tau, lam)
        sigs.append(s / scale)
        if s < thresh * scale:
            zp = _polish_root(z, GJ, GK, t0, tau, lam) if polish else complex(z)
            phys.append(zp)
    phys = np.array(phys, complex)
    if len(phys):
        phys = phys[np.argsort(-phys.real)]
        keep = []                                   # drop near-duplicates
        for z in phys:
            if not any(abs(z - w) < 1e-5 for w in keep):
                keep.append(z)
        phys = np.array(keep, complex)
    return phys[:k], cand, np.asarray(sigs)


def rightmost_real_root(GJ, GK, t0, tau, lam, x_hi=0.10, x_lo=-1.2, n=180):
    """Rightmost REAL characteristic root via a sigma_min(T_P(x)) scan on the real
    axis. sigma_min (from SVD) is overflow-free -- UNLIKE det/slogdet, which blows
    up for a 500x500 matrix. A real root is a local minimum of sigma_min that dips
    to ~0; scan from x_hi downward and return the first such (rightmost), polished
    and verified. Guarantees the real fold-mode even if it sits below the reduced-IG
    candidates."""
    xs = np.linspace(x_hi, x_lo, n)
    sig = np.array([sigmin_TP(x, GJ, GK, t0, tau, lam) for x in xs])
    for i in range(1, n - 1):
        if sig[i] <= sig[i - 1] and sig[i] <= sig[i + 1]:      # local minimum
            z = _polish_root(complex(xs[i], 0.0), GJ, GK, t0, tau, lam)
            scale = max(abs(t0 * z.real + 1.0), 1e-6)
            if abs(z.imag) < 1e-3 and \
               sigmin_TP(z.real, GJ, GK, t0, tau, lam) < 1e-3 * scale:
                return complex(z.real, 0.0)                     # genuine real root
    return None


def rightmost_real_and_complex(coup, u, beta, lam, tau, t0,
                               im_tol=1e-3, **kw):
    """
    Return (re_real, re_cplx, im_at_cplx): the rightmost real-part among real
    roots (|Im|<im_tol) and among complex roots (|Im|>=im_tol). For the Hopf test.
    """
    GJ, GK = GJ_GK(coup, u, beta, lam)
    ev, sat = beyn_roots(GJ, GK, t0, tau, lam, **kw)
    if len(ev) == 0:
        return -np.inf, -np.inf, 0.0, sat
    isc = np.abs(ev.imag) >= im_tol
    re_real = float(ev.real[~isc].max()) if np.any(~isc) else -np.inf
    re_cplx = -np.inf; im_c = 0.0
    if np.any(isc):
        j = int(np.argmax(np.where(isc, ev.real, -np.inf)))
        re_cplx = float(ev.real[j]); im_c = float(abs(ev.imag[j]))
    return re_real, re_cplx, im_c, sat
