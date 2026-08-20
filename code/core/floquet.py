"""
Floquet multipliers of the pacemaker periodic orbit.

Method
------
The linearized DDE around a T-periodic orbit u*(t):
    t0 * dw/dt = -w(t) + (1-lam) J D(t) w(t) + lam K D(t-tau) w(t-tau)

D(t)     = diag(beta*(1 - tanh^2(beta u*(t))))
D(t-tau) = diag(beta*(1 - tanh^2(beta u*(t-tau))))

The monodromy operator Phi(T) maps an initial function segment
  (w(s), s in [-tau,0])  ->  (w(T+s), s in [-tau,0])

Floquet multipliers = eigenvalues of Phi(T).
Trivial multiplier mu=1 (from phase shift) is a correctness check.

Implementation
--------------
1. Run DDE to obtain a periodic orbit stored as buf_orbit (L+1, N) at
   a reference time, plus the orbit period T (from simulation).
2. Build the monodromy matvec:
   - Given initial segment v = (v_0,...,v_M) in R^{N(M+1)} (pseudospectral),
     reconstruct the history buffer w_hist of length L+1.
   - Integrate the variational DDE (Euler) for one period T,
     reading u*(t) and u*(t-tau) from the stored periodic orbit.
   - Return the resulting history segment as the monodromy output.
3. Apply Arnoldi (eigs with which='LM') to get leading multipliers.

Limitation: storing the full N*(L_period+1) orbit is O(N*T/dt) memory.
For N=10^4, T=100, dt=0.1: 10^6 entries ~ 8 MB — acceptable.
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
from tqdm import tqdm

from couplings import Couplings
import config as cfg_mod


def store_orbit(lam: float, coup: Couplings, cfg: dict,
                buf_initial: np.ndarray, T_period: float) -> np.ndarray:
    """
    Integrate the DDE for one period T_period starting from buf_initial,
    storing the full trajectory as orbit_buf of shape (L_period+1, N).

    buf_initial : (L+1, N) history buffer at the start of the period
    Returns     : orbit_buf (L_period+1, N)  [u*(t), t in [0, T_period]]
    """
    N    = coup.N
    L    = cfg["L_buf"]
    dt   = cfg["dt"]
    beta = cfg["beta"]
    t0   = cfg["t0"]

    n_period = int(round(T_period / dt))
    orbit    = np.empty((n_period + 1, N), dtype=np.float64)

    buf     = buf_initial.copy()
    buf_len = L + 1

    for step in range(n_period + 1):
        pos_now   = step      % buf_len
        pos_delay = (step - L) % buf_len
        pos_next  = (step + 1) % buf_len

        u_now = buf[pos_now]
        orbit[step] = u_now.copy()

        if step < n_period:
            u_delay = buf[pos_delay]
            f       = coup.field_dde(u_now, u_delay, lam, beta)
            buf[pos_next] = u_now + (dt / t0) * f

    return orbit


def _monodromy_matvec(v_flat: np.ndarray, orbit: np.ndarray,
                      lam: float, coup: Couplings, cfg: dict,
                      T_period: float) -> np.ndarray:
    """
    Apply Phi(T) to v_flat = flattened (L+1, N) initial segment.
    Integrate variational DDE over one period using the stored orbit.
    Returns flattened output segment of same shape.
    """
    N    = coup.N
    L    = cfg["L_buf"]
    dt   = cfg["dt"]
    beta = cfg["beta"]
    t0   = cfg["t0"]

    n_period = int(round(T_period / dt))
    L_orbit  = len(orbit) - 1  # should equal n_period
    buf_len  = L + 1

    # Initialize variational history buffer
    w_buf = v_flat.reshape(buf_len, N).copy()

    for step in range(n_period):
        pos_now   = step      % buf_len
        pos_delay = (step - L) % buf_len
        pos_next  = (step + 1) % buf_len

        w_now   = w_buf[pos_now]
        w_delay = w_buf[pos_delay]

        # Gain at current and delayed base orbit
        u_now   = orbit[step % (L_orbit + 1)]
        u_delay = orbit[(step - L) % (L_orbit + 1)]
        gain_now   = coup.compute_gain(u_now,   beta)
        gain_delay = coup.compute_gain(u_delay, beta)

        # Variational equation field:
        # t0 dw/dt = -w + (1-lam) J (gain_now * w_now) + lam K (gain_delay * w_delay)
        f = (-w_now
             + (1.0 - lam) * coup.apply_J(gain_now   * w_now)
             +        lam  * coup.apply_K(gain_delay  * w_delay))
        w_buf[pos_next] = w_now + (dt / t0) * f

    # Return the updated history buffer
    # The output is the segment (w(T+s), s in [-tau,0]) = w_buf in its current state
    # We need to reorder: pos_next+1 = start of the newest segment
    result = np.empty_like(w_buf)
    for j in range(buf_len):
        result[j] = w_buf[(n_period + 1 + j - buf_len) % buf_len]
    return result.reshape(-1)


def compute_floquet_multipliers(orbit: np.ndarray, lam: float,
                                coup: Couplings, cfg: dict,
                                T_period: float, k: int = 10) -> np.ndarray:
    """
    Compute the leading Floquet multipliers of the periodic orbit via Arnoldi.

    Parameters
    ----------
    orbit    : (n_period+1, N) stored periodic orbit from store_orbit()
    lam      : mixing parameter
    coup     : Couplings
    cfg      : config dict
    T_period : period of the orbit
    k        : number of multipliers to compute

    Returns
    -------
    multipliers : (k,) complex128, sorted by |mu| descending
    """
    L    = cfg["L_buf"]
    N    = coup.N
    dim  = (L + 1) * N

    def mv(v):
        if np.iscomplexobj(v):
            real_part = _monodromy_matvec(v.real, orbit, lam, coup, cfg, T_period)
            imag_part = _monodromy_matvec(v.imag, orbit, lam, coup, cfg, T_period)
            return real_part + 1j * imag_part
        return _monodromy_matvec(v, orbit, lam, coup, cfg, T_period)

    A_op = LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)
    ncv  = min(max(3 * k, 30), dim - 1)

    try:
        mults, _ = eigs(A_op, k=k, which='LM', ncv=ncv, maxiter=200,
                        tol=1e-8, return_eigenvectors=True)
    except Exception as e:
        print(f"[floquet] eigs failed: {e}")
        return np.array([])

    idx = np.argsort(-np.abs(mults))
    return mults[idx]


def floquet_from_simulation(lam: float, coup: Couplings, cfg: dict,
                            T_period: float, buf_on_orbit: np.ndarray,
                            k: int = 10) -> dict:
    """
    Full Floquet pipeline given a point on the orbit.

    Parameters
    ----------
    T_period    : period in time units
    buf_on_orbit: (L+1, N) history buffer at the start of a period

    Returns
    -------
    dict with multipliers, |mu|, trivial_check
    """
    print(f"[floquet] Storing orbit (T={T_period:.2f}, "
          f"{int(round(T_period/cfg['dt']))+1} steps) ...", flush=True)
    orbit = store_orbit(lam, coup, cfg, buf_on_orbit, T_period)

    print(f"[floquet] Computing {k} Floquet multipliers (dim={(cfg['L_buf']+1)*coup.N}) ...",
          flush=True)
    mults = compute_floquet_multipliers(orbit, lam, coup, cfg, T_period, k=k)

    abs_mu = np.abs(mults)
    trivial = float(abs_mu.max())   # should be ~1 for a genuine limit cycle

    return dict(
        multipliers  = mults,
        abs_mu       = abs_mu,
        trivial_check = trivial,
        T_period     = T_period,
        lam          = lam,
    )


def save_floquet(result: dict, cfg: dict):
    tag  = cfg_mod.param_tag(cfg)
    lam  = result.get("lam", 0.0)
    path = cfg_mod.out_path(cfg, "floquet",
                            f"floquet_{tag}_lam{lam:.3f}.npz")
    np.savez(path,
             multipliers  = result["multipliers"],
             abs_mu       = result["abs_mu"],
             trivial_check= np.array([result["trivial_check"]]),
             T_period     = np.array([result["T_period"]]),
             lam          = np.array([lam]))
    print(f"[floquet] Saved → {path}")
    return path
