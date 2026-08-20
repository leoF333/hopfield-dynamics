"""
Nonlinear DDE integrator with ring-buffer history.

Dynamics:
    t0 * du/dt = -u(t) + (1-lam) J tanh(beta u(t)) + lam K tanh(beta u(t-tau))

Integration: Euler with fixed step dt.
History buffer: circular array of shape (L+1, N), L = tau/dt (integer).
  At step n:
    u_now   = buf[n % (L+1)]
    u_delay = buf[(n - L) % (L+1)]
    u_new   = u_now + (dt/t0) * field(u_now, u_delay)
    write to buf[(n+1) % (L+1)]

MLX is used for the J and K matvecs inside field().

Measurements (after transient)
-------------------------------
  m1(t) = (1/N) xi^0 . tanh(beta u(t))   [1st pattern overlap]
  A     = (max(m1) - min(m1)) / 2         [amplitude]
  T_osc = period of m1 oscillation        [via autocorrelation peak]

Hysteresis sweep
----------------
Run increasing-lambda and decreasing-lambda sweeps; record A(lambda) on both.
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
from tqdm import tqdm

import config as cfg_mod
from couplings import Couplings


# ----------------------------------------------------------------- integrator

class DDEIntegrator:
    """
    Fixed-step Euler DDE integrator with ring-buffer history.
    Uses MLX (via coup.field_dde) for the heavy matvec operations.
    """

    def __init__(self, coup: Couplings, cfg: dict):
        self.coup  = coup
        self.N     = coup.N
        self.beta  = cfg["beta"]
        self.t0    = cfg["t0"]
        self.dt    = cfg["dt"]
        self.L     = cfg["L_buf"]          # tau / dt (integer)

    def run(self, lam: float, u0_hist: np.ndarray,
            t_total: float, callback=None) -> dict:
        """
        Integrate the DDE for t_total time units.

        Parameters
        ----------
        lam      : mixing parameter
        u0_hist  : (L+1, N) or (N,) initial history.
                   If (N,), uses constant history u0_hist throughout.
        t_total  : total integration time (s)
        callback : optional callable(step, u_now) -> None  (for recording)

        Returns
        -------
        dict with keys: u_final, m1, t_arr (if callback captures them)
        """
        N   = self.N
        L   = self.L
        dt  = self.dt
        beta= self.beta
        t0  = self.t0
        n_steps = int(t_total / dt)
        buf_len = L + 1

        # Initialize history buffer
        buf = np.empty((buf_len, N), dtype=np.float64)
        if u0_hist.ndim == 1:
            buf[:] = u0_hist[np.newaxis, :]
        else:
            assert u0_hist.shape == (buf_len, N), \
                f"u0_hist must have shape ({buf_len}, {N})"
            buf[:] = u0_hist

        # Observables
        coup   = self.coup
        xi0    = coup._xi_np[0]       # first pattern for m1 measurement
        m1_arr = np.empty(n_steps, dtype=np.float64)

        for step in range(n_steps):
            pos_now   = step      % buf_len
            pos_delay = (step - L) % buf_len
            pos_next  = (step + 1) % buf_len

            u_now   = buf[pos_now]
            u_delay = buf[pos_delay]

            # DDE right-hand side (MLX-backed)
            f = coup.field_dde(u_now, u_delay, lam, beta)

            # Euler step
            buf[pos_next] = u_now + (dt / t0) * f

            # Record m1
            m1_arr[step] = float(xi0 @ np.tanh(beta * u_now) / N)

            if callback is not None:
                callback(step, buf[pos_next])

        # Time array (one step per measurement)
        t_arr = np.arange(n_steps) * dt

        return dict(
            u_final = buf[(n_steps) % buf_len].copy(),
            buf_final = buf.copy(),      # full circular buffer at end
            m1      = m1_arr,
            t_arr   = t_arr,
        )


# ----------------------------------------------------------------- measurements

def measure_oscillation(m1: np.ndarray, dt: float,
                        t_transient: float) -> dict:
    """
    Measure amplitude A and period T_osc of m1(t) after discarding transient.

    Uses the autocorrelation to find the dominant period.
    A = (max - min) / 2  after transient.
    """
    skip = int(t_transient / dt)
    m1_steady = m1[skip:]
    if len(m1_steady) < 10:
        return dict(A=np.nan, T_osc=np.nan, mean=np.nan, fixed_point=True)

    A    = float((m1_steady.max() - m1_steady.min()) / 2.0)
    mean = float(m1_steady.mean())
    fp_threshold = 0.02

    if A < fp_threshold:
        return dict(A=float(A), T_osc=np.nan, mean=mean, fixed_point=True)

    # Period via autocorrelation
    x    = m1_steady - mean
    n    = len(x)
    corr = np.correlate(x, x, mode='full')[n - 1:]
    # normalize
    corr = corr / (corr[0] + 1e-14)

    # Find first peak after lag 1
    peaks = []
    for i in range(2, min(len(corr) - 1, n // 2)):
        if corr[i] > corr[i - 1] and corr[i] > corr[i + 1] and corr[i] > 0.2:
            peaks.append(i)
    T_osc = float(peaks[0] * dt) if peaks else np.nan

    return dict(A=float(A), T_osc=T_osc, mean=mean, fixed_point=A < fp_threshold)


# ----------------------------------------------------------------- hysteresis sweep

def sweep_single_direction(coup: Couplings, cfg: dict,
                            lam_values: np.ndarray,
                            direction: str = "up") -> dict:
    """
    Sweep lambda in one direction, recording A and T_osc at each point.

    Parameters
    ----------
    lam_values : sorted array of lambda values (ascending for 'up', descending for 'down')
    direction  : 'up' or 'down' (just for labeling)

    Returns
    -------
    dict: lam, A, T_osc, m1_mean, fixed_point
    """
    N   = coup.N
    L   = cfg["L_buf"]
    t_tr= cfg["t_transient"]
    t_ms= cfg["t_measure"]
    dt  = cfg["dt"]

    intgr = DDEIntegrator(coup, cfg)

    # Start from near-pattern-1 state
    u_init  = 0.99 * coup._xi_np[0].copy()
    buf_cur = np.tile(u_init, (L + 1, 1))

    lam_arr  = []
    A_arr    = []
    T_arr    = []
    mean_arr = []
    fp_arr   = []

    for lam in tqdm(lam_values, desc=f"sim sweep {direction}"):
        # Transient: use existing buf_cur as history
        result_tr = intgr.run(lam, buf_cur, t_total=t_tr)
        # Measure: continue from transient end
        result_ms = intgr.run(lam, result_tr["buf_final"], t_total=t_ms)

        meas = measure_oscillation(result_ms["m1"], dt, t_transient=0.0)

        lam_arr.append(float(lam))
        A_arr.append(meas["A"])
        T_arr.append(meas["T_osc"])
        mean_arr.append(meas["mean"])
        fp_arr.append(meas["fixed_point"])

        # Carry buf_final as new initial condition
        buf_cur = result_ms["buf_final"]

        print(f"  lam={lam:.3f}  A={meas['A']:.4f}  "
              f"T={meas['T_osc']!s:>8}  fp={meas['fixed_point']}", flush=True)

    return dict(
        lam       = np.array(lam_arr),
        A         = np.array(A_arr),
        T_osc     = np.array(T_arr),
        m1_mean   = np.array(mean_arr),
        fixed_point = np.array(fp_arr),
        direction = direction,
    )


def hysteresis_sweep(coup: Couplings, cfg: dict,
                     lam_lo: float = 0.1, lam_hi: float = 0.9,
                     n_pts: int = 20) -> tuple[dict, dict]:
    """
    Increasing and decreasing lambda sweeps for hysteresis detection.

    Returns (sweep_up, sweep_down).
    """
    lam_up   = np.linspace(lam_lo, lam_hi, n_pts)
    lam_down = lam_up[::-1].copy()

    print("=== Lambda sweep: increasing ===")
    up   = sweep_single_direction(coup, cfg, lam_up,   "up")
    print("=== Lambda sweep: decreasing ===")
    down = sweep_single_direction(coup, cfg, lam_down, "down")
    return up, down


# ----------------------------------------------------------------- full trajectory

def run_trajectory(lam: float, coup: Couplings, cfg: dict,
                   save_every: int = 10) -> dict:
    """
    Long trajectory near bifurcation for phase portrait and space-time raster.

    Returns dict with:
      t_arr   : (n_pts,) time values (after transient)
      m_nu    : (n_pts, n_ov) overlap time series
      u_snap  : (n_snaps, N) snapshots of u(t) for phase portraits
    """
    N     = coup.N
    n_ov  = cfg.get("n_overlaps", 5)
    beta  = cfg["beta"]
    dt    = cfg["dt"]
    t_tr  = cfg["t_transient"]
    t_ms  = cfg["t_measure"]

    intgr = DDEIntegrator(coup, cfg)
    u0    = 0.99 * coup._xi_np[0].copy()

    # Transient
    res_tr = intgr.run(lam, u0, t_total=t_tr)
    buf_after_tr = res_tr["buf_final"]

    # Measurement run — record m_nu at each step
    n_steps = int(t_ms / dt)
    m_nu_arr = np.empty((n_steps, n_ov))
    u_snaps  = []
    xi_np    = coup._xi_np[:n_ov]

    L       = cfg["L_buf"]
    buf_len = L + 1
    buf     = buf_after_tr.copy()

    for step in range(n_steps):
        pos_now   = step      % buf_len
        pos_delay = (step - L) % buf_len
        pos_next  = (step + 1) % buf_len

        u_now   = buf[pos_now]
        u_delay = buf[pos_delay]

        tu_now        = np.tanh(beta * u_now)
        m_nu_arr[step] = xi_np @ tu_now / N

        f                   = coup.field_dde(u_now, u_delay, lam, beta)
        buf[pos_next]       = u_now + (dt / cfg["t0"]) * f

        if step % save_every == 0:
            u_snaps.append(buf[pos_next].copy())

    t_arr = np.arange(n_steps) * dt
    return dict(
        t_arr   = t_arr,
        m_nu    = m_nu_arr,
        u_snaps = np.array(u_snaps),
    )


def save_sweep(sweep: dict, label: str, cfg: dict):
    """Save sweep dict to NPZ."""
    tag  = cfg_mod.param_tag(cfg)
    path = cfg_mod.out_path(cfg, "simulation", f"sweep_{label}_{tag}.npz")
    np.savez(path, **{k: v for k, v in sweep.items()
                      if isinstance(v, (np.ndarray, float, str))})
    print(f"[simulate_dde] Sweep saved → {path}")
    return path
