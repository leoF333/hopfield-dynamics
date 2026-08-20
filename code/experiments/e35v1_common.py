"""Shared machinery for the E35-V1 gate (design != evaluation).

Frozen protocol: roadmaps/E35_V1_PROTOCOL_DRAFT.md (sections 1-8 + appendix A).
Everything here is target-free: no function reads hidden frames except the
explicit metric evaluators, which are only called by the reporting layer.

Language convention: prose FR in the .md deliverables, code/labels/prints EN.
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

DATA_DIR = ROOT / "results" / "8_mhn_video" / "data"
FIG_DIR = ROOT / "results" / "8_mhn_video" / "figures"
FROZEN_CONFIG = DATA_DIR / "E35_V1_frozen_config.json"
DESIGN_NPZ = DATA_DIR / "E35_V1_design.npz"
EVAL_NPZ = DATA_DIR / "E35_V1_eval.npz"

from e35_video_tools import (  # noqa: E402
    FramePreprocessor,
    loop_displacement_statistics,
    make_nonlinear_advection_loop,
    seam_diagnostics,
    split_keyframes,
)
from mhn_reduced import MhnContext, factorial_architectures  # noqa: E402
from e35_dynamics import (  # noqa: E402
    MhnReducedDDE,
    build_baseline_predictions,
    classify_cycle,
    evaluate_predictions,
    extract_peak_intervals,
    integrate_chunked,
)

# ------------------------------------------------------------------ protocol
DESIGN_SEEDS = (1, 2, 3, 4, 5, 6)          # protocol 1
EVAL_SEEDS = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110)
ARCH_IDS = ("JH_KH", "JP_KH", "JH_KP", "JP_KP")   # protocol 3 / R2.2
CLOCKS = ("CLK_U", "CLK_P")                        # A.5
BETA_PRIME_GRID = (5.0, 10.0, 20.0, 40.0)          # A.5
TSVD_SIGMAS = (0.1, 0.2, 0.3)                      # A.5 (D0 rule)
TSVD_NTRIALS = 8
TSVD_RNG_SEED = 20260725
BOOT_RNG_SEED = 20260725                           # A.7
N_BOOT = 10000

# A.6 fixed dynamics hyper-parameters
LAM = 0.9
TAU = 10.0
BETA = 20.0
DT = 0.05
T0 = 1.0
K_STEP = 4
HEIGHT = 64
WIDTH = 64
RECORD_EVERY = 1
TRANSIENT_FRACTION = 0.5
CHUNK_TIME = 25.0
TSVD_RTOL = None                                   # R2.5 machine Moore-Penrose

# A.3 pre-declared generator grid (fixed non-swept parameters)
WINDING_GRID = ((1, 1), (2, 1), (3, 2))
TEXTURE_SIGMA_GRID = (1.5, 2.5, 3.5)
CURVE_AMPLITUDE = (6.0, 4.0)
SPEED_MODULATION = 0.35
CONTRAST = 1.0

# protocol 2 frozen headroom criteria (never weakened)
HEADROOM_DB_MIN = 3.0
RESIDUAL_MIN = 0.05


# ------------------------------------------------------------------ helpers
def config_hash(cfg: dict) -> str:
    """Content hash over the frozen configuration (sorted JSON, sha256/16)."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    return obj


def build_loop(seed: int, winding, texture_sigma: float, p: int, k: int):
    """Render one closed nonlinear-advection loop (P*k frames, [0,1])."""
    return make_nonlinear_advection_loop(
        p * k,
        height=HEIGHT,
        width=WIDTH,
        winding=tuple(winding),
        curve_amplitude=CURVE_AMPLITUDE,
        speed_modulation=SPEED_MODULATION,
        texture_sigma=float(texture_sigma),
        contrast=CONTRAST,
        seed=int(seed),
    )


def prepare(frames, k, beta=BETA):
    """Keyframe split, keyframe-only preprocessor, patterns, Gram context."""
    split = split_keyframes(frames, k)
    pre = FramePreprocessor.fit_from_keyframes(split.keyframes, binary=False)
    patterns = pre.encode(split.keyframes).reshape(len(split.keyframes), -1)
    context = MhnContext.from_patterns(patterns, beta, tsvd_rtol=TSVD_RTOL)
    return split, pre, patterns, context


def out_of_span_residual(patterns, hidden_frames, pre):
    """A.2 relative out-of-span residual of the encoded hidden frames.

    ``residual = ||v - P_span(X) v||_2 / ||v||_2`` with ``X`` the keyframe
    pattern matrix.  Uses hidden frames only as a reported CHARACTERISTIC of
    the generator, on design seeds (protocol 1).
    """
    xi = np.asarray(patterns, dtype=np.float64)          # (P, N)
    _, s, vt = np.linalg.svd(xi, full_matrices=False)
    tol = np.finfo(np.float64).eps * max(xi.shape) * float(s[0])
    basis = vt[s > tol]                                   # (r, N) orthonormal
    v = pre.encode(hidden_frames).reshape(len(hidden_frames), -1)
    proj = (v @ basis.T) @ basis
    num = np.linalg.norm(v - proj, axis=1)
    den = np.maximum(np.linalg.norm(v, axis=1), np.finfo(float).tiny)
    return num / den


def baseline_frame_metrics(split):
    """B0/B1/B2 per-hidden-frame metrics (endpoint-only baselines, R2.1)."""
    preds = build_baseline_predictions(split)
    return preds, {name: evaluate_predictions(split.hidden_frames, pred)
                   for name, pred in preds.items()}


def run_arm(context, arch_id, t_total, p, *, lam=LAM, tau=TAU, dt=DT,
            t0=T0, chunk_time=CHUNK_TIME, record_every=RECORD_EVERY):
    """Integrate one J x K arm and return ``(t, a, m)``.

    ``a0 = 0.95 e_0 + 0.05 e_1`` (A.6, identical to V0).
    """
    arch = factorial_architectures(context)[arch_id]
    system = MhnReducedDDE(arch, lam, tau, t0)
    a0 = np.zeros(p, dtype=np.float64)
    a0[0], a0[1] = 0.95, 0.05
    sol = integrate_chunked(system, a0, da_hist0=np.zeros_like(a0),
                            t_total=float(t_total), dt=dt,
                            record_every=record_every, chunk_time=chunk_time)
    t, a = sol["t"], sol["a"]
    m = context.physical_overlaps_batch(a)
    return t, a, m


def truncated_pinv_apply(context, rank, vector):
    """Apply the rank-r truncated pseudoinverse ``G^dagger_r`` to a P-vector."""
    u = context.singular_vectors[:, :rank]
    s = context.singular_values[:rank]
    return u @ ((u.T @ vector) / s)


def _noisy_key_recalls(split, patterns, beta, rng):
    """Noisy KEYFRAME recalls only (D0 section 4 / A.5).  Never touches targets.

    Returns ``(m_noisy, key_index)`` with ``m_noisy.shape == (P*3*8, P)``.  One
    single noise realisation is shared by every candidate rank / beta' so that
    the comparison is paired; the D0 selection rule itself is unchanged.
    """
    xi = np.asarray(patterns, dtype=np.float64)
    p, n = xi.shape
    key_index = np.repeat(np.arange(p), len(TSVD_SIGMAS) * TSVD_NTRIALS)
    sigma_col = np.tile(
        np.repeat(np.asarray(TSVD_SIGMAS), TSVD_NTRIALS), p)[:, None]
    out = np.empty((len(key_index), p), dtype=np.float64)
    step = 300
    for start in range(0, len(key_index), step):
        stop = min(len(key_index), start + step)
        u = xi[key_index[start:stop]] + sigma_col[start:stop] * \
            rng.standard_normal((stop - start, n))
        np.tanh(beta * u, out=u)
        out[start:stop] = (u @ xi.T) / n
    return out, key_index


def _key_recon_mse(coeffs, patterns, keyframes, key_index):
    """Mean MSE of ``decode(X coeff)`` against the true keyframes (batched)."""
    xi = np.asarray(patterns, dtype=np.float64)
    flat_keys = keyframes.reshape(len(keyframes), -1)
    total, count = 0.0, 0
    step = 300
    for start in range(0, len(coeffs), step):
        stop = min(len(coeffs), start + step)
        rec = coeffs[start:stop] @ xi
        rec = np.clip(0.5 * (rec + 1.0), 0.0, 1.0)
        rec -= flat_keys[key_index[start:stop]]
        total += float(np.sum(rec * rec))
        count += rec.size
    return total / count


def select_tsvd_rank(split, pre, patterns, context, p, beta=BETA, rng_seed=None):
    """D0 section 4 rule: rank from NOISY KEYFRAME recalls only, never targets."""
    rng = np.random.default_rng(
        TSVD_RNG_SEED if rng_seed is None else rng_seed)
    m_noisy, key_index = _noisy_key_recalls(split, patterns, beta, rng)
    ranks = np.arange(1, p + 1)
    curve = np.zeros(len(ranks))
    u_all = context.singular_vectors
    s_all = context.singular_values
    for ri, r in enumerate(ranks):
        u = u_all[:, :r]
        coeffs = ((m_noisy @ u) / s_all[:r]) @ u.T
        curve[ri] = _key_recon_mse(coeffs, patterns, split.keyframes, key_index)
    best = np.flatnonzero(curve <= curve.min() + 1e-9)
    return int(ranks[best[0]]), curve, ranks


def select_beta_prime(split, pre, patterns, context, p, rng_seed=None):
    """Prespecified readout-temperature scan, same keyframe-only rule (A.5)."""
    curve = np.zeros(len(BETA_PRIME_GRID))
    base = TSVD_RNG_SEED if rng_seed is None else rng_seed
    for bi, bprime in enumerate(BETA_PRIME_GRID):
        rng = np.random.default_rng(base + 1000)
        m_noisy, key_index = _noisy_key_recalls(split, patterns, bprime, rng)
        coeffs = context.apply_pinv(m_noisy.T).T
        curve[bi] = _key_recon_mse(coeffs, patterns, split.keyframes, key_index)
    return float(BETA_PRIME_GRID[int(np.argmin(curve))]), curve


# ------------------------------------------------------------------ readout
def _latest_intervals(t, m, transient_fraction=TRANSIENT_FRACTION):
    latest = {}
    for iv in extract_peak_intervals(t, m, transient_fraction=transient_fraction):
        latest[iv.source] = iv
    return latest


def reading_time(latest, t, m, mu, offset, k, clock, p):
    """CLK-U uniform time / CLK-P prespecified overlap phase (A.5, R2.3)."""
    iv = latest.get(int(mu))
    if iv is None:
        return None
    q = float(offset) / k
    if clock == "CLK_U":
        return iv.start_time + q * (iv.end_time - iv.start_time)
    lo, hi = iv.start_index, iv.end_index + 1
    tt = t[lo:hi]
    if tt.size == 0:
        return None
    nxt = (int(mu) + 1) % p
    num = np.clip(m[lo:hi, nxt], 0.0, None)
    den = np.clip(m[lo:hi, int(mu)], 0.0, None) + num
    phi = np.maximum.accumulate(np.where(den > 1e-12, num / den, 0.0))
    j = int(np.searchsorted(phi, q))
    if j == 0:
        return float(tt[0])
    if j >= len(tt):
        return float(tt[-1])
    p0, p1 = phi[j - 1], phi[j]
    if p1 <= p0:
        return float(tt[j - 1])
    return float(tt[j - 1] + (q - p0) / (p1 - p0) * (tt[j] - tt[j - 1]))


def interp_state(t, a, ts):
    """Linear interpolation of the coefficient trajectory at one time.

    Vectorised over the P columns (``np.interp`` per column would rescan the
    whole trajectory P times per hidden frame, which dominates the run).
    """
    j = int(np.searchsorted(t, ts))
    if j <= 0:
        return np.array(a[0], copy=True)
    if j >= len(t):
        return np.array(a[-1], copy=True)
    t0, t1 = t[j - 1], t[j]
    if t1 <= t0:
        return np.array(a[j - 1], copy=True)
    w = (ts - t0) / (t1 - t0)
    return (1.0 - w) * a[j - 1] + w * a[j]


def decode_arm(t, a, m, split, patterns, pre, context, *, tsvd_rank,
               beta_prime, decoders, clocks=CLOCKS, beta=BETA,
               transient_fraction=TRANSIENT_FRACTION):
    """Decode every hidden frame for each (decoder, clock) of one arm.

    Returns ``{(decoder, clock): (predictions, valid)}``.  Hidden target pixels
    are never read here.
    """
    xi = np.asarray(patterns, dtype=np.float64)
    p, n = xi.shape
    shape = split.keyframes.shape[1:]
    seg = np.asarray(split.hidden_segment)
    off = np.asarray(split.hidden_offset)
    n_hidden = len(split.hidden_indices)
    latest = _latest_intervals(t, m, transient_fraction)

    def a_at(ts):
        return interp_state(t, a, ts)

    def m_at(a_star, bprime):
        return xi @ np.tanh(bprime * (xi.T @ a_star)) / n

    out = {}
    for clock in clocks:
        times = np.full(n_hidden, np.nan)
        states = np.zeros((n_hidden, p))
        ok = np.zeros(n_hidden, dtype=bool)
        for idx, (mu, offset) in enumerate(zip(seg, off)):
            ts = reading_time(latest, t, m, mu, offset, split.k, clock, p)
            if ts is None:
                continue
            times[idx] = ts
            states[idx] = a_at(ts)
            ok[idx] = True
        for dec in decoders:
            pred = np.full((n_hidden,) + shape, np.nan)
            for idx in np.flatnonzero(ok):
                a_star = states[idx]
                if dec == "D1_Xa":
                    coeff = a_star
                elif dec == "D2c_XGdm_clip":
                    coeff = context.apply_pinv(m_at(a_star, beta))
                elif dec == "D3_TSVD":
                    coeff = truncated_pinv_apply(
                        context, int(tsvd_rank), m_at(a_star, beta))
                elif dec == "D4_betaprime":
                    coeff = context.apply_pinv(m_at(a_star, beta_prime))
                else:
                    raise ValueError(f"unknown decoder {dec!r}")
                pred[idx] = pre.decode((xi.T @ coeff).reshape(shape))
            out[(dec, clock)] = (pred, ok.copy())
        out[("_times", clock)] = (times, ok.copy())
    return out


# ------------------------------------------------------------------ bootstrap
def per_seed_block_means(delta_frames, segments):
    """Mean paired difference per transition block, for one seed."""
    blocks = np.unique(segments)
    return np.array([np.nanmean(delta_frames[segments == b]) for b in blocks])


def hierarchical_bootstrap(block_means_per_seed, n_boot=N_BOOT,
                           rng_seed=BOOT_RNG_SEED):
    """A.7: resample seeds with replacement, then transitions within seed."""
    rng = np.random.default_rng(rng_seed)
    n_seeds = len(block_means_per_seed)
    seed_means = np.array([np.nanmean(b) for b in block_means_per_seed])
    point = float(np.nanmean(seed_means))
    draws = np.empty(n_boot)
    for it in range(n_boot):
        pick = rng.integers(0, n_seeds, size=n_seeds)
        vals = np.empty(n_seeds)
        for j, s in enumerate(pick):
            blocks = block_means_per_seed[s]
            sub = rng.integers(0, len(blocks), size=len(blocks))
            vals[j] = np.nanmean(blocks[sub])
        draws[it] = np.nanmean(vals)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), draws
