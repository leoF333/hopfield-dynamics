"""E35-D0 readout diagnostics on the stored JP_KP V0 trajectory (no new dataset).

Replays the JP_KP arm of the certified E35c production POC exactly (same config
and seed as the production npz), certifies the replay against the stored physical
overlaps, then evaluates a FROZEN list of prespecified decoders (readouts of the
same trajectory) under two prespecified clocks, computes MSE/PSNR/SSIM per hidden
frame, the pseudoinverse amplification trace, a keyframe-only TSVD rank selection,
a readout beta scan, and transition-block bootstrap CIs.

Protocol frozen in roadmaps/E35_D0_PROTOCOL.md BEFORE any hidden-frame metric.
The historical V0 NO-GO (JP_KP quality << B1/B2) is conserved regardless.

Language convention: prose FR in docs, code/labels/prints EN.
Run: /opt/homebrew/Caskroom/miniforge/base/envs/mcmc_env/bin/python src/e35d0_readout_diagnostics.py
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from e35_video_tools import FramePreprocessor, make_cyclic_loop, split_keyframes  # noqa: E402
from mhn_reduced import MhnContext, factorial_architectures  # noqa: E402
from e35_dynamics import (  # noqa: E402
    MhnReducedDDE,
    integrate_chunked,
    extract_peak_intervals,
    evaluate_predictions,
)

# ---- single-process, at most a couple of threads (E34/E36 own most cores) ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

NPZ = sorted(glob.glob(os.path.join(
    ROOT, "results/8_mhn_video/data/E35c_production_*.npz")))[-1]
OUT_NPZ = os.path.join(ROOT, "results/8_mhn_video/data/E35_D0_diagnostics.npz")

RNG_SEED = 20260724
TSVD_RANKS = np.arange(1, 17)
TSVD_SIGMAS = (0.1, 0.2, 0.3)
TSVD_NTRIALS = 8
BETA_GRID = (5.0, 10.0, 20.0, 40.0)
N_BOOT = 10000
REF_PSNR = 28.05874523047549   # D1 CLK-U reference (stored JP_KP)
REF_MSE = 0.0016309371360412525


def config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def truncated_pinv_apply(context, ranks_keep, vector):
    """Apply G^{dagger}_r (rank-r truncated pseudoinverse) to a P-vector."""
    u = context.singular_vectors[:, :ranks_keep]           # (P, r)
    s = context.singular_values[:ranks_keep]               # (r,)
    return u @ ((u.T @ vector) / s)


def main() -> int:
    t_wall = time.time()
    d = np.load(NPZ, allow_pickle=True)
    cfg = json.loads(str(d["config_json"]))
    print("npz:", os.path.basename(NPZ))
    print("config:", {k: cfg[k] for k in
                      ("p", "k", "lam", "tau", "dt", "t_total", "kind",
                       "seed", "binary", "beta", "tsvd_rtol")})
    chash = config_hash(cfg)
    print("config_hash:", chash)

    # ---- rebuild inputs exactly as e35c.run() ----
    frames = make_cyclic_loop(cfg["p"] * cfg["k"], height=cfg["height"],
                              width=cfg["width"], kind=cfg["kind"], seed=cfg["seed"])
    assert np.allclose(frames, d["frames"]), "video mismatch"
    split = split_keyframes(frames, cfg["k"])
    pre = FramePreprocessor.fit_from_keyframes(split.keyframes, binary=cfg["binary"])
    patterns = pre.encode(split.keyframes).reshape(cfg["p"], -1)
    assert np.allclose(patterns, d["patterns"]), "patterns mismatch"
    context = MhnContext.from_patterns(patterns, cfg["beta"],
                                       tsvd_rtol=cfg.get("tsvd_rtol"))
    diag = context.diagnostics
    print(f"Gram: rank={diag.rank} cond={diag.condition_retained:.3e} "
          f"MP_residual={diag.moore_penrose_residual:.3e}")

    arch = factorial_architectures(context)["JP_KP"]
    system = MhnReducedDDE(arch, cfg["lam"], cfg["tau"], cfg["t0"])
    a0 = np.zeros(cfg["p"]); a0[0] = 0.95; a0[1] = 0.05
    sol = integrate_chunked(system, a0, da_hist0=np.zeros_like(a0),
                            t_total=cfg["t_total"], dt=cfg["dt"],
                            record_every=cfg["record_every"],
                            chunk_time=cfg["chunk_time"])
    t, a = sol["t"], sol["a"]
    m = context.physical_overlaps_batch(a)
    i_arch = list(d["architecture"]).index("JP_KP")
    replay_err_m = float(np.max(np.abs(m - d["physical_overlaps"][i_arch])))
    print(f"REPLAY CHECK: max|m - stored m| = {replay_err_m:.3e}")
    if replay_err_m > 1e-9:
        print("STOP: replay certification failed (> 1e-9). No metrics produced.")
        return 1

    P = cfg["p"]; k = cfg["k"]; shape = split.keyframes.shape[1:]
    beta = cfg["beta"]; xi = patterns  # (P, N)
    truth = split.hidden_frames  # (48, H, W) in [0,1]
    n_hidden = len(split.hidden_indices)
    seg = np.asarray(split.hidden_segment)      # transition block index per frame
    off = np.asarray(split.hidden_offset)

    # ---- keyframe-only TSVD rank selection (D3) ----
    rng = np.random.default_rng(RNG_SEED)
    N = xi.shape[1]
    rank_mse = np.zeros(len(TSVD_RANKS))
    for ri, r in enumerate(TSVD_RANKS):
        errs = []
        rng_r = np.random.default_rng(RNG_SEED + int(r))
        for mu in range(P):
            key_true = split.keyframes[mu]        # [0,1]
            base = xi[mu]                          # [-1,1] pattern
            for sigma in TSVD_SIGMAS:
                for _ in range(TSVD_NTRIALS):
                    u_noisy = base + sigma * rng_r.standard_normal(N)
                    m_noisy = xi @ np.tanh(beta * u_noisy) / N
                    coeff = truncated_pinv_apply(context, r, m_noisy)
                    frame_dec = pre.decode((xi.T @ coeff).reshape(shape))
                    errs.append(np.mean((frame_dec - key_true) ** 2))
        rank_mse[ri] = float(np.mean(errs))
    best = np.where(rank_mse <= rank_mse.min() + 1e-9)[0]
    tsvd_rank_selected = int(TSVD_RANKS[best[0]])   # smallest r on tie
    print(f"TSVD rank selection (keyframes only): r* = {tsvd_rank_selected} "
          f"(min keyframe-recon MSE = {rank_mse.min():.6e})")

    # ---- keyframe-only beta selection for D4 ----
    beta_key_mse = np.zeros(len(BETA_GRID))
    for bi, bprime in enumerate(BETA_GRID):
        errs = []
        rng_b = np.random.default_rng(RNG_SEED + 1000 + int(bprime))
        for mu in range(P):
            key_true = split.keyframes[mu]
            base = xi[mu]
            for sigma in TSVD_SIGMAS:
                for _ in range(TSVD_NTRIALS):
                    u_noisy = base + sigma * rng_b.standard_normal(N)
                    m_noisy = xi @ np.tanh(bprime * u_noisy) / N
                    coeff = context.apply_pinv(m_noisy)
                    frame_dec = pre.decode((xi.T @ coeff).reshape(shape))
                    errs.append(np.mean((frame_dec - key_true) ** 2))
        beta_key_mse[bi] = float(np.mean(errs))
    beta_scan_selected = float(BETA_GRID[int(np.argmin(beta_key_mse))])
    print(f"Readout beta selection (keyframes only): beta* = {beta_scan_selected} "
          f"among {BETA_GRID}")

    # ---- per-clock reading times, then per-decoder readouts ----
    intervals = extract_peak_intervals(
        t, m, transient_fraction=cfg["transient_fraction"])
    latest = {}
    for it in intervals:
        latest[it.source] = it

    def a_at(t_star):
        return np.array([np.interp(t_star, t, a[:, c]) for c in range(P)])

    def m_at(a_star, bprime):
        return xi @ np.tanh(bprime * (xi.T @ a_star)) / N

    def reading_time(mu, offset, clock):
        it = latest.get(int(mu))
        if it is None:
            return None
        q = float(offset) / k
        if clock == "CLK_U":
            return it.start_time + q * (it.end_time - it.start_time)
        # CLK_P overlap phase
        sel = (t >= it.start_time) & (t <= it.end_time)
        tt = t[sel]
        mu_n = (int(mu) + 1) % P
        num = np.clip(m[sel, mu_n], 0, None)
        den = np.clip(m[sel, int(mu)], 0, None) + num
        phi = np.maximum.accumulate(np.where(den > 1e-12, num / den, 0.0))
        j = int(np.searchsorted(phi, q))
        if j == 0:
            return tt[0]
        if j >= len(tt):
            return tt[-1]
        p0, p1 = phi[j - 1], phi[j]
        return tt[j - 1] if p1 <= p0 else tt[j - 1] + (q - p0) / (p1 - p0) * (tt[j] - tt[j - 1])

    decoders = ["D1_Xa", "D2c_XGdm_clip", "D2n_XGdm_noclip", "D3_TSVD",
                f"D4_beta{int(beta_scan_selected)}", "D5_norm2"]
    clocks = ["CLK_U", "CLK_P"]

    def decode_frame(dec, a_star, mu, mu_n):
        if dec == "D1_Xa":
            return pre.decode((xi.T @ a_star).reshape(shape))
        if dec.startswith("D2c"):
            coeff = context.apply_pinv(m_at(a_star, beta))
            return pre.decode((xi.T @ coeff).reshape(shape))
        if dec.startswith("D2n"):
            coeff = context.apply_pinv(m_at(a_star, beta))
            val = 0.5 * ((xi.T @ coeff).reshape(shape) + 1.0)  # NO clipping
            return val
        if dec.startswith("D3"):
            coeff = truncated_pinv_apply(context, tsvd_rank_selected, m_at(a_star, beta))
            return pre.decode((xi.T @ coeff).reshape(shape))
        if dec.startswith("D4"):
            coeff = context.apply_pinv(m_at(a_star, beta_scan_selected))
            return pre.decode((xi.T @ coeff).reshape(shape))
        if dec.startswith("D5"):
            c2 = np.zeros(P); s = a_star[mu] + a_star[mu_n]
            if s > 1e-9:
                c2[mu], c2[mu_n] = a_star[mu] / s, a_star[mu_n] / s
            return pre.decode((xi.T @ c2).reshape(shape))
        raise ValueError(dec)

    n_dec, n_clk = len(decoders), len(clocks)
    mse = np.full((n_dec, n_clk, n_hidden), np.nan)
    psnr = np.full((n_dec, n_clk, n_hidden), np.nan)
    ssim = np.full((n_dec, n_clk, n_hidden), np.nan)

    for ci, clock in enumerate(clocks):
        for di, dec in enumerate(decoders):
            pred = np.full((n_hidden,) + shape, np.nan)
            valid = np.zeros(n_hidden, bool)
            for out, (mu, offset) in enumerate(zip(seg, off)):
                ts = reading_time(mu, offset, clock)
                if ts is None:
                    continue
                a_star = a_at(ts)
                mu_n = (int(mu) + 1) % P
                pred[out] = decode_frame(dec, a_star, int(mu), mu_n)
                valid[out] = True
            met = evaluate_predictions(truth, pred, valid=valid)
            mse[di, ci] = met["mse"]
            psnr[di, ci] = met["psnr"]
            ssim[di, ci] = met["ssim"]

    # ---- D1 CLK_U certification vs stored ----
    d1u = i_arch  # stored JP_KP is D1 uniform-time
    stored_psnr = np.nanmean(d["dynamic_psnr"][d1u])
    stored_mse = np.nanmean(d["dynamic_mse"][d1u])
    repro_psnr = np.nanmean(psnr[0, 0])
    repro_mse = np.nanmean(mse[0, 0])
    print(f"D1 CLK_U reproduction: PSNR {repro_psnr:.3f} (stored {stored_psnr:.3f}), "
          f"MSE {repro_mse:.6f} (stored {stored_mse:.6f})")

    # ---- amplification trace along the full stored trajectory ----
    m_stored = d["physical_overlaps"][i_arch]                  # (samples, P)
    gdm = context.apply_pinv(m_stored.T).T                     # (samples, P)
    num = np.linalg.norm(gdm, axis=1)
    den = np.linalg.norm(m_stored, axis=1)
    amp = num / np.maximum(den, np.finfo(float).tiny)
    # transition mask: within any peak-to-peak interval
    trans_mask = np.zeros(len(t), bool)
    for it in intervals:
        trans_mask[it.start_index:it.end_index + 1] = True
    amp_summary = dict(
        median=float(np.median(amp)), q05=float(np.quantile(amp, 0.05)),
        q95=float(np.quantile(amp, 0.95)), max=float(np.max(amp)),
        mean_transition=float(np.mean(amp[trans_mask])) if trans_mask.any() else float("nan"),
        mean_plateau=float(np.mean(amp[~trans_mask])) if (~trans_mask).any() else float("nan"),
    )
    print("amplification |G^dag m|/|m|:", {kk: round(vv, 3) for kk, vv in amp_summary.items()})

    # ---- baselines per frame ----
    b1_psnr = d["B1_LINEAR_psnr"]; b2_psnr = d["B2_MOTION_psnr"]

    # ---- transition-block bootstrap ----
    blocks = np.unique(seg)
    rng_boot = np.random.default_rng(RNG_SEED)

    def block_means(delta_frame):
        return np.array([np.nanmean(delta_frame[seg == b]) for b in blocks])

    def boot_ci(delta_frame):
        dblk = block_means(delta_frame)
        idx = rng_boot.integers(0, len(dblk), size=(N_BOOT, len(dblk)))
        boot = np.nanmean(dblk[idx], axis=1)
        return (float(np.nanmean(dblk)), float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
                int(np.sum(dblk > 0)), len(dblk))

    boot_vs_B1 = np.zeros((n_dec, n_clk, 5))
    boot_vs_B2 = np.zeros((n_dec, n_clk, 5))
    boot_vs_ref = np.zeros((n_dec, n_clk, 5))
    fav_B1 = np.zeros((n_dec, n_clk), int)
    fav_B2 = np.zeros((n_dec, n_clk), int)
    ref_frame_psnr = psnr[0, 0]  # D1 CLK_U per-frame
    for di in range(n_dec):
        for ci in range(n_clk):
            b1 = boot_ci(psnr[di, ci] - b1_psnr)
            b2 = boot_ci(psnr[di, ci] - b2_psnr)
            rf = boot_ci(psnr[di, ci] - ref_frame_psnr)
            boot_vs_B1[di, ci] = b1[:3] + (b1[3], b1[4])
            boot_vs_B2[di, ci] = b2[:3] + (b2[3], b2[4])
            boot_vs_ref[di, ci] = rf[:3] + (rf[3], rf[4])
            fav_B1[di, ci] = b1[3]
            fav_B2[di, ci] = b2[3]

    mean_psnr = np.nanmean(psnr, axis=2)
    mean_mse = np.nanmean(mse, axis=2)
    mean_ssim = np.nanmean(ssim, axis=2)

    # ---- gate verdict ----
    verdict = "NEGATIVE"
    best_line = None
    for di in range(n_dec):
        for ci in range(n_clk):
            dpsnr = mean_psnr[di, ci] - REF_PSNR
            dmse = mean_mse[di, ci] - REF_MSE
            ci_lo = boot_vs_ref[di, ci, 1]
            if dpsnr >= 3.0 and dmse <= 0 and ci_lo > 0:
                verdict = "POSITIVE"
                best_line = (decoders[di], clocks[ci], dpsnr, dmse)
    if verdict != "POSITIVE":
        limited = False
        for di in range(n_dec):
            for ci in range(n_clk):
                if mean_psnr[di, ci] - REF_PSNR >= 1.0:
                    limited = True
                    if best_line is None:
                        best_line = (decoders[di], clocks[ci],
                                     mean_psnr[di, ci] - REF_PSNR,
                                     mean_mse[di, ci] - REF_MSE)
        verdict = "LIMITED" if limited else "NEGATIVE"
    print(f"\nGATE D0 VERDICT: {verdict}  best={best_line}")

    print(f"\n{'decoder':<20}{'clock':<8}{'MSE':>12}{'PSNR':>9}{'SSIM':>8}"
          f"{'favB1':>7}{'favB2':>7}")
    for di in range(n_dec):
        for ci in range(n_clk):
            print(f"{decoders[di]:<20}{clocks[ci]:<8}{mean_mse[di, ci]:>12.6f}"
                  f"{mean_psnr[di, ci]:>9.3f}{mean_ssim[di, ci]:>8.4f}"
                  f"{fav_B1[di, ci]:>7d}{fav_B2[di, ci]:>7d}")
    print(f"\nB0_HOLD  MSE {np.mean(d['B0_HOLD_mse']):.6f} PSNR "
          f"{np.mean(d['B0_HOLD_psnr']):.3f} SSIM {np.mean(d['B0_HOLD_ssim']):.4f}")
    print(f"B1_LINEAR MSE {np.mean(d['B1_LINEAR_mse']):.6f} PSNR "
          f"{np.mean(b1_psnr):.3f} SSIM {np.mean(d['B1_LINEAR_ssim']):.4f}")
    print(f"B2_MOTION MSE {np.mean(d['B2_MOTION_mse']):.6f} PSNR "
          f"{np.mean(b2_psnr):.3f} SSIM {np.mean(d['B2_MOTION_ssim']):.4f}")

    wall = time.time() - t_wall
    np.savez_compressed(
        OUT_NPZ,
        decoders=np.array(decoders), clocks=np.array(clocks),
        mse=mse, psnr=psnr, ssim=ssim,
        mean_mse=mean_mse, mean_psnr=mean_psnr, mean_ssim=mean_ssim,
        hidden_segment=seg, hidden_offset=off,
        amplification_trace=amp, amplification_time=t,
        amplification_transition_mask=trans_mask,
        amplification_summary=json.dumps(amp_summary),
        tsvd_rank_curve=rank_mse, tsvd_rank_grid=TSVD_RANKS,
        tsvd_rank_selected=tsvd_rank_selected,
        beta_scan_grid=np.array(BETA_GRID), beta_scan_key_mse=beta_key_mse,
        beta_scan_selected=beta_scan_selected,
        boot_ci_vs_B1=boot_vs_B1, boot_ci_vs_B2=boot_vs_B2,
        boot_ci_vs_ref=boot_vs_ref,
        favorable_blocks_vs_B1=fav_B1, favorable_blocks_vs_B2=fav_B2,
        B0_psnr=d["B0_HOLD_psnr"], B1_psnr=b1_psnr, B2_psnr=b2_psnr,
        B0_mse=d["B0_HOLD_mse"], B1_mse=d["B1_LINEAR_mse"], B2_mse=d["B2_MOTION_mse"],
        gram_rank=diag.rank, gram_cond=diag.condition_retained,
        gram_mp_residual=diag.moore_penrose_residual,
        replay_err_m=replay_err_m, verdict=verdict,
        ref_psnr=REF_PSNR, ref_mse=REF_MSE,
        config_hash=chash, source_npz=os.path.basename(NPZ),
        wall_seconds=wall,
    )
    print(f"\nsaved: {OUT_NPZ}")
    print(f"wall time: {wall:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
