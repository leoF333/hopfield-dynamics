"""E35-V1 figures (design characterisation + evaluation verdict).

Outputs (dpi >= 170, English labels):
  results/8_mhn_video/figures/E35_V1_design.png
  results/8_mhn_video/figures/E35_V1_eval.png

Run: python src/e35v1_figures.py
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from e35v1_common import (  # noqa: E402
    ARCH_IDS, CLOCKS, DESIGN_NPZ, EVAL_NPZ, FIG_DIR, HEADROOM_DB_MIN,
    RESIDUAL_MIN,
)

DPI = 180
C_DYN = "#1b7837"
C_B0 = "#8c8c8c"
C_B1 = "#d95f02"
C_B2 = "#7570b3"
C_B3 = "#a6611a"


def figure_design() -> None:
    d = np.load(DESIGN_NPZ, allow_pickle=True)
    table = json.loads(str(d["grid_table_json"]))
    factorial = json.loads(str(d["factorial_json"]))
    scores = json.loads(str(d["arch_scores_json"]))
    chosen = json.loads(str(d["chosen_json"]))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (a) generator grid: headroom vs out-of-span residual
    ax = axes[0, 0]
    is_frozen = lambda e: (list(e["winding"]) == list(chosen["winding"])
                           and e["texture_sigma"] == chosen["texture_sigma"])
    seen = {"pass": False, "fail": False, "frozen": False}
    # Spread the labels inside each winding group (same headroom, close in x):
    # leftmost point labelled to its left, middle above, rightmost to its right.
    place = {}
    for w in {tuple(e["winding"]) for e in table}:
        grp = sorted([e for e in table if tuple(e["winding"]) == w],
                     key=lambda e: e["mean_resid"])
        for e, spec in zip(grp, [(-9, 0, "right"), (0, 11, "center"),
                                 (9, 0, "left")]):
            place[(w, e["texture_sigma"])] = spec
    for e in table:
        if is_frozen(e):
            ax.scatter(e["mean_resid"], e["headroom"], s=320, marker="*",
                       c="#ffd166", edgecolors="k", linewidths=1.1, zorder=5,
                       label=None if seen["frozen"] else "frozen setting")
            seen["frozen"] = True
        elif e["pass_both"]:
            ax.scatter(e["mean_resid"], e["headroom"], s=85, marker="o",
                       c=C_DYN, edgecolors="k", zorder=4,
                       label=None if seen["pass"] else "meets both criteria")
            seen["pass"] = True
        else:
            ax.scatter(e["mean_resid"], e["headroom"], s=70, marker="x",
                       c="#9a9a9a", zorder=3,
                       label=None if seen["fail"] else "rejected")
            seen["fail"] = True
        dx, dy, ha = place[(tuple(e["winding"]), e["texture_sigma"])]
        ax.annotate(f"w={tuple(e['winding'])}, s={e['texture_sigma']:g}",
                    (e["mean_resid"], e["headroom"]), fontsize=6.5, ha=ha,
                    va="center" if dy == 0 else "bottom",
                    xytext=(dx, dy + (3 if is_frozen(e) else 0)),
                    textcoords="offset points")
    ax.axhline(HEADROOM_DB_MIN, color=C_B1, ls="--", lw=1.2,
               label=f"frozen: B2-B1 >= {HEADROOM_DB_MIN} dB")
    ax.axvline(RESIDUAL_MIN, color=C_B2, ls="--", lw=1.2,
               label=f"frozen: residual >= {RESIDUAL_MIN}")
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 1.2)
    ax.set_ylim(-3.6, 13.8)
    ax.set_xlabel("mean out-of-span residual of hidden frames (A.2, relative)")
    ax.set_ylabel("headroom B2 - B1 (dB)")
    ax.set_title("(a) Generator grid on design seeds (A.3), frozen criteria")
    ax.legend(fontsize=6.8, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)

    # (b) J x K factorial cycle coverage
    ax = axes[0, 1]
    width = 0.35
    xs = np.arange(len(ARCH_IDS))
    cov = [scores[a]["coverage"] for a in ARCH_IDS]
    fwd = [scores[a]["forward"] for a in ARCH_IDS]
    ax.bar(xs - width / 2, cov, width, label="coverage fraction", color=C_DYN)
    ax.bar(xs + width / 2, fwd, width, label="forward fraction", color=C_B2)
    for i, a in enumerate(ARCH_IDS):
        vf = scores[a]["valid_fraction"]
        ax.text(i, 1.03, f"valid cycle\n{vf*100:.0f}% of seeds", ha="center",
                fontsize=6.8)
    ax.axhline(0.95, color=C_B1, ls=":", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(ARCH_IDS)
    ax.set_ylim(0, 1.62)
    ax.set_ylabel("mean over the 6 design seeds (fraction)")
    ax.set_title("(b) J x K factorial, A.4 score (no image metric)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.25, axis="y")

    # (c) TSVD rank curve (keyframe-only)
    ax = axes[1, 0]
    ranks = d["tsvd_rank_grid"]
    curve = d["tsvd_rank_curve"]
    r_star = int(d["tsvd_rank_selected"])
    ax.semilogy(ranks, curve, color="k", lw=1.4)
    ax.axvline(r_star, color=C_DYN, lw=1.6,
               label=f"selected r* = {r_star}")
    ax.scatter([ranks[-1]], [curve[-1]], color=C_B1, zorder=4,
               label=f"full rank P={ranks[-1]}")
    ax.set_xlabel("TSVD rank r")
    ax.set_ylabel("keyframe reconstruction MSE (noisy recalls)")
    ax.set_title("(c) Decoder calibration D0 rule: keyframes only")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    # (d) beta' readout scan
    ax = axes[1, 1]
    bg = d["beta_prime_grid"]
    bc = d["beta_prime_curve"]
    ax.plot(bg, bc, "o-", color="k", lw=1.4)
    b_star = float(d["beta_prime_selected"])
    ax.axvline(b_star, color=C_DYN, lw=1.6, label=f"selected beta' = {b_star:g}")
    ax.set_xlabel("readout temperature beta'")
    ax.set_ylabel("keyframe reconstruction MSE")
    ax.set_title("(d) Prespecified beta' scan (reported, not promotable)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"E35-V1 DESIGN (design seeds only) -- frozen generator: nonlinear "
        f"advection, winding={tuple(chosen['winding'])}, "
        f"texture_sigma={chosen['texture_sigma']:g}, "
        f"headroom {chosen['headroom']:.2f} dB, residual {chosen['mean_resid']:.4f}"
        f"  |  config hash {str(d['config_hash'])}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = FIG_DIR / "E35_V1_design.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved {out}")


def figure_eval() -> None:
    d = np.load(EVAL_NPZ, allow_pickle=True)
    seeds = d["eval_seeds"]
    decoders = [str(x) for x in d["decoders"]]
    arms = [str(x) for x in d["arms"]]
    stats = json.loads(str(d["stats_json"]))
    cycles = json.loads(str(d["cycle_json"]))
    nom_dec = str(d["nominal_decoder"])
    nom_clk = str(d["nominal_clock"])
    di, ci = decoders.index(nom_dec), list(CLOCKS).index(nom_clk)
    seg = d["hidden_segment"]

    dyn_psnr = d["dyn_psnr"][0, di, ci]        # (seeds, hidden)
    b3_psnr = d["dyn_psnr"][-1, di, ci] if len(arms) > 1 else dyn_psnr
    base_psnr = d["base_psnr"]                 # (3, seeds, hidden)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (a) per-seed mean PSNR
    ax = axes[0, 0]
    x = np.arange(len(seeds))
    with np.errstate(invalid="ignore"):
        b3_mean = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan
                            for r in b3_psnr])
        dyn_mean = np.nanmean(dyn_psnr, axis=1)
        b0_mean = np.nanmean(base_psnr[0], axis=1)
    ax.fill_between(x, dyn_mean, b0_mean, color=C_DYN, alpha=0.12, zorder=1)
    for name, arr, col in [("B2 motion", base_psnr[2], C_B2),
                           ("B1 linear", base_psnr[1], C_B1),
                           ("B0 hold", base_psnr[0], C_B0)]:
        ax.plot(x, np.nanmean(arr, axis=1), "o-", color=col, label=name, lw=1.5,
                ms=4, zorder=3)
    ax.plot(x, dyn_mean, "o-", color=C_DYN, lw=1.8, ms=4, zorder=4,
            label=f"dyn {arms[0]} (frozen)")
    # B3 = JH_KH is undecodable on most seeds: markers only, never a line that
    # would suggest an interpolated value where no frame could be decoded.
    n_b3_seeds = int(np.isfinite(b3_mean).sum())
    n_b3_frames = int(np.isfinite(b3_psnr).sum())
    ax.plot(x, b3_mean, "D", color=C_B3, ms=5, zorder=4,
            label=f"B3 Hebb JH_KH ({n_b3_seeds}/{len(seeds)} seeds decodable)")
    ax.annotate(
        f"B3 decodable on only {n_b3_frames}/{b3_psnr.size} hidden frames\n"
        f"(no marker = no valid recall interval on that seed)",
        xy=(0.02, 0.03), xycoords="axes fraction", fontsize=6.5, color=C_B3)
    ax.annotate("dyn below B0 hold", xy=(x[len(x) // 2], 0.5 * (
        dyn_mean[len(x) // 2] + b0_mean[len(x) // 2])), fontsize=7,
        color=C_DYN, ha="center", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=7)
    ax.set_xlabel("evaluation seed")
    ax.set_ylabel("mean PSNR over hidden frames (dB)")
    ax.set_ylim(6.5, 51.0)
    ax.set_title(f"(a) Per-seed mean PSNR ({nom_dec}, {nom_clk})")
    ax.legend(fontsize=6.8, loc="upper center", ncol=2)
    ax.grid(alpha=0.25)

    # (b) paired dPSNR with hierarchical CIs
    ax = axes[0, 1]
    labels, means, los, his, cols = [], [], [], [], []
    for clock in CLOCKS:
        for dec in decoders:
            for ref, col in (("B1", C_B1), ("B2", C_B2)):
                s = stats[f"{dec}|{clock}|vs{ref}"]
                labels.append(f"{dec.split('_')[0]}/{clock[-1]} vs {ref}")
                means.append(s["mean"])
                los.append(s["mean"] - s["ci_low"])
                his.append(s["ci_high"] - s["mean"])
                cols.append(col)
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=[los, his], fmt="o", ms=4, lw=1.2,
                ecolor="k", mfc="none", ls="none")
    for i, c in enumerate(cols):
        ax.scatter([means[i]], [y[i]], color=c, zorder=4, s=28)
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("paired dPSNR (dB), hierarchical 95% CI (seeds then transitions)")
    ax.set_title("(b) Dynamic readout vs B1 (orange) and B2 (purple)")
    ax.grid(alpha=0.25, axis="x")

    # (c) per-transition paired difference, nominal decoder
    ax = axes[1, 0]
    blocks = np.unique(seg)
    for si in range(len(seeds)):
        db1 = np.array([np.nanmean((dyn_psnr[si] - base_psnr[1, si])[seg == b])
                        for b in blocks])
        ax.plot(blocks, db1, lw=0.7, alpha=0.55, color=C_B1)
    mean_db1 = np.nanmean([[np.nanmean((dyn_psnr[si] - base_psnr[1, si])[seg == b])
                            for b in blocks] for si in range(len(seeds))], axis=0)
    ax.plot(blocks, mean_db1, color="k", lw=1.8, label="mean over eval seeds")
    ax.axhline(0.0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("transition block (bootstrap unit, R2.6)")
    ax.set_ylabel("dPSNR(dyn - B1) (dB)")
    ax.set_title(f"(c) Paired difference per transition ({nom_dec}, {nom_clk})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (d) cycle diagnostics on eval seeds
    ax = axes[1, 1]
    frozen_arch = str(d["frozen_architecture"])
    cov = [r["coverage_fraction"] for r in cycles if r["architecture"] == frozen_arch]
    fwd = [r["forward_fraction"] for r in cycles if r["architecture"] == frozen_arch]
    cov3 = [r["coverage_fraction"] for r in cycles if r["architecture"] == "JH_KH"]
    w = 0.27
    ax.bar(x - w, cov, w, color=C_DYN, label=f"{frozen_arch} coverage")
    ax.bar(x, fwd, w, color=C_B2, label=f"{frozen_arch} forward")
    if cov3:
        ax.bar(x + w, cov3, w, color=C_B3, label="JH_KH coverage (B3)")
    ax.axhline(0.95, color=C_B1, ls=":", lw=1.0, label="0.95 fwd threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=7)
    ax.set_xlabel("evaluation seed")
    ax.set_ylabel("fraction of the P=100 stored keyframes")
    ax.set_ylim(0, 1.42)
    ax.set_title(f"(d) Cycle validity on eval seeds "
                 f"(valid {float(d['valid_fraction'])*100:.0f}%, >=80% required)")
    ax.legend(fontsize=6.5, loc="upper center", ncol=4, framealpha=0.95)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"E35-V1 EVALUATION (eval seeds {int(seeds[0])}-{int(seeds[-1])}, opened "
        f"once) -- VERDICT {str(d['verdict'])}  |  frozen arch {frozen_arch}, "
        f"{nom_dec}, {nom_clk}, config hash {str(d['config_hash'])}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = FIG_DIR / "E35_V1_eval.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved {out}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if DESIGN_NPZ.exists():
        figure_design()
    if EVAL_NPZ.exists():
        figure_eval()
    return 0


if __name__ == "__main__":
    sys.exit(main())
