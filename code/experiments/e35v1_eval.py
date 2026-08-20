"""E35-V1 EVALUATION phase -- eval_seeds {101..110}, opened ONCE.

This script REFUSES to run unless the content hash recomputed from
``E35_V1_frozen_config.json`` matches the hash stored in that file and in
``E35_V1_design.npz``.  Nothing is selected here: generator, architecture,
decoder, TSVD rank, clocks, hyper-parameters and the decision rule all come
from the frozen file.

Run: python src/e35v1_eval.py
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
import time

import numpy as np

from e35v1_common import (
    ARCH_IDS, CLOCKS, DESIGN_NPZ, EVAL_NPZ, FROZEN_CONFIG, N_BOOT,
    baseline_frame_metrics, build_loop, classify_cycle, config_hash,
    decode_arm, evaluate_predictions, hierarchical_bootstrap, json_ready,
    out_of_span_residual, per_seed_block_means, prepare, run_arm,
)

BASELINES = ("B0_HOLD", "B1_LINEAR", "B2_MOTION")


def load_frozen() -> dict:
    if not FROZEN_CONFIG.exists():
        raise SystemExit(f"REFUSED: {FROZEN_CONFIG} does not exist. "
                         "Run the design phase first.")
    payload = json.loads(FROZEN_CONFIG.read_text())
    cfg, stored = payload["config"], payload["config_hash"]
    recomputed = config_hash(cfg)
    if recomputed != stored:
        raise SystemExit(
            f"REFUSED: config hash mismatch. file={stored} recomputed={recomputed}. "
            "The frozen configuration has been altered after freezing.")
    if DESIGN_NPZ.exists():
        design_hash = str(np.load(DESIGN_NPZ, allow_pickle=True)["config_hash"])
        if design_hash != stored:
            raise SystemExit(
                f"REFUSED: design npz hash {design_hash} != frozen hash {stored}.")
    else:
        raise SystemExit(f"REFUSED: {DESIGN_NPZ} missing.")
    print(f"frozen config hash CERTIFIED: {stored}")
    return cfg


def main() -> int:
    wall = time.time()
    cfg = load_frozen()
    seeds = list(cfg["eval_seeds"])
    p, k = int(cfg["p"]), int(cfg["k"])
    frozen_arch = str(cfg["architecture"])
    decoders = list(cfg["decoders_reported"])
    nominal_dec = str(cfg["decoder_nominal"])
    nominal_clock = str(cfg["clock_nominal"])
    r_star = int(cfg["tsvd_rank"])
    beta_prime = float(cfg["beta_prime"])
    gen = cfg["generator"]
    arms = [frozen_arch] if frozen_arch == "JH_KH" else [frozen_arch, "JH_KH"]
    print(f"eval seeds {seeds} | arch {frozen_arch} (+ B3=JH_KH) | "
          f"decoder {nominal_dec} r*={r_star} | clock {nominal_clock} | "
          f"t_total={cfg['t_total']}")

    n_hidden = p * (k - 1)
    n_s, n_a, n_d, n_c = len(seeds), len(arms), len(decoders), len(CLOCKS)
    dyn = {m: np.full((n_a, n_d, n_c, n_s, n_hidden), np.nan)
           for m in ("mse", "psnr", "ssim", "pearson")}
    base = {m: np.full((len(BASELINES), n_s, n_hidden), np.nan)
            for m in ("mse", "psnr", "ssim", "pearson")}
    dyn_valid = np.zeros((n_a, n_c, n_s, n_hidden), dtype=bool)
    cycle_rows, gram_rows = [], []
    residual_mean = np.zeros(n_s)
    headroom = np.zeros(n_s)
    segments = None

    for si, seed in enumerate(seeds):
        t_seed = time.time()
        frames = build_loop(seed, gen["winding"], gen["texture_sigma"], p, k)
        split, pre, patterns, context = prepare(frames, k, beta=cfg["beta"])
        segments = np.asarray(split.hidden_segment)
        _, bmet = baseline_frame_metrics(split)
        for bi, name in enumerate(BASELINES):
            for m in base:
                base[m][bi, si] = bmet[name][m]
        residual_mean[si] = float(np.mean(
            out_of_span_residual(patterns, split.hidden_frames, pre)))
        headroom[si] = bmet["B2_MOTION"]["mean_psnr"] - bmet["B1_LINEAR"]["mean_psnr"]
        d = context.diagnostics
        gram_rows.append(dict(seed=seed, rank=d.rank,
                              cond=d.condition_retained,
                              mp_residual=d.moore_penrose_residual))

        for ai, arch in enumerate(arms):
            t, a, m = run_arm(context, arch, cfg["t_total"], p,
                              lam=cfg["lam"], tau=cfg["tau"], dt=cfg["dt"],
                              t0=cfg["t0"], chunk_time=cfg["chunk_time"],
                              record_every=cfg["record_every"])
            cyc = classify_cycle(t, m,
                                 transient_fraction=cfg["transient_fraction"])
            cycle_rows.append(dict(seed=seed, architecture=arch,
                                   **cyc.to_dict()))
            out = decode_arm(t, a, m, split, patterns, pre, context,
                             tsvd_rank=r_star, beta_prime=beta_prime,
                             decoders=decoders, beta=cfg["beta"],
                             transient_fraction=cfg["transient_fraction"])
            for ci, clock in enumerate(CLOCKS):
                dyn_valid[ai, ci, si] = out[("_times", clock)][1]
                for di, dec in enumerate(decoders):
                    pred, ok = out[(dec, clock)]
                    met = evaluate_predictions(split.hidden_frames, pred,
                                               valid=ok)
                    for mm in dyn:
                        dyn[mm][ai, di, ci, si] = met[mm]
        print(f"  seed {seed} ({time.time() - t_seed:.0f} s): "
              + " | ".join(
                  f"{arms[ai]} cov="
                  f"{[r for r in cycle_rows if r['seed'] == seed and r['architecture'] == arms[ai]][0]['coverage_fraction']:.2f}"
                  for ai in range(n_a))
              + f" | B1 {bmet['B1_LINEAR']['mean_psnr']:.2f} dB "
                f"B2 {bmet['B2_MOTION']['mean_psnr']:.2f} dB "
                f"dyn({nominal_dec},{nominal_clock}) "
                f"{np.nanmean(dyn['psnr'][0, decoders.index(nominal_dec), CLOCKS.index(nominal_clock), si]):.2f} dB")

    # ------------------------------------------------------- paired statistics
    b1_psnr = base["psnr"][BASELINES.index("B1_LINEAR")]
    b2_psnr = base["psnr"][BASELINES.index("B2_MOTION")]
    stats = {}
    for ci, clock in enumerate(CLOCKS):
        for di, dec in enumerate(decoders):
            dyn_psnr = dyn["psnr"][0, di, ci]          # frozen arm
            for ref_name, ref in (("B1", b1_psnr), ("B2", b2_psnr)):
                blocks = [per_seed_block_means(dyn_psnr[si] - ref[si], segments)
                          for si in range(n_s)]
                blocks = [b[np.isfinite(b)] for b in blocks]
                point, lo, hi, _ = hierarchical_bootstrap(blocks, n_boot=N_BOOT)
                per_seed = np.array([np.nanmean(b) for b in blocks])
                stats[f"{dec}|{clock}|vs{ref_name}"] = dict(
                    mean=point, ci_low=lo, ci_high=hi,
                    seeds_favorable=int(np.sum(per_seed > 0)), n_seeds=n_s,
                    per_seed=per_seed.tolist(),
                    blocks_favorable=int(sum(int(np.sum(b > 0)) for b in blocks)),
                    n_blocks=int(sum(len(b) for b in blocks)),
                )

    # ------------------------------------------------------- verdict (frozen 5)
    frozen_cycles = [r for r in cycle_rows if r["architecture"] == frozen_arch]
    valid_fraction = float(np.mean([r["valid_cycle"] for r in frozen_cycles]))
    key1 = f"{nominal_dec}|{nominal_clock}|vsB1"
    key2 = f"{nominal_dec}|{nominal_clock}|vsB2"
    s1, s2 = stats[key1], stats[key2]
    cond_cycle = valid_fraction >= 0.80
    cond_b1 = (s1["mean"] > 0.0) and (s1["ci_low"] > 0.0)
    cond_seeds = s1["seeds_favorable"] >= 8
    not_below_b2 = s2["ci_high"] >= 0.0
    if not cond_cycle:
        verdict = "INDETERMINE"
    elif cond_b1 and cond_seeds and not_below_b2:
        verdict = "GO_STRICT"
    elif cond_b1 and cond_seeds:
        verdict = "GO_FAIBLE"
    else:
        verdict = "NO_GO"
    print("\n" + "=" * 78)
    print(f"FROZEN RULE (section 5) on {nominal_dec} / {nominal_clock}:")
    print(f"  (1) cycle valid on {valid_fraction*100:.0f}% of eval seeds "
          f"(>=80% required) -> {cond_cycle}")
    print(f"  (2) dPSNR(dyn-B1) = {s1['mean']:+.3f} dB "
          f"CI95 [{s1['ci_low']:+.3f},{s1['ci_high']:+.3f}] -> {cond_b1}")
    print(f"  (3) dPSNR(dyn-B2) = {s2['mean']:+.3f} dB "
          f"CI95 [{s2['ci_low']:+.3f},{s2['ci_high']:+.3f}] "
          f"-> not below B2: {not_below_b2}")
    print(f"  (4) seeds favorable vs B1 = {s1['seeds_favorable']}/{n_s} "
          f"(>=8 required) -> {cond_seeds}")
    print(f"VERDICT V1 = {verdict}")
    print("The V0 NO-GO (orbit POC) is CONSERVED regardless of this verdict.")
    print("=" * 78)

    print(f"\n{'arm':<14}{'decoder':<16}{'clock':<8}{'MSE':>11}{'PSNR':>9}"
          f"{'SSIM':>8}{'Pearson':>9}{'valid':>8}")
    for ai, arch in enumerate(arms):
        for di, dec in enumerate(decoders):
            for ci, clock in enumerate(CLOCKS):
                lab = "dyn" if ai == 0 else "B3"
                print(f"{lab + ':' + arch:<14}" + f"{dec:<16}{clock:<8}"
                      f"{np.nanmean(dyn['mse'][ai, di, ci]):>11.6f}"
                      f"{np.nanmean(dyn['psnr'][ai, di, ci]):>9.3f}"
                      f"{np.nanmean(dyn['ssim'][ai, di, ci]):>8.4f}"
                      f"{np.nanmean(dyn['pearson'][ai, di, ci]):>9.4f}"
                      f"{int(np.count_nonzero(dyn_valid[ai, ci])):>8d}")
    for bi, name in enumerate(BASELINES):
        print(f"{'--':<14}{name:<16}{'--':<8}"
              f"{np.nanmean(base['mse'][bi]):>11.6f}"
              f"{np.nanmean(base['psnr'][bi]):>9.3f}"
              f"{np.nanmean(base['ssim'][bi]):>8.4f}"
              f"{np.nanmean(base['pearson'][bi]):>9.4f}")

    np.savez_compressed(
        EVAL_NPZ,
        eval_seeds=np.array(seeds), arms=np.array(arms),
        decoders=np.array(decoders), clocks=np.array(CLOCKS),
        baselines=np.array(BASELINES),
        hidden_segment=segments,
        dyn_mse=dyn["mse"], dyn_psnr=dyn["psnr"], dyn_ssim=dyn["ssim"],
        dyn_pearson=dyn["pearson"], dyn_valid=dyn_valid,
        base_mse=base["mse"], base_psnr=base["psnr"], base_ssim=base["ssim"],
        base_pearson=base["pearson"],
        residual_mean=residual_mean, headroom_db=headroom,
        cycle_json=json.dumps(json_ready(cycle_rows)),
        gram_json=json.dumps(json_ready(gram_rows)),
        stats_json=json.dumps(json_ready(stats)),
        verdict=verdict, valid_fraction=valid_fraction,
        frozen_architecture=frozen_arch,
        nominal_decoder=nominal_dec, nominal_clock=nominal_clock,
        config_hash=config_hash(cfg),
        config_json=json.dumps(cfg, sort_keys=True),
        n_boot=N_BOOT, wall_seconds=time.time() - wall,
    )
    print(f"\nsaved {EVAL_NPZ}\nwall {time.time() - wall:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
