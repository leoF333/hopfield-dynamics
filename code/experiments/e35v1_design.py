"""E35-V1 DESIGN phase (design_seeds {1..6} ONLY).

Stages (frozen protocol roadmaps/E35_V1_PROTOCOL_DRAFT.md, appendix A):
  headroom : characterise the pre-declared generator grid A.3 on the design
             seeds (B2-B1 headroom, out-of-span residual A.2, cond(G),
             geometric keyframe displacement).  No dynamics, no decision on a
             dynamic performance.
  select   : viability pre-screen in the A.3 parsimony order, then the full
             J x K factorial (A.4 cycle-coverage score) on the frozen setting.
  freeze   : keyframe-only decoder calibration (TSVD r*, beta' scan, A.5),
             write E35_V1_frozen_config.json (content hash) and
             E35_V1_design.npz.

Nothing here ever looks at an eval seed.  Hidden design frames are read only to
report the generator CHARACTERISTICS demanded by protocol sections 1-2.

Run: python src/e35v1_design.py --stage headroom
"""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import argparse
import json
import sys
import time

import numpy as np

from e35v1_common import (
    ARCH_IDS, BETA, CHUNK_TIME, CONTRAST, CURVE_AMPLITUDE, DATA_DIR,
    DESIGN_NPZ, DESIGN_SEEDS, DT, EVAL_SEEDS, FROZEN_CONFIG, HEADROOM_DB_MIN,
    HEIGHT, K_STEP, LAM, RECORD_EVERY, RESIDUAL_MIN, SPEED_MODULATION, T0,
    TAU, TEXTURE_SIGMA_GRID, TRANSIENT_FRACTION, TSVD_RNG_SEED, TSVD_RTOL,
    WIDTH, WINDING_GRID, baseline_frame_metrics, build_loop, classify_cycle,
    config_hash, json_ready, loop_displacement_statistics,
    out_of_span_residual, prepare, run_arm, seam_diagnostics,
    select_beta_prime, select_tsvd_rank,
)

P = 100
T_TOTAL = 9000.0   # A.6, derived from the benchmark-measured recall period
STAGE1 = DATA_DIR / "_E35_V1_stage_headroom.npz"
STAGE2 = DATA_DIR / "_E35_V1_stage_select.npz"
SCREEN_NPZ = DATA_DIR / "_E35_V1_stage_screen.npz"


def grid_settings():
    """A.3 pre-declared grid, ordered by the A.3 parsimony rule."""
    rows = []
    for winding in WINDING_GRID:
        disp = loop_displacement_statistics(
            P * K_STEP, K_STEP, height=HEIGHT, width=WIDTH, winding=winding,
            curve_amplitude=CURVE_AMPLITUDE, speed_modulation=SPEED_MODULATION,
        )
        for sigma in TEXTURE_SIGMA_GRID:
            rows.append(dict(winding=winding, texture_sigma=sigma,
                             mean_disp=disp["mean_keyframe_displacement"],
                             min_disp=disp["min_keyframe_displacement"],
                             max_disp=disp["max_keyframe_displacement"],
                             path_length=disp["path_length"]))
    # parsimony: smallest mean displacement first, then LARGEST texture_sigma
    rows.sort(key=lambda r: (r["mean_disp"], -r["texture_sigma"]))
    return rows


# ------------------------------------------------------------------ headroom
def stage_headroom() -> int:
    wall = time.time()
    rows = grid_settings()
    print(f"A.3 grid: {len(rows)} settings x {len(DESIGN_SEEDS)} design seeds "
          f"| P={P} k={K_STEP} N={HEIGHT}x{WIDTH}")
    print(f"frozen criteria: B2-B1 >= {HEADROOM_DB_MIN} dB AND "
          f"mean out-of-span residual >= {RESIDUAL_MIN}")
    print(f"{'winding':<10}{'sigma':>6}{'meanDisp':>10}{'B0':>8}{'B1':>8}"
          f"{'B2':>8}{'B2-B1':>8}{'resid':>8}{'condG':>11}{'seam':>8}")
    table = []
    for row in rows:
        per_seed = {kk: [] for kk in
                    ("b0", "b1", "b2", "resid", "cond", "seam", "rank")}
        for seed in DESIGN_SEEDS:
            frames = build_loop(seed, row["winding"], row["texture_sigma"],
                                P, K_STEP)
            split, pre, patterns, context = prepare(frames, K_STEP)
            _, bmet = baseline_frame_metrics(split)
            resid = out_of_span_residual(patterns, split.hidden_frames, pre)
            per_seed["b0"].append(bmet["B0_HOLD"]["mean_psnr"])
            per_seed["b1"].append(bmet["B1_LINEAR"]["mean_psnr"])
            per_seed["b2"].append(bmet["B2_MOTION"]["mean_psnr"])
            per_seed["resid"].append(float(np.mean(resid)))
            per_seed["cond"].append(context.diagnostics.condition_retained)
            per_seed["rank"].append(context.diagnostics.rank)
            per_seed["seam"].append(seam_diagnostics(frames)["seam_ratio"])
        entry = dict(row)
        for kk, vv in per_seed.items():
            entry[f"mean_{kk}"] = float(np.mean(vv))
        entry["headroom"] = entry["mean_b2"] - entry["mean_b1"]
        entry["pass_headroom"] = bool(entry["headroom"] >= HEADROOM_DB_MIN)
        entry["pass_residual"] = bool(entry["mean_resid"] >= RESIDUAL_MIN)
        entry["pass_both"] = bool(entry["pass_headroom"] and entry["pass_residual"])
        table.append(entry)
        print(f"{str(row['winding']):<10}{row['texture_sigma']:>6.1f}"
              f"{row['mean_disp']:>10.3f}{entry['mean_b0']:>8.2f}"
              f"{entry['mean_b1']:>8.2f}{entry['mean_b2']:>8.2f}"
              f"{entry['headroom']:>8.2f}{entry['mean_resid']:>8.4f}"
              f"{entry['mean_cond']:>11.3e}{entry['mean_seam']:>8.4f}"
              f"  {'PASS' if entry['pass_both'] else 'fail'}")

    candidates = [e for e in table if e["pass_both"]]
    print(f"\n{len(candidates)}/{len(table)} settings satisfy BOTH frozen criteria.")
    if not candidates:
        print("STOP (protocol section 2): no setting meets the frozen headroom "
              "criteria and they must NEVER be weakened.")
    else:
        print("A.3 parsimony order of the viable-candidate queue "
              "(smallest mean displacement, then largest texture_sigma):")
        for i, e in enumerate(candidates):
            print(f"  {i+1}. winding={e['winding']} sigma={e['texture_sigma']} "
                  f"meanDisp={e['mean_disp']:.3f} headroom={e['headroom']:.2f} dB "
                  f"resid={e['mean_resid']:.4f}")
    np.savez_compressed(STAGE1, table_json=json.dumps(json_ready(table)),
                        wall_seconds=time.time() - wall)
    print(f"saved {STAGE1}  ({time.time() - wall:.1f} s)")
    return 0 if candidates else 2


# ------------------------------------------------------------------- select
def _factorial_on_seed(setting, seed):
    frames = build_loop(seed, setting["winding"], setting["texture_sigma"],
                        P, K_STEP)
    _, _, patterns, context = prepare(frames, K_STEP)
    out = {}
    for arch in ARCH_IDS:
        t, a, m = run_arm(context, arch, T_TOTAL, P)
        out[arch] = classify_cycle(t, m, transient_fraction=TRANSIENT_FRACTION)
    return out


def stage_screen() -> int:
    """A.3 viability pre-screen only (cheap gate before the full factorial)."""
    wall = time.time()
    table = json.loads(str(np.load(STAGE1, allow_pickle=True)["table_json"]))
    candidates = [e for e in table if e["pass_both"]]
    if not candidates:
        print("STOP: stage headroom produced no candidate.")
        return 2
    chosen, screen_log = None, []
    for cand in candidates:
        print(f"\nPRE-SCREEN winding={cand['winding']} "
              f"sigma={cand['texture_sigma']} on design seed "
              f"{DESIGN_SEEDS[0]} (4 arms, t_total={T_TOTAL:g})")
        cyc = _factorial_on_seed(cand, DESIGN_SEEDS[0])
        any_valid = False
        for arch in ARCH_IDS:
            c = cyc[arch]
            print(f"    {arch}: coverage={c.coverage_fraction:.3f} "
                  f"fwd={c.forward_fraction:.3f} tours={c.tours_estimate:.2f} "
                  f"cv={c.period_cv} valid={c.valid_cycle} ({c.reason})")
            any_valid |= bool(c.valid_cycle)
            screen_log.append(dict(winding=cand["winding"],
                                   texture_sigma=cand["texture_sigma"],
                                   seed=DESIGN_SEEDS[0], architecture=arch,
                                   **c.to_dict()))
        if any_valid:
            chosen = cand
            print("    -> structurally viable, retained (A.3 parsimony order).")
            break
        print("    -> not viable, moving to the next candidate.")
    np.savez_compressed(SCREEN_NPZ,
                        screen_json=json.dumps(json_ready(screen_log)),
                        chosen_json=json.dumps(json_ready(chosen or {})),
                        wall_seconds=time.time() - wall)
    print(f"saved {SCREEN_NPZ}  ({time.time() - wall:.1f} s)")
    return 0 if chosen else 3


def stage_select() -> int:
    wall = time.time()
    s0 = np.load(SCREEN_NPZ, allow_pickle=True)
    screen_log = json.loads(str(s0["screen_json"]))
    chosen = json.loads(str(s0["chosen_json"]))
    if not chosen:
        print("STOP: no candidate of the A.3 grid is structurally viable.")
        return 3

    print(f"FULL J x K FACTORIAL on the {len(DESIGN_SEEDS)} design seeds "
          f"(A.4 score = mean cycle coverage)")
    rows = []
    for seed in DESIGN_SEEDS:
        t_seed = time.time()
        cyc = _factorial_on_seed(chosen, seed)
        for arch in ARCH_IDS:
            rows.append(dict(seed=seed, architecture=arch, **cyc[arch].to_dict()))
        print(f"  seed {seed} ({time.time() - t_seed:.0f} s): " + " | ".join(
            f"{a} cov={cyc[a].coverage_fraction:.2f} "
            f"valid={int(cyc[a].valid_cycle)}" for a in ARCH_IDS))

    scores = {}
    for arch in ARCH_IDS:
        sub = [r for r in rows if r["architecture"] == arch]
        scores[arch] = dict(
            coverage=float(np.mean([r["coverage_fraction"] for r in sub])),
            forward=float(np.mean([r["forward_fraction"] for r in sub])),
            period_cv=float(np.nanmean([r["period_cv"] for r in sub])),
            valid_fraction=float(np.mean([r["valid_cycle"] for r in sub])),
        )
    print(f"\n{'arch':<8}{'coverage':>10}{'forward':>10}{'period_cv':>12}"
          f"{'valid_frac':>12}")
    for arch in ARCH_IDS:
        s = scores[arch]
        print(f"{arch:<8}{s['coverage']:>10.4f}{s['forward']:>10.4f}"
              f"{s['period_cv']:>12.5f}{s['valid_fraction']:>12.2f}")

    def key(arch):
        s = scores[arch]
        cv = s["period_cv"] if np.isfinite(s["period_cv"]) else np.inf
        return (-s["coverage"], -s["forward"], cv, arch)

    frozen_arch = sorted(ARCH_IDS, key=key)[0]
    print(f"\nA.4 FROZEN ARCHITECTURE = {frozen_arch} "
          f"(coverage {scores[frozen_arch]['coverage']:.4f}); no image metric "
          f"entered this score.")
    np.savez_compressed(
        STAGE2,
        screen_json=json.dumps(json_ready(screen_log)),
        chosen_json=json.dumps(json_ready(chosen)),
        factorial_json=json.dumps(json_ready(rows)),
        scores_json=json.dumps(json_ready(scores)),
        frozen_architecture=frozen_arch,
        wall_seconds=time.time() - wall,
    )
    print(f"saved {STAGE2}  ({time.time() - wall:.1f} s)")
    return 0


# -------------------------------------------------------------------- freeze
def stage_freeze() -> int:
    wall = time.time()
    table = json.loads(str(np.load(STAGE1, allow_pickle=True)["table_json"]))
    s2 = np.load(STAGE2, allow_pickle=True)
    chosen = json.loads(str(s2["chosen_json"]))
    frozen_arch = str(s2["frozen_architecture"])
    factorial = json.loads(str(s2["factorial_json"]))
    scores = json.loads(str(s2["scores_json"]))
    if not chosen:
        print("STOP: no frozen generator setting.")
        return 3

    print(f"Decoder calibration (A.5): keyframe-only, design seeds "
          f"{DESIGN_SEEDS}, rng {TSVD_RNG_SEED}")
    rank_curves, beta_curves = [], []
    diag = []
    for seed in DESIGN_SEEDS:
        frames = build_loop(seed, chosen["winding"], chosen["texture_sigma"],
                            P, K_STEP)
        split, pre, patterns, context = prepare(frames, K_STEP)
        _, curve, ranks = select_tsvd_rank(split, pre, patterns, context, P,
                                           rng_seed=TSVD_RNG_SEED + seed)
        _, bcurve = select_beta_prime(split, pre, patterns, context, P,
                                      rng_seed=TSVD_RNG_SEED + seed)
        rank_curves.append(curve)
        beta_curves.append(bcurve)
        diag.append(dict(seed=seed, rank=context.diagnostics.rank,
                         cond=context.diagnostics.condition_retained,
                         mp_residual=context.diagnostics.moore_penrose_residual,
                         singular_max=context.diagnostics.singular_max,
                         singular_min=context.diagnostics.singular_min_retained))
        print(f"  seed {seed}: cond(G)={context.diagnostics.condition_retained:.3e} "
              f"rank={context.diagnostics.rank} "
              f"MP_res={context.diagnostics.moore_penrose_residual:.2e} "
              f"({time.time() - wall:.0f} s)")
    rank_curve = np.mean(np.array(rank_curves), axis=0)
    beta_curve = np.mean(np.array(beta_curves), axis=0)
    ranks = np.arange(1, P + 1)
    best = np.flatnonzero(rank_curve <= rank_curve.min() + 1e-9)
    r_star = int(ranks[best[0]])
    from e35v1_common import BETA_PRIME_GRID
    beta_star = float(BETA_PRIME_GRID[int(np.argmin(beta_curve))])
    print(f"\nTSVD rank (D0 rule, noisy KEYFRAME recalls only): r* = {r_star} "
          f"(key MSE {rank_curve.min():.6e}; full rank P={P} gives "
          f"{rank_curve[-1]:.6e})")
    print(f"readout beta' scan {BETA_PRIME_GRID} -> beta* = {beta_star} "
          f"(key MSE {beta_curve.min():.6e})")

    cfg = dict(
        experiment="E35_V1",
        protocol="roadmaps/E35_V1_PROTOCOL_DRAFT.md (sections 1-8 + appendix A)",
        generator=dict(
            kind="nonlinear_advection_torus",
            winding=list(chosen["winding"]),
            texture_sigma=float(chosen["texture_sigma"]),
            curve_amplitude=list(CURVE_AMPLITUDE),
            speed_modulation=SPEED_MODULATION,
            contrast=CONTRAST,
            height=HEIGHT, width=WIDTH, n_frames=P * K_STEP,
        ),
        architecture=frozen_arch,
        architecture_factorial=list(ARCH_IDS),
        decoder_nominal="D3_TSVD",
        tsvd_rank=r_star,
        beta_prime=beta_star,
        decoders_reported=["D1_Xa", "D2c_XGdm_clip", "D3_TSVD", "D4_betaprime"],
        clock_nominal="CLK_U",
        clocks=["CLK_U", "CLK_P"],
        lam=LAM, tau=TAU, beta=BETA, dt=DT, t0=T0, k=K_STEP, p=P,
        n=HEIGHT * WIDTH, t_total=T_TOTAL, chunk_time=CHUNK_TIME,
        record_every=RECORD_EVERY, transient_fraction=TRANSIENT_FRACTION,
        tsvd_rtol=TSVD_RTOL, binary=False,
        a0="0.95*e0+0.05*e1",
        design_seeds=list(DESIGN_SEEDS), eval_seeds=list(EVAL_SEEDS),
        headroom_db_min=HEADROOM_DB_MIN, residual_min=RESIDUAL_MIN,
        headroom_measured_db=float(chosen["headroom"]),
        residual_measured=float(chosen["mean_resid"]),
        boot=dict(n_boot=10000, rng_seed=20260725, unit="transition",
                  hierarchical=True),
        tsvd_rng_seed=TSVD_RNG_SEED,
    )
    chash = config_hash(cfg)
    payload = dict(config=cfg, config_hash=chash)
    FROZEN_CONFIG.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nFROZEN CONFIG HASH = {chash}\nwritten {FROZEN_CONFIG}")

    np.savez_compressed(
        DESIGN_NPZ,
        grid_table_json=json.dumps(json_ready(table)),
        screen_json=str(s2["screen_json"]),
        factorial_json=json.dumps(json_ready(factorial)),
        arch_scores_json=json.dumps(json_ready(scores)),
        gram_diag_json=json.dumps(json_ready(diag)),
        chosen_json=json.dumps(json_ready(chosen)),
        frozen_architecture=frozen_arch,
        tsvd_rank_grid=ranks, tsvd_rank_curve=rank_curve,
        tsvd_rank_curves_per_seed=np.array(rank_curves),
        tsvd_rank_selected=r_star,
        beta_prime_grid=np.array(BETA_PRIME_GRID),
        beta_prime_curve=beta_curve, beta_prime_selected=beta_star,
        design_seeds=np.array(DESIGN_SEEDS), eval_seeds=np.array(EVAL_SEEDS),
        config_json=json.dumps(cfg, sort_keys=True), config_hash=chash,
        wall_seconds=time.time() - wall,
    )
    print(f"saved {DESIGN_NPZ}  ({time.time() - wall:.1f} s)")
    for path in (STAGE1, STAGE2):
        print(f"  (intermediate kept: {path.name})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("headroom", "screen", "select", "freeze"))
    args = ap.parse_args()
    return {"headroom": stage_headroom, "screen": stage_screen,
            "select": stage_select, "freeze": stage_freeze}[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
