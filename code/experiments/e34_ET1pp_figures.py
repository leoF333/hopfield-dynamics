"""Figure for E34 ET1'' -- the necklace brick tested by W^u at N=2000, P=100.

Supersedes e34_ET1prime_figures.py, which was written for the original
edge-tracking design of ET1': at N=2000 that design cannot localise the saddle
(Newton stalls at residual ~1e-3 from every dwell seed, see
E34_dwell_seed_test), so the test was re-scoped to shooting W^u directly from
the arclength fold partner.

Panels (dpi 170, English labels), all built from artefacts already on disk --
no dynamics is re-run here:

  (a) outcome grid over (mu, delta) after the follow-up diagnostics, with the
      omega-limit of each W^u branch written in the cell;
  (b) the certified fold partner: Gram separation from the node and unstable
      rate z_u against delta;
  (c) mu=91, the last surviving bead: leading-bead staircase of the forward
      landing -- a travelling orbit with ring step +1;
  (d) audit of the E24 threshold table: how far above lam_c[mu] a STABLE
      equilibrium carrying identity mu still exists, bead by bead.
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/2_cycle_rappel_snic/data/E34"
FIGS = ROOT / "results/2_cycle_rappel_snic/figures/E34"

STATUS_COLOR = {"necklace_brick_positive": "#2ca02c",
                "cycle_closure": "#1f77b4",
                "indeterminate": "#bdbdbd"}
STATUS_LABEL = {"necklace_brick_positive": "brick holds",
                "cycle_closure": "closes onto the recall cycle",
                "indeterminate": "indeterminate"}


def _latest(pattern: str) -> Path:
    hits = sorted(DATA.glob(pattern))
    if not hits:
        raise SystemExit(f"no file matching {pattern} in {DATA}")
    return hits[-1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=FIGS / "figE34_ET1pp.png")
    args = p.parse_args(argv)

    summ = json.loads(_latest("E34_ET1pp_wu_summary_*.json").read_text())
    audit = json.loads(_latest("E34_threshold_table_audit_*.json").read_text())
    cyc = json.loads(_latest("E34_ET1pp_followup_cycle_mu91_*.json").read_text())
    recell = json.loads(_latest("E34_ET1pp_followup_recell_mu28_*.json").read_text())
    land = json.loads(_latest("E34_ET1pp_followup_land_mu1_*.json").read_text())

    # cell revisions established by the follow-up diagnostics (RESULTS sec. 3)
    follow = {
        (1, 0.005): dict(status="necklace_brick_positive", minus=1,
                         plus=int(land["identify"]["best"]),
                         note="landing identified"),
        (91, 0.005): dict(status="cycle_closure", minus=91, plus=None,
                          note="no bead alive"),
        (91, 0.01): dict(status="cycle_closure", minus=91, plus=None,
                         note="no bead alive"),
        (91, 0.02): dict(status="cycle_closure", minus=91, plus=None,
                         note="fallback localisation"),
        (28, 0.02): dict(status="necklace_brick_positive",
                         minus=int(recell["landing_minus"]),
                         plus=int(recell["landing_plus"]),
                         note="re-run at measured fold"),
    }
    rows = summ["rows"]
    order = {28: 0, 1: 1, 91: 2}
    links = sorted({int(r["mu"]) for r in rows}, key=lambda m: order.get(m, m))
    deltas = sorted({round(float(r["delta"]), 4) for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # --- (a) outcome grid -------------------------------------------------
    for i, mu in enumerate(links):
        for j, dl in enumerate(deltas):
            rec = next(r for r in rows if int(r["mu"]) == mu
                       and abs(float(r["delta"]) - dl) < 1e-9)
            rev = follow.get((mu, round(dl, 4)))
            if rev is None:
                st = rec.get("status")
                note = ("" if st == "necklace_brick_positive"
                        else ("mis-parameterised:\ntable $\\lambda_c$ off by "
                              "$1.3\\times10^{-2}$"
                              if st == "arclength_failed" else str(st)))
                rev = dict(status=("necklace_brick_positive"
                                   if st == "necklace_brick_positive"
                                   else "indeterminate"),
                           minus=rec.get("landing_minus"),
                           plus=rec.get("landing_plus"), note=note)
            col = STATUS_COLOR[rev["status"]]
            ax_a.add_patch(Rectangle((j - 0.46, i - 0.42), 0.92, 0.84,
                                     facecolor=col, alpha=0.28, edgecolor=col,
                                     linewidth=1.6))
            if rev["status"] == "cycle_closure":
                txt = f"$W^u_-\\to$ {rev['minus']}\n$W^u_+\\to$ cycle"
            elif rev["plus"] is not None:
                txt = f"$W^u_-\\to$ {rev['minus']}\n$W^u_+\\to$ {rev['plus']}"
            else:
                txt = "—"
            ax_a.text(j, i + (0.08 if rev["note"] else 0.0), txt, ha="center",
                      va="center", fontsize=10.5)
            if rev["note"]:
                ax_a.text(j, i - 0.26, rev["note"], ha="center", va="center",
                          fontsize=7.4, style="italic", color="#444444")
    ax_a.set_xticks(range(len(deltas)))
    ax_a.set_xticklabels([f"{d:g}" for d in deltas])
    ax_a.set_yticks(range(len(links)))
    ax_a.set_yticklabels([f"$\\mu$ = {m}" for m in links])
    ax_a.set_xlim(-0.55, len(deltas) - 0.45)
    ax_a.set_ylim(-0.55, len(links) - 0.45)
    ax_a.set_xlabel(r"$\delta = \lambda_c(\mu) - \lambda$")
    ax_a.set_title("(a) $\\omega$-limits of $W^u$(fold partner)\n"
                   "each cell robust over 2 signs $\\times$ 2 $\\epsilon$ "
                   "$\\times$ 2 $dt$", fontsize=11)
    handles = [Rectangle((0, 0), 1, 1, facecolor=STATUS_COLOR[k], alpha=0.32,
                         edgecolor=STATUS_COLOR[k])
               for k in ("necklace_brick_positive", "cycle_closure")]
    ax_a.legend(handles, [STATUS_LABEL[k] for k in
                          ("necklace_brick_positive", "cycle_closure")],
                fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=2, frameon=False)

    # --- (b) the certified fold partner ----------------------------------
    marks = {28: "s", 1: "o", 91: "^"}
    cols = {28: "#d62728", 1: "#2ca02c", 91: "#1f77b4"}
    ax_b2 = ax_b.twinx()
    for mu in links:
        cells = [r for r in rows if int(r["mu"]) == mu and r.get("index_one")]
        if mu == 28 and recell.get("index_one"):
            cells = [recell]
        if not cells:
            continue
        cells = sorted(cells, key=lambda r: float(r["delta"]))
        d = [float(r["delta"]) for r in cells]
        ax_b.plot(d, [float(r["d_saddle_node"]) for r in cells],
                  marker=marks[mu], color=cols[mu], lw=1.7,
                  label=f"$\\mu$ = {mu}")
        zu = [float(r["z_u"]) if np.isfinite(float(r.get("z_u") or np.nan))
              else float(r["real_root"]) for r in cells]
        ax_b2.plot(d, zu, marker=marks[mu], color=cols[mu], lw=1.2, ls="--",
                   alpha=0.6)
    ax_b.set_xlabel(r"$\delta = \lambda_c(\mu) - \lambda$")
    ax_b.set_ylabel(r"$d_G$(saddle, node)   [solid]")
    ax_b2.set_ylabel(r"unstable rate $z_u$   [dashed]")
    ax_b.set_title("(b) the fold partner emerging from the fold\n"
                   "every saddle machine-exact and certified index 1",
                   fontsize=11)
    ax_b.legend(fontsize=9, loc="upper left")
    ax_b.grid(alpha=0.25)

    # --- (c) mu=91 forward landing ---------------------------------------
    tr = cyc["transitions"]
    t = [x[0] for x in tr]
    lead = [x[2] for x in tr]
    ax_c.step(t, lead, where="post", color="#1f77b4", lw=1.8)
    ax_c.plot(t, lead, "o", ms=3.2, color="#1f77b4")
    ax_c.set_xlabel("time $t$")
    ax_c.set_ylabel(r"leading bead $\arg\max_\nu |m_\nu|$")
    ax_c.set_title(
        "(c) $\\mu$ = 91 (last surviving bead): forward $W^u$ landing\n"
        f"ring step {cyc['ring_steps']}, median link {cyc['link_duration']:.1f}, "
        f"mean lead amplitude {cyc['lead_amp_mean']:.3f}", fontsize=11)
    ax_c.grid(alpha=0.25)

    # --- (d) threshold-table audit ---------------------------------------
    lam_c, off, late, mus = [], [], [], []
    for r in audit["rows"]:
        if r.get("verdict") == "no_threshold":
            continue
        ok = [float(k.strip("+")) for k, v in r["probes"].items()
              if v["ok"] and v["identity"] and v["eigmax"] is not None
              and v["eigmax"] < 0]
        lam_c.append(float(r["lam_c"]))
        off.append(max(ok) if ok else 0.0)
        late.append(bool(ok))
        mus.append(int(r["mu"]))
    lam_c, off = np.array(lam_c), np.array(off)
    late, mus = np.array(late), np.array(mus)
    ax_d.scatter(lam_c[~late], off[~late], s=18, color="#999999",
                 label=f"table consistent ({int((~late).sum())})")
    ax_d.scatter(lam_c[late], off[late], s=36, color="#d62728",
                 label=f"stable bead above $\\lambda_c$ ({int(late.sum())})")
    for mu_hi in (28, 8):
        sel = np.where(mus == mu_hi)[0]
        if sel.size and late[sel[0]]:
            ax_d.annotate(f"$\\mu$={mu_hi}", (lam_c[sel[0]], off[sel[0]]),
                          textcoords="offset points", xytext=(7, 5),
                          fontsize=9, color="#d62728")
    ax_d.set_yscale("symlog", linthresh=5e-4)
    ax_d.set_xlabel(r"tabulated $\lambda_c(\mu)$  (E24 table)")
    ax_d.set_ylabel("largest probe offset with a stable bead")
    ax_d.set_title("(d) audit of the per-bead threshold table\n"
                   f"{audit['n_late']}/{audit['P']} beads still carry a stable "
                   "identity-$\\mu$ equilibrium above $\\lambda_c$", fontsize=11)
    ax_d.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax_d.grid(alpha=0.25)

    fig.suptitle("E34 / ET1'' — necklace brick at $N$=2000, $P$=100, seed 42, "
                 r"$\beta$=20, $\tau$=10: $W^u$ of the certified fold partner",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.952))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
