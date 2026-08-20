#!/usr/bin/env python3
"""Rebuild the eight requested article panels without modifying source folders.

Raw-data regeneration is used for N6, N2(a), and the current Figure 6 (N1).
All other components are high-resolution crops of the user-selected source
figures.  Every source and crop is recorded in manifest.json.
"""

from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(CHICAGO / "4_campaigns/figure_panels/work/mplconfig"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 by folder reorganisation -----------------------------
# All absolute paths below now resolve from a single root, overridable with the
# CHICAGO_ROOT environment variable.  See ../../REORGANISATION_LOG.md
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
# -----------------------------------------------------------------------------



OUT = CHICAGO / "4_campaigns/figure_panels"
COMPONENTS = OUT / "components"
PANELS = OUT / "panels"
QA = OUT / "qa"
NUMERICS = CHICAGO / "4_campaigns/numerical_workplan"
CYCLE_DATA = CHICAGO / "3_numerics/results/2_cycle_rappel_snic/data"

# ARTICLE_ROOT retired by the 2026-08-19 reorganisation
SOURCE_FIGURES = ARTICLE_ROOT / "figures à ne pas oublier dans l'article"
REDACTOR_FIGURES = CHICAGO / "4_campaigns/manuscript_audits/agent_redacteur/figures"

for directory in (COMPONENTS, PANELS, QA):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    source: Path
    box: tuple[float, float, float, float]
    method: str = "crop"


SPECS = {
    # E23 A
    "q_a": ComponentSpec(
        "q_a", SOURCE_FIGURES / "figQ_chaos_nature.png",
        (0.000, 0.035, 0.333, 0.500),
    ),
    "q_c": ComponentSpec(
        "q_c", SOURCE_FIGURES / "figQ_chaos_nature.png",
        (0.667, 0.035, 1.000, 0.500),
    ),
    "q_f": ComponentSpec(
        "q_f", SOURCE_FIGURES / "figQ_chaos_nature.png",
        (0.667, 0.500, 1.000, 1.000),
    ),
    # Global mechanism
    "bif_schematic": ComponentSpec(
        "bif_schematic", SOURCE_FIGURES / "diagramme de bifurcation v1.png",
        (0.000, 0.000, 1.000, 1.000),
    ),
    "mechanism_a": ComponentSpec(
        "mechanism_a", SOURCE_FIGURES / "fig1_mechanism.png",
        (0.000, 0.000, 1.000, 0.540),
    ),
    "mechanism_b": ComponentSpec(
        "mechanism_b", SOURCE_FIGURES / "fig1_mechanism.png",
        (0.000, 0.540, 0.535, 1.000),
    ),
    "mechanism_c": ComponentSpec(
        "mechanism_c", SOURCE_FIGURES / "fig1_mechanism.png",
        (0.550, 0.540, 1.000, 1.000),
    ),
    # Static folds and memory distortion
    "front_birth_left": ComponentSpec(
        "front_birth_left", SOURCE_FIGURES / "figS_front_birth.png",
        (0.000, 0.000, 0.500, 1.000),
    ),
    "n4_fig2_a": ComponentSpec(
        "n4_fig2_a", SOURCE_FIGURES / "N4_Figure2.png",
        (0.000, 0.000, 0.500, 0.500),
    ),
    "branch_geometry_a": ComponentSpec(
        "branch_geometry_a", SOURCE_FIGURES / "figU_branch_geometry.png",
        (0.000, 0.000, 0.500, 0.500),
    ),
    "fold_n10000_left": ComponentSpec(
        "fold_n10000_left", SOURCE_FIGURES / "fold_N10000_a0.05.png",
        (0.000, 0.000, 0.500, 1.000),
    ),
    "fold_n10000_right": ComponentSpec(
        "fold_n10000_right", SOURCE_FIGURES / "fold_N10000_a0.05.png",
        (0.500, 0.000, 1.000, 1.000),
    ),
    # Threshold distributions and finite-size laws
    "alpha_law_hist": ComponentSpec(
        "alpha_law_hist", SOURCE_FIGURES / "figZ_alpha_law.png",
        (0.000, 0.075, 0.465, 0.510),
    ),
    "alpha_law_cdf": ComponentSpec(
        "alpha_law_cdf", SOURCE_FIGURES / "figZ_alpha_law.png",
        (0.000, 0.545, 0.465, 1.000),
    ),
    "ae_width": ComponentSpec(
        "ae_width", SOURCE_FIGURES / "figAE_threshold_scaling.png",
        (0.000, 0.080, 0.335, 1.000),
    ),
    "ae_gap": ComponentSpec(
        "ae_gap", SOURCE_FIGURES / "figAE_threshold_scaling.png",
        (0.340, 0.080, 0.680, 1.000),
    ),
    "ae_normal": ComponentSpec(
        "ae_normal", SOURCE_FIGURES / "figAE_threshold_scaling.png",
        (0.690, 0.080, 1.000, 1.000),
    ),
    "n3_gaussian_evt": ComponentSpec(
        "n3_gaussian_evt", SOURCE_FIGURES / "N3_E30_seed_aware.png",
        (0.000, 0.500, 0.500, 1.000),
    ),
    "n4_fig3_a": ComponentSpec(
        "n4_fig3_a", SOURCE_FIGURES / "N4_Figure3.png",
        (0.000, 0.000, 0.333, 0.500),
    ),
    "n4_fig3_b": ComponentSpec(
        "n4_fig3_b", SOURCE_FIGURES / "N4_Figure3.png",
        (0.333, 0.000, 0.667, 0.500),
    ),
    "threshold_size_width": ComponentSpec(
        "threshold_size_width", REDACTOR_FIGURES / "panel_thresholds.png",
        (0.000, 0.235, 0.430, 0.505),
    ),
    "threshold_load_center": ComponentSpec(
        "threshold_load_center", REDACTOR_FIGURES / "panel_thresholds.png",
        (0.435, 0.235, 0.900, 0.505),
    ),
    # SNIC signatures
    "n4_fig4_c": ComponentSpec(
        "n4_fig4_c", SOURCE_FIGURES / "N4_Figure4.png",
        (0.000, 0.500, 0.500, 1.000),
    ),
    "depinning_left": ComponentSpec(
        "depinning_left", SOURCE_FIGURES / "figB_depinning_transition.png",
        (0.000, 0.260, 0.500, 1.000),
    ),
    "depinning_right": ComponentSpec(
        "depinning_right", SOURCE_FIGURES / "figB_depinning_transition.png",
        (0.500, 0.260, 1.000, 1.000),
    ),
    "e34_a": ComponentSpec(
        "e34_a", SOURCE_FIGURES / "figE34_invariant_circle.png",
        (0.000, 0.045, 0.500, 0.500),
    ),
    "e34_b": ComponentSpec(
        "e34_b", SOURCE_FIGURES / "figE34_invariant_circle.png",
        (0.500, 0.045, 1.000, 0.500),
    ),
    "e34_d": ComponentSpec(
        "e34_d", SOURCE_FIGURES / "figE34_invariant_circle.png",
        (0.500, 0.500, 1.000, 1.000),
    ),
    "pacemaker_full": ComponentSpec(
        "pacemaker_full", REDACTOR_FIGURES / "fig_pacemaker.png",
        (0.000, 0.000, 1.000, 1.000),
    ),
    "cycle_exact_recall": ComponentSpec(
        "cycle_exact_recall", REDACTOR_FIGURES / "figAA_cycle_vs_ghosts.png",
        (0.000, 0.500, 0.500, 1.000),
    ),
    "cycle_ghost_bottleneck": ComponentSpec(
        "cycle_ghost_bottleneck", REDACTOR_FIGURES / "figAA_cycle_vs_ghosts.png",
        (0.500, 0.500, 1.000, 1.000),
    ),
    # Delay
    "n5a_left": ComponentSpec(
        "n5a_left", SOURCE_FIGURES / "N5A_delay_boundary.png",
        (0.000, 0.000, 0.500, 1.000),
    ),
    "desync_order": ComponentSpec(
        "desync_order", REDACTOR_FIGURES / "figP_desync_cdf.png",
        (0.000, 0.000, 0.318, 1.000),
    ),
    "desync_front": ComponentSpec(
        "desync_front", REDACTOR_FIGURES / "figP_desync_cdf.png",
        (0.318, 0.000, 0.642, 1.000),
    ),
    "desync_kymograph": ComponentSpec(
        "desync_kymograph", REDACTOR_FIGURES / "figP_desync_cdf.png",
        (0.642, 0.000, 1.000, 1.000),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pil_font(size: int, bold: bool = False):
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def trim_white(image: Image.Image, padding: int = 16) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background).convert("L")
    # Ignore nearly white anti-aliasing and the white canvas.
    difference = difference.point(lambda value: 255 if value > 10 else 0)
    box = difference.getbbox()
    if box is None:
        return rgb
    left = max(0, box[0] - padding)
    top = max(0, box[1] - padding)
    right = min(rgb.width, box[2] + padding)
    bottom = min(rgb.height, box[3] + padding)
    return rgb.crop((left, top, right, bottom))


def crop_component(spec: ComponentSpec) -> Image.Image:
    image = Image.open(spec.source).convert("RGB")
    left, top, right, bottom = spec.box
    box = (
        round(left * image.width),
        round(top * image.height),
        round(right * image.width),
        round(bottom * image.height),
    )
    return trim_white(image.crop(box), padding=18)


def plot_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_mpl_component(fig, name: str) -> Path:
    path = COMPONENTS / f"{name}.png"
    fig.savefig(path, dpi=430)
    plt.close(fig)
    return path


def regenerate_n6() -> tuple[Image.Image, list[Path]]:
    seeds = [42, 43, 44, 45, 46]
    sources = [
        NUMERICS / "runs/N6" /
        f"lyapunov_N2000_P100_s{seed}_lam0p310_dt0p01.npz"
        for seed in seeds
    ]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(seeds)))
    fig, ax = plt.subplots(figsize=(4.45, 3.35))
    indices = np.arange(1, 9)
    for seed, source, color in zip(seeds, sources, colors):
        values = np.asarray(np.load(source)["lyapunov"], dtype=float)
        ax.plot(indices, values, "o-", ms=3.7, lw=1.15, color=color, label=f"seed {seed}")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.axvspan(0.5, 3.5, color="#d9ecdf", alpha=0.65, zorder=-1)
    ax.set(
        xlabel="Lyapunov index $i$",
        ylabel=r"$\Lambda_i$",
        title=r"Disorder-robust hyperchaos at $\lambda=0.31$",
        xticks=indices,
    )
    ax.legend(ncol=2, handlelength=1.4, columnspacing=0.8)
    ax.grid(color="0.88", linewidth=0.5)
    fig.tight_layout()
    path = save_mpl_component(fig, "n6_spectrum_raw")
    return trim_white(Image.open(path).convert("RGB"), 12), sources


def regenerate_n2_without_seed51() -> tuple[Image.Image, list[Path]]:
    seeds = [42, 47, 52, 60]
    sources = [
        NUMERICS / "runs/N2" /
        f"seed_N2000_P100_s{seed}_production.json"
        for seed in seeds
    ]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(seeds)))
    fig, ax = plt.subplots(figsize=(4.55, 3.45))
    for seed, source, color in zip(seeds, sources, colors):
        summary = json.loads(source.read_text(encoding="utf-8"))
        rows = sorted(
            (
                row for row in summary["dynamic_rows"]
                if row["dt"] == 0.01 and not row["censored"]
            ),
            key=lambda row: row["delta"],
        )
        delta = np.asarray([row["delta"] for row in rows], dtype=float)
        time = np.asarray([row["critical_median"] for row in rows], dtype=float)
        ax.loglog(delta, time, "o", ms=3.8, color=color, label=f"seed {seed}")
        design = np.column_stack([np.ones_like(delta), delta ** -0.5])
        offset, amplitude = np.linalg.lstsq(design, time, rcond=None)[0]
        grid = np.geomspace(delta.min(), delta.max(), 240)
        ax.loglog(grid, offset + amplitude * grid ** -0.5, "-", lw=1.0, color=color)
    ax.set(
        xlabel=r"$\delta=\lambda-\lambda_*^{\rm stat}$",
        ylabel="critical-bond passage time",
        title="Cycle-side slowing across disorder",
    )
    ax.legend(ncol=2)
    ax.grid(which="both", color="0.88", linewidth=0.45)
    fig.tight_layout()
    path = save_mpl_component(fig, "n2a_without_seed51_raw")
    return trim_white(Image.open(path).convert("RGB"), 12), sources


def regenerate_n2_static_softening() -> tuple[Image.Image, list[Path]]:
    seeds = [42, 47, 51, 52, 60]
    sources = [
        NUMERICS / "runs/N2" /
        f"seed_N2000_P100_s{seed}_production.json"
        for seed in seeds
    ]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(seeds)))
    fig, ax = plt.subplots(figsize=(4.55, 3.45))
    for seed, source, color in zip(seeds, sources, colors):
        summary = json.loads(source.read_text(encoding="utf-8"))
        rows = sorted(summary["static_rows"], key=lambda row: row["delta"])
        delta = np.asarray([row["delta"] for row in rows], dtype=float)
        rate = -np.asarray([row["leading_real"] for row in rows], dtype=float)
        ax.loglog(
            delta, rate, "o-", ms=3.5, lw=0.85,
            color=color, label=f"seed {seed}",
        )
    guide_x = np.geomspace(1e-5, 1e-4, 150)
    ax.loglog(
        guide_x, 0.32 * np.sqrt(guide_x),
        "k--", lw=1.0, label=r"slope $1/2$",
    )
    ax.set(
        xlabel=r"$\delta=\lambda_*^{\rm stat}-\lambda$",
        ylabel=r"$-\mathrm{Re}\,z_{\rm real}$",
        title="Static softening beyond the local window",
    )
    ax.legend(ncol=2)
    ax.grid(which="both", color="0.88", linewidth=0.45)
    fig.tight_layout()
    path = save_mpl_component(fig, "n2_static_softening_raw")
    return trim_white(Image.open(path).convert("RGB"), 12), sources


def regenerate_n4_critical_slowing() -> tuple[Image.Image, list[Path]]:
    source = CYCLE_DATA / "E29_cycle_ghosts.npz"
    data = np.load(source)
    delta = np.asarray(data["lams"], dtype=float) - float(data["lam_star"])
    period = np.asarray(data["Ts"], dtype=float)
    keep = delta > 0
    fig, ax = plt.subplots(figsize=(4.55, 3.45))
    ax.loglog(delta[keep], period[keep], "o-", ms=4.2, lw=1.2, color="#0072B2")
    ax.set(
        xlabel=r"$\lambda-\lambda_*$",
        ylabel="tour period",
        title="Cycle-side critical slowing",
    )
    ax.grid(which="both", color="0.88", linewidth=0.45)
    fig.tight_layout()
    path = save_mpl_component(fig, "n4_critical_slowing_raw")
    return trim_white(Image.open(path).convert("RGB"), 12), [source]


def load_n1_rows() -> tuple[list[dict], Path]:
    source = NUMERICS / "reports/N1_static_extreme_vs_dynamic_onset.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, source


def regenerate_n1() -> tuple[Image.Image, Image.Image, list[Path]]:
    rows, source = load_n1_rows()
    retained = [
        row for row in rows
        if str(row["admissible_for_equality_test"]).strip().lower()
        in {"true", "1", "yes"}
    ]
    static = np.asarray([float(row["static_extreme_lambda"]) for row in retained])
    dynamic = np.asarray([float(row["dynamic_onset_midpoint"]) for row in retained])
    errors = 0.5 * np.asarray([
        float(row["dynamic_onset_high"]) - float(row["dynamic_onset_low"])
        for row in retained
    ])
    seeds = np.asarray([int(row["seed"]) for row in retained])
    blue = "#0072B2"
    tolerance = 1e-3

    fig, ax = plt.subplots(figsize=(4.05, 3.75))
    line = np.linspace(0.310, 0.342, 300)
    ax.fill_between(line, line - tolerance, line + tolerance, color="0.92", linewidth=0)
    ax.plot(line, line, color="0.25", lw=0.9)
    ax.errorbar(
        static, dynamic, yerr=errors, fmt="o", ms=5.0,
        color=blue, ecolor=blue, elinewidth=0.8, capsize=1.8,
    )
    agreement = int(np.sum(np.abs(dynamic - static) < tolerance))
    ax.text(
        0.03, 0.97, rf"{agreement}/{len(retained)} admissible seeds agree at $10^{{-3}}$",
        transform=ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "0.75", "lw": 0.6},
    )
    ax.set(
        xlim=(0.310, 0.342), ylim=(0.310, 0.342),
        xlabel=r"static extreme $\lambda_{\rm stat}$",
        ylabel=r"dynamic onset $\lambda_{\rm dyn}$",
        title="Sample-wise static-dynamic identity",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.88", linewidth=0.45)
    fig.tight_layout()
    path_identity = save_mpl_component(fig, "n1_identity_raw")

    fig, ax = plt.subplots(figsize=(4.45, 2.9))
    residual = 1e3 * (dynamic - static)
    ax.axhspan(-1, 1, color="0.92")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.errorbar(
        seeds, residual, yerr=1e3 * errors, fmt="o", ms=4.8,
        color=blue, ecolor=blue, elinewidth=0.75, capsize=1.8,
    )
    ax.set(
        xlim=(41.3, 61.7), ylim=(-1.15, 1.15),
        xticks=np.arange(42, 62, 2), yticks=[-1, 0, 1],
        xlabel="disorder seed",
        ylabel=r"$10^3(\lambda_{\rm dyn}-\lambda_{\rm stat})$",
        title="Residuals remain within the resolution band",
    )
    ax.grid(axis="y", color="0.88", linewidth=0.45)
    fig.tight_layout()
    path_residual = save_mpl_component(fig, "n1_residual_raw")
    return (
        trim_white(Image.open(path_identity).convert("RGB"), 12),
        trim_white(Image.open(path_residual).convert("RGB"), 12),
        [source],
    )


def add_letter(image: Image.Image, letter: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = pil_font(max(28, round(min(result.size) * 0.050)), bold=True)
    text = f"({letter})"
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    padding = max(8, round(height * 0.25))
    draw.rounded_rectangle(
        (4, 4, 4 + width + 2 * padding, 4 + height + 2 * padding),
        radius=padding,
        fill="white",
        outline="#D4DAE0",
        width=max(1, padding // 5),
    )
    draw.text((4 + padding, 4 + padding - bbox[1]), text, font=font, fill="#151F28")
    return result


def replace_top_title(image: Image.Image, title: str) -> Image.Image:
    """Replace a clipped multi-panel title without touching the plotted data."""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    band_height = max(92, round(result.height * 0.085))
    draw.rectangle((0, 0, result.width, band_height), fill="white")
    font = pil_font(max(34, round(result.width * 0.027)))
    bbox = draw.textbbox((0, 0), title, font=font)
    x = (result.width - (bbox[2] - bbox[0])) // 2
    y = max(8, (band_height - (bbox[3] - bbox[1])) // 2 - bbox[1])
    draw.text((x, y), title, font=font, fill="#111111")
    return result


def retitle_component(
    image: Image.Image,
    title: str,
    cut_fraction: float,
) -> Image.Image:
    """Remove an inherited subfigure label and prepend a clean panel title."""
    cut = max(1, round(image.height * cut_fraction))
    body = image.crop((0, cut, image.width, image.height))
    band_height = max(72, round(image.height * 0.090))
    result = Image.new("RGB", (image.width, band_height + body.height), "white")
    result.paste(body, (0, band_height))
    draw = ImageDraw.Draw(result)
    font = pil_font(max(30, round(image.width * 0.027)))
    bbox = draw.textbbox((0, 0), title, font=font)
    x = (result.width - (bbox[2] - bbox[0])) // 2
    y = max(7, (band_height - (bbox[3] - bbox[1])) // 2 - bbox[1])
    draw.text((x, y), title, font=font, fill="#111111")
    return result


def compose_panel(
    filename: str,
    title: str,
    rows: list[list[str]],
    row_heights: list[int],
    images: dict[str, Image.Image],
    start_letter: int = 0,
) -> Path:
    width = 5400
    margin = 70
    gutter = 44
    header = 155
    title_font = pil_font(64, bold=True)
    canvas_height = header + margin + sum(row_heights) + gutter * (len(rows) - 1)
    canvas = Image.new("RGB", (width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 45), title, font=title_font, fill="#17212B")
    draw.line((margin, 128, width - margin, 128), fill="#CBD3DB", width=3)

    letter_index = start_letter
    y = header
    for row, row_height in zip(rows, row_heights):
        n = len(row)
        cell_width = (width - 2 * margin - gutter * (n - 1)) // n
        for column, key in enumerate(row):
            x = margin + column * (cell_width + gutter)
            image = add_letter(images[key], chr(ord("a") + letter_index))
            letter_index += 1
            contained = ImageOps.contain(
                image, (cell_width, row_height), Image.Resampling.LANCZOS
            )
            px = x + (cell_width - contained.width) // 2
            py = y + (row_height - contained.height) // 2
            canvas.paste(contained, (px, py))
            draw.rectangle(
                (x, y, x + cell_width, y + row_height),
                outline="#D7DEE5",
                width=2,
            )
        y += row_height + gutter

    path = PANELS / filename
    canvas.save(path, dpi=(300, 300), optimize=True)
    return path


def make_contact_sheet(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    thumb_width = 1100
    title_height = 65
    gap = 30
    cols = 2
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumbs.append(image.resize((thumb_width, round(image.height * ratio)), Image.Resampling.LANCZOS))
    row_heights = []
    for start in range(0, len(thumbs), cols):
        row_heights.append(max(image.height for image in thumbs[start:start + cols]) + title_height)
    canvas = Image.new(
        "RGB",
        (
            gap + cols * (thumb_width + gap),
            gap + sum(row_heights) + gap * (len(row_heights) - 1),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = pil_font(28, bold=True)
    y = gap
    for row_index, row_height in enumerate(row_heights):
        for col in range(cols):
            index = row_index * cols + col
            if index >= len(paths):
                break
            x = gap + col * (thumb_width + gap)
            draw.text((x, y), paths[index].name, font=label_font, fill="#17212B")
            canvas.paste(thumbs[index], (x, y + title_height))
        y += row_height + gap
    path = QA / "all_panels_contact_sheet.png"
    canvas.save(path, dpi=(150, 150), optimize=True)
    return path


def main():
    plot_style()
    images: dict[str, Image.Image] = {}
    manifest_components: dict[str, dict] = {}

    for name, spec in SPECS.items():
        image = crop_component(spec)
        if name == "threshold_size_width":
            image = retitle_component(
                image,
                "System size sets the width",
                cut_fraction=0.075,
            )
        elif name == "threshold_load_center":
            image = retitle_component(
                image,
                "Load sets the centre",
                cut_fraction=0.075,
            )
        elif name == "desync_order":
            image = retitle_component(
                image,
                "Desynchronization order parameter R(τ)",
                cut_fraction=0.045,
            )
        elif name == "desync_front":
            image = retitle_component(
                image,
                "Front width and participation",
                cut_fraction=0.060,
            )
        elif name == "desync_kymograph":
            image = retitle_component(
                image,
                r"Zero-delay kymograph",
                cut_fraction=0.060,
            )
        elif name == "e34_a":
            image = retitle_component(
                image,
                "The last two beads close the invariant circle",
                cut_fraction=0.105,
            )
        elif name == "e34_b":
            image = retitle_component(
                image,
                "The same connection, unrolled",
                cut_fraction=0.105,
            )
        elif name == "e34_d":
            image = retitle_component(
                image,
                "A surviving bead tours the whole ring",
                cut_fraction=0.105,
            )
        elif name == "depinning_left":
            image = retitle_component(
                image,
                "Cycle period diverges at depinning",
                cut_fraction=0.0,
            )
        elif name == "depinning_right":
            image = retitle_component(
                image,
                "The weakest bond exhibits SNIC slowing",
                cut_fraction=0.0,
            )
        path = COMPONENTS / f"{name}.png"
        image.save(path, dpi=(300, 300), optimize=True)
        images[name] = image
        manifest_components[name] = {
            "method": (
                "crop_with_clean_title"
                if name in {
                    "threshold_size_width",
                    "threshold_load_center",
                    "desync_order",
                    "desync_front",
                    "desync_kymograph",
                    "e34_a",
                    "e34_b",
                    "e34_d",
                    "depinning_left",
                    "depinning_right",
                }
                else spec.method
            ),
            "source": str(spec.source),
            "source_sha256": sha256(spec.source),
            "normalized_crop": list(spec.box),
            "output": str(path),
        }

    n6_image, n6_sources = regenerate_n6()
    images["n6_raw"] = n6_image
    manifest_components["n6_raw"] = {
        "method": "regenerated_from_npz",
        "sources": [str(path) for path in n6_sources],
        "source_sha256": {str(path): sha256(path) for path in n6_sources},
        "output": str(COMPONENTS / "n6_spectrum_raw.png"),
    }

    n2_image, n2_sources = regenerate_n2_without_seed51()
    images["n2a_raw_no_seed51"] = n2_image
    manifest_components["n2a_raw_no_seed51"] = {
        "method": "regenerated_from_json",
        "excluded_by_user_request": [51],
        "included_seeds": [42, 47, 52, 60],
        "sources": [str(path) for path in n2_sources],
        "source_sha256": {str(path): sha256(path) for path in n2_sources},
        "output": str(COMPONENTS / "n2a_without_seed51_raw.png"),
    }

    n2_static_image, n2_static_sources = regenerate_n2_static_softening()
    images["n2_c"] = n2_static_image
    manifest_components["n2_c"] = {
        "method": "regenerated_from_json",
        "included_seeds": [42, 47, 51, 52, 60],
        "sources": [str(path) for path in n2_static_sources],
        "source_sha256": {
            str(path): sha256(path) for path in n2_static_sources
        },
        "output": str(COMPONENTS / "n2_static_softening_raw.png"),
    }

    n4_image, n4_sources = regenerate_n4_critical_slowing()
    images["n4_fig4_b"] = n4_image
    manifest_components["n4_fig4_b"] = {
        "method": "regenerated_from_npz",
        "sources": [str(path) for path in n4_sources],
        "source_sha256": {
            str(path): sha256(path) for path in n4_sources
        },
        "output": str(COMPONENTS / "n4_critical_slowing_raw.png"),
    }

    n1_identity, n1_residual, n1_sources = regenerate_n1()
    images["n1_identity_raw"] = n1_identity
    images["n1_residual_raw"] = n1_residual
    for key, output in [
        ("n1_identity_raw", COMPONENTS / "n1_identity_raw.png"),
        ("n1_residual_raw", COMPONENTS / "n1_residual_raw.png"),
    ]:
        manifest_components[key] = {
            "method": "regenerated_from_csv",
            "sources": [str(path) for path in n1_sources],
            "source_sha256": {str(path): sha256(path) for path in n1_sources},
            "output": str(output),
        }

    panel_paths = [
        compose_panel(
            "Panel01_lyapunov_chaos.png",
            "Chaotic recall is robust and high-dimensional",
            [
                ["n6_raw", "q_a"],
                ["q_c", "q_f"],
            ],
            [1300, 1300],
            images,
        ),
        compose_panel(
            "Panel02_global_bifurcation.png",
            "From quenched memory folds to the global recall cycle",
            [
                ["bif_schematic", "mechanism_a"],
                ["mechanism_b", "mechanism_c"],
            ],
            [1300, 1300],
            images,
        ),
        compose_panel(
            "Panel03_static_folds_memory_distortion.png",
            "Static folds reshape memories before they disappear",
            [
                ["front_birth_left", "n4_fig2_a", "branch_geometry_a"],
                ["fold_n10000_left", "fold_n10000_right"],
            ],
            [1250, 1250],
            images,
        ),
        compose_panel(
            "Panel04_threshold_distributions.png",
            "Thresholds form a finite-size random ensemble",
            [
                ["alpha_law_hist", "alpha_law_cdf", "ae_normal"],
                ["ae_width", "n4_fig3_a", "ae_gap", "n4_fig3_b"],
                [
                    "n3_gaussian_evt",
                    "threshold_size_width",
                    "threshold_load_center",
                ],
            ],
            [1150, 1050, 1250],
            images,
        ),
        compose_panel(
            "Panel07_extreme_selection.png",
            "The largest static threshold selects collective cycle onset",
            [
                ["n4_fig4_c", "n1_identity_raw"],
                ["n1_residual_raw"],
            ],
            [1250, 900],
            images,
        ),
        compose_panel(
            "Panel05_snic_characterization.png",
            "The collective transition is a saddle-node on an invariant circle",
            [
                ["n2a_raw_no_seed51", "n4_fig4_b", "depinning_left"],
                ["n2_c", "depinning_right"],
                ["pacemaker_full"],
            ],
            [1120, 1120, 900],
            images,
        ),
        compose_panel(
            "Panel08_invariant_circle.png",
            "The recall orbit persists as a single invariant circle",
            [
                ["e34_a", "e34_b", "e34_d"],
                ["cycle_exact_recall", "cycle_ghost_bottleneck"],
            ],
            [1120, 1250],
            images,
        ),
        compose_panel(
            "Panel06_short_delay.png",
            "A finite delay synchronizes the propagating recall front",
            [
                ["n5a_left", "desync_order"],
                ["desync_front", "desync_kymograph"],
            ],
            [1150, 1150],
            images,
        ),
    ]

    contact_sheet = make_contact_sheet(panel_paths)
    manifest = {
        "created_in": str(OUT),
        "source_policy": (
            "All Desktop and historical numerical folders were read only. "
            "Every output was written under the Codex output directory."
        ),
        "components": manifest_components,
        "panels": [str(path) for path in panel_paths],
        "contact_sheet": str(contact_sheet),
        "current_figure_mapping": {
            "Figure 6": (
                "N1 static extreme versus dynamic onset, regenerated from the "
                "N1 CSV as two components and moved to the extreme-selection "
                "panel."
            ),
            "Figure 13": (
                "E36 implicit-delay panels intentionally omitted after the "
                "user correction; the short-delay panel now uses figP_desync_cdf."
            ),
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n".join(str(path) for path in panel_paths))


if __name__ == "__main__":
    main()
