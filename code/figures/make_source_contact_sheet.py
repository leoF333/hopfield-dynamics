#!/usr/bin/env python3
"""Build a read-only visual inventory of the requested source figures."""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 by folder reorganisation -----------------------------
# All absolute paths below now resolve from a single root, overridable with the
# CHICAGO_ROOT environment variable.  See ../../REORGANISATION_LOG.md
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
# -----------------------------------------------------------------------------



SOURCE_DIR = CHICAGO / "4_campaigns/figure_sources"
EXTRA_DIR = CHICAGO / "4_campaigns/manuscript_audits/agent_redacteur/figures"
OUTPUT = CHICAGO / "4_campaigns/figure_panels/qa/source_contact_sheet.png"

FILES = [
    SOURCE_DIR / "N6_multiseed_lyapunov.png",
    SOURCE_DIR / "figQ_chaos_nature.png",
    SOURCE_DIR / "diagramme de bifurcation v1.png",
    SOURCE_DIR / "fig1_mechanism.png",
    SOURCE_DIR / "figS_front_birth.png",
    SOURCE_DIR / "N4_Figure2.png",
    SOURCE_DIR / "figU_branch_geometry.png",
    SOURCE_DIR / "fold_N10000_a0.05.png",
    SOURCE_DIR / "figZ_alpha_law.png",
    SOURCE_DIR / "figAE_threshold_scaling.png",
    SOURCE_DIR / "N3_E30_seed_aware.png",
    SOURCE_DIR / "N4_Figure3.png",
    SOURCE_DIR / "N2_critical_laws.png",
    SOURCE_DIR / "N4_Figure4.png",
    SOURCE_DIR / "figB_depinning_transition.png",
    SOURCE_DIR / "figE34_invariant_circle.png",
    SOURCE_DIR / "N5A_delay_boundary.png",
    EXTRA_DIR / "E36_implicit_delay.png",
]


def font(size: int):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main():
    thumb_w, thumb_h = 520, 330
    label_h, gap = 52, 22
    cols = 3
    rows = (len(FILES) + cols - 1) // cols
    canvas = Image.new(
        "RGB",
        (
            gap + cols * (thumb_w + gap),
            gap + rows * (thumb_h + label_h + gap),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(18)
    meta_font = font(14)

    for index, path in enumerate(FILES):
        row, col = divmod(index, cols)
        x0 = gap + col * (thumb_w + gap)
        y0 = gap + row * (thumb_h + label_h + gap)
        image = Image.open(path).convert("RGB")
        original_size = image.size
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        px = x0 + (thumb_w - image.width) // 2
        py = y0 + (thumb_h - image.height) // 2
        canvas.paste(image, (px, py))
        draw.rectangle(
            (x0, y0, x0 + thumb_w, y0 + thumb_h),
            outline="#B9C2CC",
            width=2,
        )
        draw.text(
            (x0, y0 + thumb_h + 5),
            f"{index + 1:02d}. {path.name}",
            fill="#15212B",
            font=label_font,
        )
        draw.text(
            (x0, y0 + thumb_h + 29),
            f"{original_size[0]} x {original_size[1]} px",
            fill="#637181",
            font=meta_font,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(150, 150))


if __name__ == "__main__":
    main()
