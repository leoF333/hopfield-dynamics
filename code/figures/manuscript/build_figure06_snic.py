#!/usr/bin/env python3
"""
Rebuild the SNIC panel (manuscript Figure 6) with three sub-panels removed.

Dropped, at the author's request, as the least load-bearing of the seven:
  * the cycle-side tour period versus lambda - lambda*        (former panel b)
  * the 'three clocks' half of the pacemaker component        (former panel f, left)
  * the band of front-displacement exponents                  (former panel g, right)

What remains keeps its original reading order and is re-lettered a-f:
  (a) cycle-side critical slowing across four disorder realizations   [N2]
  (b) the cycle period diverging at depinning                         [E1/E8c/E9]
  (c) static softening of the leading real eigenvalue beyond the local window [N2]
  (d) the weakest bond, and only the weakest bond, showing SNIC slowing [E9]
  (e) the escape time collapsing across three delays                  [E1/E4]
  (f) Floquet multipliers fleeing the unit circle as the cycle dies    [E11/E13]

Layout goes from four stacked rows to a 2 x 3 grid: the same evidence, half the
vertical space.

Components are reused verbatim from 4_campaigns/figure_panels/components/, so no
number is recomputed here; only (e) and (f) are re-cropped out of a wider source
and given a clean title.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

COMPONENTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("components")
FLOQUET = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figE_floquet_spectrum_v2.png")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("Figure06_snic.png")

TITLE = "The collective transition is a saddle-node on an invariant circle"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def pil_font(size: int, bold: bool = False):
    for path in FONT_CANDIDATES:
        p = Path(path)
        if not p.exists():
            continue
        if bold and "Bold" not in path:
            bold_path = path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
            if Path(bold_path).exists():
                return ImageFont.truetype(bold_path, size)
        if bold and "Bold" not in path:
            continue
        if not bold and "Bold" in path:
            continue
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def trim_white(image: Image.Image, padding: int = 16) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, background).convert("L").point(
        lambda v: 255 if v > 10 else 0)
    box = diff.getbbox()
    if box is None:
        return rgb
    return rgb.crop((max(0, box[0] - padding), max(0, box[1] - padding),
                     min(rgb.width, box[2] + padding),
                     min(rgb.height, box[3] + padding)))


def vertical_gutter(image: Image.Image, lo: float = 0.35, hi: float = 0.65) -> int:
    """Return the x of the widest all-white column band in [lo, hi] of the width."""
    a = np.asarray(image.convert("L"))
    white = (a > 246).all(axis=0)
    x0, x1 = int(lo * image.width), int(hi * image.width)
    best, run_start, best_len = (x0 + x1) // 2, None, 0
    for x in range(x0, x1):
        if white[x]:
            if run_start is None:
                run_start = x
        else:
            if run_start is not None and x - run_start > best_len:
                best_len, best = x - run_start, (run_start + x) // 2
            run_start = None
    if run_start is not None and x1 - run_start > best_len:
        best = (run_start + x1) // 2
    return best


def retitle(image: Image.Image, title: str, cut_fraction: float) -> Image.Image:
    """Drop an inherited sub-figure label and prepend a clean panel title."""
    cut = max(1, round(image.height * cut_fraction))
    body = image.crop((0, cut, image.width, image.height))
    band = max(72, round(image.height * 0.090))
    result = Image.new("RGB", (image.width, band + body.height), "white")
    result.paste(body, (0, band))
    draw = ImageDraw.Draw(result)
    font = pil_font(max(30, round(image.width * 0.027)))
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.text(((result.width - (bbox[2] - bbox[0])) // 2,
               max(7, (band - (bbox[3] - bbox[1])) // 2 - bbox[1])),
              title, font=font, fill="#111111")
    return result


def add_letter(image: Image.Image, letter: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = pil_font(max(28, round(min(result.size) * 0.050)), bold=True)
    text = f"({letter})"
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(8, round(h * 0.25))
    draw.rounded_rectangle((4, 4, 4 + w + 2 * pad, 4 + h + 2 * pad), radius=pad,
                           fill="white", outline="#D4DAE0", width=max(1, pad // 5))
    draw.text((4 + pad, 4 + pad - bbox[1]), text, font=font, fill="#151F28")
    return result


def erase_slate_annotation(image: Image.Image, box: tuple[int, int, int, int]
                           ) -> Image.Image:
    """Whiten the slate-grey in-plot annotation inside *box*, leaving data alone.

    The annotation collides with the legend in the inherited component.  Its ink
    is a distinct blue-grey (111, 123, 131); black legend text and the coloured
    markers are neutral or saturated, so a chroma test removes only the text.
    """
    a = np.asarray(image).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    depth = 255.0 - r
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(depth > 1, (b - r) / np.maximum(depth, 1), 0.0)
        gratio = np.where(depth > 1, (g - r) / np.maximum(depth, 1), 0.0)
    mask = ((depth > 25) & (np.abs(ratio - 0.139) < 0.055)
            & (np.abs(gratio - 0.083) < 0.05) & (b <= 245))
    window = np.zeros_like(mask)
    x0, y0, x1, y1 = box
    window[y0:y1, x0:x1] = True
    out = np.asarray(image).copy()
    out[mask & window] = [255, 255, 255]
    return Image.fromarray(out)


def erase_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Paint a rectangle white (used to drop a clipped stray glyph)."""
    out = image.copy()
    ImageDraw.Draw(out).rectangle(box, fill="white")
    return out


def compose(rows, row_heights, images, title, out: Path) -> Path:
    width, margin, gutter, header = 5400, 70, 44, 155
    canvas_height = header + margin + sum(row_heights) + gutter * (len(rows) - 1)
    canvas = Image.new("RGB", (width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 45), title, font=pil_font(64, bold=True), fill="#17212B")
    draw.line((margin, 128, width - margin, 128), fill="#CBD3DB", width=3)

    index, y = 0, header
    for row, row_height in zip(rows, row_heights):
        n = len(row)
        cell_width = (width - 2 * margin - gutter * (n - 1)) // n
        for column, key in enumerate(row):
            x = margin + column * (cell_width + gutter)
            image = add_letter(images[key], chr(ord("a") + index))
            index += 1
            contained = ImageOps.contain(image, (cell_width, row_height),
                                         Image.Resampling.LANCZOS)
            canvas.paste(contained, (x + (cell_width - contained.width) // 2,
                                     y + (row_height - contained.height) // 2))
            draw.rectangle((x, y, x + cell_width, y + row_height),
                           outline="#D7DEE5", width=2)
        y += row_height + gutter
    canvas.save(out, dpi=(300, 300), optimize=True)
    return out


def main() -> None:
    images = {
        "n2a": Image.open(COMPONENTS / "n2a_without_seed51_raw.png").convert("RGB"),
        "depin_left": Image.open(COMPONENTS / "depinning_left.png").convert("RGB"),
        # A clipped stray glyph sits above the axes of this component; it carries
        # no information and is painted out.
        "n2_c": erase_box(
            Image.open(COMPONENTS / "n2_c.png").convert("RGB"),
            (152, 130, 208, 168)),
        "depin_right": Image.open(COMPONENTS / "depinning_right.png").convert("RGB"),
    }

    # (e) the right half of the pacemaker component: the escape-time collapse.
    pace = Image.open(COMPONENTS / "pacemaker_full.png").convert("RGB")
    pace = erase_slate_annotation(pace, (pace.width // 2, 170, pace.width, 350))
    split = vertical_gutter(pace, 0.42, 0.58)
    escape = trim_white(pace.crop((split, 0, pace.width, pace.height)), padding=10)
    images["escape"] = retitle(
        escape, "One escape time: the three delays collapse", cut_fraction=0.115)

    # (f) the left half of the E11/E13 Floquet figure, below its shared suptitle.
    flq = Image.open(FLOQUET).convert("RGB")
    split = vertical_gutter(flq, 0.44, 0.56)
    left = flq.crop((0, round(flq.height * 0.14), split, flq.height))
    left = trim_white(left, padding=10)
    images["floquet"] = retitle(
        left, "No multiplier approaches the unit circle as the cycle dies",
        cut_fraction=0.135)

    for key, image in images.items():
        print(f"  {key:12s} {image.width} x {image.height}")

    path = compose(
        [["n2a", "depin_left", "n2_c"],
         ["depin_right", "escape", "floquet"]],
        [1250, 1250],
        images, TITLE, OUT,
    )
    final = Image.open(path)
    print(f"wrote {path}  {final.width} x {final.height}")


if __name__ == "__main__":
    main()
