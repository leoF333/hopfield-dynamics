"""
Render REPORT_static_bifurcation.md to a self-contained HTML with typeset math.

No LaTeX/pandoc needed: math is left verbatim and rendered in-browser by MathJax
(so $$...$$, \\boxed, \\begin{cases}, matrices, etc. all compile). Figures are
base64-embedded so the single .html file is fully portable.

Usage:  conda run -n mcmc_env python build_report_html.py
Output: results/REPORT_static_bifurcation.html
"""

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os, re, base64
import markdown

SRC     = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(SRC), "results")
MD_PATH  = os.path.join(RESULTS, "REPORT_static_bifurcation.md")
OUT_PATH = os.path.join(RESULTS, "REPORT_static_bifurcation.html")

FIGURES = [
    ("N=10000/lambda_c_vs_alpha.png",
     "Fig 4 (§5.3) — THE summary result: λ_c(α) at N=10⁴. Clean low-α curve (blue, 5 seeds) "
     "≈5% below the legacy analytic; the near-capacity α=0.10 point (red) collapses far below "
     "with a huge error bar — single-realization numerics break down near α_c≈0.138."),
    ("N=10000/fold_N10000_a0.05.png",
     "Fig 5 (§5.3) — α=0.05 multi-seed (clean): eig_max(M) rises smoothly to 0 (fold) and m1 curls; "
     "λ_c = 0.263 ± 0.002 across 5 seeds."),
    ("N=10000/fold_N10000_a0.1.png",
     "Fig 6 (§5.4) — α=0.10 multi-seed (near-capacity): folds scatter 0.03–0.20; a few seeds start "
     "with eig_max>0 (spurious unstable states) — the rough capacity landscape."),
    ("fixed_point/fig1_branch_N2000_a0.050_tau10_b20.0_s42.png",
     "Fig 1 (§5.1) — Memory branch (N=2000, α=0.05): overlaps m_ν vs λ, ending at the fold curl."),
    ("stability/fig3_maxre_N2000_a0.050_tau10_b20.0_s42.png",
     "Fig 3 (§5.1) — Stability margin max Re(z): spurious plateau then the real mode crossing 0 at "
     "the fold (saddle-node)."),
    ("N=10000/fig2_physical_roots_near_lc.png",
     "Fig 2 (§5.1) — EXACT physical characteristic roots (no artefacts) at six λ near λ_c, "
     "N=10⁴, via the reduced P×P spectrum. The real fold-mode (red) climbs to 0 at the fold "
     "while the complex modes (blue) stay left (Re ≤ −0.11) — saddle-node, no Hopf, certified "
     "to σ_min(Δ)≈10⁻¹⁷. (The λ=0.25 panel's real modes, ≈−0.13, were not captured by the "
     "automatic root-finder; the trend across the other panels is unambiguous.)"),
    ("stability/lambda_c_grid_v2_N2000.png",
     "Fig 7 (§5.2) — λ_c(α,τ) grid v2 (robust branch + exact reduced spectrum, τ=0–15, "
     "α=0.01–0.10): five flat lines — λ_c exactly τ-independent per α, all 50 points "
     "saddle-node/SNIC, no Hopf. α=0.10 dashed (near-capacity, indicative)."),
    ("stability/finite_size_scaling_a0.050_tau10_b20.0.png",
     "Fig 8 (§5.5) — Finite-size scaling (α=0.05): real vs complex modes; single-seed fold scatter."),
]


def protect_math(text):
    """Replace $$...$$ and $...$ spans with placeholders so Markdown leaves them
    alone (underscores, backslashes, etc.). Returns (text, display, inline)."""
    display, inline = [], []

    def grab_disp(m):
        display.append(m.group(0))
        return f"@@MJD{len(display)-1}@@"

    def grab_inl(m):
        inline.append(m.group(0))
        return f"@@MJI{len(inline)-1}@@"

    text = re.sub(r"\$\$.*?\$\$", grab_disp, text, flags=re.DOTALL)   # display first
    text = re.sub(r"\$[^$\n]+?\$", grab_inl, text)                    # then inline
    return text, display, inline


def restore_math(html, display, inline):
    for i, s in enumerate(display):
        html = html.replace(f"@@MJD{i}@@", s)
    for i, s in enumerate(inline):
        html = html.replace(f"@@MJI{i}@@", s)
    return html


def b64_img(rel):
    with open(os.path.join(RESULTS, rel), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Static fixed-point bifurcation — report</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    packages: {'[+]': ['ams', 'boldsymbol']}
  },
  loader: { load: ['[tex]/ams', '[tex]/boldsymbol'] }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root { --fg:#1a1a1a; --muted:#555; --rule:#e2e2e2; --accent:#2257a8; --bg:#fdfdfc; }
  html { font-size: 17px; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family: Georgia, 'Times New Roman', serif; line-height:1.62; }
  main { max-width: 820px; margin: 0 auto; padding: 56px 28px 120px; }
  h1,h2,h3 { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
             line-height:1.25; color:#111; }
  h1 { font-size: 1.9rem; margin:0 0 .2em; }
  h2 { font-size: 1.35rem; margin:2.2em 0 .6em; padding-top:.5em;
       border-top:1px solid var(--rule); }
  h3 { font-size: 1.08rem; margin:1.6em 0 .4em; color:#333; }
  p, li { color:var(--fg); }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size:.86em;
         background:#f0f0ee; padding:1px 5px; border-radius:4px; }
  pre { background:#f6f6f4; border:1px solid var(--rule); border-radius:8px;
        padding:14px 16px; overflow-x:auto; }
  pre code { background:none; padding:0; font-size:.84em; }
  blockquote { margin:1.2em 0; padding:.6em 1.1em; border-left:4px solid var(--accent);
               background:#f3f7fc; color:#243; border-radius:0 6px 6px 0; }
  blockquote p { margin:.4em 0; }
  table { border-collapse:collapse; margin:1.2em 0; width:100%; font-size:.92rem;
          font-family: -apple-system,'Helvetica Neue',Arial,sans-serif; }
  th,td { border:1px solid var(--rule); padding:7px 11px; text-align:left; }
  th { background:#f1f1ee; }
  tr:nth-child(even) td { background:#faf9f7; }
  hr { border:none; border-top:1px solid var(--rule); margin:2.4em 0; }
  figure { margin:1.8em 0; text-align:center; }
  figure img { max-width:100%; border:1px solid var(--rule); border-radius:8px;
               background:#fff; padding:6px; }
  figcaption { font-size:.86rem; color:var(--muted); margin-top:.5em;
               font-family:-apple-system,'Helvetica Neue',Arial,sans-serif; }
  .mjx-container { overflow-x:auto; overflow-y:hidden; max-width:100%; }
</style>
</head>
<body>
<main>
__BODY__
</main>
</body>
</html>
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=MD_PATH, help="markdown source")
    ap.add_argument("--out", default=OUT_PATH, help="html output")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the embedded figure gallery")
    a = ap.parse_args()

    raw = open(a.md, encoding="utf-8").read()
    protected, display, inline = protect_math(raw)
    body = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = restore_math(body, display, inline)

    if not a.no_figures:
        gallery = ['<hr>', '<h2 id="figures">Figures</h2>']
        for rel, cap in FIGURES:
            try:
                data = b64_img(rel)
            except FileNotFoundError:
                gallery.append(f"<p><em>(missing figure: {rel})</em></p>")
                continue
            gallery.append(
                f'<figure><img src="data:image/png;base64,{data}" alt="{cap}">'
                f'<figcaption>{cap}</figcaption></figure>')
        body += "\n" + "\n".join(gallery)

    html = TEMPLATE.replace("__BODY__", body)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
