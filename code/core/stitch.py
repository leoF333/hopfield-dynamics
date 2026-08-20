
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import os
from PIL import Image

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 (folder reorganisation); see REORGANISATION_LOG.md ---
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
CHICAGO_S = str(CHICAGO)
# ---------------------------------------------------------------------------


def main():
    alpha_list = [0.01, 0.05, 0.1]
    tau_list = [1, 10, 20]
    outdir = CHICAGO_S + "/3_numerics/results"
    
    images = []
    width = 0
    height = 0
    
    for alpha in alpha_list:
        row_imgs = []
        for tau in tau_list:
            if alpha == 0.1 and tau == 20:
                name = f"stability_scissors_N300_tau20_a0.100.png"
            else:
                name = f"stability_scissors_N1000_tau{tau}_a{alpha:.3f}.png"
            path = os.path.join(outdir, name)
            img = Image.open(path)
            row_imgs.append(img)
            width = img.width
            height = img.height
        images.append(row_imgs)
        
    synth_img = Image.new('RGB', (width * 3, height * 3))
    
    for i in range(3):
        for j in range(3):
            synth_img.paste(images[i][j], (j * width, i * height))
            
    # Add borders to separate the grid better if needed
    synth_img.save(os.path.join(outdir, "synthesis_grid_N1000_hybrid.png"))
    print("Saved synthesis_grid_N1000_hybrid.png")

if __name__ == "__main__":
    main()
