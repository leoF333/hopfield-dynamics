"""Run in a normal macOS Terminal before selecting V5_BACKEND=mlx."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import os
import time

os.environ["V5_BACKEND"] = "mlx"

import numpy as np

from couplings import Couplings, make_patterns


def main() -> None:
    N, P = 2000, 100
    xi, xis = make_patterns(N, P, 42)
    coupling = Couplings(xi, xis)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(N)
    observed = coupling.apply_C(x, 0.37)
    overlaps = xi @ x / N
    expected = (1.0 - 0.37) * (xi.T @ overlaps) + 0.37 * (xis.T @ overlaps)
    relative = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    started = time.perf_counter()
    for _ in range(100):
        coupling.apply_C(x, 0.37)
    elapsed = time.perf_counter() - started
    print({
        "backend": coupling.backend,
        "relative_error_vs_float64": relative,
        "100_pattern_projections_seconds": elapsed,
        "passed": relative < 5e-5,
    })
    if relative >= 5e-5:
        raise SystemExit("MLX preflight failed the numerical comparison")


if __name__ == "__main__":
    main()

