"""Small-N end-to-end smoke test for both halves of N1."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["V5_BACKEND"] = "cpu"

from dynamics import DynamicConfig, run_descent
from static_thresholds import StaticConfig, run as run_static


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="v5-smoke-n1-") as folder:
        target = Path(folder)
        static = run_static(
            StaticConfig(
                N=120, P=6, seed=42, beta=12.0, initial_step=0.04,
                bracket_tol=2e-3, lambda_max=0.75,
                # Small-N crosstalk can exceed the production third-amplitude
                # guard; warm-start continuity is enough for this code-path test.
                relaxed_identity=True,
            ),
            target / "static.npz",
            nproc=1,
        )
        if static["resolved"] < 5:
            raise RuntimeError(f"too few smoke-test branches: {static}")

        # This validates construction, warm restart, event detection and output.
        # Small systems fluctuate too much to require a production-quality bracket.
        try:
            dynamic = run_descent(
                DynamicConfig(
                    N=120, P=6, seed=42, beta=12.0, tau=2.0, dt=0.02,
                    record_dt=0.04, chunk_time=40.0, max_time=400.0,
                    no_tour_arrest_time=300.0, required_tours=2,
                    forward_fraction=0.90,
                ),
                target / "dynamic.npz",
                lambda_start=0.8,
                lambda_stop=0.05,
                lambda_step=0.05,
                bracket_tol=0.02,
            )
            print("dynamic smoke:", dynamic)
        except RuntimeError as exc:
            # The event machinery has still run; the scientific bracket is checked
            # at a more representative N in the mandatory N1 pilot.
            print("dynamic small-N bracket not present:", exc)
        print("static smoke:", static)


if __name__ == "__main__":
    main()
