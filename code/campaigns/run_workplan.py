"""Preflight and dispatch for the N1--N10 numerical work plan."""
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
import subprocess
import sys
from pathlib import Path

from campaign_specs import SPECS
from v5_paths import ROOT


def command_for(campaign: str, stage: str, extra: list[str]) -> list[str]:
    smoke = ["--smoke"] if stage == "smoke" else []
    scripts = {
        "N2": "n2_local_laws.py",
        "N3": "n3_reanalyse.py",
        "N4": "n4_figures.py",
        "N5A": "n5a_delay_scan.py",
        "N5B": "n5b_long_delay.py",
        "N6": "n6_lyapunov_campaign.py",
        "N7": "n7_correlations.py",
        "N8": "n8_highload.py",
        "N9": "n9_stationarity.py",
        "N10": "n10_audit.py",
    }
    if campaign == "N1":
        if stage == "smoke":
            return [sys.executable, str(ROOT / "tests" / "smoke_n1.py"), *extra]
        raise SystemExit(
            "N1 production has independent static and dynamic components. "
            "Use ./run.sh N1-static ... and ./run.sh N1-dynamic ... so each "
            "heavy dynamic seed is explicitly timed and logged."
        )
    script = scripts[campaign]
    command = [sys.executable, str(ROOT / "src" / script)]
    if campaign == "N3":
        command.extend(["--bootstrap", "100" if stage == "smoke" else "5000"])
    elif campaign != "N4":
        command.extend(smoke)
    command.extend(extra)
    return command


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] == "list":
        print(json.dumps(SPECS, indent=2, ensure_ascii=False))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", choices=list(SPECS))
    parser.add_argument("--stage", choices=["smoke", "pilot", "production"],
                        default="smoke")
    parser.add_argument("--confirm-heavy", action="store_true")
    args, extra = parser.parse_known_args()
    spec = SPECS[args.campaign]
    print(f"{args.campaign}: {spec['why']}")
    print(f"Estimated production time: {spec['estimated_production_time']}")
    if args.stage == "production" and not args.confirm_heavy:
        raise SystemExit(
            "Production run not started. Re-run with --confirm-heavy after "
            "reviewing the estimate above."
        )
    command = command_for(args.campaign, args.stage, extra)
    print("Launching:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

