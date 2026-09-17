#!/usr/bin/env python3
"""Hash-check research data and figures. --record creates a new inventory deliberately."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/portfolio-sha256.json"

def hashes():
    result = {}
    for base in (ROOT / "data", ROOT / "figures"):
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p == MANIFEST or p.name == ".DS_Store" or "__pycache__" in p.parts:
                continue
            result[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record",action="store_true")
    args=parser.parse_args()
    actual=hashes()
    if args.record:
        MANIFEST.write_text(json.dumps(actual,indent=2)+"\n")
        print(f"Recorded {len(actual)} files")
    else:
        expected=json.loads(MANIFEST.read_text())
        mismatch=[p for p in set(actual)|set(expected) if actual.get(p)!=expected.get(p)]
        if mismatch:
            print("Integrity mismatch:\n"+"\n".join(sorted(mismatch)))
            raise SystemExit(1)
        print(f"Verified {len(actual)} research data and figure files")
