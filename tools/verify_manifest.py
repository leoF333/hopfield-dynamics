#!/usr/bin/env python3
"""Check the shipped tree against data/MANIFEST_included.csv.

Reports files that are missing, files whose size changed, and files present in the tree but
absent from the manifest. Exit code 0 if the tree matches, 1 otherwise.

Usage
-----
    python tools/verify_manifest.py
    python tools/verify_manifest.py --update      # rewrite the manifest from the tree
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "MANIFEST_included.csv"
IGNORE_PARTS = {"__pycache__", ".git", ".venv"}


def walk() -> dict[str, int]:
    out: dict[str, int] = {}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if IGNORE_PARTS & set(rel.parts) or p.name == ".DS_Store":
            continue
        out[rel.as_posix()] = p.stat().st_size
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true",
                    help="rewrite the manifest from the current tree instead of checking it")
    args = ap.parse_args()

    tree = walk()

    if args.update:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "bytes"])
            w.writeheader()
            for path in sorted(tree):
                w.writerow({"path": path, "bytes": tree[path]})
        print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(tree)} entries")
        return 0

    if not MANIFEST.exists():
        print(f"manifest not found: {MANIFEST}", file=sys.stderr)
        return 1

    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        listed = {r["path"]: int(r["bytes"]) for r in csv.DictReader(fh)}

    missing = sorted(set(listed) - set(tree))
    extra = sorted(set(tree) - set(listed))
    changed = sorted(p for p in set(listed) & set(tree) if listed[p] != tree[p])

    print(f"manifest: {len(listed)} entries   tree: {len(tree)} files")
    for label, items in (("missing", missing), ("size changed", changed), ("untracked", extra)):
        if items:
            print(f"\n{label} ({len(items)}):")
            for p in items[:40]:
                print("  ", p)
            if len(items) > 40:
                print(f"   … and {len(items) - 40} more")

    if not (missing or changed or extra):
        print("\ntree matches the manifest")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
