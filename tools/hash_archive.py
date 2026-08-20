#!/usr/bin/env python3
"""Fill the sha256 column of data/MANIFEST_excluded.csv from a local copy of the archive.

The manifest lists every archived file that is not shipped in this repository, by its path
relative to the archive root. Hashing ~13 GB over a network mount is slow, so the column
ships empty; run this against a local copy and it will fill in.

Resumable: rows that already carry a hash are skipped, and the file is rewritten after
every batch, so interrupting it costs at most one batch.

Usage
-----
    python tools/hash_archive.py /path/to/Chicago
    python tools/hash_archive.py /path/to/Chicago --batch 200 --limit 5000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

FIELDS = ["archive_path", "bytes", "sha256", "reason"]


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while (block := fh.read(chunk)):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive_root", type=Path,
                    help="local path to the Chicago archive root")
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "MANIFEST_excluded.csv")
    ap.add_argument("--batch", type=int, default=100,
                    help="rewrite the manifest every N hashed files (default 100)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = no limit)")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not args.archive_root.is_dir():
        print(f"archive root not found: {args.archive_root}", file=sys.stderr)
        return 1

    with args.manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    todo = [r for r in rows if not r.get("sha256")]
    print(f"{len(rows)} rows, {len(todo)} still to hash")

    def flush() -> None:
        tmp = args.manifest.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(args.manifest)

    done = missing = 0
    for row in todo:
        if args.limit and done >= args.limit:
            break
        p = args.archive_root / row["archive_path"]
        if not p.is_file():
            missing += 1
            continue
        row["sha256"] = sha256(p)
        done += 1
        if done % args.batch == 0:
            flush()
            print(f"  hashed {done}/{len(todo)}", flush=True)

    flush()
    print(f"done: {done} hashed, {missing} not found on disk, "
          f"{sum(1 for r in rows if not r.get('sha256'))} still empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
