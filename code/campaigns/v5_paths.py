"""Paths, immutable-reference checks, and common provenance for V5 campaigns."""
from __future__ import annotations

# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------


import hashlib
import json
import os
import platform
import subprocess
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import os as _os
from pathlib import Path as _Path
# --- PATCHED 2026-08-19 by folder reorganisation -----------------------------
# All absolute paths below now resolve from a single root, overridable with the
# CHICAGO_ROOT environment variable.  See ../../REORGANISATION_LOG.md
CHICAGO = _Path(_os.environ.get("CHICAGO_ROOT", "/Users/leoflack/Desktop/Recherche/Chicago"))
# -----------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "manifests"
LOGS = ROOT / "logs"
CACHE = ROOT / "cache"

REFERENCE_ROOT = CHICAGO / "3_numerics"
REFERENCE_SRC = REFERENCE_ROOT / "src"
WORKPLAN = ROOT.parent / "paper_v5" / "NUMERICAL_WORKPLAN.md"


def prepare_paths() -> None:
    for path in (RUNS, FIGURES, REPORTS, MANIFESTS, LOGS, CACHE / "matplotlib"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))


def activate_reference_modules() -> None:
    """Append, never prepend, the read-only historical source tree.

    The local `couplings` module must remain first so CPU mode never imports MLX.
    """
    here = str(Path(__file__).resolve().parent)
    ref = str(REFERENCE_SRC)
    if here not in sys.path:
        sys.path.insert(0, here)
    if ref not in sys.path:
        sys.path.append(ref)


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def reference_snapshot(relative_paths: Iterable[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for relative in relative_paths:
        path = REFERENCE_ROOT / relative
        snapshot[relative] = sha256(path) if path.is_file() else "MISSING"
    return snapshot


def environment_manifest() -> dict:
    def version(module: str) -> str:
        try:
            mod = __import__(module)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - recorded for diagnostics
            return f"unavailable: {type(exc).__name__}: {exc}"

    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=False, timeout=5
        ).stdout.strip() or "not-a-git-worktree"
    except Exception:
        git = "unavailable"
    local_sources = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted((ROOT / "src").glob("*.py"))
    }
    reference_sources = {}
    baseline = MANIFESTS / "reference_baseline.json"
    if baseline.exists():
        try:
            reference_sources = json.loads(
                baseline.read_text(encoding="utf-8")
            ).get("sha256", {})
        except (OSError, json.JSONDecodeError):
            reference_sources = {"baseline": "UNREADABLE"}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backend": os.environ.get("V5_BACKEND", "cpu"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "git_commit": git,
        "workplan_sha256": sha256(WORKPLAN),
        "reference_root": str(REFERENCE_ROOT),
        "local_source_sha256": local_sources,
        "reference_source_sha256": reference_sources,
    }


def write_json(path: Path, payload: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            return clean(value.tolist())
        if isinstance(value, complex):
            return [clean(value.real), clean(value.imag)]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


prepare_paths()
