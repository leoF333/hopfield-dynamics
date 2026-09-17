# Verified entry points

This public-facing layer was prepared on 17 September 2026 from a copy of the existing research repository. The original research directory was not edited. Original commits and research data are retained.

## Supported introductory workflow

- `python examples/quickstart.py`: replot the stored reference thresholds and integrate a new N=300, P=9 recall example. Output is separate from research data.
- `python examples/quickstart.py --data-only`: replot stored data without simulation.
- `python -m unittest discover -s tests -v`: six fast tests of the core numerical implementation and reference data.
- `python tools/verify_portfolio_data.py`: verify hashes of the shipped research data and figures.

Validation environment: Python 3.12 on macOS arm64, CPU, float64. Exact installed versions are recorded in `requirements-tested.txt`. Timing and output values from the demonstration are recorded in `examples/reference-summary.json`.

## What is not certified by this release

The historical E-series and N-series campaigns were not all rerun. Some preserve original archive-relative paths or depend on inputs not included in this compact checkout. The original runbook is retained for scientific provenance, but its existence is not a claim that every command is portable. Start with the verified commands above.

The original manuscript/figure discrepancies in `OPEN_ITEMS.md` remain visible. This release improves access and verifies the introductory workflow; it does not resolve those scientific questions or change their status.

## Original manifests

`data/MANIFEST_included.csv`, `tools/verify_manifest.py` and `docs/ARCHIVE_README.md` describe an earlier packaging snapshot. They are retained as historical records. The current data-and-figure inventory is `data/portfolio-sha256.json`, checked by `tools/verify_portfolio_data.py`. It covers research outputs, not generated demo files or mutable documentation.
