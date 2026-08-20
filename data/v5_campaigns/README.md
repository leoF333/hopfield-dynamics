# Version 5 numerical campaigns

This directory is the writable, reproducible workspace for tasks N1--N10 in
`../paper_v5/NUMERICAL_WORKPLAN.md`.

The current execution state, obtained results and exact remaining work are maintained
in `RUN_STATUS.md`.  Read that file first after any interruption or restart.

The historical project at

`/Users/leoflack/Desktop/Recherche/Chicago/numerics copie théorie claude + implémentation GPT`

is a read-only reference.  No script in this directory writes below that path.

## Execution environments

All commands use the conda environment `mcmc_env`.

- `V5_BACKEND=cpu` selects the independent NumPy/float64 implementation.  This is
  the mandatory validation backend and the fallback used inside the Codex sandbox.
- `V5_BACKEND=mlx` selects MLX for the repeated pattern projections while retaining
  float64 states, diagnostics and saved data.  It is intended for a normal macOS
  Terminal session with access to Metal.

Run the smoke suite:

```bash
./run.sh smoke
```

Validate MLX from a normal macOS Terminal before enabling it:

```bash
./run.sh mlx-preflight
```

List the campaigns and their configured production grids:

```bash
./run.sh list
```

Run a production campaign only after its smoke test and benchmark have passed.
N1 is deliberately split into independent static and dynamic commands:

```bash
./run.sh N1-static --N 2000 --P 100 --seed 42 --nproc 4
./run.sh N1-dynamic --N 2000 --P 100 --seed 42
./run.sh N5A --stage production --confirm-heavy
```

Raw results are written below `runs/`; derived figures below `figures/`; logs and
machine-readable manifests below `logs/` and `manifests/`.
