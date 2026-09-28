# Notebooks

Analysis and exploration notebooks. Notebooks that produce a paper figure
live in that figure's folder instead (`figures/figN/figN.ipynb`).

## Conventions

- Name notebooks `NN_short_description.ipynb` so they sort in the order
  they should be read, e.g. `01_trough_detection_check.ipynb`,
  `02_phase_offset_sweep.ipynb`, `03_jitter_frequency_sweep.ipynb`.
- Start from `00_template.ipynb`. Its first cell finds the repo root, sets up
  imports, and turns on autoreload so edits to `flutter/` show up without
  restarting the kernel.
- Put reusable functions in the `flutter/` package, not in notebook cells.
  If two notebooks need the same function, move it into `flutter/`.
- Outputs are stripped on commit (see below). A notebook should run
  top-to-bottom on a fresh kernel.

## Output stripping

After creating the environment, run once in the repo:

    nbstripout --install

This removes cell outputs and execution counts when you commit, which keeps
the repo small and diffs readable. Your local copy keeps its outputs.
