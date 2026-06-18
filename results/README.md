# Thesis Results

This directory is a curated snapshot of the experiment artifacts that are most
relevant for the bachelor thesis. It is intentionally smaller and cleaner than
the original working `runs/` directories.

The three main blocks are:

- [`regularization/`](regularization/README.md)
  Regularization and robustness study outputs, including summary CSVs, LaTeX
  tables, and the exported thesis figures.
- [`slice_distribution/`](slice_distribution/README.md)
  Evaluation across different slice counts plus appendix figures and a fixed
  split snapshot with baseline comparisons.
- [`ultrasound_finetune/`](ultrasound_finetune/README.md)
  Comparison between the baseline model and a finetuned
  `ultrasound_probe` variant, including aggregate evaluation CSVs and renamed
  qualitative PDF figures.

The machine-readable entry point is [`index.json`](index.json).
