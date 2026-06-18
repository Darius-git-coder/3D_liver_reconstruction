# Regularization

This folder contains the curated outputs of the regularization study used in
the thesis discussion of stability, convergence, sampling variability, and
robustness.

Included:

- `study_summary.json`
  Clean summary of the study design, selected test cases, and headline
  settings.
- `split_case_ids.json`
  Case-ID-only version of the evaluation split used for this study.
- `selected_cases.csv`
  The explicit case subset evaluated in the thesis plots and tables.
- `summary/*.csv`
  Long-form summary tables for convergence, noise, sampling, sensitivity, and
  robustness.
- `tables/*.tex`
  Exported LaTeX tables used to support thesis table generation.
- `figures/*.pdf`
  Exported figures, including `thesis_overview.pdf`,
  `convergence_dual_metrics.pdf`, and `robustness_primary_metric.pdf`.

Source script in this repository:

- `scripts/evaluate_regularization_behavior.py`
