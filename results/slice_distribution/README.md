# Slice Distribution

This folder documents the slice-count evaluation used to study how
reconstruction quality changes with the number of observed slices.

Included:

- `evaluation_summary.json`
  Compact summary for the uniform slice-count evaluations at 32, 64, and 128
  slices, plus the 64-slice comparison against a regression baseline.
- `eval_uniform/`
  Aggregate metric snapshots for the uniform evaluations.
- `fixed_split_case_ids.json`
  Case-ID-only description of the fixed split used for the later comparison
  snapshot.
- `fixed_split_eval/`
  Summary JSON/CSV files for the fixed-split evaluation against nearest and
  regression baselines.
- `appendix_figures/`
  Selected images used in the thesis appendix, including MIP comparisons and
  visual sanity checks.

This block is useful both for thesis figures and for answering the practical
question of how quickly quality saturates as slice coverage increases.
