# Ultrasound Finetuning

This folder contains the curated comparison between the baseline reconstruction
model and the model finetuned under the more ultrasound-like
`ultrasound_probe` geometry.

Included:

- `experiment_summary.json`
  Clean summary of the finetuning protocol, evaluation seeds, and aggregate
  key metrics.
- `eval_case_ids.json`
  Case-ID-only version of the evaluation split.
- `eval/metrics_summary_by_model.csv`
  Aggregate metric summary by model.
- `eval/metrics_summary_by_model_seed.csv`
  Aggregate metric summary broken down by evaluation seed.
- `eval/metrics_per_case.csv`
  Per-case evaluation table.
- `finetune/history.json`
  Training history of the finetuning stage.
- `finetune/loss_history.png`
  Training-curve plot for the finetuning run.
- `qualitative_cases.json`
  Clean manifest of the selected qualitative cases and the renamed PDF files.
- `qualitative/`
  Renamed PDF exports for best, average, and worst cases at 32, 64, and 128
  slices.

Thesis figure mapping:

- `qualitative/slices_64/average_matrix.pdf`
  Corresponds to the average-case matrix figure under `ultrasound_probe`.
- `qualitative/slices_32/worst_comparison.pdf`
  Corresponds to the worst-case comparison figure under `ultrasound_probe`.

Source scripts in this repository:

- `scripts/compare_best_run_ultrasound.py`
- `scripts/run_multislice_ultrasound_eval_and_export.py`
- `scripts/export_thesis_multislice_qualitative_figures.py`
