# Liver Sparse Reconstruction Core

Core research repository for sparse-to-dense 3D liver reconstruction from CT volumes with synthetically generated sparse slice observations.

This project contains the standalone codebase for:

- preprocessing liver CT volumes into a consistent cubic representation,
- generating sparse slice observations on the fly,
- training 3D reconstruction and inpainting networks,
- evaluating reconstruction quality on fixed cohort splits,
- running ablation studies and benchmark experiments,
- producing quantitative and qualitative analysis artifacts for thesis work.

The repository is intentionally focused on the reconstruction core and excludes thesis documents, generated run artifacts, and model checkpoints.

## Highlights

- Fixed `train/val/test` split workflow via persisted split manifests
- Multiple model variants: `partial`, `baseline`, and `gated`
- Hard data-consistency constraint on known voxels
- Cohort-level evaluation with case-wise metrics and summary statistics
- Benchmarking, visualization, and regularization-behavior analysis scripts
- Reproducibility metadata stored alongside training and evaluation runs

## Repository Layout

- `inpainting3d/`
  Core package with data handling, sparse acquisition, models, losses, metrics, statistics, split management, preprocessing, and utilities.
- `scripts/`
  Training, evaluation, visualization, benchmarking, split creation, ablation, and analysis entry points.
- `configs/`
  Example configs for smoke, short, and final ablation stages.

## Data Assumptions

This repository expects dense 3D liver CT volumes stored as `.nii.gz` files.

In the typical workflow, the offline preprocessing pipeline has already:

- normalized intensities,
- aligned volumes to a canonical orientation,
- cropped and padded the liver region,
- resampled each case to a fixed cubic shape such as `64^3` or `128^3`.

Sparse observations are not stored as separate files. They are generated at runtime from the dense target volumes by `simulate_sparse_acquisition(...)`, which is why large evaluation studies can still take substantial time.

## Installation

Python `>= 3.10` is required.

Create a virtual environment and install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If you prefer requirements-based installation:

```powershell
python -m pip install -r requirements.txt
```

## Recommended Workflow

### 1. Create a persistent cohort split

Create the split once and reuse it across all experiments:

```powershell
python scripts/create_data_split.py `
  --data ".\data\preprocessed_liver_ct_128\*.nii.gz" `
  --out ".\runs\splits\liver_seed1337_split_128.json" `
  --seed 1337 `
  --train_ratio 0.7 `
  --val_ratio 0.15 `
  --test_ratio 0.15
```

Generated split artifacts:

- `split_manifest.json`
- `split_manifest_train.txt`
- `split_manifest_val.txt`
- `split_manifest_test.txt`

### 2. Train a reconstruction model

Example training run with a `PartialResUNet3D` model:

```powershell
python scripts/train_inpainting.py `
  --data ".\data\preprocessed_liver_ct_128\*.nii.gz" `
  --split_file ".\runs\splits\liver_seed1337_split_128.json" `
  --out ".\runs\partialresunet3d_full\train" `
  --model partial `
  --dim 128 `
  --epochs 500 `
  --batch 2 `
  --slices_min 8 `
  --slices_max 128 `
  --w_grad 0.05 `
  --w_ms_ssim 0.2 `
  --amp `
  --deterministic
```

Training stores:

- `config.json`
- `history.json`
- `reproducibility.json`
- `best_model.pt`
- `checkpoints/last.pt`
- run-local split manifest artifacts

### 3. Evaluate on the fixed test split

Example evaluation of a trained checkpoint:

```powershell
python scripts/evaluate_inpainting.py `
  --data ".\data\preprocessed_liver_ct_128\*.nii.gz" `
  --split_file ".\runs\splits\liver_seed1337_split_128.json" `
  --split test `
  --weights ".\runs\partialresunet3d_full\train\best_model.pt" `
  --out ".\runs\partialresunet3d_full\eval" `
  --model partial `
  --dim 128 `
  --init_feat 32 `
  --batch 1 `
  --slices 64
```

Outputs:

- `metrics_per_case.csv`
- `metrics_summary.csv`
- `metrics_summary.json`
- `reproducibility.json`

### 4. Run qualitative and runtime analysis

Benchmark runtime on a subset of the test cohort:

```powershell
python scripts/benchmark_inference.py `
  --data ".\data\preprocessed_liver_ct_128\*.nii.gz" `
  --split_file ".\runs\splits\liver_seed1337_split_128.json" `
  --split test `
  --weights ".\runs\partialresunet3d_full\train\best_model.pt" `
  --out ".\runs\partialresunet3d_full\benchmark" `
  --model partial `
  --dim 128 `
  --init_feat 32 `
  --slices 64 `
  --num_cases 10
```

Create case-level comparison plots against the regression baseline:

```powershell
python scripts/visualize_comparison.py `
  --volume ".\data\preprocessed_liver_ct_128\case_001.nii.gz" `
  --weights ".\runs\partialresunet3d_full\train\best_model.pt" `
  --out ".\runs\partialresunet3d_full\visuals\case_001" `
  --model partial `
  --dim 128 `
  --init_feat 32 `
  --slices 64
```

## Advanced Analysis

### Regularization-Behavior Study

`scripts/evaluate_regularization_behavior.py` is intended for thesis-grade empirical analysis rather than quick model evaluation.

It can study:

- information convergence across different slice counts,
- robustness to noisy sparse inputs,
- sensitivity to repeated random sampling geometries,
- output stability under small input perturbations,
- robustness across multiple checkpoints or model variants.

Because each study evaluates many combinations of cases, slice counts, noise levels, and repetitions, this script can require hundreds of forward passes and substantial CPU-side sparse acquisition time.

Typical outputs include:

- raw per-case CSV files,
- grouped summary CSV files,
- LaTeX tables,
- publication-ready PNG plots,
- `study_index.json`,
- `study_config.json`,
- `reproducibility.json`.

### Invariance and Sanity Scripts

Additional scripts such as `scripts/invariance_test.py` and `scripts/unit_test_synthetic.py` support targeted inspection and synthetic sanity checks.

## Experiment Suites

Three staged suite configurations are included:

- `configs/ablation_smoke.example.json`
- `configs/ablation_short.example.json`
- `configs/ablation_final.example.json`

Recommended order:

1. `smoke`
   Fast sanity check across all variants with small budgets
2. `short`
   Reduced but representative comparison run
3. `final`
   Thesis-grade final comparison on the main variants

Convenience entry points:

```powershell
.\scripts\run_ablation_suite.ps1 -Stage smoke
```

or

```powershell
python scripts/run_experiment_suite.py --config ".\configs\ablation_smoke.example.json"
```

## Supported Model Variants

- `partial`
  Partial-convolution based residual U-Net
- `baseline`
  Standard residual U-Net without partial convolutions
- `gated`
  Gated convolution variant

Relevant toggles:

- hard constraint enabled by default, disable with `--no_hard_constraint`
- remove gradient term with `--w_grad 0.0`
- remove MS-SSIM term with `--w_ms_ssim 0.0`
- optional SSIM term via `--w_ssim ...`

## Reproducibility

This repository is designed around explicit experiment traceability.

Training and evaluation runs store:

- command-line configuration,
- selected split manifest,
- seed values,
- Python, NumPy, and PyTorch versions,
- working directory,
- Git metadata when available.

This makes it easier to compare checkpoints fairly and to report thesis results on a fixed cohort split instead of a moving ad hoc selection.

## Main Entry Points

- `scripts/create_data_split.py`
- `scripts/train_inpainting.py`
- `scripts/evaluate_inpainting.py`
- `scripts/evaluate_regularization_behavior.py`
- `scripts/benchmark_inference.py`
- `scripts/visualize_comparison.py`
- `scripts/visualize_model_comparison.py`
- `scripts/run_experiment_suite.py`
- `scripts/run_ablation_suite.ps1`
- `scripts/invariance_test.py`
- `scripts/unit_test_synthetic.py`

## Scope and Status

This is research code intended for controlled experimentation on sparse-to-dense liver reconstruction. It is suitable for thesis work, ablation studies, and reproducible model comparison, but it is not packaged as a clinical or production inference system.
