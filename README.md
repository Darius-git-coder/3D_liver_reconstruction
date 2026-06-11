# Liver Sparse Reconstruction Core

Companion repository for the bachelor thesis on sparse-to-dense 3D liver reconstruction from CT data.

The repository contains the code that was used for preprocessing, split generation, model training, evaluation, regularization analysis, qualitative figure export, and the controlled `ultrasound_probe` finetuning study. In addition, it includes a curated `results/` directory with the thesis-relevant artifacts that are cited or discussed in the written work.

## Scope

Included:

- the core Python package for data handling, models, losses, metrics, and reproducibility helpers,
- the scripts required for the experimental workflow described in the thesis,
- public example configurations for the final `128^3` reference regime,
- curated evaluation artifacts for ablation, regularization, ultrasound-probe finetuning, and appendix figures.

Not included:

- raw medical image data,
- large training checkpoints,
- machine-local configuration files,
- exploratory runs that were not part of the documented thesis workflow.

## Repository Layout

- `inpainting3d/`  
  Core package with data loading, sparse acquisition, model definitions, losses, metrics, statistics, split helpers, and utilities.
- `scripts/`  
  Entry points for preprocessing, split creation, training, evaluation, regularization analysis, qualitative exports, and ultrasound-probe experiments.
- `configs/`  
  Public reference configurations for the final thesis setup.
- `results/`  
  Curated thesis artifacts with human-readable summaries and sanitized metadata.

## Installation

Python `>= 3.10` is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively:

```powershell
python -m pip install -r requirements.txt
```

## Data Requirements

The workflow expects dense liver CT volumes as `.nii.gz` files. The repository does not ship any patient data.

The thesis experiments were run on a preprocessed `128^3` dataset derived from official liver CT data sources. For a fresh setup, prepare a local directory such as:

```text
./data/pipeline_preprocessed_data_128/
```

Sparse observations are generated online from the dense target volumes during training and evaluation.

## Thesis Workflow

### 1. Preprocess dense liver CT volumes

```powershell
python scripts/preprocessing_data_pipeline.py `
  --source ".\data\raw_liver_ct" `
  --target_dim 128 `
  --output_dir ".\data\pipeline_preprocessed_data_128"
```

### 2. Create the persisted train/val/test split

```powershell
python scripts/create_data_split.py `
  --data ".\data\pipeline_preprocessed_data_128\*.nii.gz" `
  --out ".\runs\splits\liver_seed1337_split_128.json" `
  --seed 1337 `
  --train_ratio 0.7 `
  --val_ratio 0.15 `
  --test_ratio 0.15
```

### 3. Run the final ablation suite

```powershell
python scripts/run_experiment_suite.py `
  --config ".\configs\ablation_final.example.json"
```

This reproduces the five final ablation variants used in the thesis:

- `standard_unet3d_no_partialconv`
- `partialresunet3d_full`
- `partialresunet3d_no_hard_constraint`
- `partialresunet3d_no_gradient_loss`
- `partialresunet3d_no_ms_ssim`

### 4. Run the regularization analysis of the selected reference model

```powershell
python scripts/evaluate_regularization_behavior.py `
  --data ".\data\pipeline_preprocessed_data_128\*.nii.gz" `
  --weights ".\runs\ablation_final_128\standard_unet3d_no_partialconv\train\best_model.pt" `
  --split_file ".\runs\splits\liver_seed1337_split_128.json" `
  --split test `
  --model baseline `
  --dim 128 `
  --init_feat 32 `
  --out ".\runs\ablation_final_128\standard_unet3d_no_partialconv\thesis_regularization_full"
```

### 5. Run the controlled `ultrasound_probe` finetuning study

```powershell
python scripts/compare_best_run_ultrasound.py `
  --data ".\data\pipeline_preprocessed_data_128\*.nii.gz"
```

The helper script launches the finetuning and evaluation steps for the documented ultrasound-like observation geometry.

## Curated Results

The `results/` directory is the public-facing artifact layer of the repository.

- `results/ablation/`  
  Final ablation summary table and suite-level metrics.
- `results/splits/`  
  Sanitized documentation of the exact case split used in the thesis.
- `results/regularization/`  
  Plots, summary tables, raw CSV exports, and LaTeX tables for the empirical regularization analysis.
- `results/ultrasound_probe/`  
  Finetuning history, evaluation summaries, and qualitative comparison figures for the `ultrasound_probe` study.
- `results/slice_distribution/`  
  Appendix figures and summary files for the classical comparison against the regression baseline.

## Reproducibility Notes

The thesis uses one persisted split, fixed seeds, and a single documented `128^3` reference regime. Public example configurations are sanitized to remove machine-local paths, while `results/` preserves the experiment structure and the reported outputs in a portable form.

For a compact overview of the final setup, start with:

- `configs/ablation_final.example.json`
- `configs/liver_example.yaml`
- `results/manifest.json`
