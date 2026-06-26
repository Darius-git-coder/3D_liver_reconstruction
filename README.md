# 3D Liver Reconstruction

Companion repository for a bachelor thesis on sparse-to-dense 3D liver
reconstruction from CT volumes with synthetically generated sparse
observations.

This snapshot is meant to be publishable and readable: it contains the
reconstruction code, portable example configurations, and a curated
[`results/`](results/README.md) folder with the thesis-relevant artifacts that
are worth inspecting without shipping the entire private working tree.

## Included

- `inpainting3d/`: core package for data handling, sparse acquisition,
  reconstruction models, losses, metrics, and reproducibility helpers
- `scripts/`: training, evaluation, benchmarking, robustness, and qualitative
  comparison entry points
- `configs/`: portable example configurations for smoke, short, and final
  experiment suites
- [`results/`](results/README.md): curated thesis artifacts for
  regularization analysis, slice-count evaluation, and ultrasound-probe
  finetuning

## Intentionally Excluded

- raw medical data
- preprocessed `.nii.gz` archives
- trained weights and checkpoints
- full `runs/` directories
- machine-local `*.local*.json` files

## Repository Layout

- `inpainting3d/`
  Python package with acquisition simulation, models, losses, metrics, split
  handling, preprocessing helpers, and utilities.
- `scripts/`
  CLI entry points for preprocessing, split creation, training, evaluation,
  regularization studies, ultrasound-style comparison, and qualitative export.
- `configs/`
  Example configurations for reproducible suite runs. See
  [`configs/README.md`](configs/README.md).
- `results/`
  Cleaned thesis artifact snapshot with README files and summary JSONs. See
  [`results/README.md`](results/README.md).

## Data Assumptions

The code expects liver-focused dense CT volumes stored as `.nii.gz` files.
Within the thesis workflow these volumes were normalized, spatially aligned,
cropped around the organ of interest, and resampled to fixed cubic grids such
as `64^3` or `128^3`.

Sparse observations are not stored as separate datasets. They are generated at
runtime from dense targets by the acquisition simulation code, which keeps the
problem setup explicit and easy to vary.

## Installation

The public repository snapshot requires Python `>= 3.10`.
The archived thesis run environment used Python `3.9.0` together with
PyTorch `2.5.1+cu121`, while the cleaned publication snapshot now contains
modern PEP 604 type hints and therefore declares the stricter interpreter
requirement. The pinned dependency set below reflects the documented thesis
package versions; on GPU systems, install the matching PyTorch wheel variant
for your CUDA setup if needed.

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

## Quick Start

### 1. Preprocess dense CT volumes

```powershell
python scripts/preprocessing_data_pipeline.py `
  --source ".\raw_liver_ct" `
  --output_root ".\data" `
  --target_dim 128
```

### 2. Create a persistent train/val/test split

```powershell
python scripts/create_data_split.py `
  --data ".\data\pipeline_preprocessed_data_128\*.nii.gz" `
  --out ".\runs\splits\liver_seed1337_split_128.json" `
  --seed 1337 `
  --train_ratio 0.7 `
  --val_ratio 0.15 `
  --test_ratio 0.15
```

### 3. Train a baseline reconstruction model

```powershell
python scripts/train_inpainting.py `
  --data ".\data\pipeline_preprocessed_data_128\*.nii.gz" `
  --split_file ".\runs\splits\liver_seed1337_split_128.json" `
  --out ".\runs\standard_unet3d\train" `
  --model baseline `
  --dim 128 `
  --epochs 150 `
  --batch 2 `
  --slices_min 8 `
  --slices_max 128 `
  --w_grad 0.05 `
  --w_ms_ssim 0.2 `
  --amp `
  --deterministic
```

### 4. Evaluate on the test split

```powershell
python scripts/evaluate_inpainting.py `
  --data ".\data\pipeline_preprocessed_data_128\*.nii.gz" `
  --split_file ".\runs\splits\liver_seed1337_split_128.json" `
  --split test `
  --weights ".\runs\standard_unet3d\train\best_model.pt" `
  --out ".\runs\standard_unet3d\eval" `
  --model baseline `
  --dim 128 `
  --init_feat 32 `
  --batch 1 `
  --slices 64
```

## Thesis-Relevant Workflows

The most thesis-specific scripts are:

- `scripts/run_experiment_suite.py` for structured ablation runs
- `scripts/evaluate_regularization_behavior.py` for the regularization study
- `scripts/generate_ct_method_figures.py` for CT-based method figures used to
  illustrate oriented slice geometry, Sparse Volume / Known Mask, and the
  Hard-Constraint in thesis Chapter 5
- `scripts/prepare_thesis_real_figures.py` for thesis figures based on real CT
  examples, curated reconstruction outputs, and patient US/CT verification
  plots
- `scripts/generate_thesis_concept_figures.py` for schematic figures such as
  CT-vs-ultrasound motivation, oriented slice geometry, Sparse Volume /
  Known Mask, Hard-Constraint, and pipeline overviews
- `scripts/compare_best_run_ultrasound.py` for finetuning under
  `ultrasound_probe` geometry
- `scripts/run_multislice_ultrasound_eval_and_export.py` and
  `scripts/export_thesis_multislice_qualitative_figures.py` for qualitative
  exports across multiple slice counts

The curated outputs already shipped in [`results/`](results/README.md) are the
small subset of those experiments that directly support the
written thesis.

## Notes on Reproducibility

- Example configs are tracked, machine-local configs are not.
- The public artifact snapshot uses cleaned summary JSONs and selected CSV/PDF
  outputs instead of entire private `runs/` trees.
- Model weights are intentionally excluded, so a full rerun still requires
  access to compatible data and local training.

## Status

This is research code for controlled reconstruction experiments. It is suitable
for thesis documentation, method comparison, and follow-up experiments. It is
not a clinical product and not packaged as a production inference system.
