# Liver Sparse Reconstruction Core

Core repository for the CT-based sparse-to-dense 3D liver reconstruction pipeline.

This repository contains the minimal code needed to:

- preprocess dense liver CT volumes,
- generate synthetic sparse slice observations,
- train 3D inpainting/reconstruction models,
- evaluate reconstruction quality on a fixed cohort split,
- run cohort-wide baseline and ablation suites,
- benchmark inference runtime,
- inspect robustness and qualitative results.

It intentionally excludes:

- patient-specific US/CT verification,
- US adapter experiments,
- thesis documents and auxiliary tooling.

## Structure

- `inpainting3d/`: core package for data handling, split management, models, losses, metrics, statistics, preprocessing, and utilities
- `scripts/`: training, evaluation, split creation, ablation runners, benchmarking, invariance checks, visualization, and a synthetic unit demo
- `configs/`: example experiment configuration and ablation-suite template

## Setup

Create an environment and install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Split Policy

Earlier versions of the training script created an implicit random `85/15` `train/val` split, while `scripts/evaluate_inpainting.py` evaluated whatever the input glob matched. This repository now supports an explicit persisted `train/val/test` split manifest that should be created once and then reused for all baselines and ablations.

`scripts/evaluate_inpainting.py` now treats the persisted split manifest as the default-safe mode. Without `--split_file`, standalone evaluation aborts unless you explicitly opt into the legacy all-data mode with `--allow_all_data_eval`.

Recommended workflow:

1. Create the split once with `scripts/create_data_split.py`.
2. Reuse the same `split_file` for all training runs.
3. Evaluate every model on the exact same `test` split.

Persisted split artifacts:

- `split_manifest.json`
- `split_manifest_train.txt`
- `split_manifest_val.txt`
- `split_manifest_test.txt`

If you do not pass `--split_file` to `scripts/train_inpainting.py`, the script will still create a manifest inside the run directory so the used split is documented.

## Reproducible Workflow

1. Create and persist the cohort split:

```powershell
python scripts/create_data_split.py `
  --data ".\data\preprocessed_liver_ct_64\*.nii.gz" `
  --out ".\runs\splits\liver_seed1337_split.json" `
  --seed 1337 `
  --train_ratio 0.7 `
  --val_ratio 0.15 `
  --test_ratio 0.15
```

2. Train a reference `PartialResUNet3D` model on the fixed split:

```powershell
python scripts/train_inpainting.py `
  --data ".\data\preprocessed_liver_ct_64\*.nii.gz" `
  --split_file ".\runs\splits\liver_seed1337_split.json" `
  --out ".\runs\partialresunet3d_full\train" `
  --model partial `
  --dim 64 `
  --epochs 500 `
  --batch 2 `
  --slices_min 8 `
  --slices_max 64 `
  --w_grad 0.05 `
  --w_ms_ssim 0.2 `
  --amp `
  --deterministic
```

3. Evaluate the best checkpoint on the persisted `test` split:

```powershell
python scripts/evaluate_inpainting.py `
  --data ".\data\preprocessed_liver_ct_64\*.nii.gz" `
  --split_file ".\runs\splits\liver_seed1337_split.json" `
  --split test `
  --weights ".\runs\partialresunet3d_full\train\best_model.pt" `
  --out ".\runs\partialresunet3d_full\eval" `
  --model partial `
  --batch 1
```

The `--split test` argument is already the default. Keeping it in the example makes the intended thesis protocol explicit.

4. Run the full baseline and ablation suite on the same split:

```powershell
.\scripts\run_ablation_suite.ps1 -Stage smoke
```

or

```powershell
python scripts/run_experiment_suite.py --config ".\configs\ablation_smoke.example.json"
```

## Staged Ablation Workflow

Three ready-to-use suite configs are included:

- `configs/ablation_smoke.example.json`
- `configs/ablation_short.example.json`
- `configs/ablation_final.example.json`

Recommended execution order:

1. Smoke suite:
   small sanity run across all variants, including `gated`, with reduced resolution, 2 epochs, and limited case counts

```powershell
.\scripts\run_ablation_suite.ps1 -Stage smoke
```

2. Short suite:
   trend check across all variants on the fixed split with reduced epochs and full case-wise evaluation

```powershell
.\scripts\run_ablation_suite.ps1 -Stage short
```

3. Final suite:
   full thesis-grade comparison on the five core variants only

```powershell
.\scripts\run_ablation_suite.ps1 -Stage final
```

If you want to run `gated` in the final phase as well, copy the `gated_unet3d` block from `configs/ablation_short.example.json` into the final config or run it separately as an additional experiment.

## Comparison and Ablation Setup

The staged configs follow this policy:

- smoke: all variants, including `gated`
- short: all variants, including `gated`
- final: five core variants only

The covered variants are:

- `PartialResUNet3D` full model
- `PartialResUNet3D` without hard constraint
- `PartialResUNet3D` without gradient loss
- `PartialResUNet3D` without MS-SSIM
- `Standard U-Net / DeepResUNet3D` without partial convolutions
- `GatedResUNet3D` as an optional additional comparison in smoke and short runs

Hard-constraint toggle:

- default: known voxels are enforced exactly
- disable with `--no_hard_constraint`

Model toggle:

- `--model partial`
- `--model baseline`
- `--model gated`

Loss toggles:

- no gradient loss: `--w_grad 0.0`
- no MS-SSIM: `--w_ms_ssim 0.0`
- optional SSIM term: `--w_ssim ...`

## Evaluation Outputs

`scripts/evaluate_inpainting.py` now reports case-wise statistics instead of batch-level means.

By default it also requires a split manifest and therefore runs against the persisted evaluation split rather than the full data glob.

Per-case artifact:

- `metrics_per_case.csv`

Cohort summary artifacts:

- `metrics_summary.csv`
- `metrics_summary.json`

Reported summary statistics for each metric:

- mean
- median
- q25
- q75
- IQR
- standard deviation
- 95% confidence interval

## Reproducibility Artifacts

Training and evaluation runs now persist:

- `config.json`
- `reproducibility.json`
- `split_manifest.json`
- split text files per subset

The reproducibility metadata includes:

- seeds,
- command-line arguments,
- Python/Torch/Numpy versions,
- working directory,
- Git commit hash and dirty state when available.

Suite-level outputs:

- `suite_commands.json`
- `suite_summary.csv`
- `suite_reproducibility.json`

## Main Entry Points

- `scripts/create_data_split.py`
- `scripts/train_inpainting.py`
- `scripts/evaluate_inpainting.py`
- `scripts/run_experiment_suite.py`
- `scripts/run_ablation_suite.ps1`
- `scripts/benchmark_inference.py`
- `scripts/invariance_test.py`
- `scripts/visualize_comparison.py`
- `scripts/unit_test_synthetic.py`

## Notes

- The training setup learns from dense CT volumes with synthetically generated sparse observations.
- Known voxels can be enforced exactly through the inpainting hard constraint in `inpainting3d/utils.py`.
- The repository is kept intentionally small so it can serve as a clean thesis or paper code base.
