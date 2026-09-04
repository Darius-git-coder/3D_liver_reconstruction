# Experiment: ResUNet vs Diffusion vs Flow Matching

This branch adds a separate experimental path for comparing the existing
deterministic ResUNet reconstruction baseline with two conditional generative
models:

- `resunet`: existing deterministic DeepResUNet3D trained with
  `scripts/train_inpainting.py`
- `diffusion`: conditional 3D DDPM-style inpainting model with a cosine noise
  schedule and DDIM sampling
- `flow_matching`: conditional rectified-flow / flow-matching model trained to
  predict the transport velocity from noise to the dense CT target

All three methods can use the same dense CT data glob, persistent
train/validation/test split, sparse-slice simulation, hard constraint, and
evaluation metrics. That keeps the comparison focused on the reconstruction
model rather than on data leakage or split changes.

## Important Scope Note

The diffusion arm is a compact research implementation inspired by modern DDPM
practice. It is not a downloaded medical foundation model. For a thesis or
paper, describe it as a conditional 3D DDPM-style baseline unless you replace
it with a cited external SOTA implementation.

## Scientific Comparison Protocol

Use this protocol when the comparison should be reportable in a thesis:

1. Create one persistent split manifest and keep it unchanged for all methods.
   Use the training split for gradient updates, the validation split for model
   selection, and the test split only once for final reporting.
2. Use identical sparse-slice simulation settings for all methods. The example
   config fixes the test-time sparse pattern with `eval_seed=9001`.
3. Repeat training with multiple model seeds. The scientific example uses
   `1337`, `2024`, and `3407`; this captures seed-to-seed variability instead
   of relying on one lucky run.
4. Use one primary endpoint before looking at the test results. The proposed
   primary metric is `rmse_missing`, because the missing region is the actual
   inpainting target. Secondary metrics are `mae_missing`, `ssim_missing`,
   `psnr_missing`, `rmse_all`, `mae_all`, and `ssim_all`.
5. Compare methods with paired case-level statistics. The helper
   `scripts/summarize_method_comparison.py` joins metrics by
   `(train_seed, case_id)`, computes bootstrap confidence intervals, and runs a
   two-sided Wilcoxon signed-rank test against the chosen baseline.
6. Report training settings, data split counts, number of parameters if used,
   GPU, training time, sampling steps, and whether hard constraints were active.
7. Keep the language precise: the current models are experimental conditional
   3D baselines, not a claim that a full external SOTA architecture was
   reproduced.

## Main Files

- `inpainting3d/generative.py`
  Time-conditioned 3D U-Net, DDPM schedule/loss/sampler, and flow-matching
  loss/sampler.
- `scripts/train_generative_inpainting.py`
  Training entry point for `--objective diffusion` and
  `--objective flow_matching`.
- `scripts/evaluate_generative_inpainting.py`
  Test-split evaluation for trained generative checkpoints.
- `scripts/run_resunet_diffusion_flowmatching_suite.py`
  Suite runner that can train/evaluate all enabled methods across multiple
  seeds and then call the scientific summary script.
- `scripts/summarize_method_comparison.py`
  Aggregates per-case metrics across methods, computes bootstrap confidence
  intervals, and performs paired Wilcoxon tests.
- `configs/resunet_diffusion_flowmatching.example.json`
  Portable single-seed example configuration.
- `configs/resunet_diffusion_flowmatching_scientific.example.json`
  Multi-seed scientific comparison template.

## Example

Create the split once if it does not exist yet:

```powershell
python scripts/create_data_split.py `
  --data "E:/Bachelorarbeit_Daten/pipeline_preprocessed_data_128/*.nii.gz" `
  --out "./runs/splits/liver_seed1337_split_128.json" `
  --seed 1337 `
  --train_ratio 0.7 `
  --val_ratio 0.15 `
  --test_ratio 0.15
```

Dry-run the scientific command suite:

```powershell
python scripts/run_resunet_diffusion_flowmatching_suite.py `
  --config configs/resunet_diffusion_flowmatching_scientific.example.json `
  --dry_run
```

Run the scientific suite:

```powershell
python scripts/run_resunet_diffusion_flowmatching_suite.py `
  --config configs/resunet_diffusion_flowmatching_scientific.example.json
```

This writes one folder per method and training seed, plus:

- `runs/resunet_diffusion_flowmatching_scientific/suite_manifest.json`
- `runs/resunet_diffusion_flowmatching_scientific/scientific_summary/all_case_metrics.csv`
- `runs/resunet_diffusion_flowmatching_scientific/scientific_summary/method_metric_summary.csv`
- `runs/resunet_diffusion_flowmatching_scientific/scientific_summary/pairwise_tests.csv`
- `runs/resunet_diffusion_flowmatching_scientific/scientific_summary/scientific_comparison_summary.md`

Run only the generative diffusion arm manually:

```powershell
python scripts/train_generative_inpainting.py `
  --objective diffusion `
  --data "E:/Bachelorarbeit_Daten/pipeline_preprocessed_data_128/*.nii.gz" `
  --split_file "./runs/splits/liver_seed1337_split_128.json" `
  --out "./runs/resunet_diffusion_flowmatching/diffusion/train" `
  --dim 128 `
  --batch 1 `
  --epochs 150 `
  --init_feat 16 `
  --diffusion_steps 1000 `
  --sample_steps 50 `
  --amp `
  --deterministic
```

Run only the flow-matching arm manually:

```powershell
python scripts/train_generative_inpainting.py `
  --objective flow_matching `
  --data "E:/Bachelorarbeit_Daten/pipeline_preprocessed_data_128/*.nii.gz" `
  --split_file "./runs/splits/liver_seed1337_split_128.json" `
  --out "./runs/resunet_diffusion_flowmatching/flow_matching/train" `
  --dim 128 `
  --batch 1 `
  --epochs 150 `
  --init_feat 16 `
  --sample_steps 50 `
  --amp `
  --deterministic
```

Evaluate a trained generative checkpoint:

```powershell
python scripts/evaluate_generative_inpainting.py `
  --objective diffusion `
  --data "E:/Bachelorarbeit_Daten/pipeline_preprocessed_data_128/*.nii.gz" `
  --split_file "./runs/splits/liver_seed1337_split_128.json" `
  --split test `
  --weights "./runs/resunet_diffusion_flowmatching/diffusion/train/best_model.pt" `
  --out "./runs/resunet_diffusion_flowmatching/diffusion/eval" `
  --dim 128 `
  --init_feat 16 `
  --sample_steps 50
```

Summarize already existing evaluation folders manually:

```powershell
python scripts/summarize_method_comparison.py `
  --baseline resunet `
  --out "./runs/resunet_diffusion_flowmatching/scientific_summary" `
  --input "resunet=./runs/resunet_diffusion_flowmatching/resunet/eval" `
  --input "diffusion=./runs/resunet_diffusion_flowmatching/diffusion/eval" `
  --input "flow_matching=./runs/resunet_diffusion_flowmatching/flow_matching/eval"
```

For quick debugging on CPU or a small GPU, reduce `--dim`,
`--max_train_cases`, `--max_val_cases`, `--epochs`, and `--sample_steps`.

## Method References

- Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
- Improved Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2102.09672
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
