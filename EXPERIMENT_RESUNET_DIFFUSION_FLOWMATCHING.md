# Experiment: ResUNet vs Diffusion vs Flow Matching

This branch adds a separate experimental path for comparing the existing
deterministic ResUNet reconstruction baseline with two generative conditional
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
  Optional suite runner that trains and evaluates all enabled methods from one
  JSON config.
- `configs/resunet_diffusion_flowmatching.example.json`
  Portable example configuration for the comparison.

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

Dry-run the full command suite:

```powershell
python scripts/run_resunet_diffusion_flowmatching_suite.py `
  --config configs/resunet_diffusion_flowmatching.example.json `
  --dry_run
```

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

For quick debugging on CPU or a small GPU, reduce `--dim`, `--max_train_cases`,
`--max_val_cases`, `--epochs`, and `--sample_steps`.
