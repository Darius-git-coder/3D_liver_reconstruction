# Configurations

This directory contains the public reference configurations for the documented thesis workflow.

- `ablation_final.example.json`  
  Final `128^3` ablation suite used for the thesis comparison of the five reported model variants.
- `liver_example.yaml`  
  Compact workflow overview for preprocessing, split creation, training, evaluation, regularization analysis, and the controlled `ultrasound_probe` finetuning study.

Machine-local overrides are intentionally not versioned. Adapt the data paths and output directories to your local environment before running the pipeline.
