# Configurations

This directory contains portable example configurations for the public
repository snapshot.

Included examples:

- `ablation_smoke.example.json`
- `ablation_short.example.json`
- `ablation_final.example.json`
- `ablation_suite.example.json`
- `liver_example.yaml`

These files are meant to document the structure of the experiment setup and to
serve as starting points for local runs.

Machine-specific files such as `*.local*.json` are intentionally not part of
the publishable snapshot. When you need absolute paths, a specific Python
interpreter, or workstation-local output locations, copy an example file and
adapt it locally.
