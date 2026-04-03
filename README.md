# Experimental Neural Cryptography Lab

Research sandbox for neural cryptography experiments. This project is for reproducible experimentation only.

## Security warning

This repository is **not** production cryptography and must not be used to protect real data.
Use standard, audited cryptography (for example AES-GCM or ChaCha20-Poly1305) for real systems.

## What it includes

- Adversarial neural cryptography (Alice-Bob-Eve) training pipeline
- Stronger-attacker evaluation pipeline (baseline, restarted, known-plaintext, chosen-plaintext)
- Tree Parity Machine (TPM) synchronization simulator with observer attacker
- Standard AEAD baselines (AES-GCM and ChaCha20-Poly1305)
- Reproducible run artifacts: resolved config, metrics, logs, checkpoints, plots, summaries

## Requirements

- Windows with local Python `3.13.5`
- `pip`

## Setup (Windows)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install -e .[dev]
py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The last command installs a CPU-only PyTorch build (no NVIDIA CUDA runtime packages).

## CLI quickstart

Train ANC:

```powershell
py -3.13 -m src.training.train_anc --config configs/anc_small.yaml
```

Evaluate trained checkpoint directory:

```powershell
py -3.13 -m src.evaluation.eval_anc --checkpoint outputs\anc\<run_dir> --config configs/anc_eval.yaml
```

Run TPM simulation:

```powershell
py -3.13 -m src.tpm.run_tpm --config configs/tpm_default.yaml
```

Run baseline benchmark:

```powershell
py -3.13 -m src.baseline.benchmark --config configs/baseline.yaml
```

Run tests:

```powershell
py -3.13 -m pytest -q
```

## Configuration and overrides

All commands use YAML config files in `configs/`.
For ANC and ANC evaluation, override values using repeated `--override key=value`.

Example:

```powershell
py -3.13 -m src.training.train_anc --config configs/anc_small.yaml --override epochs=5 --override learning_rate=0.0005
```

## Output artifacts

Each run creates a unique run directory with mode-specific files. ANC runs include:

- `resolved_config.json`
- `metrics.json`
- `train_log.csv`
- `checkpoints/`
- `training_curves.png`
- `summary.md`

Evaluation, TPM, and baseline modes write equivalent structured artifacts in their run directories.

## Repository layout

```text
configs/
src/
  baseline/
  data/
  evaluation/
  models/
  tpm/
  training/
  utils/
tests/
outputs/
```
