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

- Linux, WSL, or another POSIX-like shell
- Python `3.10+`
- `pip`

## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The last command installs a CPU-only PyTorch build. Use a CUDA-specific PyTorch install command only if you intentionally want GPU packages.

## CLI quickstart

Train ANC:

```bash
anc-train --config configs/anc_small.yaml
```

Evaluate trained checkpoint directory:

```bash
anc-eval --checkpoint outputs/anc/<run_dir> --config configs/anc_eval.yaml
```

Run TPM simulation:

```bash
anc-tpm --config configs/tpm_default.yaml
```

Run baseline benchmark:

```bash
anc-baseline --config configs/baseline.yaml
```

Test a trained checkpoint with plaintext typed from the keyboard:

```bash
anc-demo --checkpoint outputs/anc/<run_dir>
```

Run tests:

```bash
pytest -q
```

The original module entrypoints still work, for example `python -m src.training.train_anc --config configs/anc_small.yaml`.

## Configuration and overrides

All commands use YAML config files in `configs/`.
For ANC and ANC evaluation, override values using repeated `--override key=value`.

Example:

```bash
anc-train --config configs/anc_small.yaml --override epochs=5 --override learning_rate=0.0005
```

## Output artifacts

Each run creates a unique run directory with mode-specific files. ANC runs include:

- `resolved_config.json`
- `metrics.json`
- `train_log.csv`
- `checkpoints/`
- `training_curves.png`, or `training_curves.csv` when Matplotlib is unavailable
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
