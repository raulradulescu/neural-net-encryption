# Implementation Tracker

Status legend: done, partial, not started

## 1. Experiment Modes

| Feature | Status | Notes |
|---|---|---|
| Alice-Bob-Eve ANC training pipeline | done | `src/training/train_anc.py`, MLP models in `src/models/anc.py` |
| Alternating Alice/Bob vs Eve optimization | done | Train script supports joint training |
| 16-bit and 32-bit plaintext/key settings | done | Configurable via YAML (`anc_small.yaml`) |
| Stronger-attacker evaluation pipeline | done | `src/evaluation/eval_anc.py` |
| Baseline neural Eve | done | Default eval mode |
| Restarted Eve (multiple random inits) | done | Eval supports restarts |
| Known-plaintext evaluation | done | Eval mode |
| Chosen-plaintext evaluation | done | Eval mode |
| TPM synchronization simulator | done | `src/tpm/simulator.py`, `src/tpm/run_tpm.py` |
| TPM observer attacker | done | Observer tracked during sync |
| Standard AEAD baselines (AES-GCM, ChaCha20) | done | `src/baseline/aead.py`, `src/baseline/benchmark.py` |

## 2. Configuration

| Feature | Status | Notes |
|---|---|---|
| YAML config files | done | `configs/` directory |
| CLI override support (`--override key=value`) | done | Train and eval scripts |
| Configurable: plaintext/key length, batch, epochs, lr, model dims, seed, Eve restarts, TPM params, output dir | done | See `configs/shared.yaml` and mode-specific configs |

## 3. Reproducibility

| Feature | Status | Notes |
|---|---|---|
| Resolved config saved per run | done | `resolved_config.json` |
| Random seed stored and applied | done | `src/utils/seed.py` |
| Unique run directory per execution | done | Timestamped dirs under `outputs/` |
| Training logs written to disk | done | `train_log.csv` |
| Model checkpoints saved | done | `checkpoints/` in run dir |

## 4. Metrics and Outputs

| Feature | Status | Notes |
|---|---|---|
| Bob reconstruction accuracy | done | |
| Eve reconstruction accuracy | done | |
| Bob/Eve bit error rate | done | |
| Ciphertext bit balance | done | |
| Plaintext-bit flip sensitivity | done | |
| Key-bit flip sensitivity | done | |
| Training loss curves | done | |
| TPM sync success rate + avg rounds | done | |
| Baseline encrypt/decrypt correctness | done | |
| Runtime/throughput per mode | done | |
| JSON metrics summary | done | `metrics.json` |
| CSV epoch logs | done | `train_log.csv` |
| Training curve plot | done | `training_curves.png` |
| Markdown run summary | done | `summary.md` |

## 5. Tests

| Test Suite | Status | Notes |
|---|---|---|
| Bit utilities (encode/decode, shape, flip) | done | `tests/test_bits.py` |
| Model I/O shape tests | done | `tests/test_models_shapes.py` |
| Loss / training step tests | done | `tests/test_training_step.py` |
| Checkpoint save/load tests | done | `tests/test_checkpoint.py` |
| TPM tests (init, sync, observer) | done | `tests/test_tpm.py` |
| Baseline crypto tests (round-trip, error handling) | done | `tests/test_baseline_crypto.py` |
| Smoke tests (end-to-end small runs) | done | `tests/test_smoke.py` |

All 15 tests passing.

## 6. Documentation

| Item | Status | Notes |
|---|---|---|
| Setup instructions | done | README.md |
| Dependency list | done | `pyproject.toml` |
| Quickstart commands | done | README.md |
| Experiment mode explanations | done | README.md |
| Security warning (not production crypto) | done | README.md |

## 7. Error Handling

| Feature | Status | Notes |
|---|---|---|
| Invalid config fails with clear message | done | `src/utils/errors.py` |
| Unsupported mode fails early | done | |
| Missing checkpoint path readable error | done | |
| Baseline crypto failures don't silently continue | done | |

## 8. Nice-to-Have (Post-v1)

| Feature | Status | Notes |
|---|---|---|
| More attacker architectures | not started | |
| Hyperparameter sweep scripts | not started | |
| Richer plots dashboard | not started | |
| Notebook-based analysis | not started | |
| Docker support | not started | |
| Web UI for browsing experiment runs | not started | |
