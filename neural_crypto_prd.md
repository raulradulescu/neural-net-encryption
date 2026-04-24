# Product Requirements Document

## Project title
Experimental Neural Cryptography Lab

## Purpose
Build a research-oriented application that implements and evaluates neural-network-based cryptographic experiments. The system is not intended for production security use. It must support three tracks:

1. adversarial neural cryptography (Alice-Bob-Eve),
2. neural synchronization / Tree Parity Machine (TPM) key exchange,
3. comparison against standard authenticated encryption baselines.

The goal is to let a developer or researcher reproduce known experimental setups, run stronger attackers, and measure whether the learned system provides any meaningful secrecy under defined threat models.

## Product goals
- Implement a reproducible sandbox for neural cryptography experiments.
- Make threat models explicit and configurable.
- Provide baseline comparisons against standard crypto.
- Provide metrics, logs, plots, and saved checkpoints for each run.
- Make it easy to add stronger attackers and additional experiments later.

## Non-goals
- Do not market the learned system as secure real-world encryption.
- Do not replace standard ciphers such as AES-GCM or ChaCha20-Poly1305.
- Do not build a network service or messaging product.
- Do not optimize for large-scale distributed training in v1.

## Target users
- students,
- researchers,
- security engineers exploring adversarial learning,
- CTF / academic teams studying ML and cryptography intersections.

## Core requirements

### 1. Experiment modes
The application must support the following modes:

#### 1.1 Alice-Bob-Eve adversarial neural cryptography
- Alice receives plaintext bits and a shared secret key.
- Alice outputs ciphertext.
- Bob receives ciphertext and the shared secret key and reconstructs plaintext.
- Eve receives ciphertext and attempts to reconstruct plaintext.
- The system must support training Alice/Bob jointly against Eve.
- The system must support fixed bit-length experiments, at minimum 16-bit and 32-bit plaintext/key settings.

#### 1.2 Stronger-attacker evaluation
- The system must support multiple Eve strategies.
- At minimum include:
  - baseline neural Eve,
  - restarted neural Eve with multiple random initializations,
  - known-plaintext evaluation mode,
  - chosen-plaintext style evaluation mode where Eve can query many plaintext/ciphertext pairs from the same trained model.
- The evaluation pipeline must make it easy to see whether security claims only hold against a weak Eve.

#### 1.3 Tree Parity Machine synchronization
- Implement a discrete-weight TPM synchronization protocol simulation.
- Two parties must synchronize over public interaction.
- The run must report number of rounds until synchronization.
- The run must support an attacker observer trying to synchronize as well.
- Parameters such as hidden units, weight range, and input dimension must be configurable.

#### 1.4 Standard crypto baseline
- Include a benchmark path using a standard authenticated encryption primitive from a trusted library.
- At minimum support one of:
  - AES-GCM,
  - ChaCha20-Poly1305.
- This mode is used only as a comparison baseline for correctness, throughput, and ciphertext properties.

### 2. Configurability
The application must expose experiment parameters through config files and CLI flags.

Required configurable values:
- plaintext length,
- key length,
- batch size,
- number of epochs,
- learning rate,
- model width/depth,
- random seed,
- number of Eve restarts,
- TPM parameters,
- output directory.

Preferred format:
- YAML or JSON config files,
- CLI override support.

### 3. Reproducibility
- Every run must store the full resolved configuration.
- Every run must store the random seed.
- Every run must create a unique run directory.
- Training logs and final metrics must be written to disk.
- Saved checkpoints for Alice, Bob, and Eve must be supported.

### 4. Metrics and outputs
The system must report the following metrics where applicable:
- Bob reconstruction accuracy,
- Eve reconstruction accuracy,
- Bob bit error rate,
- Eve bit error rate,
- ciphertext bit balance,
- plaintext-bit flip sensitivity,
- key-bit flip sensitivity,
- training loss curves,
- TPM synchronization success rate,
- TPM average rounds to synchronize,
- baseline encryption/decryption correctness,
- basic runtime / throughput per mode.

The system should generate:
- JSON metrics summary,
- CSV or JSONL epoch logs,
- at least one training curve plot,
- final markdown or text summary for the run.

## Functional requirements

### 5. Training pipeline
- Provide a training script for Alice-Bob-Eve mode.
- Support alternating optimization between Alice/Bob and Eve.
- Support train and eval splits generated from random bitstrings.
- Support CPU execution by default.
- GPU support is optional but should work if CUDA is available.

### 6. Evaluation pipeline
- Provide a separate evaluation script that loads trained checkpoints.
- Evaluation must be runnable without retraining.
- Evaluation must support all attacker modes.
- Evaluation must emit comparable metrics across runs.

### 7. TPM simulator
- Provide a standalone simulator for TPM synchronization.
- Allow repeated trials over a parameter grid.
- Save aggregate stats over multiple trials.

### 8. Baseline crypto benchmark
- Provide a standalone benchmark comparing:
  - neural encryption/decryption path,
  - standard AEAD path.
- The benchmark does not need to claim equal security; it only reports correctness and speed-related metrics.

## Suggested repository structure

```text
project/
  configs/
  src/
    data/
    models/
    training/
    evaluation/
    tpm/
    baseline/
    utils/
  tests/
  scripts/
  outputs/
  README.md
```

## Model expectations

### 9. Minimum v1 model design
For v1, keep the models simple and small.

Recommended:
- MLP-based Alice,
- MLP-based Bob,
- MLP-based Eve.

Avoid transformers or large sequence models in v1.

### 10. Data representation
- Represent plaintext, keys, and ciphertext internally as binary tensors.
- The code must clearly define whether values are stored as {0,1} or {-1,1}.
- Conversion utilities must be tested.

## CLI requirements
The project must expose at least these commands:

```bash
python -m src.training.train_anc --config configs/anc_small.yaml
python -m src.evaluation.eval_anc --checkpoint outputs/run_x/
python -m src.tpm.run_tpm --config configs/tpm_default.yaml
python -m src.baseline.benchmark --config configs/baseline.yaml
```

Equivalent entrypoints are acceptable if documented.

## Minimal tests for development

### 11. Unit tests
The coding agent must implement at minimum these tests:

#### 11.1 Bit utilities
- encode/decode round-trip for bit tensors,
- random bit generation returns expected shape and value set,
- bit flip utility flips exactly one requested position.

#### 11.2 Model I/O shape tests
- Alice output shape matches ciphertext length,
- Bob output shape matches plaintext length,
- Eve output shape matches plaintext length.

#### 11.3 Loss / training step tests
- one training step runs without crashing,
- gradients are non-null for trainable parameters,
- loss is finite after one forward/backward pass.

#### 11.4 Checkpoint tests
- save/load checkpoint preserves model weights,
- loaded model produces identical output for same input in eval mode.

#### 11.5 TPM tests
- TPM weights initialize in valid range,
- synchronization routine terminates or respects max-round limit,
- observer metrics are recorded.

#### 11.6 Baseline crypto tests
- plaintext decrypts back exactly after encrypt/decrypt round-trip,
- invalid nonce/key sizes raise a controlled error.

### 12. Smoke tests
Implement at minimum these smoke tests:
- small ANC training run of 1 to 3 epochs completes on CPU,
- evaluation script loads the produced checkpoint and writes metrics,
- TPM simulation completes a small batch of trials,
- baseline benchmark completes and writes output files.

### 13. Acceptance tests
The coding agent must treat the following as minimum acceptance criteria:
- a full small ANC run completes end-to-end from config,
- Bob accuracy improves above random guessing on the small training task,
- Eve metrics are reported for at least two attacker modes,
- TPM simulation produces synchronization statistics over multiple trials,
- baseline crypto path successfully encrypts and decrypts test inputs,
- all required artifacts are written to the run directory.

## Logging and artifact requirements
Each run directory must contain:
- resolved_config.json or .yaml,
- metrics.json,
- train_log.csv or .jsonl,
- model checkpoints,
- generated plots,
- short run summary.

## Error handling
- Invalid configs must fail with clear messages.
- Unsupported experiment modes must fail early.
- Missing checkpoint paths must produce a readable error.
- Baseline crypto failures must not silently continue.

## Documentation requirements
The repository must include:
- setup instructions,
- dependency list,
- quickstart commands,
- explanation of each experiment mode,
- explicit warning that this is an experimental research project and not production-grade cryptography.

## Definition of done
The v1 implementation is done when:
- all required experiment modes exist,
- the CLI works from a clean environment,
- minimal unit and smoke tests pass,
- one sample run for ANC, TPM, and baseline crypto is reproducible,
- outputs are written in a structured format,
- documentation explains how to run and interpret results.

## Nice-to-have after v1
- more attacker architectures,
- hyperparameter sweep scripts,
- richer plots dashboard,
- notebook-based analysis,
- Docker support,
- optional web UI for browsing completed experiment runs.
