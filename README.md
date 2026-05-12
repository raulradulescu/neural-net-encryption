# Experimental Neural Cryptography Lab

Experimental Neural Cryptography Lab is a research sandbox for studying how neural networks can learn encryption-like behavior, how those learned systems fail under stronger attackers, and how the results compare with standard authenticated encryption.

This repository is designed for reproducible experiments and presentations. It gives you runnable training scripts, evaluation scripts, TPM synchronization simulations, standard cryptography baselines, plots, metrics, checkpoints, and short summaries that can be used directly in a slide deck.

## Security Warning

This project is **not production cryptography**.

Do not use the neural models in this repository to protect real data. The learned Alice/Bob/Eve system is an experiment, not a cipher. For real systems, use audited standard schemes such as AES-GCM or ChaCha20-Poly1305 through established libraries.

The correct way to present this project is:

- "This is a research experiment about neural cryptography."
- "The model can learn patterns that look encryption-like under a defined threat model."
- "Security claims are limited to the attackers we actually evaluate."
- "Standard cryptography remains the correct tool for real confidentiality and integrity."

## Project At A Glance

The repository contains these runnable modes:

| Track | Purpose | Main command |
| --- | --- | --- |
| Adversarial Neural Cryptography | Train Alice and Bob to communicate while Eve tries to recover the plaintext. | `anc-train` |
| Stronger Attacker Evaluation | Reload a trained checkpoint and test multiple Eve strategies. | `anc-eval` |
| Tree Parity Machine Simulation | Simulate public synchronization between two parties and an observer attacker. | `anc-tpm` |
| Standard AEAD Baseline | Benchmark AES-GCM and ChaCha20-Poly1305 for correctness and throughput comparison. | `anc-baseline` |
| Presentation Demo | Type plaintext and a key, then show Bob and Eve outputs from a trained checkpoint. | `anc-demo` |

## The Big Idea

Classical cryptography starts from mathematically designed algorithms. This project asks a research question:

> Can neural networks learn a private communication protocol when one network is trained as an attacker?

The adversarial neural cryptography setup uses three models:

- **Alice** receives plaintext bits and a shared secret key, then outputs ciphertext.
- **Bob** receives the ciphertext and the same secret key, then reconstructs the plaintext.
- **Eve** receives only the ciphertext, then tries to guess the plaintext.

The experiment is successful only in a narrow research sense if Bob reconstructs the plaintext much better than Eve. That does not prove real cryptographic security, but it creates a useful playground for studying adversarial learning, threat models, and evaluation discipline.

## High-Level Architecture

```text
                shared key                         shared key
                    |                                  |
                    v                                  v
plaintext bits -> Alice model -> ciphertext bits -> Bob model -> reconstructed plaintext
                                  |
                                  v
                               Eve model -> guessed plaintext
```

Alice, Bob, and Eve are small multilayer perceptrons defined in `src/models/anc.py`.

The canonical bit representation for utility code is `{0, 1}`. During training, tensors are represented as floating point values containing zeros and ones so PyTorch can optimize the models.

## What Each Component Does

### Alice

Alice is the learned sender. It receives:

- a plaintext bit vector,
- a secret key bit vector.

It outputs ciphertext logits. The training loop applies `sigmoid` to turn those logits into soft ciphertext probabilities. The demo can also threshold them into hard bits for display.

### Bob

Bob is the learned receiver. It receives:

- Alice's ciphertext,
- the same secret key used by Alice.

Bob tries to reconstruct the original plaintext. Bob is rewarded when his bit predictions match the plaintext.

### Eve

Eve is the learned attacker. It receives:

- Alice's ciphertext only.

Eve does not receive the secret key. Eve tries to reconstruct the plaintext. The project uses Eve during training and also supports stronger post-training Eve strategies during evaluation.

## Training Objective

Training lives in `src/training/train_anc.py`.

For each batch:

1. Alice converts plaintext and key into ciphertext.
2. Bob tries to decode the ciphertext using the key.
3. Eve tries to decode the ciphertext without the key.
4. Alice and Bob are updated to improve Bob while making Eve worse.
5. Eve is trained separately against the latest Alice output.

The Alice/Bob loss is:

```text
bob_loss - eve_weight * eve_loss + balance_weight * balance_loss
```

Meaning:

- `bob_loss` pushes Bob toward accurate plaintext reconstruction.
- `eve_loss` is subtracted so Alice/Bob are rewarded when Eve performs worse.
- `balance_loss` discourages trivial ciphertext such as always-zero or always-one output.

Eve is then trained for `eve_steps_per_ab_step` update steps on detached ciphertext. This keeps Eve active instead of leaving Alice/Bob to beat a stale attacker.

## Stronger Attacker Evaluation

Training against one Eve is not enough. A weak Eve can make a learned protocol look safer than it is.

The evaluation pipeline in `src/evaluation/eval_anc.py` reloads a checkpoint and tests several attacker modes:

| Mode | What it tests |
| --- | --- |
| `baseline` | Uses the Eve model saved in the checkpoint. |
| `restarted` | Trains multiple fresh Eve models from different random initializations. |
| `known_plaintext` | Gives Eve many plaintext/ciphertext pairs from the trained Alice model. |
| `chosen_plaintext` | Lets Eve query many generated plaintext/ciphertext examples from the trained model. |

These modes help answer the presentation question:

> Did Alice and Bob learn something robust, or did they only beat the particular Eve used during training?

## Tree Parity Machine Track

The TPM simulator in `src/tpm/simulator.py` is a separate neural synchronization experiment.

Two parties each hold a discrete-weight Tree Parity Machine. They exchange public inputs and outputs, update their weights, and try to synchronize to the same internal state. An observer attacker sees the same public interaction and tries to synchronize too.

The simulator reports:

- whether Alice and Bob synchronized,
- how many rounds synchronization took,
- whether the observer synchronized,
- final distance between parties,
- aggregate success rates across trials.

This gives a second presentation angle: neural synchronization can create shared state through public interaction, but it must still be evaluated against observers.

## Standard Crypto Baseline

The baseline path in `src/baseline/aead.py` uses the `cryptography` package to benchmark:

- AES-GCM,
- ChaCha20-Poly1305.

The baseline is not there because neural encryption and AEAD offer the same security. They do not.

The baseline is there to keep the research honest:

- standard AEAD decrypts exactly,
- standard AEAD authenticates data,
- standard AEAD has clear key and nonce requirements,
- standard AEAD throughput gives a practical comparison point.

## Requirements

- Linux, WSL, or another POSIX-like shell
- Python `3.10+`
- `pip`
- PyTorch for ANC training, evaluation, and demo commands

The project dependencies are declared in `pyproject.toml`:

- `numpy`
- `PyYAML`
- `matplotlib`
- `cryptography`
- `pytest` for development tests
- `torch` for neural experiments

## Setup

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The last command installs a CPU-only PyTorch build. Use a CUDA-specific PyTorch command only if you intentionally want GPU packages. The current project commands are configured for CPU-friendly execution.

## Quickstart

Train an ANC checkpoint:

```bash
anc-train --config configs/anc_recommended.yaml
```

For a shorter trial run, override the epoch count:

```bash
anc-train --config configs/anc_recommended.yaml --override epochs=20
```

The training command prints a JSON object containing `run_dir`, for example `outputs/anc/anc_20260424T120000Z_anc_recommended_ab12cd34`. Save that full path for evaluation and demo commands.

Evaluate the trained checkpoint:

```bash
anc-eval --checkpoint <run_dir> --config configs/anc_eval.yaml
```

Run the presentation-friendly plaintext demo:

```bash
anc-demo \
  --checkpoint <run_dir> \
  --text "hello neural crypto" \
  --key "presentation key" \
  --ecc repeat3_hamming74 \
  --primary-path soft
```

Run the TPM simulator:

```bash
anc-tpm --config configs/tpm_default.yaml
```

Run standard crypto baselines:

```bash
anc-baseline --config configs/baseline.yaml
```

Run tests:

```bash
pytest -q
```

The module entrypoints also work:

```bash
python -m src.training.train_anc --config configs/anc_recommended.yaml
python -m src.evaluation.eval_anc --checkpoint <run_dir> --config configs/anc_eval.yaml
python -m src.tpm.run_tpm --config configs/tpm_default.yaml
python -m src.baseline.benchmark --config configs/baseline.yaml
python -m src.demo_cli --checkpoint <run_dir>
```

## Hamming Code: Getting The Message Back

The neural path can be slightly noisy. That matters because one wrong bit can break a byte, corrupt UTF-8 text, or invalidate the final padding. A demo is only convincing if Bob gets the actual message back, not just "mostly close" bits.

`anc-demo` can wrap the message in a small error-correction layer before Alice sees it:

| Mode | What it does | Best use |
| --- | --- | --- |
| `none` | Sends padded plaintext bits directly. | Raw model behavior. |
| `hamming74` | Turns every 4 data bits into a 7-bit Hamming codeword. | Corrects one bit error per codeword. |
| `repeat3_hamming74` | Repeats each Hamming bit 3 times, majority-votes, then applies Hamming correction. | Best presentation mode. |

Use this for live demos:

```bash
anc-demo --checkpoint <run_dir> --text "hello neural crypto" --key "presentation key" --ecc repeat3_hamming74 --primary-path soft
```

The tradeoff is simple: `repeat3_hamming74` sends more bits, but it gives Bob a much better chance of reconstructing the exact original text. The demo output also reports `repeat-3 votes` and `hamming fixes`, which shows how many small transport errors were cleaned up.

## Recommended Presentation Flow

Use this sequence when preparing a live demo or slide deck:

1. Introduce the research question: can Alice and Bob learn private communication while Eve attacks?
2. Show the Alice/Bob/Eve diagram.
3. Run or show results from `anc-train`.
4. Open the generated `summary.md`, `metrics.json`, and `training_curves.png`.
5. Run `anc-eval` to show stronger attackers.
6. Run `anc-demo` with `--ecc repeat3_hamming74` so the recovered text is exact, not just close.
7. Run `anc-tpm` to show the separate synchronization experiment.
8. Run `anc-baseline` to compare against real authenticated encryption.
9. Close with limitations: useful experiment, not real security.

For live demos, train the checkpoint before the presentation and keep the run directory ready. Training can take longer than is comfortable on stage.

## Configuration Files

All main commands are driven by YAML config files in `configs/`.

| File | Purpose |
| --- | --- |
| `configs/anc_recommended.yaml` | Recommended presentation ANC training config. Uses equal plaintext, key, and ciphertext lengths, which works with `anc-demo`. |
| `configs/anc_small.yaml` | Larger 64-bit ANC experiment config with mixed random, structured, and edge-case data. Good for training and evaluation, but not the default demo path because key length differs from plaintext length. |
| `configs/anc_eval.yaml` | Stronger-attacker evaluation settings. |
| `configs/tpm_default.yaml` | Tree Parity Machine simulator settings. |
| `configs/baseline.yaml` | AES-GCM, ChaCha20-Poly1305, and optional neural-path benchmark settings. |
| `configs/shared.yaml` | Shared seed and path conventions for future or cross-mode configuration. |

ANC training and ANC evaluation support repeated config overrides:

```bash
anc-train --config configs/anc_recommended.yaml \
  --override epochs=50 \
  --override learning_rate=0.0005 \
  --override seed=123
```

Overrides use `key=value` syntax, and values are parsed as YAML. For example, `epochs=5` becomes an integer and `deterministic=true` becomes a boolean.

## Important ANC Configuration Knobs

| Key | Meaning |
| --- | --- |
| `plaintext_len` | Number of plaintext bits per sample. |
| `key_len` | Number of key bits shared by Alice and Bob. |
| `ciphertext_len` | Number of bits Alice outputs. Defaults to `plaintext_len` if omitted. |
| `model_width` | Hidden layer width for Alice, Bob, and Eve. |
| `model_depth` | MLP depth for Alice, Bob, and Eve. |
| `batch_size` | Number of samples per training batch. |
| `epochs` | Number of full training passes. |
| `learning_rate` | Alice and Bob optimizer learning rate. |
| `eve_learning_rate` | Eve optimizer learning rate. |
| `eve_steps_per_ab_step` | Number of Eve updates after each Alice/Bob update. |
| `eve_weight` | Strength of the adversarial term in Alice/Bob training. |
| `balance_weight` | Strength of ciphertext balance regularization. |
| `dataset_mode` | `random` or `mixed`. |
| `dataset_structured_ratio` | Fraction of structured correlated samples in mixed mode. |
| `dataset_edge_ratio` | Fraction of edge-case samples in mixed mode. |
| `seed` | Random seed recorded for reproducibility. |
| `output_dir` | Root directory where run artifacts are written. |
| `checkpoint_every` | Frequency for writing `checkpoint_last.pt`. |

## Dataset Modes

ANC data is generated from bitstrings.

`random` mode creates independent random plaintext and key bits.

`mixed` mode combines:

- random samples,
- structured samples with local or prefix dependencies,
- edge cases such as all-zero, all-one, alternating, sparse, and almost-all-one patterns.

Mixed mode is useful for presentations because it shows that the training set is not only uniform noise. It also makes evaluation more interesting because Bob and Eve must handle simple structure and edge cases.

## Output Artifacts

Every run creates a unique timestamped directory under its configured `output_dir`.

ANC training writes:

| Artifact | Meaning |
| --- | --- |
| `resolved_config.json` | Full config used for the run, including resolved device. |
| `metrics.json` | Final metrics, best metrics, and per-epoch metrics. |
| `train_log.csv` | Per-epoch CSV log. |
| `checkpoints/checkpoint_last.pt` | Latest checkpoint. |
| `checkpoints/checkpoint_best.pt` | Checkpoint with best Bob accuracy. |
| `training_curves.png` | Plot of loss and accuracy curves. |
| `summary.md` | Short human-readable summary. |

ANC evaluation writes:

| Artifact | Meaning |
| --- | --- |
| `resolved_config.json` | Evaluation configuration. |
| `metrics.json` | Attacker-mode metrics. |
| `attacker_log.jsonl` | Training logs for restarted, known-plaintext, and chosen-plaintext attackers when applicable. |
| `attacker_curves.png` | Optional attacker adaptation loss plot when attacker logs are available. |
| `summary.md` | Human-readable attacker summary. |

TPM simulation writes:

| Artifact | Meaning |
| --- | --- |
| `resolved_config.yaml` | TPM config used for the run. |
| `metrics.json` | Aggregate synchronization and observer metrics. |
| `trials.jsonl` | Trial-by-trial results. |
| `rounds.csv` | CSV version of trial rounds and attacker results. |
| `rounds_histogram.png` | Histogram of synchronization rounds. |
| `summary.txt` | Short TPM summary. |

Baseline benchmarking writes:

| Artifact | Meaning |
| --- | --- |
| `resolved_config.yaml` | Benchmark config used for the run. |
| `metrics.json` | Correctness, throughput, runtime, and optional neural comparison metrics. |
| `results.csv` | Algorithm and payload-size benchmark results. |
| `throughput.png` | Throughput chart. |
| `summary.txt` | Short benchmark summary. |

## Metrics Glossary

| Metric | How to read it |
| --- | --- |
| `bob_accuracy` | Fraction of bits Bob reconstructs correctly. Higher is better. |
| `bob_ber` | Bob bit error rate. Lower is better. |
| `eve_accuracy` | Fraction of bits Eve reconstructs correctly. Near `0.5` means random guessing for balanced bits. |
| `eve_ber` | Eve bit error rate. Higher is better for secrecy, but only within the tested threat model. |
| `ciphertext_bit_balance` | Mean of hard ciphertext bits. Values near `0.5` avoid trivial always-zero or always-one output. |
| `ciphertext_probability_mean` | Mean of Alice's soft ciphertext probabilities. |
| `plaintext_flip_sensitivity` | Average ciphertext change when one plaintext bit is flipped. |
| `key_flip_sensitivity` | Average ciphertext change when one key bit is flipped. |
| `success_rate` | TPM fraction of trials where Alice and Bob synchronized. |
| `attacker_success_rate` | TPM fraction of trials where the observer synchronized. |
| `round_trip_ok` | Baseline crypto decrypts back to the exact plaintext. |
| `throughput_bps` | Bytes per second for baseline encryption/decryption paths. |

## How To Interpret Results

A strong ANC result for this project usually looks like:

- Bob accuracy is high.
- Bob bit error rate is low.
- Eve accuracy remains close to random guessing.
- Stronger attacker modes do not dramatically outperform baseline Eve.
- Ciphertext balance is not collapsed to all zeros or all ones.
- Flip sensitivity shows ciphertext changes when plaintext or key bits change.

A weak result can still be useful. For example:

- If Bob fails, the model has not learned a reliable transport.
- If Eve succeeds, the learned protocol does not hide the plaintext under that attacker.
- If restarted or known-plaintext Eve performs much better, the original training Eve was too weak.
- If ciphertext balance collapses, the model may have learned a trivial representation.

## Presentation Talking Points

Use these points when explaining the project:

- The project studies the intersection of machine learning and cryptography.
- Alice and Bob are optimized for communication, while Eve is optimized for attack.
- The setup is adversarial because one model's success can be another model's failure.
- A single weak attacker is not enough, so the repository includes stronger evaluation modes.
- The TPM simulator shows a different neural-inspired key synchronization idea.
- AES-GCM and ChaCha20-Poly1305 are included as practical baselines and as a reminder of what real cryptography provides.
- The main lesson is evaluation discipline: every claim depends on the threat model and the attacker strength.

## Suggested Slide Outline

1. **Title:** Experimental Neural Cryptography Lab
2. **Motivation:** Can models learn encryption-like communication?
3. **Safety Note:** Research only, not production cryptography.
4. **Alice/Bob/Eve Diagram:** Sender, receiver, attacker.
5. **Training Loop:** Bob learns to decode, Eve learns to attack, Alice learns to hide.
6. **Metrics:** Bob accuracy, Eve accuracy, BER, ciphertext balance, flip sensitivity.
7. **Stronger Attackers:** Baseline, restarted, known-plaintext, chosen-plaintext.
8. **TPM Experiment:** Synchronization with public interaction and observer attacker.
9. **Baseline Crypto:** AES-GCM and ChaCha20-Poly1305 comparison.
10. **Demo:** Human-readable plaintext through `anc-demo` with Hamming correction.
11. **Results:** Show generated summaries, metrics, and plots.
12. **Limitations:** No proof, no production use, only tested threat models.
13. **Next Steps:** More attackers, longer bit lengths, additional architectures, broader statistical analysis.

## Demo Command Script

The following sequence is convenient for a live or recorded presentation.

Train before the presentation:

```bash
anc-train --config configs/anc_recommended.yaml
```

Then evaluate the printed run directory:

```bash
anc-eval --checkpoint <run_dir> --config configs/anc_eval.yaml
```

Run a plaintext demo:

```bash
anc-demo \
  --checkpoint <run_dir> \
  --text "Neural cryptography demo" \
  --key "shared presentation secret" \
  --ecc repeat3_hamming74 \
  --primary-path soft
```

Show ciphertext blocks if you want a more technical demo:

```bash
anc-demo \
  --checkpoint <run_dir> \
  --text "Neural cryptography demo" \
  --key "shared presentation secret" \
  --show-blocks
```

Run the TPM simulator:

```bash
anc-tpm --config configs/tpm_default.yaml
```

Run the AEAD baseline:

```bash
anc-baseline --config configs/baseline.yaml
```

## Repository Layout

```text
configs/
  anc_recommended.yaml      Presentation-oriented ANC training config
  anc_small.yaml            Larger ANC experiment config
  anc_eval.yaml             Stronger-attacker evaluation config
  tpm_default.yaml          TPM simulation config
  baseline.yaml             AEAD benchmark config

docs/
  ANC_IMPLEMENTATION.md     Extra implementation notes

src/
  baseline/                 AES-GCM and ChaCha20-Poly1305 benchmark code
  data/                     Bit utility helpers
  evaluation/               ANC checkpoint evaluation and attacker modes
  models/                   Alice, Bob, and Eve model definitions
  tpm/                      Tree Parity Machine simulator
  training/                 ANC training loop and checkpoint logic
  utils/                    Config, seed, IO, plotting, and error helpers
  demo_cli.py               Presentation-friendly plaintext demo

tests/
  test_*.py                 Unit and smoke tests

outputs/
  anc/                      ANC training runs
  anc_eval/                 ANC evaluation runs
  tpm/                      TPM runs
  baseline/                 Baseline benchmark runs
```

`outputs/` is created by run commands and may not exist in a fresh clone.

## Testing

Run the full test suite:

```bash
pytest -q
```

The tests cover:

- bit conversion utilities,
- Alice/Bob/Eve model shapes,
- one-step training behavior,
- checkpoint save/load behavior,
- dataset generation,
- TPM simulator behavior,
- baseline AEAD round trips and validation,
- demo CLI helpers,
- end-to-end smoke execution.

## Limitations

This project does not provide:

- a proof of secrecy,
- authenticated encryption for neural ciphertext,
- secure key exchange for production systems,
- resistance against arbitrary cryptanalysis,
- a replacement for standard cryptographic libraries,
- large-scale distributed training.

The learned system should be described as an experiment whose conclusions are limited by its architecture, data generation, attacker models, and evaluation settings.

## Future Work Ideas

Possible extensions:

- add stronger Eve architectures,
- add more chosen-plaintext and known-plaintext attack variants,
- test longer plaintext and key lengths,
- compare MLPs with convolutional or attention-based models,
- add statistical tests for ciphertext randomness,
- add repeated-seed experiment reports,
- add richer TPM attacker strategies,
- add notebooks or generated reports for easier presentation export.

## License

See `LICENSE`.
