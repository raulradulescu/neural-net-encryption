# ANC Implementation and Error Correction

## Overview

This project implements a small Adversarial Neural Cryptography (ANC) experiment with three neural networks: Alice, Bob, and Eve. It is meant for reproducible experiments and demonstrations, not for real security. For real encryption, use audited schemes such as AES-GCM or ChaCha20-Poly1305.

## Model Roles

The models are defined in `src/models/anc.py`.

- `AliceModel` receives a plaintext bit vector and a key bit vector, then outputs ciphertext logits.
- `BobModel` receives Alice's ciphertext and the same key, then tries to reconstruct the plaintext.
- `EveModel` receives only the ciphertext and tries to guess the plaintext.

All three models are simple MLPs. Their width and depth are controlled by config values such as `model_width` and `model_depth`.

## Training Objective

Training is implemented in `src/training/train_anc.py`. For each batch:

1. Alice produces ciphertext probabilities with `sigmoid(alice(plaintext, key))`.
2. Bob tries to reconstruct the original plaintext.
3. Eve tries to recover plaintext from ciphertext alone.
4. Alice and Bob are updated with:

```text
bob_loss - eve_weight * eve_loss + balance_weight * balance_loss
```

`bob_loss` rewards correct reconstruction. The negative Eve term encourages Alice/Bob to make Eve worse. `balance_loss` pushes ciphertext probabilities toward `0.5`, which discourages trivial always-zero or always-one ciphertext.

Eve is then trained separately for `eve_steps_per_ab_step` steps against detached ciphertext. This keeps Eve as an active attacker during training.

## Demo Plaintext Flow

The demo CLI is in `src/demo_cli.py`.

```bash
anc-demo --checkpoint outputs/anc/<run_dir>
```

If `--text` or `--key` is omitted, the CLI prompts for them. Text is encoded as bytes, padded with PKCS#7 to the model block size, converted to bits, and split into fixed-size blocks matching `plaintext_len`.

The key passphrase is expanded with SHA-256 into the exact number of key bits expected by the checkpoint.

## Error Correction Modes

The neural transport can make bit mistakes, so the demo can wrap plaintext bits with an outer error-correction layer before Alice processes them.

- `none`: send the padded plaintext bits directly.
- `hamming74`: encode every 4 data bits as a 7-bit Hamming(7,4) codeword. This can correct one bit error per codeword.
- `repeat3_hamming74`: apply Hamming(7,4), then repeat every encoded bit three times. Decoding first uses majority vote, then Hamming correction.

The default is `repeat3_hamming74` because it is the most reliable for live demos. It increases message size, but helps Bob recover the original text when the neural model is slightly noisy.

## Soft vs Hard Ciphertext

The demo reports two Bob paths:

- `soft`: Bob receives Alice's sigmoid probabilities. This matches training and is usually more reliable.
- `hard`: Alice's logits are thresholded into exact bits before Bob sees them. This is easier to display but can lose useful information.

Use `--primary-path soft` for demonstrations unless you are specifically testing bit-thresholded ciphertext.

## Practical Commands

Train a recommended checkpoint:

```bash
anc-train --config configs/anc_recommended.yaml
```

Evaluate it:

```bash
anc-eval --checkpoint outputs/anc/<run_dir> --config configs/anc_eval.yaml
```

Test typed plaintext:

```bash
anc-demo --checkpoint outputs/anc/<run_dir> --ecc repeat3_hamming74 --primary-path soft
```

## Limitations

This implementation does not prove cryptographic security. It only measures performance against the Eve models and attacker modes implemented in this repository. Always report Bob accuracy, Eve accuracy, bit error rate, attacker mode, config, and random seed when presenting results.
