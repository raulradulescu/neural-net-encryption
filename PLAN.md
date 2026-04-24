# Project Plan

Last updated: 2026-04-24

## Research Notes

Adversarial Neural Cryptography (ANC) was introduced by Abadi and Andersen as an end-to-end game between Alice, Bob, and Eve, where Alice and Bob share a key and Eve only sees ciphertext. The original paper is explicit that this is weaker than formal cryptographic security because it trains against one or a few learned adversaries rather than quantifying over all efficient attackers. Source: [arXiv:1610.06918](https://arxiv.org/abs/1610.06918).

Coutinho et al. argue that the original ANC setup often lets Alice and Bob learn schemes that beat a weak Eve without becoming secure cryptosystems. Their CPA-ANC variant strengthens Eve with chosen-plaintext structure and can drive simple networks toward one-time-pad-like behavior under controlled conditions. Source: [Sensors 2018 / PMC5982701](https://pmc.ncbi.nlm.nih.gov/articles/PMC5982701/).

Survey work after 2016 frames ANC as useful research machinery, but not deployable cryptography. Reported weaknesses include repetitive ciphertext patterns, weak distinguishability, and sensitivity to attacker assumptions. Source: [Neural Networks-Based Cryptography: A Survey, IEEE Access 2021](https://doaj.org/article/bcb66f2eaa4145678ee28a26a077a85b).

Recent extensions explore multi-party and asymmetric adversarial encryption, stronger leakage/CPA experiments, and secret-sharing-style settings. These are good research directions, but they reinforce that evaluation must include stronger attackers and clear threat models. Sources: [multi-party adversarial encryption, IEEE Access 2022](https://riuma.uma.es/xmlui/handle/10630/29922) and [asymmetric adversarial neural encryption, 2023](https://ouci.dntb.gov.ua/en/works/9GkP5w09/).

## Completed Linux Refactor

- [x] Lower Python packaging requirement from `>=3.13` to `>=3.10`, matching common Linux and WSL distributions.
- [x] Add Linux-friendly console scripts: `anc-train`, `anc-eval`, `anc-tpm`, `anc-baseline`, and `anc-demo`.
- [x] Make plotting lazy and optional so Matplotlib/NumPy ABI issues do not break imports or non-plot workflows.
- [x] Add CSV fallback for training/evaluation plots when Matplotlib is unavailable.
- [x] Reuse shared artifact helpers for run-directory, JSON, and YAML writes in baseline and TPM workflows.
- [x] Update README and demo docs to prefer Linux shell commands and POSIX paths.

## Recommended Next Changes

- [ ] Add Linux CI with Python 3.10, 3.11, and 3.12, running `pytest -q` with and without PyTorch installed.
- [ ] Add a lockfile or constraints file for reproducible Linux environments, especially around NumPy, Matplotlib, PyTorch, and `cryptography`.
- [ ] Rename the import package from `src` to a real package name such as `neural_crypto` in a planned migration.
- [ ] Split optional dependencies into clearer groups: `dev`, `plot`, `torch-cpu`, and possibly `demo`.
- [ ] Add CLI smoke tests for the installed console scripts after editable install.

## Recommended ANC Improvements

- [ ] Add a formal `threat_model.md` covering passive Eve, known-plaintext, chosen-plaintext, restarted Eve, and what the project does not claim.
- [ ] Add CPA-style training, not only CPA-style evaluation, so Alice/Bob learn against structured chosen-plaintext pressure.
- [ ] Add ciphertext indistinguishability tests: same plaintext/different keys, different plaintext/same key, bit-balance, collision rate, and simple statistical batteries.
- [ ] Add nonce/probabilistic-encryption experiments instead of deterministic ciphertext-only transforms.
- [ ] Track key reuse explicitly and warn when configs imply one key can protect multiple plaintext blocks.
- [ ] Add multiple Eve architectures and capacity sweeps so results are not tied to one adversary shape.
- [ ] Report confidence intervals across seeds; single-run Bob/Eve accuracy is not enough.
- [ ] Keep AES-GCM and ChaCha20-Poly1305 as first-class baselines and state clearly that they are the only suitable choices for real encryption.

## Recommended Project Hygiene

- [ ] Move generated outputs under a configurable Linux-friendly root, defaulting to `outputs/`, and document cleanup.
- [ ] Add `ruff` or equivalent linting once the current code is stable.
- [ ] Add type checking for config dataclasses and artifact payloads.
- [ ] Add a small `make test` or `just test` wrapper for Linux coursework demos.
- [ ] Add sample expected output snippets for each CLI in the docs.
