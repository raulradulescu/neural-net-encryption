from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.training.train_anc import ANCTrainingConfig, build_models


def test_model_output_shapes() -> None:
    cfg = ANCTrainingConfig(
        plaintext_len=16,
        key_len=16,
        ciphertext_len=16,
        model_width=32,
        model_depth=2,
        batch_size=8,
        epochs=1,
        learning_rate=1e-3,
        seed=3,
        train_samples=32,
        eval_samples=16,
        output_dir="outputs/test",
        device="cpu",
    )
    models = build_models(cfg, device=torch.device("cpu"))

    batch = 5
    plaintext = torch.randint(0, 2, (batch, cfg.plaintext_len), dtype=torch.float32)
    key = torch.randint(0, 2, (batch, cfg.key_len), dtype=torch.float32)

    alice_logits = models["alice"](plaintext, key)
    bob_logits = models["bob"](torch.sigmoid(alice_logits), key)
    eve_logits = models["eve"](torch.sigmoid(alice_logits))

    assert alice_logits.shape == (batch, cfg.ciphertext_len)
    assert bob_logits.shape == (batch, cfg.plaintext_len)
    assert eve_logits.shape == (batch, cfg.plaintext_len)
