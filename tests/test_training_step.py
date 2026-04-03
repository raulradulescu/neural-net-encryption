from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.training.train_anc import ANCTrainingConfig, build_models, build_optimizers, generate_dataset, train_batch


def test_one_training_step_runs_and_has_gradients() -> None:
    cfg = ANCTrainingConfig(
        plaintext_len=16,
        key_len=16,
        ciphertext_len=16,
        model_width=32,
        model_depth=2,
        batch_size=8,
        epochs=1,
        learning_rate=1e-3,
        seed=11,
        train_samples=64,
        eval_samples=16,
        output_dir="outputs/test",
        device="cpu",
    )
    models = build_models(cfg, device=torch.device("cpu"))
    optimizers = build_optimizers(models, cfg)
    plaintext, key = generate_dataset(cfg, 8, cfg.seed, device=torch.device("cpu"))

    metrics = train_batch(models, optimizers, plaintext, key, cfg)

    assert torch.isfinite(torch.tensor(metrics["bob_loss"]))
    assert torch.isfinite(torch.tensor(metrics["eve_loss_step"]))
    assert metrics["alice_grad_norm"] > 0.0
    assert metrics["bob_grad_norm"] > 0.0
    assert metrics["eve_grad_norm"] > 0.0
