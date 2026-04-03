from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.training.train_anc import ANCTrainingConfig, build_models, build_optimizers, generate_dataset, load_checkpoint, save_checkpoint, train_batch


def test_checkpoint_round_trip_preserves_outputs(tmp_path) -> None:
    cfg = ANCTrainingConfig(
        plaintext_len=16,
        key_len=16,
        ciphertext_len=16,
        model_width=32,
        model_depth=2,
        batch_size=8,
        epochs=1,
        learning_rate=1e-3,
        seed=23,
        train_samples=64,
        eval_samples=16,
        output_dir=str(tmp_path / "out"),
        device="cpu",
    )
    models = build_models(cfg, device=torch.device("cpu"))
    optimizers = build_optimizers(models, cfg)
    plaintext, key = generate_dataset(cfg, 8, cfg.seed, device=torch.device("cpu"))
    train_batch(models, optimizers, plaintext, key, cfg)

    probe_plain = torch.randint(0, 2, (4, cfg.plaintext_len), dtype=torch.float32)
    probe_key = torch.randint(0, 2, (4, cfg.key_len), dtype=torch.float32)
    with torch.no_grad():
        original_out = models["alice"](probe_plain, probe_key)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "config": cfg.to_dict(),
            "epoch": 1,
            "model_states": {name: model.state_dict() for name, model in models.items()},
            "optimizer_states": {name: opt.state_dict() for name, opt in optimizers.items()},
        },
    )

    loaded = load_checkpoint(checkpoint_path)
    with torch.no_grad():
        loaded_out = loaded["models"]["alice"](probe_plain, probe_key)

    assert torch.allclose(original_out, loaded_out)
    assert loaded["optimizers"]["alice"].state_dict()["state"]
