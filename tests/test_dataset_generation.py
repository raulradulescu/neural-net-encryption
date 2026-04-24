from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.training.train_anc import ANCTrainingConfig, generate_dataset


def test_generate_dataset_is_deterministic_for_same_seed() -> None:
    cfg = ANCTrainingConfig(
        plaintext_len=16,
        key_len=16,
        train_samples=128,
        eval_samples=32,
        dataset_mode="mixed",
        dataset_structured_ratio=0.35,
        dataset_edge_ratio=0.10,
        device="cpu",
    )

    p1, k1 = generate_dataset(cfg, 128, seed=123, device=torch.device("cpu"))
    p2, k2 = generate_dataset(cfg, 128, seed=123, device=torch.device("cpu"))

    assert torch.equal(p1, p2)
    assert torch.equal(k1, k2)


def test_generate_dataset_mixed_contains_edge_patterns() -> None:
    cfg = ANCTrainingConfig(
        plaintext_len=16,
        key_len=16,
        train_samples=256,
        eval_samples=32,
        dataset_mode="mixed",
        dataset_structured_ratio=0.40,
        dataset_edge_ratio=0.20,
        device="cpu",
    )

    plain, key = generate_dataset(cfg, 256, seed=999, device=torch.device("cpu"))

    assert plain.shape == (256, 16)
    assert key.shape == (256, 16)

    # With edge-ratio enabled, we should reliably see all-zero and all-one rows.
    zeros_plain = (plain.sum(dim=1) == 0).any().item()
    ones_plain = (plain.sum(dim=1) == cfg.plaintext_len).any().item()
    zeros_key = (key.sum(dim=1) == 0).any().item()
    ones_key = (key.sum(dim=1) == cfg.key_len).any().item()

    assert zeros_plain and ones_plain
    assert zeros_key and ones_key
