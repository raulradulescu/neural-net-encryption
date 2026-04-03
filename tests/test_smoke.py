from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from src.baseline.aead import run_benchmark_from_config
from src.evaluation.eval_anc import ANCEvalConfig, evaluate_checkpoint
from src.tpm.simulator import run_tpm_from_config
from src.training.train_anc import ANCTrainingConfig, train_anc


def test_smoke_end_to_end(tmp_path) -> None:
    anc_cfg = ANCTrainingConfig(
        plaintext_len=16,
        key_len=16,
        ciphertext_len=16,
        model_width=48,
        model_depth=2,
        batch_size=64,
        epochs=2,
        learning_rate=1e-3,
        eve_learning_rate=1e-3,
        eve_steps_per_ab_step=1,
        balance_weight=0.01,
        eve_weight=0.3,
        grad_clip=1.0,
        train_samples=2048,
        eval_samples=512,
        seed=42,
        output_dir=str(tmp_path / "anc"),
        run_name="smoke",
        device="cpu",
        deterministic=True,
        checkpoint_every=1,
    )

    anc_result = train_anc(anc_cfg)
    anc_run_dir = Path(anc_result["run_dir"])

    assert (anc_run_dir / "resolved_config.json").exists()
    assert (anc_run_dir / "metrics.json").exists()
    assert (anc_run_dir / "train_log.csv").exists()
    assert (anc_run_dir / "checkpoints" / "checkpoint_last.pt").exists()
    assert (anc_run_dir / "checkpoints" / "checkpoint_best.pt").exists()
    assert (anc_run_dir / "training_curves.png").exists()
    assert (anc_run_dir / "summary.md").exists()

    eval_cfg = ANCEvalConfig(
        output_dir=str(tmp_path / "anc_eval"),
        seed=77,
        device="cpu",
        eval_samples=512,
        batch_size=128,
        attacker_modes=["baseline", "restarted", "known_plaintext", "chosen_plaintext"],
        restart_count=2,
        restart_epochs=2,
        known_plaintext_samples=1024,
        known_plaintext_epochs=2,
        chosen_plaintext_queries=1024,
        chosen_plaintext_epochs=2,
        attacker_learning_rate=1e-3,
    )
    eval_result = evaluate_checkpoint(anc_run_dir, eval_cfg)
    eval_run_dir = Path(eval_result["run_dir"])
    assert (eval_run_dir / "metrics.json").exists()
    assert (eval_run_dir / "summary.md").exists()
    assert "baseline" in eval_result["metrics"]["modes"]
    assert "restarted" in eval_result["metrics"]["modes"]

    tpm_config = {
        "k_hidden": 3,
        "n_inputs": 4,
        "weight_limit": 3,
        "max_rounds": 200,
        "trials": 4,
        "seed": 5,
        "output_dir": str(tmp_path / "tpm"),
    }
    tpm_cfg_path = tmp_path / "tpm.yaml"
    tpm_cfg_path.write_text(yaml.safe_dump(tpm_config), encoding="utf-8")
    tpm_result = run_tpm_from_config(tpm_cfg_path)
    assert (Path(tpm_result["run_dir"]) / "metrics.json").exists()

    baseline_config = {
        "output_dir": str(tmp_path / "baseline"),
        "seed": 9,
        "iterations": 8,
        "payload_sizes": [32],
        "algorithms": ["aesgcm", "chacha20poly1305"],
        "associated_data": "aad",
        "neural_compare": True,
        "neural_checkpoint": str(anc_run_dir),
        "neural_batch_size": 32,
        "neural_iterations": 8,
    }
    baseline_cfg_path = tmp_path / "baseline.yaml"
    baseline_cfg_path.write_text(yaml.safe_dump(baseline_config), encoding="utf-8")
    baseline_result = run_benchmark_from_config(baseline_cfg_path)
    assert (Path(baseline_result["run_dir"]) / "metrics.json").exists()
