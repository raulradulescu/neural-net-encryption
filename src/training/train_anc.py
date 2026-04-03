from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Adam

from src.models import AliceModel, BobModel, EveModel, bit_accuracy, bit_error_rate, gradient_norm, hard_bits_from_logits
from src.utils.config import load_config_with_overrides
from src.utils.io import create_run_dir, write_csv, write_json
from src.utils.plot import save_line_plot
from src.utils.seed import set_seed


@dataclass
class ANCTrainingConfig:
    plaintext_len: int = 16
    key_len: int = 16
    ciphertext_len: int | None = None
    model_width: int = 64
    model_depth: int = 2
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    eve_learning_rate: float | None = None
    eve_steps_per_ab_step: int = 1
    balance_weight: float = 0.01
    eve_weight: float = 0.5
    grad_clip: float = 1.0
    train_samples: int = 8192
    eval_samples: int = 1024
    seed: int = 7
    output_dir: str = "outputs/anc"
    run_name: str = "anc"
    device: str = "cpu"
    deterministic: bool = True
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.ciphertext_len is None:
            self.ciphertext_len = self.plaintext_len
        self.validate()

    def validate(self) -> None:
        positive_ints = {
            "plaintext_len": self.plaintext_len,
            "key_len": self.key_len,
            "ciphertext_len": self.ciphertext_len,
            "model_width": self.model_width,
            "model_depth": self.model_depth,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "eve_steps_per_ab_step": self.eve_steps_per_ab_step,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "checkpoint_every": self.checkpoint_every,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        positive_floats = {
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
        }
        for name, value in positive_floats.items():
            if float(value) <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if self.eve_learning_rate is not None and float(self.eve_learning_rate) <= 0:
            raise ValueError("eve_learning_rate must be > 0 when provided")
        if self.balance_weight < 0:
            raise ValueError("balance_weight must be >= 0")
        if self.eve_weight < 0:
            raise ValueError("eve_weight must be >= 0")
        if self.device not in {"auto", "cpu"}:
            raise ValueError("device must be one of: auto, cpu")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ANCTrainingConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping.keys()) - allowed)
        if unknown:
            raise ValueError(f"Unknown ANC config keys: {unknown}")
        return cls(**dict(mapping))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(device_name: str) -> torch.device:
    if device_name not in {"cpu", "auto"}:
        raise RuntimeError("This project is configured for CPU-only execution; set device=cpu")
    return torch.device("cpu")


def load_config_file(config_path: Path, overrides: Sequence[str] | None = None) -> ANCTrainingConfig:
    mapping = load_config_with_overrides(config_path, overrides or [])
    return ANCTrainingConfig.from_mapping(mapping)


def generate_dataset(cfg: ANCTrainingConfig, n_samples: int, seed: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    target_device = device if device is not None else torch.device("cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    plaintext = torch.randint(0, 2, (n_samples, cfg.plaintext_len), generator=generator, dtype=torch.float32)
    key = torch.randint(0, 2, (n_samples, cfg.key_len), generator=generator, dtype=torch.float32)
    return plaintext.to(target_device), key.to(target_device)


def build_models(cfg: ANCTrainingConfig, device: torch.device | None = None) -> dict[str, torch.nn.Module]:
    target_device = device if device is not None else torch.device("cpu")
    alice = AliceModel(
        plaintext_len=cfg.plaintext_len,
        key_len=cfg.key_len,
        ciphertext_len=cfg.ciphertext_len,
        width=cfg.model_width,
        depth=cfg.model_depth,
    ).to(target_device)
    bob = BobModel(
        plaintext_len=cfg.plaintext_len,
        key_len=cfg.key_len,
        ciphertext_len=cfg.ciphertext_len,
        width=cfg.model_width,
        depth=cfg.model_depth,
    ).to(target_device)
    eve = EveModel(
        plaintext_len=cfg.plaintext_len,
        ciphertext_len=cfg.ciphertext_len,
        width=cfg.model_width,
        depth=cfg.model_depth,
    ).to(target_device)
    return {"alice": alice, "bob": bob, "eve": eve}


def build_optimizers(models: Mapping[str, torch.nn.Module], cfg: ANCTrainingConfig) -> dict[str, Adam]:
    eve_lr = cfg.eve_learning_rate if cfg.eve_learning_rate is not None else cfg.learning_rate
    return {
        "alice": Adam(models["alice"].parameters(), lr=cfg.learning_rate),
        "bob": Adam(models["bob"].parameters(), lr=cfg.learning_rate),
        "eve": Adam(models["eve"].parameters(), lr=eve_lr),
    }


def _batch_slices(n_samples: int, batch_size: int, seed: int) -> Sequence[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(n_samples, generator=generator)
    return [order[i : i + batch_size] for i in range(0, n_samples, batch_size)]


def _clip_and_norm(parameters: Sequence[torch.nn.Parameter], max_norm: float) -> float:
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    return gradient_norm(parameters)


def train_batch(
    models: Mapping[str, torch.nn.Module],
    optimizers: Mapping[str, Adam],
    plaintext: Tensor,
    key: Tensor,
    cfg: ANCTrainingConfig,
) -> dict[str, float]:
    alice = models["alice"]
    bob = models["bob"]
    eve = models["eve"]
    bce = torch.nn.BCEWithLogitsLoss()

    alice.train()
    bob.train()
    eve.train()

    optimizers["alice"].zero_grad(set_to_none=True)
    optimizers["bob"].zero_grad(set_to_none=True)

    alice_logits = alice(plaintext, key)
    ciphertext_prob = torch.sigmoid(alice_logits)
    bob_logits = bob(ciphertext_prob, key)
    eve_logits_ab = eve(ciphertext_prob)

    bob_loss = bce(bob_logits, plaintext)
    eve_loss_ab = bce(eve_logits_ab, plaintext)
    balance_loss = ((ciphertext_prob - 0.5) ** 2).mean()
    ab_loss = bob_loss - cfg.eve_weight * eve_loss_ab + cfg.balance_weight * balance_loss
    ab_loss.backward()

    alice_params = [p for p in alice.parameters() if p.requires_grad]
    bob_params = [p for p in bob.parameters() if p.requires_grad]
    alice_grad_norm = _clip_and_norm(alice_params, cfg.grad_clip)
    bob_grad_norm = _clip_and_norm(bob_params, cfg.grad_clip)

    optimizers["alice"].step()
    optimizers["bob"].step()

    eve_losses: list[float] = []
    eve_grad_norm = 0.0
    for _ in range(cfg.eve_steps_per_ab_step):
        optimizers["eve"].zero_grad(set_to_none=True)
        with torch.no_grad():
            detached_cipher = torch.sigmoid(alice(plaintext, key))
        eve_logits = eve(detached_cipher)
        eve_loss = bce(eve_logits, plaintext)
        eve_loss.backward()
        eve_params = [p for p in eve.parameters() if p.requires_grad]
        eve_grad_norm = _clip_and_norm(eve_params, cfg.grad_clip)
        optimizers["eve"].step()
        eve_losses.append(float(eve_loss.item()))

    return {
        "bob_loss": float(bob_loss.item()),
        "eve_loss_ab": float(eve_loss_ab.item()),
        "eve_loss_step": float(sum(eve_losses) / len(eve_losses)),
        "balance_loss": float(balance_loss.item()),
        "alice_grad_norm": alice_grad_norm,
        "bob_grad_norm": bob_grad_norm,
        "eve_grad_norm": eve_grad_norm,
    }


@torch.no_grad()
def _flip_sensitivity(
    alice: AliceModel,
    plaintext: Tensor,
    key: Tensor,
    bit_count: int,
    flip_target: str,
) -> float:
    subset = min(plaintext.shape[0], 64)
    p = plaintext[:subset]
    k = key[:subset]
    base_cipher = hard_bits_from_logits(alice(p, k))

    distances: list[float] = []
    for idx in range(bit_count):
        if flip_target == "plaintext":
            p_flip = p.clone()
            p_flip[:, idx] = 1.0 - p_flip[:, idx]
            flip_cipher = hard_bits_from_logits(alice(p_flip, k))
        elif flip_target == "key":
            k_flip = k.clone()
            k_flip[:, idx] = 1.0 - k_flip[:, idx]
            flip_cipher = hard_bits_from_logits(alice(p, k_flip))
        else:
            raise ValueError(f"Unknown flip_target: {flip_target}")
        distances.append(float((flip_cipher != base_cipher).to(dtype=torch.float32).mean().item()))

    return float(sum(distances) / len(distances)) if distances else 0.0


@torch.no_grad()
def evaluate_models(
    models: Mapping[str, torch.nn.Module],
    plaintext: Tensor,
    key: Tensor,
    cfg: ANCTrainingConfig,
) -> dict[str, float]:
    alice: AliceModel = models["alice"]  # type: ignore[assignment]
    bob: BobModel = models["bob"]  # type: ignore[assignment]
    eve: EveModel = models["eve"]  # type: ignore[assignment]

    alice.eval()
    bob.eval()
    eve.eval()

    bce = torch.nn.BCEWithLogitsLoss()

    cipher_logits = alice(plaintext, key)
    cipher_prob = torch.sigmoid(cipher_logits)
    cipher_bits = hard_bits_from_logits(cipher_logits)

    bob_logits = bob(cipher_prob, key)
    eve_logits = eve(cipher_prob)

    bob_bits = hard_bits_from_logits(bob_logits)
    eve_bits = hard_bits_from_logits(eve_logits)

    bob_loss = bce(bob_logits, plaintext)
    eve_loss = bce(eve_logits, plaintext)

    return {
        "bob_accuracy": bit_accuracy(bob_bits, plaintext),
        "bob_ber": bit_error_rate(bob_bits, plaintext),
        "bob_loss_eval": float(bob_loss.item()),
        "eve_accuracy": bit_accuracy(eve_bits, plaintext),
        "eve_ber": bit_error_rate(eve_bits, plaintext),
        "eve_loss_eval": float(eve_loss.item()),
        "ciphertext_bit_balance": float(cipher_bits.mean().item()),
        "ciphertext_probability_mean": float(cipher_prob.mean().item()),
        "plaintext_flip_sensitivity": _flip_sensitivity(alice, plaintext, key, cfg.plaintext_len, "plaintext"),
        "key_flip_sensitivity": _flip_sensitivity(alice, plaintext, key, cfg.key_len, "key"),
    }


def save_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any] | None = None,
    *,
    config: ANCTrainingConfig | Mapping[str, Any] | None = None,
    epoch: int | None = None,
    models: Mapping[str, torch.nn.Module] | None = None,
    optimizers: Mapping[str, Adam] | None = None,
    metrics: Mapping[str, Any] | None = None,
    epoch_logs: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if payload is None:
        if config is None or epoch is None or models is None or optimizers is None:
            raise ValueError("save_checkpoint requires payload or config/epoch/models/optimizers")
        resolved_config = config.to_dict() if isinstance(config, ANCTrainingConfig) else dict(config)
        payload = {
            "config": resolved_config,
            "epoch": int(epoch),
            "model_states": {name: model.state_dict() for name, model in models.items()},
            "optimizer_states": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "metrics": dict(metrics or {}),
            "epoch_logs": [dict(row) for row in epoch_logs] if epoch_logs is not None else [],
        }

    torch.save(dict(payload), destination)
    return destination


def _resolve_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")

    options = [
        candidate / "checkpoints" / "checkpoint_best.pt",
        candidate / "checkpoints" / "checkpoint_last.pt",
        candidate / "checkpoint_best.pt",
        candidate / "checkpoint_last.pt",
        candidate / "checkpoint.pt",
    ]
    for option in options:
        if option.exists():
            return option
    raise FileNotFoundError(f"No checkpoint file found in {candidate}")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt_path = _resolve_checkpoint_path(path)
    payload = torch.load(ckpt_path, map_location=map_location)

    if "config" not in payload:
        raise ValueError(f"Invalid checkpoint format: missing config in {ckpt_path}")

    cfg = ANCTrainingConfig.from_mapping(payload["config"])
    device = torch.device(map_location if isinstance(map_location, str) else map_location.type)
    models = build_models(cfg, device=device)
    optimizers = build_optimizers(models, cfg)

    model_states = payload.get("model_states") or payload.get("models")
    optimizer_states = payload.get("optimizer_states") or payload.get("optimizers")
    if model_states is None or optimizer_states is None:
        raise ValueError(f"Invalid checkpoint format: missing model/optimizer states in {ckpt_path}")

    for name, model in models.items():
        model.load_state_dict(model_states[name])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_states[name])

    payload["config"] = cfg
    payload["models"] = models
    payload["optimizers"] = optimizers
    payload["checkpoint_path"] = str(ckpt_path)
    return payload


def _write_summary(path: Path, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# ANC Run Summary",
        "",
        f"- Bob accuracy: {metrics.get('bob_accuracy', 'n/a'):.4f}" if isinstance(metrics.get("bob_accuracy"), float) else f"- Bob accuracy: {metrics.get('bob_accuracy', 'n/a')}",
        f"- Eve accuracy: {metrics.get('eve_accuracy', 'n/a'):.4f}" if isinstance(metrics.get("eve_accuracy"), float) else f"- Eve accuracy: {metrics.get('eve_accuracy', 'n/a')}",
        f"- Bob BER: {metrics.get('bob_ber', 'n/a'):.4f}" if isinstance(metrics.get("bob_ber"), float) else f"- Bob BER: {metrics.get('bob_ber', 'n/a')}",
        f"- Eve BER: {metrics.get('eve_ber', 'n/a'):.4f}" if isinstance(metrics.get("eve_ber"), float) else f"- Eve BER: {metrics.get('eve_ber', 'n/a')}",
        f"- Ciphertext bit balance: {metrics.get('ciphertext_bit_balance', 'n/a'):.4f}" if isinstance(metrics.get("ciphertext_bit_balance"), float) else f"- Ciphertext bit balance: {metrics.get('ciphertext_bit_balance', 'n/a')}",
        f"- Plaintext flip sensitivity: {metrics.get('plaintext_flip_sensitivity', 'n/a'):.4f}" if isinstance(metrics.get("plaintext_flip_sensitivity"), float) else f"- Plaintext flip sensitivity: {metrics.get('plaintext_flip_sensitivity', 'n/a')}",
        f"- Key flip sensitivity: {metrics.get('key_flip_sensitivity', 'n/a'):.4f}" if isinstance(metrics.get("key_flip_sensitivity"), float) else f"- Key flip sensitivity: {metrics.get('key_flip_sensitivity', 'n/a')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, epoch_logs: Sequence[Mapping[str, Any]]) -> None:
    if not epoch_logs:
        return
    save_line_plot(
        path,
        {
            "bob_loss": [float(row["train_bob_loss"]) for row in epoch_logs],
            "eve_loss": [float(row["train_eve_loss_step"]) for row in epoch_logs],
            "bob_accuracy": [float(row["bob_accuracy"]) for row in epoch_logs],
            "eve_accuracy": [float(row["eve_accuracy"]) for row in epoch_logs],
        },
        title="ANC Training Curves",
        xlabel="epoch",
        ylabel="value",
    )


def train_anc(cfg: ANCTrainingConfig) -> dict[str, Any]:
    seed = set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(cfg.device)
    run_dir = create_run_dir(cfg.output_dir, prefix="anc", run_name=cfg.run_name)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(cfg, device=device)
    optimizers = build_optimizers(models, cfg)

    train_plain, train_key = generate_dataset(cfg, cfg.train_samples, seed, device=device)
    eval_plain, eval_key = generate_dataset(cfg, cfg.eval_samples, seed + 1, device=device)

    resolved_config = cfg.to_dict() | {"resolved_device": str(device)}
    write_json(run_dir / "resolved_config.json", resolved_config)

    epoch_logs: list[dict[str, float]] = []
    best_bob_accuracy = -1.0
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, cfg.epochs + 1):
        train_batch_metrics: list[dict[str, float]] = []
        for batch_ids in _batch_slices(cfg.train_samples, cfg.batch_size, seed + epoch):
            batch_plain = train_plain[batch_ids.to(device)]
            batch_key = train_key[batch_ids.to(device)]
            train_batch_metrics.append(train_batch(models, optimizers, batch_plain, batch_key, cfg))

        eval_metrics = evaluate_models(models, eval_plain, eval_key, cfg)
        epoch_row = {
            "epoch": float(epoch),
            "train_bob_loss": float(sum(m["bob_loss"] for m in train_batch_metrics) / len(train_batch_metrics)),
            "train_eve_loss_ab": float(sum(m["eve_loss_ab"] for m in train_batch_metrics) / len(train_batch_metrics)),
            "train_eve_loss_step": float(sum(m["eve_loss_step"] for m in train_batch_metrics) / len(train_batch_metrics)),
            "train_balance_loss": float(sum(m["balance_loss"] for m in train_batch_metrics) / len(train_batch_metrics)),
            "train_alice_grad_norm": float(sum(m["alice_grad_norm"] for m in train_batch_metrics) / len(train_batch_metrics)),
            "train_bob_grad_norm": float(sum(m["bob_grad_norm"] for m in train_batch_metrics) / len(train_batch_metrics)),
            "train_eve_grad_norm": float(sum(m["eve_grad_norm"] for m in train_batch_metrics) / len(train_batch_metrics)),
            **eval_metrics,
        }
        epoch_logs.append(epoch_row)

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(
                checkpoints_dir / "checkpoint_last.pt",
                config=cfg,
                epoch=epoch,
                models=models,
                optimizers=optimizers,
                metrics=epoch_row,
                epoch_logs=epoch_logs,
            )

        if epoch_row["bob_accuracy"] > best_bob_accuracy:
            best_bob_accuracy = epoch_row["bob_accuracy"]
            best_metrics = dict(epoch_row)
            save_checkpoint(
                checkpoints_dir / "checkpoint_best.pt",
                config=cfg,
                epoch=epoch,
                models=models,
                optimizers=optimizers,
                metrics=epoch_row,
                epoch_logs=epoch_logs,
            )

    final_metrics = dict(epoch_logs[-1]) if epoch_logs else {}

    write_csv(run_dir / "train_log.csv", epoch_logs)
    write_json(
        run_dir / "metrics.json",
        {
            "run_dir": str(run_dir),
            "config": resolved_config,
            "final": final_metrics,
            "best": best_metrics,
            "epochs": epoch_logs,
        },
    )
    _write_plot(run_dir / "training_curves.png", epoch_logs)
    _write_summary(run_dir / "summary.md", final_metrics)

    return {
        "run_dir": str(run_dir),
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "epoch_logs": epoch_logs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANC Alice/Bob/Eve models")
    parser.add_argument("--config", required=True, help="Path to ANC config YAML/JSON")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values using dotted key=value syntax. Example: --override epochs=5",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config_file(Path(args.config), args.override)
    result = train_anc(cfg)
    print(json.dumps({"run_dir": result["run_dir"], "final_metrics": result["final_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
