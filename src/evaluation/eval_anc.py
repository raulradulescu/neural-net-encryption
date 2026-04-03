from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Adam

from src.models import EveModel, hard_bits_from_logits
from src.training.train_anc import ANCTrainingConfig, generate_dataset, load_checkpoint, resolve_device
from src.utils.config import load_config_with_overrides
from src.utils.io import create_run_dir, write_json, write_jsonl
from src.utils.plot import save_line_plot
from src.utils.seed import set_seed


@dataclass
class ANCEvalConfig:
    checkpoint: str | None = None
    output_dir: str | None = None
    seed: int = 19
    device: str = "cpu"
    eval_samples: int = 2048
    batch_size: int = 256
    attacker_modes: list[str] | None = None
    restart_count: int = 5
    restart_epochs: int = 4
    known_plaintext_samples: int = 4096
    known_plaintext_epochs: int = 4
    chosen_plaintext_queries: int = 8192
    chosen_plaintext_epochs: int = 6
    attacker_learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.attacker_modes is None:
            self.attacker_modes = ["baseline", "restarted", "known_plaintext", "chosen_plaintext"]
        self.validate()

    def validate(self) -> None:
        if self.eval_samples <= 0:
            raise ValueError("eval_samples must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.restart_count <= 0:
            raise ValueError("restart_count must be positive")
        if self.restart_epochs <= 0:
            raise ValueError("restart_epochs must be positive")
        if self.known_plaintext_samples <= 0:
            raise ValueError("known_plaintext_samples must be positive")
        if self.known_plaintext_epochs <= 0:
            raise ValueError("known_plaintext_epochs must be positive")
        if self.chosen_plaintext_queries <= 0:
            raise ValueError("chosen_plaintext_queries must be positive")
        if self.chosen_plaintext_epochs <= 0:
            raise ValueError("chosen_plaintext_epochs must be positive")
        if self.attacker_learning_rate <= 0:
            raise ValueError("attacker_learning_rate must be positive")
        if self.device not in {"auto", "cpu"}:
            raise ValueError("device must be one of: auto, cpu")

        allowed_modes = {"baseline", "restarted", "known_plaintext", "chosen_plaintext"}
        invalid = [mode for mode in self.attacker_modes if mode not in allowed_modes]
        if invalid:
            raise ValueError(f"Unsupported attacker modes: {invalid}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ANCEvalConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping.keys()) - allowed)
        if unknown:
            raise ValueError(f"Unknown eval config keys: {unknown}")
        return cls(**dict(mapping))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_eval_config(config_path: Path | None, overrides: Sequence[str] | None = None) -> ANCEvalConfig:
    if config_path is None:
        base: dict[str, Any] = {}
    else:
        base = load_config_with_overrides(config_path, overrides or [])
    if config_path is None and overrides:
        overlay = load_config_with_overrides(Path("configs/anc_eval.yaml"), overrides)
        return ANCEvalConfig.from_mapping(overlay)
    return ANCEvalConfig.from_mapping(base)


@torch.no_grad()
def _evaluate_eve(eve: EveModel, alice, plaintext: Tensor, key: Tensor) -> dict[str, float]:
    eve.eval()
    alice.eval()
    bce = torch.nn.BCEWithLogitsLoss()

    ciphertext = torch.sigmoid(alice(plaintext, key))
    logits = eve(ciphertext)
    pred_bits = hard_bits_from_logits(logits)
    loss = bce(logits, plaintext)

    accuracy = float((pred_bits == plaintext).to(dtype=torch.float32).mean().item())
    ber = 1.0 - accuracy
    return {
        "eve_accuracy": accuracy,
        "eve_ber": ber,
        "eve_loss": float(loss.item()),
    }


def _batch_indices(n_samples: int, batch_size: int, seed: int) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(n_samples, generator=generator)
    return [order[i : i + batch_size] for i in range(0, n_samples, batch_size)]


def _train_attacker(
    alice,
    eve: EveModel,
    plaintext: Tensor,
    key: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    optimizer = Adam(eve.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logs: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for ids in _batch_indices(plaintext.shape[0], batch_size, seed + epoch):
            idx = ids.to(plaintext.device)
            p = plaintext[idx]
            k = key[idx]
            with torch.no_grad():
                c = torch.sigmoid(alice(p, k))
            optimizer.zero_grad(set_to_none=True)
            logits = eve(c)
            loss = loss_fn(logits, p)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        logs.append({"epoch": float(epoch), "eve_train_loss": float(sum(epoch_losses) / len(epoch_losses))})

    return logs


def _make_chosen_plaintext_queries(
    anc_cfg: ANCTrainingConfig,
    n_queries: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    plaintext, key = generate_dataset(anc_cfg, n_queries, seed, device=device)

    # Inject structured patterns to emulate chosen-plaintext probing.
    n_inject = min(anc_cfg.plaintext_len + 2, n_queries)
    plaintext[0] = torch.zeros(anc_cfg.plaintext_len, device=device)
    if n_queries > 1:
        plaintext[1] = torch.ones(anc_cfg.plaintext_len, device=device)
    for bit in range(max(0, n_inject - 2)):
        row = bit + 2
        pattern = torch.zeros(anc_cfg.plaintext_len, device=device)
        pattern[bit % anc_cfg.plaintext_len] = 1.0
        plaintext[row] = pattern

    return plaintext, key


def _evaluate_restarted(
    anc_cfg: ANCTrainingConfig,
    alice,
    eval_plain: Tensor,
    eval_key: Tensor,
    cfg: ANCEvalConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    restart_metrics: list[dict[str, float]] = []
    all_logs: list[dict[str, float]] = []

    train_plain, train_key = generate_dataset(anc_cfg, cfg.known_plaintext_samples, cfg.seed + 100, device=device)
    best_accuracy = -1.0
    best_index = -1

    for restart in range(cfg.restart_count):
        torch.manual_seed(cfg.seed + restart)
        eve = EveModel(
            plaintext_len=anc_cfg.plaintext_len,
            ciphertext_len=anc_cfg.ciphertext_len,
            width=anc_cfg.model_width,
            depth=anc_cfg.model_depth,
        ).to(device)
        logs = _train_attacker(
            alice,
            eve,
            train_plain,
            train_key,
            epochs=cfg.restart_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.attacker_learning_rate,
            seed=cfg.seed + restart,
        )
        for row in logs:
            all_logs.append({"mode": "restarted", "restart": float(restart), **row})

        metrics = _evaluate_eve(eve, alice, eval_plain, eval_key)
        metrics["restart"] = float(restart)
        restart_metrics.append(metrics)

        if metrics["eve_accuracy"] > best_accuracy:
            best_accuracy = metrics["eve_accuracy"]
            best_index = restart

    summary = {
        "best_restart": best_index,
        "best_eve_accuracy": best_accuracy,
        "mean_eve_accuracy": float(sum(m["eve_accuracy"] for m in restart_metrics) / len(restart_metrics)),
        "mean_eve_ber": float(sum(m["eve_ber"] for m in restart_metrics) / len(restart_metrics)),
        "restarts": restart_metrics,
    }
    return summary, all_logs


def evaluate_checkpoint(checkpoint: str | Path, eval_cfg: ANCEvalConfig) -> dict[str, Any]:
    resolved_checkpoint = Path(checkpoint)
    device = resolve_device(eval_cfg.device)
    set_seed(eval_cfg.seed)

    loaded = load_checkpoint(resolved_checkpoint, map_location=device)
    anc_cfg: ANCTrainingConfig = loaded["config"]
    alice = loaded["models"]["alice"].to(device)
    checkpoint_eve = loaded["models"]["eve"].to(device)

    eval_plain, eval_key = generate_dataset(anc_cfg, eval_cfg.eval_samples, eval_cfg.seed + 1, device=device)

    output_root = Path(eval_cfg.output_dir) if eval_cfg.output_dir else Path(resolved_checkpoint).resolve().parent
    run_dir = create_run_dir(output_root, prefix="anc_eval", run_name="eval")

    metrics: dict[str, Any] = {
        "checkpoint": str(resolved_checkpoint),
        "run_dir": str(run_dir),
        "eval_samples": eval_cfg.eval_samples,
        "attacker_modes": eval_cfg.attacker_modes,
        "modes": {},
    }
    attack_logs: list[dict[str, float]] = []

    if "baseline" in eval_cfg.attacker_modes:
        baseline_metrics = _evaluate_eve(checkpoint_eve, alice, eval_plain, eval_key)
        metrics["modes"]["baseline"] = baseline_metrics

    if "restarted" in eval_cfg.attacker_modes:
        restarted_summary, restart_logs = _evaluate_restarted(anc_cfg, alice, eval_plain, eval_key, eval_cfg, device)
        metrics["modes"]["restarted"] = restarted_summary
        attack_logs.extend(restart_logs)

    if "known_plaintext" in eval_cfg.attacker_modes:
        known_plain, known_key = generate_dataset(anc_cfg, eval_cfg.known_plaintext_samples, eval_cfg.seed + 200, device=device)
        eve = copy.deepcopy(checkpoint_eve)
        known_logs = _train_attacker(
            alice,
            eve,
            known_plain,
            known_key,
            epochs=eval_cfg.known_plaintext_epochs,
            batch_size=eval_cfg.batch_size,
            learning_rate=eval_cfg.attacker_learning_rate,
            seed=eval_cfg.seed + 201,
        )
        for row in known_logs:
            attack_logs.append({"mode": "known_plaintext", **row})
        metrics["modes"]["known_plaintext"] = _evaluate_eve(eve, alice, eval_plain, eval_key)

    if "chosen_plaintext" in eval_cfg.attacker_modes:
        chosen_plain, chosen_key = _make_chosen_plaintext_queries(anc_cfg, eval_cfg.chosen_plaintext_queries, eval_cfg.seed + 300, device)
        eve = EveModel(
            plaintext_len=anc_cfg.plaintext_len,
            ciphertext_len=anc_cfg.ciphertext_len,
            width=anc_cfg.model_width,
            depth=anc_cfg.model_depth,
        ).to(device)
        chosen_logs = _train_attacker(
            alice,
            eve,
            chosen_plain,
            chosen_key,
            epochs=eval_cfg.chosen_plaintext_epochs,
            batch_size=eval_cfg.batch_size,
            learning_rate=eval_cfg.attacker_learning_rate,
            seed=eval_cfg.seed + 301,
        )
        for row in chosen_logs:
            attack_logs.append({"mode": "chosen_plaintext", **row})
        metrics["modes"]["chosen_plaintext"] = _evaluate_eve(eve, alice, eval_plain, eval_key)

    write_json(run_dir / "resolved_config.json", {**eval_cfg.to_dict(), "resolved_device": str(device)})
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "attacker_log.jsonl", attack_logs)

    if attack_logs:
        curves: dict[str, list[float]] = {}
        for mode in sorted({str(entry["mode"]) for entry in attack_logs}):
            mode_losses = [float(entry["eve_train_loss"]) for entry in attack_logs if entry["mode"] == mode]
            if mode_losses:
                curves[f"{mode}_loss"] = mode_losses
        if curves:
            save_line_plot(
                run_dir / "attacker_curves.png",
                curves,
                title="Attacker Adaptation Loss",
                xlabel="step",
                ylabel="loss",
            )

    summary_lines = [
        "# ANC Evaluation Summary",
        "",
        f"- Checkpoint: {resolved_checkpoint}",
        f"- Eval samples: {eval_cfg.eval_samples}",
    ]
    for mode_name, mode_metrics in metrics["modes"].items():
        if isinstance(mode_metrics, dict) and "eve_accuracy" in mode_metrics:
            summary_lines.append(f"- {mode_name} eve_accuracy: {mode_metrics['eve_accuracy']:.4f}")
        elif isinstance(mode_metrics, dict) and "best_eve_accuracy" in mode_metrics:
            summary_lines.append(f"- {mode_name} best_eve_accuracy: {mode_metrics['best_eve_accuracy']:.4f}")
            summary_lines.append(f"- {mode_name} mean_eve_accuracy: {mode_metrics['mean_eve_accuracy']:.4f}")
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {"run_dir": str(run_dir), "metrics": metrics}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained ANC checkpoints against stronger attacker modes")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or ANC run directory")
    parser.add_argument("--config", default="configs/anc_eval.yaml", help="Optional eval config path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override eval config values using dotted key=value syntax",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config) if args.config else None
    eval_cfg = load_eval_config(config_path, args.override)
    eval_cfg.checkpoint = args.checkpoint
    result = evaluate_checkpoint(args.checkpoint, eval_cfg)
    print(json.dumps({"run_dir": result["run_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
