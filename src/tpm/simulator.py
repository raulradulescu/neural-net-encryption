from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import random
import statistics
from time import perf_counter
from typing import Any

from src.utils.io import create_run_dir, write_json, write_yaml

try:
    import yaml
except Exception as exc:  # pragma: no cover - dependency is present in test env
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


def _require_yaml() -> None:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load TPM configs") from _YAML_IMPORT_ERROR


def _sign(value: int) -> int:
    return 1 if value >= 0 else -1


def _bounded(value: int, limit: int) -> int:
    return max(-limit, min(limit, value))


def _matrix_copy(matrix: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in matrix]


def _weights_equal(a: list[list[int]], b: list[list[int]]) -> bool:
    return a == b


def _weight_distance(a: list[list[int]], b: list[list[int]]) -> int:
    return sum(abs(x - y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


@dataclass(frozen=True)
class TpmConfig:
    k_hidden: int
    n_inputs: int
    weight_limit: int
    max_rounds: int
    trials: int
    seed: int = 0
    output_dir: str = "outputs/tpm"
    attacker_strategy: str = "naive"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TpmConfig":
        normalized = dict(mapping)
        normalized.setdefault("seed", 0)
        normalized.setdefault("output_dir", "outputs/tpm")
        normalized.setdefault("attacker_strategy", "naive")

        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in normalized:
                    return normalized[name]
            return default

        k_hidden = pick("k_hidden", "K")
        n_inputs = pick("n_inputs", "N")
        weight_limit = pick("weight_limit", "L")
        max_rounds = pick("max_rounds")
        trials = pick("trials")

        missing = [name for name, value in {
            "k_hidden/K": k_hidden,
            "n_inputs/N": n_inputs,
            "weight_limit/L": weight_limit,
            "max_rounds": max_rounds,
            "trials": trials,
        }.items() if value is None]
        if missing:
            raise ValueError(f"Missing TPM config values: {', '.join(missing)}")

        return cls(
            k_hidden=int(k_hidden),
            n_inputs=int(n_inputs),
            weight_limit=int(weight_limit),
            max_rounds=int(max_rounds),
            trials=int(trials),
            seed=int(normalized["seed"]),
            output_dir=str(normalized["output_dir"]),
            attacker_strategy=str(normalized["attacker_strategy"]),
        )

    def validate(self) -> None:
        if self.k_hidden <= 0:
            raise ValueError("k_hidden must be positive")
        if self.n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        if self.weight_limit <= 0:
            raise ValueError("weight_limit must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if self.attacker_strategy not in {"naive"}:
            raise ValueError(f"Unsupported attacker strategy: {self.attacker_strategy}")


@dataclass
class TpmState:
    k_hidden: int
    n_inputs: int
    weight_limit: int
    weights: list[list[int]]

    @classmethod
    def random(cls, rng: random.Random, k_hidden: int, n_inputs: int, weight_limit: int) -> "TpmState":
        weights = [
            [rng.randint(-weight_limit, weight_limit) for _ in range(n_inputs)]
            for _ in range(k_hidden)
        ]
        return cls(k_hidden=k_hidden, n_inputs=n_inputs, weight_limit=weight_limit, weights=weights)

    def output(self, inputs: list[list[int]]) -> tuple[list[int], int]:
        hidden_outputs: list[int] = []
        for i in range(self.k_hidden):
            activation = sum(w * x for w, x in zip(self.weights[i], inputs[i]))
            hidden_outputs.append(_sign(activation))
        tau = math.prod(hidden_outputs)
        return hidden_outputs, tau

    def update(self, inputs: list[list[int]], tau: int, hidden_outputs: list[int]) -> None:
        for i, sigma in enumerate(hidden_outputs):
            if sigma != tau:
                continue
            for j in range(self.n_inputs):
                self.weights[i][j] = _bounded(self.weights[i][j] + tau * inputs[i][j], self.weight_limit)


@dataclass
class TpmTrialResult:
    trial_index: int
    synchronized: bool
    rounds: int
    attacker_synchronized: bool
    attacker_sync_round: int | None
    attacker_distance: int
    final_distance: int


def _generate_inputs(rng: random.Random, k_hidden: int, n_inputs: int) -> list[list[int]]:
    return [[rng.choice((-1, 1)) for _ in range(n_inputs)] for _ in range(k_hidden)]


def run_single_trial(cfg: TpmConfig, rng: random.Random, trial_index: int) -> TpmTrialResult:
    alice = TpmState.random(rng, cfg.k_hidden, cfg.n_inputs, cfg.weight_limit)
    bob = TpmState.random(rng, cfg.k_hidden, cfg.n_inputs, cfg.weight_limit)
    observer = TpmState.random(rng, cfg.k_hidden, cfg.n_inputs, cfg.weight_limit)

    attacker_sync_round: int | None = None
    synchronized = False
    rounds = cfg.max_rounds

    for round_index in range(1, cfg.max_rounds + 1):
        inputs = _generate_inputs(rng, cfg.k_hidden, cfg.n_inputs)
        alice_hidden, alice_tau = alice.output(inputs)
        bob_hidden, bob_tau = bob.output(inputs)
        observer_hidden, observer_tau = observer.output(inputs)

        if alice_tau == bob_tau:
            alice.update(inputs, alice_tau, alice_hidden)
            bob.update(inputs, bob_tau, bob_hidden)
        if observer_tau == alice_tau:
            observer.update(inputs, alice_tau, observer_hidden)

        if _weights_equal(alice.weights, bob.weights):
            synchronized = True
            rounds = round_index
            if _weights_equal(observer.weights, alice.weights):
                attacker_sync_round = round_index
            break

        if attacker_sync_round is None and _weights_equal(observer.weights, alice.weights):
            attacker_sync_round = round_index

    final_distance = _weight_distance(alice.weights, bob.weights)
    attacker_distance = _weight_distance(observer.weights, alice.weights)
    return TpmTrialResult(
        trial_index=trial_index,
        synchronized=synchronized,
        rounds=rounds,
        attacker_synchronized=attacker_sync_round is not None,
        attacker_sync_round=attacker_sync_round,
        attacker_distance=attacker_distance,
        final_distance=final_distance,
    )


def _write_summary(path: Path, cfg: TpmConfig, results: list[TpmTrialResult], metrics: dict[str, Any]) -> None:
    success_rate = metrics["success_rate"]
    attacker_rate = metrics["attacker_success_rate"]
    avg_rounds = metrics["average_rounds"]
    lines = [
        "Tree Parity Machine Simulation Summary",
        f"Trials: {cfg.trials}",
        f"K/N/L: {cfg.k_hidden}/{cfg.n_inputs}/{cfg.weight_limit}",
        f"Max rounds: {cfg.max_rounds}",
        f"Synchronization success rate: {success_rate:.3f}",
        f"Average synchronization rounds: {avg_rounds:.2f}",
        f"Observer success rate: {attacker_rate:.3f}",
        f"Median synchronization rounds: {metrics['median_rounds']:.2f}",
        f"Final maximum distance: {max(item.final_distance for item in results)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rounds_csv(path: Path, results: list[TpmTrialResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_index", "synchronized", "rounds", "attacker_synchronized", "attacker_sync_round", "attacker_distance", "final_distance"])
        for result in results:
            writer.writerow([
                result.trial_index,
                int(result.synchronized),
                result.rounds,
                int(result.attacker_synchronized),
                "" if result.attacker_sync_round is None else result.attacker_sync_round,
                result.attacker_distance,
                result.final_distance,
            ])


def _write_histogram(path: Path, results: list[TpmTrialResult]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - fallback when plotting unavailable
        path.with_suffix(".csv").write_text(
            "\n".join(f"{result.trial_index},{result.rounds}" for result in results) + "\n",
            encoding="utf-8",
        )
        return

    rounds = [result.rounds for result in results]
    plt.figure(figsize=(7, 4))
    plt.hist(rounds, bins=min(10, max(1, len(set(rounds)))))
    plt.title("TPM Synchronization Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_tpm_from_config(config_path: str | Path) -> dict[str, Any]:
    _require_yaml()
    config_path = Path(config_path)
    mapping = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("TPM config must contain a mapping")

    cfg = TpmConfig.from_mapping(mapping)
    cfg.validate()

    run_dir = create_run_dir(Path(cfg.output_dir), prefix="tpm")
    resolved = asdict(cfg)
    write_yaml(run_dir / "resolved_config.yaml", resolved)

    rng = random.Random(cfg.seed)
    trials: list[TpmTrialResult] = []
    started = perf_counter()
    for trial_index in range(cfg.trials):
        trials.append(run_single_trial(cfg, rng, trial_index))
    elapsed = perf_counter() - started

    successful = [trial for trial in trials if trial.synchronized]
    attacker_successful = [trial for trial in trials if trial.attacker_synchronized]
    rounds = [trial.rounds for trial in successful] or [cfg.max_rounds]
    attacker_rounds = [trial.attacker_sync_round for trial in attacker_successful if trial.attacker_sync_round is not None]
    metrics = {
        "seed": cfg.seed,
        "trials": cfg.trials,
        "success_count": len(successful),
        "success_rate": len(successful) / cfg.trials,
        "attacker_success_count": len(attacker_successful),
        "attacker_success_rate": len(attacker_successful) / cfg.trials,
        "average_rounds": statistics.fmean(rounds),
        "median_rounds": statistics.median(rounds),
        "min_rounds": min(rounds),
        "max_rounds": max(rounds),
        "average_attacker_distance": statistics.fmean(trial.attacker_distance for trial in trials),
        "average_final_distance": statistics.fmean(trial.final_distance for trial in trials),
        "average_attacker_sync_round": statistics.fmean(attacker_rounds) if attacker_rounds else None,
        "runtime_seconds": elapsed,
        "trials_detail": [asdict(trial) for trial in trials],
        "artifacts": {
            "resolved_config": "resolved_config.yaml",
            "metrics": "metrics.json",
            "trials": "trials.jsonl",
            "rounds_csv": "rounds.csv",
            "summary": "summary.txt",
            "histogram": "rounds_histogram.png",
        },
    }

    with (run_dir / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(json.dumps(asdict(trial), sort_keys=True) + "\n")
    write_json(run_dir / "metrics.json", metrics)
    _write_summary(run_dir / "summary.txt", cfg, trials, metrics)
    _write_rounds_csv(run_dir / "rounds.csv", trials)
    _write_histogram(run_dir / "rounds_histogram.png", trials)

    return {"run_dir": str(run_dir), "metrics": metrics}

