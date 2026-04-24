from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
import yaml

from src.utils.io import create_run_dir, write_json, write_yaml


def _coerce_bytes(value: bytes | bytearray | memoryview | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("Expected bytes-like data or string")


def _validate_nonce_and_key(algorithm: str, key: bytes, nonce: bytes) -> None:
    if algorithm == "aesgcm":
        if len(key) not in {16, 24, 32}:
            raise ValueError("AES-GCM key must be 16, 24, or 32 bytes")
        if len(nonce) != 12:
            raise ValueError("AES-GCM nonce must be 12 bytes")
    elif algorithm == "chacha20poly1305":
        if len(key) != 32:
            raise ValueError("ChaCha20-Poly1305 key must be 32 bytes")
        if len(nonce) != 12:
            raise ValueError("ChaCha20-Poly1305 nonce must be 12 bytes")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def encrypt_aesgcm(key: bytes, nonce: bytes, plaintext: bytes, associated_data: bytes | None = None) -> bytes:
    plaintext = _coerce_bytes(plaintext)
    ad = _coerce_bytes(associated_data)
    _validate_nonce_and_key("aesgcm", key, nonce)
    return AESGCM(key).encrypt(nonce, plaintext, ad)


def decrypt_aesgcm(key: bytes, nonce: bytes, ciphertext: bytes, associated_data: bytes | None = None) -> bytes:
    ciphertext = _coerce_bytes(ciphertext)
    ad = _coerce_bytes(associated_data)
    _validate_nonce_and_key("aesgcm", key, nonce)
    return AESGCM(key).decrypt(nonce, ciphertext, ad)


def encrypt_chacha20poly1305(key: bytes, nonce: bytes, plaintext: bytes, associated_data: bytes | None = None) -> bytes:
    plaintext = _coerce_bytes(plaintext)
    ad = _coerce_bytes(associated_data)
    _validate_nonce_and_key("chacha20poly1305", key, nonce)
    return ChaCha20Poly1305(key).encrypt(nonce, plaintext, ad)


def decrypt_chacha20poly1305(key: bytes, nonce: bytes, ciphertext: bytes, associated_data: bytes | None = None) -> bytes:
    ciphertext = _coerce_bytes(ciphertext)
    ad = _coerce_bytes(associated_data)
    _validate_nonce_and_key("chacha20poly1305", key, nonce)
    return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, ad)


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: str = "outputs/baseline"
    seed: int = 0
    iterations: int = 100
    payload_sizes: list[int] | None = None
    algorithms: list[str] | None = None
    associated_data: str = "baseline-aad"
    neural_compare: bool = True
    neural_checkpoint: str | None = None
    neural_plaintext_len: int = 32
    neural_key_len: int = 32
    neural_model_width: int = 64
    neural_model_depth: int = 2
    neural_batch_size: int = 256
    neural_iterations: int = 100

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "BenchmarkConfig":
        payload_sizes = mapping.get("payload_sizes") or mapping.get("plaintext_sizes") or [32, 1024, 4096]
        algorithms = mapping.get("algorithms") or ["aesgcm", "chacha20poly1305"]
        return cls(
            output_dir=str(mapping.get("output_dir", "outputs/baseline")),
            seed=int(mapping.get("seed", 0)),
            iterations=int(mapping.get("iterations", 100)),
            payload_sizes=[int(size) for size in payload_sizes],
            algorithms=[str(name).lower() for name in algorithms],
            associated_data=str(mapping.get("associated_data", "baseline-aad")),
            neural_compare=bool(mapping.get("neural_compare", True)),
            neural_checkpoint=None if mapping.get("neural_checkpoint") in {None, "", "null"} else str(mapping.get("neural_checkpoint")),
            neural_plaintext_len=int(mapping.get("neural_plaintext_len", 32)),
            neural_key_len=int(mapping.get("neural_key_len", 32)),
            neural_model_width=int(mapping.get("neural_model_width", 64)),
            neural_model_depth=int(mapping.get("neural_model_depth", 2)),
            neural_batch_size=int(mapping.get("neural_batch_size", 256)),
            neural_iterations=int(mapping.get("neural_iterations", 100)),
        )

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not self.payload_sizes:
            raise ValueError("payload_sizes must not be empty")
        if not self.algorithms:
            raise ValueError("algorithms must not be empty")
        if any(size <= 0 for size in self.payload_sizes):
            raise ValueError("payload_sizes must contain only positive integers")
        unsupported = [name for name in self.algorithms if name not in {"aesgcm", "chacha20poly1305"}]
        if unsupported:
            raise ValueError(f"Unsupported algorithms: {', '.join(unsupported)}")

        if self.neural_compare:
            if self.neural_plaintext_len <= 0 or self.neural_key_len <= 0:
                raise ValueError("neural plaintext/key lengths must be positive")
            if self.neural_model_width <= 0 or self.neural_model_depth <= 0:
                raise ValueError("neural model width/depth must be positive")
            if self.neural_batch_size <= 0 or self.neural_iterations <= 0:
                raise ValueError("neural batch size/iterations must be positive")


@dataclass
class SizeBenchmarkResult:
    algorithm: str
    payload_size: int
    iterations: int
    round_trip_ok: bool
    encrypt_seconds: float
    decrypt_seconds: float
    round_trip_seconds: float
    encrypt_throughput_bps: float
    round_trip_throughput_bps: float


def _derive_bytes(seed: int, label: str, size: int) -> bytes:
    digest = hashlib.blake2b(f"{seed}:{label}".encode("utf-8"), digest_size=32).digest()
    rng = random.Random(digest)
    return rng.randbytes(size)


def _algorithm_runner(name: str):
    if name == "aesgcm":
        return encrypt_aesgcm, decrypt_aesgcm, 16, 12
    if name == "chacha20poly1305":
        return encrypt_chacha20poly1305, decrypt_chacha20poly1305, 32, 12
    raise ValueError(f"Unsupported algorithm: {name}")


def benchmark_algorithm(algorithm: str, payload_size: int, iterations: int, seed: int, associated_data: bytes) -> SizeBenchmarkResult:
    encrypt_fn, decrypt_fn, key_size, nonce_size = _algorithm_runner(algorithm)
    key = _derive_bytes(seed, f"{algorithm}:key", key_size)
    plaintexts = [_derive_bytes(seed, f"{algorithm}:plaintext:{index}", payload_size) for index in range(iterations)]
    nonces = [index.to_bytes(nonce_size, "big") for index in range(iterations)]

    start = perf_counter()
    ciphertexts = [encrypt_fn(key, nonce, plaintext, associated_data) for nonce, plaintext in zip(nonces, plaintexts)]
    encrypt_seconds = perf_counter() - start

    start = perf_counter()
    decrypted = [decrypt_fn(key, nonce, ciphertext, associated_data) for nonce, ciphertext in zip(nonces, ciphertexts)]
    decrypt_seconds = perf_counter() - start

    round_trip_seconds = encrypt_seconds + decrypt_seconds
    round_trip_ok = decrypted == plaintexts
    total_bytes = payload_size * iterations
    encrypt_throughput_bps = total_bytes / encrypt_seconds if encrypt_seconds > 0 else float("inf")
    round_trip_throughput_bps = total_bytes / round_trip_seconds if round_trip_seconds > 0 else float("inf")

    return SizeBenchmarkResult(
        algorithm=algorithm,
        payload_size=payload_size,
        iterations=iterations,
        round_trip_ok=round_trip_ok,
        encrypt_seconds=encrypt_seconds,
        decrypt_seconds=decrypt_seconds,
        round_trip_seconds=round_trip_seconds,
        encrypt_throughput_bps=encrypt_throughput_bps,
        round_trip_throughput_bps=round_trip_throughput_bps,
    )


def _benchmark_neural_path(cfg: BenchmarkConfig) -> dict[str, Any]:
    import torch

    from src.models import AliceModel, BobModel
    from src.training.train_anc import load_checkpoint

    device = torch.device("cpu")
    if cfg.neural_checkpoint:
        loaded = load_checkpoint(cfg.neural_checkpoint, map_location=device)
        anc_cfg = loaded["config"]
        alice = loaded["models"]["alice"].to(device)
        bob = loaded["models"]["bob"].to(device)
        plaintext_len = int(anc_cfg.plaintext_len)
        key_len = int(anc_cfg.key_len)
    else:
        plaintext_len = cfg.neural_plaintext_len
        key_len = cfg.neural_key_len
        alice = AliceModel(
            plaintext_len=plaintext_len,
            key_len=key_len,
            ciphertext_len=plaintext_len,
            width=cfg.neural_model_width,
            depth=cfg.neural_model_depth,
        ).to(device)
        bob = BobModel(
            plaintext_len=plaintext_len,
            key_len=key_len,
            ciphertext_len=plaintext_len,
            width=cfg.neural_model_width,
            depth=cfg.neural_model_depth,
        ).to(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed)

    total_bits = cfg.neural_batch_size * cfg.neural_iterations * plaintext_len
    correct_bits = 0.0

    alice.eval()
    bob.eval()

    encrypt_seconds = 0.0
    decrypt_seconds = 0.0
    with torch.no_grad():
        for _ in range(cfg.neural_iterations):
            plaintext = torch.randint(0, 2, (cfg.neural_batch_size, plaintext_len), generator=generator, dtype=torch.float32, device=device)
            key = torch.randint(0, 2, (cfg.neural_batch_size, key_len), generator=generator, dtype=torch.float32, device=device)

            start = perf_counter()
            ciphertext = torch.sigmoid(alice(plaintext, key))
            encrypt_seconds += perf_counter() - start

            start = perf_counter()
            bob_logits = bob(ciphertext, key)
            decrypt_seconds += perf_counter() - start

            pred = (torch.sigmoid(bob_logits) >= 0.5).to(dtype=torch.float32)
            correct_bits += float((pred == plaintext).to(dtype=torch.float32).sum().item())

    round_trip = encrypt_seconds + decrypt_seconds
    throughput_bytes_per_sec = (total_bits / 8.0) / round_trip if round_trip > 0 else float("inf")

    return {
        "enabled": True,
        "source": "checkpoint" if cfg.neural_checkpoint else "random_init",
        "checkpoint": cfg.neural_checkpoint,
        "plaintext_len": plaintext_len,
        "key_len": key_len,
        "batch_size": cfg.neural_batch_size,
        "iterations": cfg.neural_iterations,
        "encrypt_seconds": encrypt_seconds,
        "decrypt_seconds": decrypt_seconds,
        "round_trip_seconds": round_trip,
        "throughput_bytes_per_sec": throughput_bytes_per_sec,
        "reconstruction_accuracy": correct_bits / float(total_bits),
    }


def _write_results_csv(path: Path, results: list[SizeBenchmarkResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "algorithm",
                "payload_size",
                "iterations",
                "round_trip_ok",
                "encrypt_seconds",
                "decrypt_seconds",
                "round_trip_seconds",
                "encrypt_throughput_bps",
                "round_trip_throughput_bps",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.algorithm,
                    result.payload_size,
                    result.iterations,
                    int(result.round_trip_ok),
                    f"{result.encrypt_seconds:.9f}",
                    f"{result.decrypt_seconds:.9f}",
                    f"{result.round_trip_seconds:.9f}",
                    f"{result.encrypt_throughput_bps:.3f}",
                    f"{result.round_trip_throughput_bps:.3f}",
                ]
            )


def _write_chart(path: Path, results: list[SizeBenchmarkResult], neural_metrics: dict[str, Any] | None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        path.with_suffix(".txt").write_text(
            "\n".join(f"{r.algorithm},{r.payload_size},{r.round_trip_throughput_bps:.3f}" for r in results) + "\n",
            encoding="utf-8",
        )
        return

    labels = [f"{result.algorithm}\n{result.payload_size}" for result in results]
    values = [result.round_trip_throughput_bps / 1_000_000 for result in results]

    if neural_metrics is not None:
        labels.append("neural\npath")
        values.append(float(neural_metrics["throughput_bytes_per_sec"]) / 1_000_000)

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Throughput (MB/s)")
    plt.title("AEAD vs Neural Round-Trip Throughput")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_summary(path: Path, cfg: BenchmarkConfig, results: list[SizeBenchmarkResult], elapsed: float, neural_metrics: dict[str, Any] | None) -> None:
    lines = [
        "Baseline Benchmark Summary",
        f"Algorithms: {', '.join(cfg.algorithms)}",
        f"Payload sizes: {', '.join(str(size) for size in cfg.payload_sizes)}",
        f"Iterations: {cfg.iterations}",
        f"Runtime seconds: {elapsed:.6f}",
    ]
    for result in results:
        lines.append(
            f"{result.algorithm} payload={result.payload_size} round_trip_ok={result.round_trip_ok} throughput_bps={result.round_trip_throughput_bps:.3f}"
        )
    if neural_metrics is not None:
        lines.append("Neural path comparison:")
        lines.append(
            f"source={neural_metrics['source']} throughput_bytes_per_sec={neural_metrics['throughput_bytes_per_sec']:.3f} accuracy={neural_metrics['reconstruction_accuracy']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_from_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    mapping = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Benchmark config must contain a mapping")

    cfg = BenchmarkConfig.from_mapping(mapping)
    cfg.validate()

    run_dir = create_run_dir(Path(cfg.output_dir), prefix="baseline")
    write_yaml(run_dir / "resolved_config.yaml", cfg.__dict__)

    associated_data = _coerce_bytes(cfg.associated_data)
    started = perf_counter()
    results: list[SizeBenchmarkResult] = []
    for algorithm in cfg.algorithms:
        for payload_size in cfg.payload_sizes:
            results.append(benchmark_algorithm(algorithm, payload_size, cfg.iterations, cfg.seed, associated_data))

    neural_metrics = _benchmark_neural_path(cfg) if cfg.neural_compare else None
    elapsed = perf_counter() - started

    metrics = {
        "seed": cfg.seed,
        "iterations": cfg.iterations,
        "results": [result.__dict__ for result in results],
        "correctness_rate": sum(1 for result in results if result.round_trip_ok) / len(results),
        "average_round_trip_throughput_bps": sum(result.round_trip_throughput_bps for result in results) / len(results),
        "runtime_seconds": elapsed,
        "neural_path": neural_metrics,
        "artifacts": {
            "resolved_config": "resolved_config.yaml",
            "metrics": "metrics.json",
            "results_csv": "results.csv",
            "summary": "summary.txt",
            "throughput_chart": "throughput.png",
        },
    }
    write_json(run_dir / "metrics.json", metrics)
    _write_results_csv(run_dir / "results.csv", results)
    _write_summary(run_dir / "summary.txt", cfg, results, elapsed, neural_metrics)
    _write_chart(run_dir / "throughput.png", results, neural_metrics)

    return {"run_dir": str(run_dir), "metrics": metrics}
