from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.baseline.aead import (
    decrypt_aesgcm,
    decrypt_chacha20poly1305,
    encrypt_aesgcm,
    encrypt_chacha20poly1305,
    run_benchmark_from_config,
)


class BaselineCryptoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_aesgcm_round_trip(self) -> None:
        key = b"0" * 32
        nonce = b"1" * 12
        plaintext = b"hello baseline"
        ciphertext = encrypt_aesgcm(key, nonce, plaintext, b"aad")
        self.assertEqual(decrypt_aesgcm(key, nonce, ciphertext, b"aad"), plaintext)

    def test_chacha20poly1305_round_trip(self) -> None:
        key = b"2" * 32
        nonce = b"3" * 12
        plaintext = b"hello chacha"
        ciphertext = encrypt_chacha20poly1305(key, nonce, plaintext, b"aad")
        self.assertEqual(decrypt_chacha20poly1305(key, nonce, ciphertext, b"aad"), plaintext)

    def test_invalid_nonce_or_key_size_raises_controlled_error(self) -> None:
        cases = [
            (encrypt_aesgcm, b"short", b"1" * 12, "AES-GCM key must be 16, 24, or 32 bytes"),
            (encrypt_aesgcm, b"0" * 32, b"short", "AES-GCM nonce must be 12 bytes"),
            (encrypt_chacha20poly1305, b"short", b"1" * 12, "ChaCha20-Poly1305 key must be 32 bytes"),
            (encrypt_chacha20poly1305, b"0" * 32, b"short", "ChaCha20-Poly1305 nonce must be 12 bytes"),
        ]
        for func, key, nonce, expected in cases:
            with self.subTest(func=func.__name__, expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    func(key, nonce, b"payload", b"aad")

    def test_baseline_benchmark_outputs(self) -> None:
        config = {
            "output_dir": str(self.tmp_path / "baseline-out"),
            "seed": 123,
            "iterations": 32,
            "payload_sizes": [16, 128],
            "algorithms": ["aesgcm", "chacha20poly1305"],
            "associated_data": "aad",
            "neural_compare": False,
        }
        config_path = self.tmp_path / "baseline.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        result = run_benchmark_from_config(config_path)
        run_dir = Path(result["run_dir"])

        self.assertTrue(run_dir.exists())
        self.assertTrue((run_dir / "resolved_config.yaml").exists())
        self.assertTrue((run_dir / "metrics.json").exists())
        self.assertTrue((run_dir / "results.csv").exists())
        self.assertTrue((run_dir / "summary.txt").exists())
        self.assertTrue((run_dir / "throughput.png").exists() or (run_dir / "throughput.txt").exists())

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["iterations"], 32)
        self.assertEqual(metrics["correctness_rate"], 1.0)
        self.assertEqual(len(metrics["results"]), 4)
        self.assertIn("neural_path", metrics)
        self.assertIsNone(metrics["neural_path"])

    def test_baseline_cli_entrypoint(self) -> None:
        config = {
            "output_dir": str(self.tmp_path / "baseline-cli"),
            "iterations": 4,
            "payload_sizes": [32],
            "algorithms": ["aesgcm"],
            "neural_compare": False,
        }
        config_path = self.tmp_path / "baseline-cli.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        from src.baseline.benchmark import main

        self.assertEqual(main(["--config", str(config_path)]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
