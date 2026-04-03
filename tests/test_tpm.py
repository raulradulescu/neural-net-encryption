from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

import yaml

from src.tpm.simulator import TpmState, run_tpm_from_config


class TpmTests(unittest.TestCase):
    def test_tpm_weights_initialize_in_range(self) -> None:
        rng = random.Random(123)
        state = TpmState.random(rng, k_hidden=3, n_inputs=4, weight_limit=2)
        for row in state.weights:
            for weight in row:
                self.assertGreaterEqual(weight, -2)
                self.assertLessEqual(weight, 2)

    def test_tpm_trial_and_aggregate_artifacts(self) -> None:
        config = {
            "k_hidden": 3,
            "n_inputs": 4,
            "weight_limit": 3,
            "max_rounds": 300,
            "trials": 5,
            "seed": 42,
            "output_dir": str(self.tmp_path / "tpm-out"),
        }

        config_path = self.tmp_path / "tpm.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        result = run_tpm_from_config(config_path)
        run_dir = Path(result["run_dir"])

        self.assertTrue(run_dir.exists())
        self.assertTrue((run_dir / "resolved_config.yaml").exists())
        self.assertTrue((run_dir / "metrics.json").exists())
        self.assertTrue((run_dir / "trials.jsonl").exists())
        self.assertTrue((run_dir / "summary.txt").exists())
        self.assertTrue((run_dir / "rounds.csv").exists())
        self.assertTrue((run_dir / "rounds_histogram.png").exists() or (run_dir / "rounds_histogram.csv").exists())

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["trials"], 5)
        self.assertGreaterEqual(metrics["success_rate"], 0.0)
        self.assertLessEqual(metrics["success_rate"], 1.0)
        self.assertIn("average_rounds", metrics)
        self.assertEqual(len(metrics["trials_detail"]), 5)

    def test_tpm_cli_entrypoint(self) -> None:
        config_dir = Path(self.tmp_path)
        config = {
            "K": 3,
            "N": 4,
            "L": 3,
            "max_rounds": 100,
            "trials": 2,
            "seed": 7,
            "output_dir": str(config_dir / "tpm-cli"),
        }
        config_path = config_dir / "tpm-cli.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        from src.tpm.run_tpm import main

        self.assertEqual(main(["--config", str(config_path)]), 0)

    def setUp(self) -> None:
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
