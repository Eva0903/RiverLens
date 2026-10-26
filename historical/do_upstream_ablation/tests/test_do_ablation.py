from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jinjiang.config import load_config
from jinjiang.data import PreparedDataset, make_sequences
from jinjiang.do_ablation import (ARMS, LOCKED_NAIVES, _time_block_ci,
                                 build_common_cohort, run_do_ablation)
from jinjiang.models import PredictionResult


def synthetic_data():
    times = pd.date_range("2022-01-01", periods=150, freq="4h")
    raw = pd.DataFrame({"A_DO": np.arange(150.) + 1, "B_DO": np.arange(150.) + 2,
                        "C_DO": np.arange(150.) + 3}, index=times)
    # An unavailable upstream input must remove a row from BOTH experiment arms.
    raw.loc[times[30], "A_DO"] = np.nan
    return PreparedDataset(raw, raw.copy(), {s: [f"{s}_DO"] for s in "ABC"},
        {s: {"DO": f"{s}_DO"} for s in "ABC"}, [],
        {"train_end": times[79], "validation_end": times[114], "test_end": times[-1]},
        pd.DataFrame({"source": ["synthetic"]}))


class FixedDOAblationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/do_upstream_ablation.json")
        self.data = synthetic_data()

    def test_missing_upstream_filters_both_arms_and_preserves_shared_c_features(self):
        with patch("jinjiang.do_ablation.prepare_dataset", return_value=self.data):
            _, seq, split, inputs, columns = build_common_cohort(self.config)
        local_only = make_sequences(self.data, ["C_DO"], "C_DO", 6, 6)
        self.assertLess(len(seq.y), len(local_only.y))
        self.assertEqual(inputs[ARMS[0]].shape, (len(seq.y), 6, 6))
        self.assertEqual(inputs[ARMS[1]].shape, (len(seq.y), 6, 10))
        self.assertEqual(columns[ARMS[0]], ["C_DO"])
        # The three-station arm is ordered A,B,C, independently of the union's C,A,B order.
        np.testing.assert_array_equal(inputs[ARMS[0]][:, :, 0], inputs[ARMS[1]][:, :, 2])
        np.testing.assert_array_equal(inputs[ARMS[0]][:, :, 1], inputs[ARMS[1]][:, :, 5])
        np.testing.assert_array_equal(inputs[ARMS[0]][:, :, 2:], inputs[ARMS[1]][:, :, 6:])
        for early, later in (("train", "validation"), ("validation", "test")):
            self.assertLess(seq.target_end_times[split[early]].max(), seq.target_start_times[split[later]].min())

    def test_partial_run_resume_locked_baselines_and_config_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            config["_project_root"] = tmp
            config["data_dir"] = str(Path(self.config["_project_root"]) / "data/raw")
            config["training"]["seeds"] = [42, 7]
            config["bootstrap"]["replicates"] = 20
            seen = []

            def fake_fit(x, y, persistence, seasonal_naive, split, seed, training,
                         refit_train_validation, *, locked_baseline_names):
                # Integration seam only: these tests DO NOT claim to train PyTorch.
                seen.append((seed, x.shape[-1], locked_baseline_names, split))
                base = np.column_stack([(persistence if name == "persistence" else seasonal_naive)[:, h]
                                        for h, name in enumerate(locked_baseline_names)])
                prediction = base[split["test"]] + (0.1 if x.shape[-1] == 6 else 0.2)
                return PredictionResult(prediction, [{"stage": 0., "epoch": 1., "train_loss": 1.}],
                                         1, 100, diagnostics={"baseline_names": locked_baseline_names})

            with patch("jinjiang.do_ablation.prepare_dataset", return_value=self.data), \
                 patch("jinjiang.do_ablation.fit_residual_lstm_predict", side_effect=fake_fit):
                output = run_do_ablation(config, audit_only=True)
                self.assertEqual(len(seen), 0)
                run_do_ablation(config, seeds=[42])
                self.assertEqual(len(seen), 2)
                self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "partial")
                run_do_ablation(config)
                self.assertEqual(len(seen), 4)  # seed 42 was NOT retrained
                self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "complete")
                run_do_ablation(config)
                self.assertEqual(len(seen), 4)
                for _, _, names, _ in seen:
                    self.assertEqual(names, LOCKED_NAIVES)
                metrics = pd.read_csv(output / "metrics.csv")
                self.assertEqual(len(metrics), 2 * 5 * 6)  # two arms + three shared naive references
                paired = pd.read_csv(output / "paired_seed_horizon.csv")
                self.assertEqual(len(paired), 2 * 6)
                self.assertTrue((paired.MAE_reduction > 0).all())
                config["training"]["max_epochs"] += 1
                with self.assertRaisesRegex(RuntimeError, "changed config"):
                    run_do_ablation(config)

    def test_protect_old_results(self):
        self.config["results_dir"] = "results/lstm_upgrade_paper"
        with self.assertRaisesRegex(ValueError, "results_dir"):
            run_do_ablation(self.config, audit_only=True)

    def test_invalid_seed_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "subset"):
            run_do_ablation(self.config, seeds=[12345], audit_only=True)

    def test_time_block_ci_retains_actual_time_gaps(self):
        times = pd.date_range("2022-01-01", periods=60, freq="4h").delete(range(20, 35))
        values = pd.Series(np.ones(len(times)), index=times)
        estimate, low, high = _time_block_ci(values, {"block_length_steps": 6, "replicates": 30}, np.random.default_rng(42))
        self.assertEqual((estimate, low, high), (1., 1., 1.))


if __name__ == "__main__":
    unittest.main()
