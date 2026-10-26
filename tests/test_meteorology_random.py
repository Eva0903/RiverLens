"""Common-sample, split, temporal-input and resume checks; no GPU required."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jinjiang.data import PreparedDataset, compute_split_boundaries
from jinjiang.meteorology_random import (
    ARMS, FIXED_WINDOWS, RandomMeteorologyRunner, common_meteorology_pool,
    paired_seed_gains, validate_config,
)


def config():
    return dict(
        meteorology_protocol="meteorology_random_fixed_budget_v1",
        baseline_protocol="random_window_holdout_80_10_10_v1",
        frequency="4h", target_station="C", targets=["DO"],
        target_default_windows=FIXED_WINDOWS.copy(), forecast_horizon_steps=6,
        split=dict(train_fraction=.8, validation_fraction=.1), split_seed=42,
        missing_data=dict(interpolation_policy="causal_ffill"),
        training=dict(seeds=[42], max_epochs=512, patience=100),
        bootstrap=dict(block_length_steps=6, replicates=20, seed=2026),
        interpretability=dict(enabled=False), meteorology_model="lstm",
        data_dir="unused", results_dir="results",
    )


def fixture(missing=False):
    index = pd.date_range("2022-01-01", periods=360, freq="4h")
    i = np.arange(len(index), dtype=float)
    frame = pd.DataFrame({"A_DO": i, "B_DO": i + 1, "C_DO": i + 2,
                          "W_TEM": i + 3, "W_RAIN_LAG_4H": i + 4}, index=index)
    if missing:
        frame.loc[index[160], "W_TEM"] = np.nan
    return PreparedDataset(
        frame.copy(), frame.copy(), {s: [s + "_DO"] for s in "ABC"},
        {s: {"DO": s + "_DO"} for s in "ABC"},
        ["W_TEM", "W_RAIN_LAG_4H"], compute_split_boundaries(index, .8, .1), pd.DataFrame(),
    )


class CommonPoolTests(unittest.TestCase):
    def test_rejects_chronological_split_wrong_windows_and_oversized_budget(self):
        validate_config(config())
        for change in ["split", "windows", "budget"]:
            c = config()
            if change == "split": c["split"]["train_fraction"] = .7
            elif change == "windows": c["target_default_windows"]["DO"] = 30
            else: c["training"]["max_epochs"] = 2048
            with self.assertRaises(ValueError): validate_config(c)

    def test_all_arms_share_dates_labels_and_split_even_with_missing_weather(self):
        seq, split, arms = common_meteorology_pool(fixture(True), config(), "DO")
        self.assertEqual(set(arms), set(ARMS))
        for arm in arms.values():
            np.testing.assert_array_equal(arm.y, seq.y)
            np.testing.assert_array_equal(arm.target_start_times, seq.target_start_times)
            np.testing.assert_array_equal(arm.split_labels, seq.split_labels)
            self.assertTrue(np.isfinite(arm.x).all())
            self.assertEqual(arm.x.shape[1], 84)
        ids = np.concatenate(list(split.values()))
        np.testing.assert_array_equal(np.sort(ids), np.arange(len(seq.x)))
        _, again, _ = common_meteorology_pool(fixture(True), config(), "DO")
        for name in split: np.testing.assert_array_equal(split[name], again[name])
        self.assertLess(len(seq.x), len(common_meteorology_pool(fixture(), config(), "DO")[0].x))

    def test_every_input_precedes_target_and_weather_increments_are_real(self):
        data = fixture()
        seq, split, arms = common_meteorology_pool(data, config(), "DO")
        starts = data.raw_frame.index.get_indexer(seq.target_start_times)
        self.assertTrue((seq.issue_times < seq.target_start_times).all())
        np.testing.assert_array_equal(seq.x[:, -1, 0], starts - 1)
        self.assertEqual([len(arms[a].feature_columns) for a in ARMS], [3, 4, 4, 5])
        # Random assignment mixes dates within each partition, rather than slicing a calendar.
        self.assertLess(seq.target_start_times[split["test"]].min(), seq.target_start_times[split["train"]].max())

    def test_paired_gain_uses_matching_seed_and_lead(self):
        rows = [dict(target="DO", seed=42, horizon_hours=4, model="lstm",
                     feature_set=arm, MAE=1 if arm == "all_stations" else .8,
                     RMSE=1, R2=.5) for arm in ARMS]
        result = paired_seed_gains(pd.DataFrame(rows), "lstm")
        primary = result[result.role == "primary"]
        self.assertEqual(len(primary), 3)
        np.testing.assert_allclose(primary.MAE_reduction_percent, 20)


class ResumeTest(unittest.TestCase):
    def test_partial_run_resumes_only_unfinished_arms(self):
        calls = []
        def fit(runner, method, x, sequence, split, seed):
            calls.append(method.feature_set)
            history = [] if method.model in {"persistence", "seasonal_naive"} else [
                dict(epoch=1, train_loss=.3, validation_loss=.4),
                dict(epoch=2, train_loss=.2, validation_loss=.3),
            ]
            return sequence.y[split["test"]] + .01, history, 2 if history else 0, 1, None
        with tempfile.TemporaryDirectory() as temp:
            c = config(); c["_project_root"] = temp
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch("jinjiang.meteorology_random.prepare_random_pool", side_effect=lambda data, cfg: data), \
                 patch.object(RandomMeteorologyRunner, "_fit_method", fit), \
                 redirect_stdout(io.StringIO()):
                r = RandomMeteorologyRunner(deepcopy(c), max_tasks=3); output = r.run()
                self.assertEqual(len(calls), 3)
                r = RandomMeteorologyRunner(deepcopy(c)); r.run()
                self.assertEqual(len(calls), 6)
                status = json.loads((output / "run_status.json").read_text())
                self.assertEqual(status["run_status"], "complete")
                self.assertEqual(status["completed_tasks"], 6)
                metrics = pd.read_csv(output / "metrics.csv")
                self.assertEqual(len(metrics), 36)
                self.assertFalse(metrics.duplicated(["task_id", "horizon_hours"]).any())
                diag = pd.read_csv(output / "model_diagnostics.csv")
                self.assertEqual(len(diag), 4)
                self.assertTrue((diag.max_epochs == 512).all())


if __name__ == "__main__":
    unittest.main()
