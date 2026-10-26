from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jinjiang.baseline_random import legacy_random_split, prepare_random_pool
from jinjiang.config import load_config
from jinjiang.data import PreparedDataset, compute_split_boundaries, make_sequences
from jinjiang.models import PredictionResult, _scale_x
from jinjiang.station_ablation_random import (
    ARMS, RandomStationRunner, calendar_block_ci, common_station_pool,
    paired_calendar_gains, paired_seed_gains, validate_config,
)


def fixture():
    index = pd.date_range("2022-01-01", periods=240, freq="4h")
    columns = ["A_DO", "A_WT", "B_DO", "B_WT", "B_FLOW", "C_DO", "C_WT", "C_TN"]
    frame = pd.DataFrame({c: np.arange(240) + 1. + i / 10 for i, c in enumerate(columns)}, index=index)
    return PreparedDataset(frame.copy(), frame.copy(),
        {s: [c for c in columns if c.startswith(s + "_")] for s in "ABC"},
        {s: {"DO": s + "_DO"} for s in "ABC"}, [],
        compute_split_boundaries(index, .7, .15), pd.DataFrame())


class StationRandomTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/station_ablation_random_gpu.json")
        self.data = prepare_random_pool(fixture(), self.config)
        self.sequence, self.split, self.arms = common_station_pool(self.data, self.config, "DO")

    def test_matches_01_full_pool_and_ordered_split_indices(self):
        expected = make_sequences(self.data, self.sequence.feature_columns, "C_DO", 18, 6)
        np.testing.assert_array_equal(self.sequence.x, expected.x)
        np.testing.assert_array_equal(self.sequence.y, expected.y)
        for name, ids in legacy_random_split(len(expected.x)).items():
            np.testing.assert_array_equal(self.split[name], ids)
            self.assertTrue((self.sequence.split_labels[ids] == name).all())

    def test_all_arms_share_labels_times_and_naive_predictions(self):
        self.assertEqual(tuple(self.arms), ARMS)
        for arm in self.arms.values():
            self.assertIs(arm.y, self.sequence.y)
            self.assertIs(arm.target_start_times, self.sequence.target_start_times)
            self.assertIs(arm.persistence, self.sequence.persistence)
            self.assertIs(arm.seasonal_naive, self.sequence.seasonal_naive)
            selected = [self.sequence.feature_columns.index(c) for c in arm.feature_columns]
            np.testing.assert_array_equal(arm.x, self.sequence.x[:, :, selected])

    def test_upstream_only_inputs_do_not_contain_C(self):
        for name in ["station_A", "station_B", "upstream_AB"]:
            self.assertFalse(any(c.startswith("C_") for c in self.arms[name].feature_columns))
        self.assertEqual(self.arms["local_target"].feature_columns, ["C_DO"])
        self.assertEqual(self.arms["local_multivariate"].feature_columns, ["C_DO", "C_WT", "C_TN"])

    def test_union_missingness_filters_every_arm_together(self):
        modified = replace(self.data, feature_frame=self.data.feature_frame.copy())
        modified.feature_frame.iloc[50:55, 0] = np.nan
        seq, split, arms = common_station_pool(modified, self.config, "DO")
        self.assertLess(len(seq.x), len(self.sequence.x))
        self.assertEqual(len(arms["local_target"].x), len(seq.x))
        self.assertTrue(all(np.isfinite(a.x).all() for a in arms.values()))
        self.assertEqual(sum(map(len, split.values())), len(seq.x))

    def test_imputed_targets_are_not_used_as_observed_labels(self):
        modified = replace(self.data, raw_frame=self.data.raw_frame.copy())
        time = modified.raw_frame.index[80]
        modified.raw_frame.loc[time, "C_DO"] = np.nan
        seq, _, _ = common_station_pool(modified, self.config, "DO")
        self.assertFalse(((seq.target_start_times <= time) & (seq.target_end_times >= time)).any())

    def test_scaling_is_fitted_only_on_common_training_windows(self):
        for arm in self.arms.values():
            modified = arm.x.copy()
            modified[self.split["validation"]] += 100000
            modified[self.split["test"]] += 200000
            _, before = _scale_x(arm.x, self.split["train"])
            _, after = _scale_x(modified, self.split["train"])
            np.testing.assert_array_equal(before.mean_, after.mean_)
            np.testing.assert_array_equal(before.scale_, after.scale_)

    def test_old_config_and_wrong_window_are_rejected(self):
        for key, value in [("station_ablation_protocol", "old"), ("baseline_lookback_steps", 30), ("split_seed", 7)]:
            config = json.loads(json.dumps(self.config));config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_calendar_blocks_preserve_large_gaps(self):
        times = pd.date_range("2022-01-01", periods=12, freq="4h").append(pd.date_range("2022-01-21", periods=12, freq="4h"))
        result = calendar_block_ci(pd.Series(.5, index=times), 6, 100, 2026)
        self.assertEqual(result["n_calendar_blocks"], 22)
        self.assertEqual(result["n_nonempty_calendar_blocks"], 4)
        self.assertAlmostEqual(result["MAE_reduction"], .5)
        self.assertAlmostEqual(result["ci95_lower"], .5)
        self.assertAlmostEqual(result["ci95_upper"], .5)

    def test_seed_gain_direction_and_percent(self):
        metrics = pd.DataFrame([
            dict(target="DO", seed=42, horizon_hours=4, feature_set="local_multivariate", model="lstm", MAE=2., RMSE=3., R2=.5),
            dict(target="DO", seed=42, horizon_hours=4, feature_set="all_stations", model="lstm", MAE=1., RMSE=2., R2=.8),
        ])
        row = paired_seed_gains(metrics, "lstm").iloc[0]
        self.assertAlmostEqual(row.MAE_reduction, 1.)
        self.assertAlmostEqual(row.MAE_reduction_percent, 50.)
        self.assertAlmostEqual(row.R2_gain, .3)

    def test_resume_repair_shared_fit_inputs_and_complete_status(self):
        config = json.loads(json.dumps(self.config));config["targets"] = ["DO"]
        config["training"]["seeds"] = [42];config["bootstrap"]["replicates"] = 10
        def fake_fit(architecture, x, y, split, seed, training):
            self.assertEqual(architecture, "lstm")
            self.assertEqual(training["hidden_size"], 64)
            for name in self.split:
                np.testing.assert_array_equal(split[name], self.split[name])
            self.assertTrue(any(np.array_equal(x, arm.x) for arm in self.arms.values()))
            np.testing.assert_array_equal(y, self.sequence.y)
            return PredictionResult(y[split["test"]] + 1. / x.shape[2], [], 1, 100)
        with tempfile.TemporaryDirectory() as directory:
            config["results_dir"] = directory
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch.object(RandomStationRunner, "_environment_manifest", return_value={}), \
                 patch("jinjiang.station_ablation_random._line_figure"), \
                 patch("jinjiang.runner.fit_torch_predict", side_effect=fake_fit) as fit:
                out = RandomStationRunner(config, max_tasks=4).run()
                self.assertEqual(json.loads((out/"run_status.json").read_text())["run_status"], "partial")
                # Simulate a write interrupted before the completion marker.
                frame = pd.read_csv(out / "metrics.csv"); orphan = frame.tail(6).copy();orphan["task_id"] = "orphan"
                orphan.to_csv(out/"metrics.csv", mode="a", header=False, index=False)
                RandomStationRunner(config).run()
                RandomStationRunner(config).run()
                self.assertEqual(fit.call_count, 6)
            status = json.loads((out/"run_status.json").read_text())
            self.assertEqual(status["run_status"], "complete")
            self.assertEqual(status["completed_tasks"], 9)
            self.assertEqual(len(pd.read_csv(out/"metrics.csv")), 54)
            groups = pd.read_csv(out/"input_groups.csv")
            self.assertEqual(len(groups), 9)
            self.assertEqual(groups.test_ordered_indices_sha256.nunique(), 1)
            summary = pd.read_csv(out/"upstream_gain_summary.csv")
            self.assertGreater(summary.iloc[0].MAE_reduction, 0)
            self.assertEqual(summary.iloc[0].n_paired_seeds, 1)
            self.assertEqual(summary.iloc[0].status, "complete")
            (out/"data_fingerprints.json").write_text('{"changed.xlsx":"different"}')
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch.object(RandomStationRunner, "_environment_manifest", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "source/data changed"):
                    RandomStationRunner(config)

    def test_audit_only_never_fits_any_model(self):
        config = json.loads(json.dumps(self.config));config["targets"] = ["DO"]
        with tempfile.TemporaryDirectory() as directory:
            config["results_dir"] = directory
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch.object(RandomStationRunner, "_environment_manifest", return_value={}), \
                 patch.object(RandomStationRunner, "_fit_method", side_effect=AssertionError("Unexpected fit")):
                out = RandomStationRunner(config, max_tasks=0).run()
            self.assertEqual(json.loads((out/"run_status.json").read_text())["run_status"], "audit_only")
            self.assertFalse((out/"metrics.csv").exists())
            self.assertEqual(len(pd.read_csv(out/"sample_manifest.csv")), len(self.sequence.x))

    def test_keyboard_interrupt_records_progress_and_releases_lock(self):
        config = json.loads(json.dumps(self.config));config["targets"] = ["DO"]
        config["training"]["seeds"] = [42]
        with tempfile.TemporaryDirectory() as directory:
            config["results_dir"] = directory
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch.object(RandomStationRunner, "_environment_manifest", return_value={}), \
                 patch("jinjiang.runner.fit_torch_predict", side_effect=KeyboardInterrupt):
                runner = RandomStationRunner(config)
                with self.assertRaises(KeyboardInterrupt):
                    runner.run()
                self.assertIsNone(runner._lock_handle)
                status = json.loads((runner.output/"run_status.json").read_text())
                self.assertEqual(status["completed_tasks"], 2)
                self.assertEqual(status["run_status"], "partial")


if __name__ == "__main__":
    unittest.main()
