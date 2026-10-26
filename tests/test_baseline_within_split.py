from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jinjiang.baseline_within_split import (
    make_within_split_sequences, raw_partitions, support_tables, training_mase_scale,
)
from jinjiang.data import PreparedDataset, compute_split_boundaries, make_sequences, official_split
from jinjiang.models import _scale_x
from jinjiang.config import load_config
from jinjiang.models import PredictionResult
from jinjiang.runner import ExperimentRunner


def fixture() -> PreparedDataset:
    index = pd.date_range("2022-01-01", periods=240, freq="4h")
    frame = pd.DataFrame({"A_DO": np.arange(240) + 1., "C_DO": np.arange(240) + 2.}, index=index)
    return PreparedDataset(
        raw_frame=frame.copy(), feature_frame=frame.copy(),
        station_columns={"A": ["A_DO"], "B": [], "C": ["C_DO"]},
        target_columns={"A": {"DO": "A_DO"}, "B": {}, "C": {"DO": "C_DO"}},
        weather_columns=[], split_boundaries=compute_split_boundaries(index, .7, .15),
        audit=pd.DataFrame(),
    )


class WithinSplitTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.seq = make_within_split_sequences(self.data, ["A_DO", "C_DO"], "C_DO", 6, 6)

    def test_exact_first_and_last_windows_and_counts(self):
        split = official_split(self.seq)
        self.assertEqual({key: len(value) for key, value in split.items()},
                         {"train": 157, "validation": 25, "test": 25})
        for name, times in raw_partitions(self.data).items():
            first, last = split[name][0], split[name][-1]
            np.testing.assert_array_equal(self.seq.x[first], self.data.feature_frame.loc[times[:6]].values)
            np.testing.assert_array_equal(self.seq.y[first], self.data.raw_frame.loc[times[6:12], "C_DO"].values)
            self.assertEqual(self.seq.target_end_times[last], times[-1])

    def test_full_support_is_disjoint_but_internal_windows_overlap(self):
        audit, manifest = support_tables(self.data, self.seq, 6, 6, {"target": "DO"})
        self.assertTrue(audit.full_window_contained.all())
        self.assertFalse(audit.cross_partition_overlap.any())
        for name in ("train", "validation", "test"):
            rows = manifest[manifest.data_split == name]
            self.assertLess(rows.iloc[1].input_start_time, rows.iloc[0].issue_time)
        for a, b in (("train", "validation"), ("validation", "test")):
            self.assertLess(manifest[manifest.data_split == a].target_end_time.max(),
                            manifest[manifest.data_split == b].input_start_time.min())

    def test_old_constructor_remains_unchanged_and_can_borrow_past_inputs(self):
        old = make_sequences(self.data, ["A_DO", "C_DO"], "C_DO", 6, 6)
        old_split = official_split(old)
        self.assertEqual(len(old_split["validation"]), 31)
        self.assertEqual(len(official_split(self.seq)["validation"]), 25)

    def test_future_partition_changes_cannot_change_training_sequences_or_scaler(self):
        altered = replace(self.data, raw_frame=self.data.raw_frame.copy(), feature_frame=self.data.feature_frame.copy())
        future = altered.raw_frame.index > altered.split_boundaries["train_end"]
        altered.raw_frame.loc[future] += 100000
        altered.feature_frame.loc[future] += 100000
        other = make_within_split_sequences(altered, ["A_DO", "C_DO"], "C_DO", 6, 6)
        train = official_split(self.seq)["train"]
        np.testing.assert_array_equal(self.seq.x[train], other.x[train])
        np.testing.assert_array_equal(self.seq.y[train], other.y[train])
        _, scaler = _scale_x(self.seq.x, train)
        _, other_scaler = _scale_x(other.x, train)
        np.testing.assert_array_equal(scaler.mean_, other_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, other_scaler.scale_)

    def test_naive_forecasts_only_use_internal_available_history(self):
        split = official_split(self.seq)
        for name, times in raw_partitions(self.data).items():
            i = split[name][0]
            expected = self.data.feature_frame.loc[times[:6], "C_DO"].values
            np.testing.assert_array_equal(self.seq.seasonal_naive[i], expected)
            np.testing.assert_array_equal(self.seq.persistence[i], np.repeat(expected[-1], 6))

    def test_raw_missing_targets_are_not_scored_even_if_features_filled(self):
        altered = replace(self.data, raw_frame=self.data.raw_frame.copy())
        bad_time = altered.raw_frame.index[12]
        altered.raw_frame.loc[bad_time, "C_DO"] = np.nan
        seq = make_within_split_sequences(altered, ["A_DO", "C_DO"], "C_DO", 6, 6)
        self.assertFalse(((seq.target_start_times <= bad_time) & (seq.target_end_times >= bad_time)).any())

    def test_missing_inputs_produce_common_support_for_every_method(self):
        altered = replace(self.data, feature_frame=self.data.feature_frame.copy())
        altered.feature_frame.iloc[20, 0] = np.nan
        seq = make_within_split_sequences(altered, ["A_DO", "C_DO"], "C_DO", 6, 6)
        self.assertEqual(len(seq.x), len(self.seq.x) - 6)
        self.assertEqual(len(seq.x), len(seq.persistence))
        self.assertEqual(len(seq.x), len(seq.seasonal_naive))

    def test_mase_is_invariant_to_datetime_resolution(self):
        train = official_split(self.seq)["train"]
        micros = replace(self.seq, target_start_times=self.seq.target_start_times.as_unit("us"))
        self.assertEqual(training_mase_scale(self.seq, train), 1.)
        self.assertEqual(training_mase_scale(micros, train), 1.)

    def test_audit_rejects_a_cross_boundary_window(self):
        old = make_sequences(self.data, ["A_DO", "C_DO"], "C_DO", 6, 6)
        with self.assertRaisesRegex(AssertionError, "cross"):
            support_tables(self.data, old, 6, 6, {"target": "DO"})

    def test_too_short_partition_has_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            make_within_split_sequences(self.data, ["A_DO"], "C_DO", 36, 6)

    def test_runner_partial_resume_has_no_random_diagnostic_and_uses_shared_support(self):
        # Mock only the neural fit; this tests orchestration, not neural accuracy or GPU execution.
        config = load_config("configs/baseline_within_split_gpu.json")
        config["targets"] = ["DO"]
        config["baseline_models"] = ["lstm"]
        config["baseline_lookback_steps"] = 6
        config["training"]["seeds"] = [42]
        config["bootstrap"]["replicates"] = 20

        def fake_fit(architecture, x, y, split, seed, training):
            self.assertEqual(architecture, "lstm")
            self.assertEqual({k: len(v) for k, v in split.items()},
                             {"train": 157, "validation": 25, "test": 25})
            np.testing.assert_array_equal(x, self.seq.x)
            return PredictionResult(y[split["test"]] + .1, [], 1, 100)

        with tempfile.TemporaryDirectory() as directory:
            config["results_dir"] = directory
            with patch("jinjiang.runner.prepare_dataset", return_value=self.data), \
                 patch("jinjiang.runner.generate_experiment_figures"), \
                 patch.object(ExperimentRunner, "_environment_manifest", return_value={}), \
                 patch("jinjiang.runner.fit_torch_predict", side_effect=fake_fit) as fit:
                first = ExperimentRunner(config, "01_baseline", max_tasks=2)
                output = first.run()
                self.assertEqual(json.loads((output / "run_status.json").read_text())["run_status"], "partial")
                self.assertEqual(fit.call_count, 0)
                resumed = ExperimentRunner(config, "01_baseline")
                resumed.run()
                self.assertEqual(fit.call_count, 1)
                ExperimentRunner(config, "01_baseline").run()
                self.assertEqual(fit.call_count, 1)
            status = json.loads((output / "run_status.json").read_text())
            self.assertEqual(status["run_status"], "complete")
            self.assertEqual(status["completed_tasks"], 3)
            self.assertEqual(status["expected_tasks"], 3)
            metrics = pd.read_csv(output / "metrics.csv")
            self.assertEqual(len(metrics), 18)
            self.assertEqual(set(metrics.subexperiment), {"unified_model_benchmark"})
            self.assertTrue(np.isfinite(metrics.MASE).all())
            self.assertFalse((output / "split_optimism_summary.csv").exists())
            self.assertEqual(len(pd.read_csv(output / "sample_manifest.csv")), 207)
            self.assertEqual(len(pd.read_csv(output / "sample_support_audit.csv")), 3)


if __name__ == "__main__":
    unittest.main()
