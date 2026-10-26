from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from jinjiang.baseline_random import (
    legacy_random_split, prepare_random_pool, random_support_tables,
    conditional_time_block_comparisons, RandomBaselineRunner,
)
from jinjiang.config import load_config
from jinjiang.data import PreparedDataset, compute_split_boundaries, make_sequences
from jinjiang.models import PredictionResult, _scale_x


def fixture():
    index = pd.date_range("2022-01-01", periods=240, freq="4h")
    frame = pd.DataFrame({"A_DO": np.arange(240) + 1., "C_DO": np.arange(240) + 2.}, index=index)
    return PreparedDataset(
        raw_frame=frame.copy(), feature_frame=frame.copy(),
        station_columns={"A": ["A_DO"], "B": [], "C": ["C_DO"]},
        target_columns={"A": {"DO": "A_DO"}, "B": {}, "C": {"DO": "C_DO"}}, weather_columns=[],
        split_boundaries=compute_split_boundaries(index, .7, .15), audit=pd.DataFrame(),
    )


class RandomBaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/baseline_random_gpu.json")
        self.data = prepare_random_pool(fixture(), self.config)
        self.sequence = make_sequences(self.data, ["A_DO", "C_DO"], "C_DO", 6, 6)
        self.split = legacy_random_split(len(self.sequence.x))

    def test_split_exactly_reproduces_two_old_sklearn_calls(self):
        n = len(self.sequence.x)
        train, rest = train_test_split(np.arange(n), test_size=.2, random_state=42)
        validation, test = train_test_split(rest, test_size=.5, random_state=42)
        for name, indices in zip(("train", "validation", "test"), (train, validation, test)):
            np.testing.assert_array_equal(self.split[name], indices)
        self.assertEqual(len(set(np.concatenate(list(self.split.values())))), n)

    def test_all_windows_including_former_boundaries_are_available(self):
        self.assertEqual(len(self.sequence.x), 240 - 6 - 6 + 1)
        np.testing.assert_array_equal(self.sequence.x[0, :, 0], np.arange(1., 7.))
        np.testing.assert_array_equal(self.sequence.y[0], np.arange(8., 14.))
        self.assertTrue((np.diff(self.sequence.issue_times) == pd.Timedelta(hours=4)).all())

    def test_audit_distinguishes_unique_ids_from_shared_target_times(self):
        audit, manifest = random_support_tables(self.data, self.sequence, self.split, {"target": "DO"})
        self.assertTrue(audit.window_ids_disjoint.all())
        self.assertFalse(audit.temporal_independence.any())
        self.assertEqual(len(manifest), len(self.sequence.x))
        self.assertGreater(audit.loc[audit.data_split == "test", "fraction_target_times_seen_in_training_labels"].iloc[0], .9)

    def test_scaler_uses_training_windows_only(self):
        altered = self.sequence.x.copy()
        altered[self.split["test"]] += 100000
        _, a = _scale_x(self.sequence.x, self.split["train"])
        _, b = _scale_x(altered, self.split["train"])
        np.testing.assert_array_equal(a.mean_, b.mean_)
        np.testing.assert_array_equal(a.scale_, b.scale_)

    def test_missing_labels_remain_observed_only(self):
        data = replace(self.data, raw_frame=self.data.raw_frame.copy())
        time = data.raw_frame.index[24]
        data.raw_frame.loc[time, "C_DO"] = np.nan
        seq = make_sequences(data, ["A_DO", "C_DO"], "C_DO", 6, 6)
        self.assertFalse(((seq.target_start_times <= time) & (seq.target_end_times >= time)).any())

    def test_block_ci_preserves_calendar_gaps_and_has_correct_point_estimate(self):
        times = pd.date_range("2022-01-01", periods=12, freq="4h").append(
            pd.date_range("2022-01-21", periods=12, freq="4h"))
        frames = []
        for model, feature, error in [("persistence", "persistence", 1.), ("ridge", "all_stations", .5)]:
            frames.append(pd.DataFrame(dict(target="DO", horizon_hours=4, seed=42, valid_time=times,
                                           model=model, feature_set=feature, absolute_error=error, squared_error=error**2)))
        result = conditional_time_block_comparisons(pd.concat(frames), replicates=100)
        row = result[(result.loss == "absolute_error") & (result.reference_model == "persistence")].iloc[0]
        self.assertEqual(row.n_calendar_blocks, 22)
        self.assertEqual(row.n_nonempty_calendar_blocks, 4)
        self.assertAlmostEqual(row.competitor_minus_reference_loss, -.5)
        self.assertAlmostEqual(row.ci95_lower, -.5)
        self.assertAlmostEqual(row.ci95_upper, -.5)

    def test_partial_run_resume_and_metadata(self):
        config = json.loads(json.dumps(self.config))
        config["targets"] = ["DO"]; config["training"]["seeds"] = [42]
        config["baseline_models"] = ["lstm"]; config["baseline_lookback_steps"] = 6
        config["bootstrap"]["replicates"] = 20

        def fake_fit(architecture, x, y, split, seed, training):
            np.testing.assert_array_equal(x, self.sequence.x)
            for name in self.split:
                np.testing.assert_array_equal(split[name], self.split[name])
            return PredictionResult(y[split["test"]] + .1, [], 1, 100)

        with tempfile.TemporaryDirectory() as directory:
            config["results_dir"] = directory
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch.object(RandomBaselineRunner, "_environment_manifest", return_value={}), \
                 patch("jinjiang.baseline_random._line_figure"), \
                 patch("jinjiang.runner.fit_torch_predict", side_effect=fake_fit) as fit:
                out = RandomBaselineRunner(config, max_tasks=2).run()
                self.assertEqual(json.loads((out / "run_status.json").read_text())["run_status"], "partial")
                RandomBaselineRunner(config).run()
                RandomBaselineRunner(config).run()
                self.assertEqual(fit.call_count, 1)
            (out / "data_fingerprints.json").write_text('{"tampered.xlsx": "different"}')
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch.object(RandomBaselineRunner, "_environment_manifest", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "Raw data changed"):
                    RandomBaselineRunner(config)
            status = json.loads((out / "run_status.json").read_text())
            self.assertEqual(status["run_status"], "complete")
            self.assertEqual(status["completed_tasks"], 3)
            metrics = pd.read_csv(out / "metrics.csv")
            self.assertEqual(len(metrics), 18)
            self.assertEqual(metrics.split_scheme.nunique(), 1)
            self.assertEqual(len(pd.read_csv(out / "sample_manifest.csv")), len(self.sequence.x))
            self.assertFalse((out / "split_optimism_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
