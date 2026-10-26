from __future__ import annotations

import unittest
import tempfile

import numpy as np
import pandas as pd

from jinjiang.config import load_config, resolve_path
from jinjiang.data import (
    feature_set_columns,
    make_sequences,
    official_split,
    prepare_dataset,
    rolling_origin_datasets,
    short_gap_causal_ffill,
    short_gap_linear,
)
from jinjiang.evaluation import mean_sd_table, moving_block_bootstrap, skill_scores_table
from jinjiang.lstm_upgrade import _candidate_x
from jinjiang.models import choose_validation_baseline, optimal_residual_gates


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.make_synthetic_data import generate
        cls._fixture_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._fixture_dir.cleanup)
        generate(cls._fixture_dir.name)
        cls.config = load_config("configs/smoke.json")
        cls.config["data_dir"] = cls._fixture_dir.name
        cls.data = prepare_dataset(resolve_path(cls.config, cls.config["data_dir"]), cls.config)

    def test_long_gap_is_not_partially_filled(self):
        index = pd.date_range("2022-01-01", periods=12, freq="4h")
        series = pd.Series(np.arange(12, dtype=float), index=index)
        series.iloc[2:4] = np.nan
        series.iloc[6:10] = np.nan
        filled = short_gap_linear(series, max_steps=3)
        self.assertTrue(filled.iloc[2:4].notna().all())
        self.assertTrue(filled.iloc[6:10].isna().all())

    def test_raw_targets_are_not_replaced_by_imputed_values(self):
        target = self.data.target_columns["C"]["DO"]
        self.assertGreaterEqual(
            self.data.raw_frame[target].isna().sum(),
            self.data.feature_frame[target].isna().sum(),
        )

    def test_causal_fill_never_uses_future_value(self):
        index = pd.date_range("2022-01-01", periods=4, freq="4h")
        series = pd.Series([1.0, np.nan, np.nan, 10.0], index=index)
        filled = short_gap_causal_ffill(series, max_steps=2)
        np.testing.assert_allclose(filled.to_numpy(), [1.0, 1.0, 1.0, 10.0])

    def test_official_sequence_splits_are_temporally_separated(self):
        columns = feature_set_columns(self.data, "all_stations", "C", "DO")
        sequence = make_sequences(self.data, columns, self.data.target_columns["C"]["DO"], 6, 2)
        split = official_split(sequence)
        self.assertLess(
            sequence.target_end_times[split["train"]].max(),
            sequence.target_start_times[split["validation"]].min(),
        )
        self.assertLess(
            sequence.target_end_times[split["validation"]].max(),
            sequence.target_start_times[split["test"]].min(),
        )

    def test_seasonal_naive_uses_only_available_previous_day_values(self):
        target = self.data.target_columns["C"]["DO"]
        columns = feature_set_columns(self.data, "all_stations", "C", "DO")
        sequence = make_sequences(self.data, columns, target, 6, 2)
        start = sequence.target_start_times[0]
        expected = self.data.feature_frame.loc[
            [start - pd.Timedelta(hours=24), start - pd.Timedelta(hours=20)], target
        ].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(sequence.seasonal_naive[0], expected)

    def test_rolling_folds_use_disjoint_test_blocks_and_expanding_training(self):
        datasets = rolling_origin_datasets(self.data, self.config)
        self.assertEqual(len(datasets), 2)
        previous_test_end = None
        previous_train_end = None
        for fold_data in datasets:
            boundaries = fold_data.split_boundaries
            self.assertLess(boundaries["train_end"], boundaries["validation_end"])
            self.assertLess(boundaries["validation_end"], boundaries["test_end"])
            if previous_test_end is not None:
                self.assertLess(previous_test_end, boundaries["test_end"])
                self.assertLess(previous_train_end, boundaries["train_end"])
            previous_test_end = boundaries["test_end"]
            previous_train_end = boundaries["train_end"]
    def test_uploaded_schema_is_recognized(self):
        self.assertEqual(set(self.data.station_columns), {"A", "B", "C"})
        for station in "ABC":
            self.assertTrue(set(["DO", "NH3N", "CODMN", "TP"]).issubset(self.data.target_columns[station]))
        self.assertIn("W_RAIN_ANTE_24H", self.data.weather_columns)

    def test_complete_window_grid(self):
        self.assertEqual(self.config["window_grid"], [6, 12])
        paper = load_config("configs/paper.json")
        self.assertEqual(paper["window_grid"], list(range(6, 85, 6)))
        self.assertEqual(paper["missing_data"]["interpolation_policy"], "causal_ffill")

    def test_summary_headers_are_single_level(self):
        rows = []
        for seed, value in [(1, 0.5), (2, 0.7)]:
            rows.append({
                "experiment": "x", "subexperiment": "y", "fold": 0,
                "split_scheme": "chronological", "target_station": "C", "target": "DO",
                "feature_set": "all_stations", "model": "lstm", "lookback_steps": 6,
                "lookback_days": 1.0, "horizon_step": 1, "horizon_hours": 4,
                "seed": seed, "MSE": 1.0, "RMSE": 1.0, "MAE": 1.0,
                "nRMSE_test_sd": 1.0, "MASE": 1.0, "MAPE_percent": 1.0,
                "sMAPE_percent": 1.0, "R2": value,
            })
        summary = mean_sd_table(pd.DataFrame(rows))
        self.assertIn("R2_mean", summary.columns)
        self.assertIn("R2_std", summary.columns)

    def test_skill_scores_and_bootstrap_p_value_are_finite(self):
        common = {
            "experiment": "x", "subexperiment": "y", "fold": 0,
            "split_scheme": "chronological", "target_station": "C", "target": "DO",
            "seed": 42, "lookback_steps": 6, "lookback_days": 1.0,
            "forecast_horizon_steps": 1, "horizon_step": 1, "horizon_hours": 4,
            "RMSE": 1.0, "R2": 0.0,
        }
        metrics = pd.DataFrame([
            {**common, "feature_set": "persistence", "model": "persistence", "MAE": 2.0, "MSE": 4.0},
            {**common, "feature_set": "seasonal_naive", "model": "seasonal_naive", "MAE": 1.5, "MSE": 3.0},
            {**common, "feature_set": "all_stations", "model": "lstm", "MAE": 1.0, "MSE": 2.0},
        ])
        skills = skill_scores_table(metrics)
        learned = skills[skills["model"] == "lstm"].iloc[0]
        self.assertAlmostEqual(learned["MAE_skill_vs_best_naive"], 1 - 1.0 / 1.5)
        _, _, _, p_value = moving_block_bootstrap(
            np.ones(30), block_length=3, replicates=100, rng=np.random.default_rng(42)
        )
        self.assertGreater(p_value, 0)

    def test_validation_baseline_selection_does_not_use_test_targets(self):
        y = np.array([[1.0], [2.0], [100.0], [100.0]], dtype=np.float32)
        persistence = np.array([[0.0], [2.0], [0.0], [0.0]], dtype=np.float32)
        seasonal = np.array([[1.0], [9.0], [100.0], [100.0]], dtype=np.float32)
        selected, names = choose_validation_baseline(
            y, persistence, seasonal, np.array([0, 1])
        )
        self.assertEqual(names, ["persistence"])
        np.testing.assert_allclose(selected[:, 0], persistence[:, 0])

    def test_residual_gate_can_exactly_recover_naive_forecast(self):
        y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        baseline = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        harmful_residual = np.full_like(y, 100.0)
        gates = optimal_residual_gates(y, baseline, harmful_residual)
        np.testing.assert_allclose(gates, [0.0])

    def test_lstm_candidates_share_identical_sequence_support(self):
        feature_names = ["local_target", "cross_station_target", "process_focused", "all_stations"]
        feature_sets = {
            name: feature_set_columns(self.data, name, "C", "DO") for name in feature_names
        }
        union_columns = list(dict.fromkeys(
            column for columns in feature_sets.values() for column in columns
        ))
        sequence = make_sequences(
            self.data,
            union_columns,
            self.data.target_columns["C"]["DO"],
            18,
            2,
        )
        local = _candidate_x(sequence, union_columns, feature_sets["local_target"], 6, True, True)
        all_stations = _candidate_x(
            sequence, union_columns, feature_sets["all_stations"], 18, True, True
        )
        self.assertEqual(local.shape[0], all_stations.shape[0])
        self.assertEqual(local.shape[0], len(sequence.issue_times))
        self.assertEqual(local.shape[1], 6)
        self.assertEqual(all_stations.shape[1], 18)


if __name__ == "__main__":
    unittest.main()
