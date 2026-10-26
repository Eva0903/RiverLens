from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jinjiang.baseline_random import legacy_random_split, prepare_random_pool
from jinjiang.config import load_config
from jinjiang.data import PreparedDataset, compute_split_boundaries, make_sequences
from jinjiang.models import PredictionResult, _scale_x
from jinjiang.window_horizon_random import (
    RandomWindowRunner, assign_pool, fit_with_validation, make_common_pool,
    select_windows, describe_selected_test_windows, validate_config, window_skills,
)


def fixture():
    index = pd.date_range("2022-01-01", periods=300, freq="4h")
    columns = ["A_DO", "A_WT", "B_DO", "B_FLOW", "C_DO", "C_WT", "C_TN"]
    frame = pd.DataFrame({c: np.arange(300) + i / 10 + 1 for i, c in enumerate(columns)}, index=index)
    return PreparedDataset(frame.copy(), frame.copy(),
        {s: [c for c in columns if c.startswith(s + "_")] for s in "ABC"},
        {s: {"DO": s + "_DO"} for s in "ABC"}, [],
        compute_split_boundaries(index, .7, .15), pd.DataFrame())


class WindowHorizonTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/window_horizon_random_main_gpu.json")
        self.data = prepare_random_pool(fixture(), self.config)
        self.sequence, self.split = assign_pool(self.data, self.config, "DO")

    def test_main_and_long_configs_have_predefined_ranges(self):
        self.assertEqual(self.config["window_grid"], [3] + list(range(6,85,6)))
        long = load_config("configs/window_horizon_random_long_gpu.json")
        validate_config(long)
        self.assertEqual(long["window_grid"], [18,42,84,126,168])
        self.assertEqual(long["support_lookback_steps"], 168)
        self.assertEqual(self.sequence.y.shape[1], 18)
        self.assertEqual(len(self.sequence.x), 300 - 84 - 18 + 1)

    def test_new_builder_matches_old_builder_through_24h(self):
        old = make_sequences(self.data, self.sequence.feature_columns, "C_DO", 84, 6)
        new = make_common_pool(self.data, "DO", 84, 6)
        for name in ["x", "y", "persistence", "seasonal_naive"]:
            np.testing.assert_array_equal(getattr(old,name), getattr(new,name))

    def test_72h_seasonal_naive_repeats_only_last_observed_day(self):
        index = self.data.raw_frame.index
        i = index.get_loc(self.sequence.target_start_times[0])
        expected = self.data.feature_frame.C_DO.iloc[i-6:i].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(self.sequence.seasonal_naive[0], np.tile(expected, 3))
        self.assertLess(expected.max(), self.sequence.y[0,0])
        # Poison every future feature value for this forecast: first baseline must not change.
        changed = replace(self.data, feature_frame=self.data.feature_frame.copy())
        changed.feature_frame.loc[index[i]:, "C_DO"] = 99999
        after = make_common_pool(changed, "DO", 84, 18)
        np.testing.assert_array_equal(after.seasonal_naive[0], self.sequence.seasonal_naive[0])

    def test_missing_labels_never_become_scoring_targets(self):
        changed = replace(self.data, raw_frame=self.data.raw_frame.copy())
        time = changed.raw_frame.index[140];changed.raw_frame.loc[time,"C_DO"] = np.nan
        seq = make_common_pool(changed,"DO",84,18)
        self.assertFalse(((seq.target_start_times <= time)&(seq.target_end_times >= time)).any())

    def test_all_windows_share_full_support_and_exact_split(self):
        for name, values in legacy_random_split(len(self.sequence.x)).items():
            np.testing.assert_array_equal(self.split[name], values)
        short = self.sequence.x[:, -3:]
        np.testing.assert_array_equal(short[0,:,0], np.arange(82,85))
        # Missing history outside the short input must still remove that common sample.
        changed = replace(self.data, feature_frame=self.data.feature_frame.copy())
        changed.feature_frame.iloc[10,0] = np.nan
        seq, _ = assign_pool(changed,self.config,"DO")
        self.assertNotIn(self.sequence.target_start_times[0],seq.target_start_times)

    def test_same_fitted_model_returns_validation_and_test_in_correct_order(self):
        def fake(architecture,x,y,split,seed,training):
            np.testing.assert_array_equal(split["train"],self.split["train"])
            np.testing.assert_array_equal(split["validation"],self.split["validation"])
            np.testing.assert_array_equal(split["test"],np.r_[self.split["validation"],self.split["test"]])
            return PredictionResult(y[split["test"]]+.25,[],2,100)
        with patch("jinjiang.window_horizon_random.fit_torch_predict",side_effect=fake) as fit:
            result=fit_with_validation("lstm",self.sequence.x,self.sequence,self.split,42,self.config["training"])
        self.assertEqual(fit.call_count,1)
        np.testing.assert_array_equal(result.prediction,self.sequence.y[self.split["test"]]+.25)
        np.testing.assert_array_equal(result.validation_prediction,self.sequence.y[self.split["validation"]]+.25)

    def test_scaling_uses_only_common_training_windows(self):
        x=self.sequence.x.copy();x[self.split["test"]]+=1e5;x[self.split["validation"]]+=2e5
        _,a=_scale_x(self.sequence.x,self.split["train"]);_,b=_scale_x(x,self.split["train"])
        np.testing.assert_array_equal(a.mean_,b.mean_)
        np.testing.assert_array_equal(a.scale_,b.scale_)

    def test_poisoned_test_labels_do_not_change_real_ridge_validation_predictions(self):
        x = self.sequence.x[:, -3:]
        original = fit_with_validation("ridge", x, self.sequence, self.split, 42, self.config["training"])
        y = self.sequence.y.copy();y[self.split["test"]] += 100000
        changed = replace(self.sequence, y=y)
        after = fit_with_validation("ridge", x, changed, self.split, 42, self.config["training"])
        np.testing.assert_array_equal(original.validation_prediction, after.validation_prediction)

    def selection_fixture(self):
        c=json.loads(json.dumps(self.config));c.update(targets=["DO"],smoke_test=True,window_grid=[3,6,18]);c["training"]["seeds"]=[42,7]
        rows=[]
        for window,error in [(3,1.015),(6,1.),(18,1.10)]:
            for seed in c["training"]["seeds"]:
                for hour in range(4,73,4):
                    rows.append(dict(target="DO",model="lstm",lookback_steps=window,seed=seed,horizon_hours=hour,MAE=error,RMSE=error*2,R2=.9))
        return c,pd.DataFrame(rows)

    def test_selection_uses_validation_and_chooses_shortest_near_optimal(self):
        c,validation=self.selection_fixture()
        summary,selected,status=select_windows(validation,c)
        self.assertTrue(status.status.eq("complete").all())
        self.assertEqual(len(selected),19)
        self.assertTrue(selected.best_lookback_steps.eq(6).all())
        self.assertTrue(selected.shortest_near_optimal_steps.eq(3).all())
        # Test errors favor a different window; selected lengths remain validation-defined.
        test=validation.copy();test.loc[test.lookback_steps==18,"MAE"] = .001
        report=describe_selected_test_windows(test,selected)
        self.assertEqual(set(report.selected_lookback_steps),{3,6})

    def test_partial_validation_grid_does_not_publish_a_best_window(self):
        c,validation=self.selection_fixture()
        _,selected,status=select_windows(validation.iloc[:-1],c)
        self.assertTrue(selected.empty)
        self.assertTrue(status.status.eq("pending_all_windows_and_seeds").all())

    def test_skill_matches_baseline_once_per_seed_across_windows(self):
        rows=[]
        for model,window,error in [("persistence",0,2.),("seasonal_naive",0,1.),("lstm",3,.5),("lstm",6,.6)]:
            rows.append(dict(target="DO",seed=42,horizon_hours=4,model=model,lookback_steps=window,MAE=error,MSE=error**2))
        skills=window_skills(pd.DataFrame(rows))
        self.assertEqual(len(skills),2)
        np.testing.assert_allclose(skills.MAE_skill_vs_best_naive,[.5,.4])

    def test_old_config_and_wrong_phase_support_rejected(self):
        for key,value in [("window_horizon_protocol","old"),("forecast_horizon_steps",6),("support_lookback_steps",168),("split_seed",7)]:
            c=json.loads(json.dumps(self.config));c[key]=value
            with self.assertRaises(ValueError):validate_config(c)

    def test_partial_resume_repairs_missing_validation_output(self):
        c=json.loads(json.dumps(self.config));c.update(targets=["DO"],window_grid=[3,18,84],smoke_test=True);c["training"]["seeds"]=[42]
        def fake(architecture,x,y,split,seed,training):
            self.assertIn(x.shape[1],[3,18,84]);self.assertEqual(y.shape[1],18)
            return PredictionResult(y[split["test"]]+1./x.shape[1],
                                    [{"epoch":1,"train_loss":1.,"validation_loss":1.}],1,100)
        with tempfile.TemporaryDirectory() as directory:
            c["results_dir"]=directory
            with patch("jinjiang.runner.prepare_dataset",return_value=fixture()), \
                 patch.object(RandomWindowRunner,"_environment_manifest",return_value={}), \
                 patch("jinjiang.window_horizon_random.render_figures"), \
                 patch("jinjiang.window_horizon_random.fit_torch_predict",side_effect=fake) as fit:
                out=RandomWindowRunner(c,max_tasks=3).run()
                self.assertEqual(json.loads((out/"run_status.json").read_text())["run_status"],"partial")
                self.assertTrue(pd.read_csv(out/"window_selection_status.csv").status.str.startswith("pending").all())
                # A completed task with a missing validation lead must be reopened.
                v=pd.read_csv(out/"validation_metrics.csv");v.iloc[:-1].to_csv(out/"validation_metrics.csv",index=False)
                RandomWindowRunner(c).run();RandomWindowRunner(c).run()
                self.assertEqual(fit.call_count,4) # first short task was correctly redone
            metrics=pd.read_csv(out/"metrics.csv");self.assertEqual(len(metrics),90)
            self.assertEqual(metrics.task_id.nunique(),5)
            status=json.loads((out/"run_status.json").read_text());self.assertEqual(status["run_status"],"complete")
            self.assertEqual(status["expected_lstm_tasks"],3)
            self.assertEqual(len(pd.read_csv(out/"validation_window_selection.csv")),19)
            self.assertEqual(len(pd.read_csv(out/"test_metrics_for_validation_selected_windows.csv")),38)
            audit=pd.read_csv(out/"window_input_audit.csv")
            self.assertEqual(audit.groupby("data_split").ordered_sample_indices_sha256.nunique().max(),1)

    def test_audit_only_never_fits_and_ctrlC_keeps_completed_baselines(self):
        c=json.loads(json.dumps(self.config));c["targets"]=["DO"];c["training"]["seeds"]=[42]
        with tempfile.TemporaryDirectory() as directory:
            c["results_dir"]=directory
            with patch("jinjiang.runner.prepare_dataset",return_value=fixture()), \
                 patch.object(RandomWindowRunner,"_environment_manifest",return_value={}), \
                 patch("jinjiang.window_horizon_random.fit_torch_predict",side_effect=KeyboardInterrupt):
                out=RandomWindowRunner(c,max_tasks=0).run()
                self.assertEqual(json.loads((out/"run_status.json").read_text())["run_status"],"audit_only")
                runner=RandomWindowRunner(c)
                with self.assertRaises(KeyboardInterrupt):runner.run()
                self.assertIsNone(runner._lock_handle)
                status=json.loads((out/"run_status.json").read_text())
                self.assertEqual(status["completed_tasks"],2)
                self.assertEqual(status["run_status"],"partial")

    def test_validation_scored_count_is_distinct_from_test_partition_size(self):
        c=load_config("configs/window_horizon_random_long_gpu.json")
        c.update(targets=["DO"],window_grid=[18],smoke_test=True);c["training"]["seeds"]=[42]
        def fake(architecture,x,y,split,seed,training):
            return PredictionResult(y[split["test"]]+.1,[],1,100)
        with tempfile.TemporaryDirectory() as directory:
            c["results_dir"]=directory
            with patch("jinjiang.runner.prepare_dataset",return_value=fixture()), \
                 patch.object(RandomWindowRunner,"_environment_manifest",return_value={}), \
                 patch("jinjiang.window_horizon_random.render_figures"), \
                 patch("jinjiang.window_horizon_random.fit_torch_predict",side_effect=fake):
                out=RandomWindowRunner(c).run()
            validation=pd.read_csv(out/"validation_metrics.csv")
            self.assertTrue(validation.n_scored.eq(11).all())
            self.assertTrue(validation.n_validation.eq(11).all())
            self.assertTrue(validation.n_test.eq(12).all())
            self.assertTrue(validation.evaluation_split.eq("validation").all())


if __name__ == "__main__":
    unittest.main()
