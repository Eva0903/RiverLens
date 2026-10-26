"""Permutation axes, training-free defaults, replay caps, and checkpoint fidelity."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from jinjiang.interpretation_random import (
    CASES,InterpretationRunner,condition_masks,groups,permutation_scores,
    perturbation_specs,verify_prediction,load_predictor,validate_spec,
)


def spec():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/interpretation_random.json").read_text())


def fixture():
    rng=np.random.default_rng(3)
    columns=["W_RAIN_4H","W_RAIN_LAG_4H","W_TEM_MEAN","W_RHU_MEAN","W_PRS_MEAN",
             "W_WIND_SPEED_MAX","W_WIND_DIR_SIN","C_DO"]
    x=rng.normal(size=(40,12,len(columns))).astype(np.float32)
    def predict(values):
        return np.repeat(values[:,-1,2,None],6,axis=1)
    y=predict(x)+.1
    seq=SimpleNamespace(x=x,y=y,feature_columns=columns)
    split=dict(train=np.arange(25),validation=np.arange(25,32),test=np.arange(32,40))
    return seq,split,predict


def controller(temp):
    seq,split,predict=fixture()
    r=object.__new__(InterpretationRunner)
    r.spec=spec();r.config=dict(training=dict(device="cpu"))
    r.output=Path(temp);r.protocol_hash="testhash";r.replays_this_run=0;r.failures={}
    r.sequence=lambda s,t:(seq,split)
    r.case_meta=lambda s,t,seed:dict(station=s,target=t,seed=seed,source_task_id=f"source|{s}|{t}|{seed}")
    r.reference_predictions=lambda s,t,seed,seq,split:predict(seq.x[split["test"]])
    r.checkpoint_valid=lambda s,t,seed:(r.checkpoint_dir(s,t,seed)/"metadata.json").exists()
    return r,predict


class PermutationTests(unittest.TestCase):
    def test_declared_scope_and_grouped_rain_wind(self):
        validate_spec(spec())
        bad=spec();bad["automatic_replay"]=True
        with self.assertRaises(ValueError):validate_spec(bad)
        bad=spec();bad["interpretation_seeds"]=[99]
        with self.assertRaises(ValueError):validate_spec(bad)
        columns=["A_DO","B_DO","C_DO","W_RAIN_4H","W_RAIN_LAG_4H","W_RAIN_ANTE_24H",
                 "W_TEM_MEAN","W_WIND_SPEED_MAX","W_WIND_DIR_SIN","W_WIND_DIR_COS"]
        g=groups(columns)
        self.assertEqual(g["rain"],[3,4,5]);self.assertEqual(g["wind"],[7,8,9])
        self.assertEqual(g["weather_all"],list(range(3,10)))

    def test_time_axis_active_feature_and_no_mutation(self):
        seq,split,predict=fixture()
        original=seq.x.copy()
        raw,summary=permutation_scores(seq.x,seq.y,predict,seq.feature_columns,3,17)
        np.testing.assert_array_equal(seq.x,original)
        summary=summary[summary.horizon_hours==0]
        temp=summary[(summary.scope=="variable_group")&(summary.group=="air_temperature")]
        self.assertGreater(temp.MAE_increase.iloc[0],.1)
        rain=summary[(summary.scope=="variable_group")&(summary.group=="rain")]
        self.assertAlmostEqual(rain.MAE_increase.iloc[0],0.)
        older=summary[(summary.scope=="variable_by_history")&(summary.group=="air_temperature")&(summary.age_band=="24_72h")]
        self.assertAlmostEqual(older.MAE_increase.iloc[0],0.)
        recent=summary[(summary.scope=="variable_by_history")&(summary.group=="air_temperature")&(summary.age_band=="0_24h")]
        self.assertGreater(recent.MAE_increase.iloc[0],.1)
        same=permutation_scores(seq.x,seq.y,predict,seq.feature_columns,3,17)[0]
        np.testing.assert_array_equal(raw.MAE_increase,same.MAE_increase)

    def test_age_bins_are_exhaustive_and_issue_time_is_age_zero(self):
        seq,_,_=fixture()
        specs=perturbation_specs(seq.feature_columns,84)
        bins=[s for s in specs if s["scope"]=="history_band"]
        np.testing.assert_array_equal(np.sort(np.concatenate([b["times"] for b in bins])),np.arange(84))
        self.assertEqual(len(bins[0]["times"]),6)
        self.assertEqual(bins[0]["times"][-1],83)
        self.assertTrue(all(i<83 for i in bins[-1]["times"]))

    def test_strata_cutpoints_use_training_values_and_pair_rain(self):
        train=np.arange(9,dtype=float)
        result=condition_masks(train,np.array([-100.,4.,100.]))
        self.assertAlmostEqual(result[0][3],8/3)
        self.assertAlmostEqual(result[-1][2],16/3)
        np.testing.assert_array_equal(result[0][1],[True,False,False])
        np.testing.assert_array_equal(result[-1][1],[False,False,True])
        rain=condition_masks(np.zeros(10),np.array([0.,.01]),True)
        np.testing.assert_array_equal(rain[0][1],[True,False])
        np.testing.assert_array_equal(rain[1][1],[False,True])


class ControlTests(unittest.TestCase):
    def test_missing_models_never_trigger_implicit_training(self):
        with tempfile.TemporaryDirectory() as temp:
            r,_=controller(temp)
            with patch.object(r,"recover",side_effect=AssertionError("No training authorized")), \
                 patch("jinjiang.interpretation_random.load_predictor",side_effect=AssertionError("No model exists")):
                done,missing=r.explain()
            self.assertEqual(done,[]);self.assertEqual(len(missing),6)
            self.assertEqual(r.replays_this_run,0)

    def test_explicit_replay_cap_and_completed_attribution_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            r,predict=controller(temp)
            calls=[]
            def recover(s,t,seed,seq,split):
                directory=r.checkpoint_dir(s,t,seed);directory.mkdir(parents=True)
                (directory/"metadata.json").write_text('{}')
                r.replays_this_run+=1;calls.append((s,t))
            with patch.object(r,"recover",side_effect=recover), \
                 patch("jinjiang.interpretation_random.load_predictor",return_value=predict):
                done,missing=r.explain(replay_missing=True,max_replays=1)
                self.assertEqual(len(done),1);self.assertEqual(len(missing),5)
                self.assertEqual(len(calls),1)
                # Existing commit reuses attribution output and never calls the scorer.
                with patch("jinjiang.interpretation_random.permutation_scores",side_effect=AssertionError("Already complete")):
                    done2,missing2=r.explain()
                self.assertEqual(done2,done);self.assertEqual(missing2,missing)

    def test_original_prediction_mismatch_blocks_attribution(self):
        with tempfile.TemporaryDirectory() as temp:
            r,predict=controller(temp)
            directory=r.checkpoint_dir("C","DO",42);directory.mkdir(parents=True)
            (directory/"metadata.json").write_text('{}')
            with patch("jinjiang.interpretation_random.load_predictor",return_value=lambda x:predict(x)+1):
                with self.assertRaisesRegex(ValueError,"original predictions"):
                    r.explain()
            self.assertFalse((r.output/"attributions/C_DO_seed42/commit.json").exists())
        verify_prediction(np.array([[1.]],np.float32),np.array([[1.]],np.float32))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,"Optional tiny CPU Torch replay; torch is not installed here")
    def test_real_cpu_torch_capture_checkpoint_and_reload(self):
        from jinjiang.models import fit_torch_predict
        seq,split,_=fixture()
        training=dict(hidden_size=4,num_layers=1,dropout=0.,batch_size=8,max_epochs=2,
                      patience=2,learning_rate=.001,device="cpu",deterministic="strict",seeds=[42])
        result=fit_torch_predict("lstm",seq.x,seq.y,split,42,training)
        with tempfile.TemporaryDirectory() as temp:
            r,_=controller(temp);r.config=dict(training=training)
            meta=dict(station="C",target="DO",seed=42,source_task_id="cpu_fixture",best_epoch=result.best_epoch,
                      parameter_count=result.parameter_count,lookback_steps=12)
            r.case_meta=lambda *args:meta
            r.reference_predictions=lambda *args:result.prediction
            InterpretationRunner.recover(r,"C","DO",42,seq,split)
            self.assertTrue(InterpretationRunner.checkpoint_valid(r,"C","DO",42))
            predictor=load_predictor(r.checkpoint_dir("C","DO",42),"cpu")
            verify_prediction(predictor(seq.x[split["test"]]),result.prediction)


if __name__=="__main__":unittest.main()
