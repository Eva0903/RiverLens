"""Post-hoc weather diagnostics; cached-model permutation, never implicit training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .baseline_random import prepare_random_pool, SPLITS
from .config import resolve_path
from .data import prepare_dataset
from .models import fit_torch_predict
from .station_rotation_random import (
    RandomStationRotationRunner, SOURCES, SEEDS, ARMS, FIXED_WINDOWS,
    common_station_pool, digest, verify_task_tables,
)

PROTOCOL = "weather_interpretation_fixed_models_v1"
FULL = ARMS[1]
CASES = [("C", "DO"), ("C", "NH3N"), ("C", "TP"), ("C", "CODMN"), ("A", "CODMN"), ("B", "CODMN")]
AGE_BINS = [("0_24h", 0, 24), ("24_72h", 24, 72), ("72_168h", 72, 168), ("ge168h", 168, np.inf)]


def json_write(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def csv_write(path, frame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def validate_spec(c):
    if c.get("interpretation_protocol") != PROTOCOL:
        raise ValueError("Use configs/interpretation_random.json.")
    if c.get("cases") != [dict(station=s, target=t) for s,t in CASES]:
        raise ValueError("Keep the six declared station/target cases; do not select cases by new test results.")
    if c.get("interpretation_seeds") != [42]:
        raise ValueError("This small diagnostic fixes seed42; it does not claim five-seed attribution stability.")
    if c.get("permutation_repeats") != 10 or c.get("permutation_seed") != 2026:
        raise ValueError("Use ten fixed permutations, seed2026.")
    if c.get("min_stratum_origins") != 15 or c.get("automatic_replay") is not False:
        raise ValueError("Use the declared small-stratum flag and disable automatic replay.")


def groups(columns):
    result = {f"station_{s}": [i for i,c in enumerate(columns) if c.startswith(s+"_")] for s in "ABC"}
    result.update(rain=[i for i,c in enumerate(columns) if c.startswith("W_RAIN")],
        air_temperature=[i for i,c in enumerate(columns) if c == "W_TEM_MEAN"],
        humidity=[i for i,c in enumerate(columns) if c == "W_RHU_MEAN"],
        pressure=[i for i,c in enumerate(columns) if c == "W_PRS_MEAN"],
        wind=[i for i,c in enumerate(columns) if c.startswith("W_WIND")])
    result["weather_all"] = [i for i,c in enumerate(columns) if c.startswith("W_")]
    return {k:v for k,v in result.items() if v}


def perturbation_specs(columns, lookback):
    g = groups(columns)
    ages = 4 * np.arange(lookback-1, -1, -1)
    all_times = np.arange(lookback)
    specs = [dict(scope="variable_group", group=k, age_band="all", times=all_times, columns=v)
             for k,v in g.items()]
    for name,low,high in AGE_BINS:
        times = np.flatnonzero((ages >= low) & (ages < high))
        if not len(times):
            continue
        for family in ["weather_all", "rain", "air_temperature", "humidity", "pressure", "wind"]:
            if family not in g:
                continue
            specs.append(dict(scope="history_band" if family == "weather_all" else "variable_by_history",
                              group=family, age_band=name, times=times, columns=g[family]))
    return specs


def permutation_scores(x, y, predict, columns, repeats=10, seed=2026):
    """Jointly move a selected group/segment between intact held-out windows."""
    baseline = predict(x)
    if baseline.shape != y.shape or not np.isfinite(baseline).all():
        raise ValueError("Predictor must return finite original-scale six-lead forecasts.")
    base_mae = np.abs(baseline-y).mean(axis=0)
    base_rmse = np.sqrt(((baseline-y)**2).mean(axis=0))
    rng = np.random.default_rng(seed)
    donors = [rng.permutation(len(x)) for _ in range(repeats)]
    rows=[]
    for spec in perturbation_specs(columns,x.shape[1]):
        ix=np.ix_(np.arange(len(x)),spec["times"],spec["columns"])
        for repeat, order in enumerate(donors):
            changed=x.copy()
            changed[ix]=x[order][:,spec["times"]][:,:,spec["columns"]]
            predicted=predict(changed)
            if predicted.shape != y.shape or not np.isfinite(predicted).all():
                raise ValueError("Invalid perturbed prediction.")
            mae=np.abs(predicted-y).mean(axis=0)
            rmse=np.sqrt(((predicted-y)**2).mean(axis=0))
            for lead in range(7):
                b=float(base_mae.mean() if lead==0 else base_mae[lead-1])
                pert=float(mae.mean() if lead==0 else mae[lead-1])
                rows.append(dict(scope=spec["scope"],group=spec["group"],age_band=spec["age_band"],
                    horizon_hours=lead*4,repeat=repeat,baseline_MAE=b,perturbed_MAE=pert,
                    MAE_increase=pert-b,MAE_increase_percent=100*(pert/b-1) if b>0 else np.nan,
                    RMSE_increase=float((rmse-base_rmse).mean() if lead==0 else rmse[lead-1]-base_rmse[lead-1]),
                    n_time_steps=len(spec["times"]),n_features=len(spec["columns"]),n_test_origins=len(x)))
    frame=pd.DataFrame(rows)
    keys=["scope","group","age_band","horizon_hours","n_time_steps","n_features","n_test_origins"]
    summary=frame.groupby(keys,as_index=False).agg(MAE_increase=("MAE_increase","mean"),
        permutation_sd=("MAE_increase","std"),MAE_increase_percent=("MAE_increase_percent","mean"),
        RMSE_increase=("RMSE_increase","mean"),baseline_MAE=("baseline_MAE","mean"),
        n_permutations=("repeat","size"))
    summary["uncertainty_scope"]="permutation_randomness_for_one_fitted_seed_not_model_or_split_uncertainty"
    return frame,summary


def condition_masks(train_values, test_values, rain=False):
    if rain:
        return [("dry", test_values<=0, 0.,0.),("wet", test_values>0,0.,np.inf)]
    cut=np.unique(np.quantile(train_values,[1/3,2/3]))
    if len(cut)==1 and np.all(train_values==cut[0]):
        return [("constant_training_condition",np.ones(len(test_values),dtype=bool),float(cut[0]),float(cut[0]))]
    edges=np.r_[-np.inf,cut,np.inf]
    names=["low","middle","high"] if len(cut)==2 else ["lower","higher"]
    return [(names[i],(test_values>edges[i])&(test_values<=edges[i+1]),float(edges[i]),float(edges[i+1])) for i in range(len(edges)-1)]


class InterpretationRunner(RandomStationRotationRunner):
    def __init__(self,spec):
        validate_spec(spec)
        self.spec=spec
        root=spec["_project_root"]
        source05=resolve_path(spec,spec["source_05_dir"])
        source_config=json.loads((source05/"run_config.json").read_text(encoding="utf-8-sig"))
        self.config={**source_config,"_project_root":root,"data_dir":spec["data_dir"],
                     "reference_04_dir":spec["source_04_dir"]}
        self.clean_config={k:v for k,v in spec.items() if not k.startswith("_")}
        self.config_sha256=hashlib.sha256(json.dumps(self.clean_config,sort_keys=True).encode()).hexdigest()
        self.protocol_hash=self.config_sha256[:16]
        self.experiment="06_interpretation_random"
        self.output=resolve_path(spec,spec["results_dir"])/self.experiment
        self.output.mkdir(parents=True,exist_ok=True)
        self._lock_handle=None
        self.reference_verified=False
        self.reused_metrics,self.reused_predictions,self.reused_diagnostics=(pd.DataFrame() for _ in range(3))
        self.replays_this_run=0
        self.failures={}
        self._acquire_run_lock()
        try:
            self._guard("run_config.json",self.clean_config)
            self.source_hashes={n:digest(Path(__file__).with_name(n)) for n in SOURCES+["station_rotation_random.py","interpretation_random.py"]}
            data_dir=resolve_path(spec,spec["data_dir"])
            self.data_hashes={p.name:digest(p) for p in sorted(data_dir.glob("*.xlsx"))}
            self._guard("source_fingerprints.json",self.source_hashes)
            self._guard("data_fingerprints.json",self.data_hashes)
            read=lambda n:json.loads((source05/n).read_text(encoding="utf-8-sig"))
            status=read("run_status.json")
            if status.get("run_status")!="complete" or status.get("completed_new_lstm_tasks")!=80 or status.get("unresolved_failed_tasks")!=0:
                raise ValueError("Use the complete formal 05 result directory (80 new tasks, zero failures).")
            if read("data_fingerprints.json")!=self.data_hashes:
                raise ValueError("Raw data differ from 05.")
            for n,v in read("source_fingerprints.json").items():
                if self.source_hashes.get(n)!=v:
                    raise ValueError(f"05 source mismatch: {n}.")
            self.data=prepare_random_pool(prepare_dataset(data_dir,self.config),self.config)
            self._load_reference()  # audited read-only 04 reference
            m=pd.read_csv(source05/"metrics.csv")
            p=pd.read_csv(source05/"predictions.csv",parse_dates=["target_start_time","valid_time"])
            h=pd.read_csv(source05/"training_history.csv")
            markers=set((source05/"completed_tasks.txt").read_text(encoding="utf-8").splitlines())
            if verify_task_tables(m,p,h,markers)!=markers or len(markers)!=80 or len(m)!=480 or set(m.target_station)!={"A","B"}:
                raise ValueError("05 must contain eighty complete new A/B LSTM tasks.")
            self.metrics=pd.concat([m,self.reused_metrics],ignore_index=True)
            self.predictions=pd.concat([p,self.reused_predictions],ignore_index=True)
            self.predictions["target_start_time"]=pd.to_datetime(self.predictions.target_start_time)
            if self.metrics.task_id.nunique()!=120 or len(self.metrics)!=720:
                raise ValueError("Expected 120 complete tasks and 720 metric rows.")
            self.manifest=pd.read_csv(source05/"sample_manifest.csv")
            self.input_groups=pd.read_csv(source05/"input_groups.csv")
            names=["run_config.json","run_status.json","metrics.csv","predictions.csv","training_history.csv",
                   "sample_manifest.csv","input_groups.csv","completed_tasks.txt","data_fingerprints.json","source_fingerprints.json"]
            self._guard("reference_05_provenance.json",dict(source_directory=str(source05),
                files_sha256={n:digest(source05/n) for n in names},verified_new_fits=80))
            json_write(self.output/"environment.json",self._environment_manifest())
        except BaseException:
            self._release_run_lock()
            raise

    def sequence(self,station,target):
        seq,split,arms=common_station_pool(self.data,self.config,station,target)
        seq=arms[FULL]
        old=self.manifest[(self.manifest.target_station==station)&(self.manifest.target==target)].sort_values("sample_index")
        if len(old)!=len(seq.x) or not np.array_equal(old.data_split.to_numpy(),seq.split_labels):
            raise ValueError(f"Sample split mismatch for {station}/{target}.")
        for col,current in [("target_start_time",seq.target_start_times),("issue_time",seq.issue_times),("target_end_time",seq.target_end_times)]:
            if not np.array_equal(pd.to_datetime(old[col]).to_numpy(dtype="datetime64[ns]"),current.to_numpy(dtype="datetime64[ns]")):
                raise ValueError(f"Timestamp mismatch: {station}/{target}/{col}.")
        previous=self.input_groups[(self.input_groups.target_station==station)&(self.input_groups.target==target)&(self.input_groups.feature_set==FULL)]
        if len(previous)!=1 or previous.iloc[0].feature_columns!="|".join(seq.feature_columns):
            raise ValueError("Feature columns differ from 05.")
        for name in SPLITS:
            expected=hashlib.sha256(np.asarray(split[name],dtype="<i8").tobytes()).hexdigest()
            if previous.iloc[0][name+"_ordered_indices_sha256"]!=expected:
                raise ValueError("Split index order differs from 05.")
        for seed in SEEDS:
            expected=self.reference_predictions(station,target,seed,seq,split)
            if expected.shape!=seq.y[split["test"]].shape:
                raise ValueError("Incomplete held-out reference predictions.")
        return seq,split

    def reference_predictions(self,station,target,seed,seq,split):
        p=self.predictions[(self.predictions.target_station==station)&(self.predictions.target==target)&
                           (self.predictions.seed==seed)&(self.predictions.feature_set==FULL)]
        values=p.pivot(index="target_start_time",columns="horizon_step",values="y_pred").reindex(seq.target_start_times[split["test"]])
        observed=p.pivot(index="target_start_time",columns="horizon_step",values="y_true").reindex(seq.target_start_times[split["test"]])
        if len(p)!=6*len(split["test"]) or not np.array_equal(observed.to_numpy(dtype=np.float32),seq.y[split["test"]]):
            raise ValueError("Observed test labels differ from source results.")
        return values.to_numpy(dtype=np.float32)

    def case_meta(self,station,target,seed):
        selected=self.metrics[(self.metrics.target_station==station)&(self.metrics.target==target)]
        means=selected.groupby(["feature_set","seed"]).MAE.mean().unstack("feature_set")
        row=selected[(selected.feature_set==FULL)&(selected.seed==seed)].iloc[0]
        return dict(station=station,target=target,seed=seed,source_task_id=row.task_id,
            best_epoch=int(row.best_epoch),parameter_count=int(row.parameter_count),lookback_steps=int(row.lookback_steps),
            selected_seed_MAE_gain_percent=float(100*(1-means.loc[seed,FULL]/means.loc[seed,ARMS[0]])),
            pooled_five_seed_MAE_gain_percent=float(100*(1-means[FULL].mean()/means[ARMS[0]].mean())),
            positive_seeds=int((means[FULL]<means[ARMS[0]]).sum()),interpretation_scope="one_fixed_seed_model_not_five_seed_attribution")

    def checkpoint_dir(self,station,target,seed):
        return self.output/"checkpoints"/f"{station}_{target}_seed{seed}"

    def checkpoint_valid(self,station,target,seed):
        directory=self.checkpoint_dir(station,target,seed)
        try:
            meta=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
            if meta["case"]!=self.case_meta(station,target,seed) or meta["protocol_hash"]!=self.protocol_hash:
                return False
            return all(digest(directory/n)==v for n,v in meta["files_sha256"].items()) and set(meta["files_sha256"])=={"model.pt","scalers.npz"}
        except (OSError,ValueError,KeyError,TypeError):
            return False

    def audit(self):
        rows=[]
        for case in self.spec["cases"]:
            s,t=case["station"],case["target"]
            seq,split=self.sequence(s,t)
            for seed in self.spec["interpretation_seeds"]:
                meta=self.case_meta(s,t,seed)
                meta.update(checkpoint_available=self.checkpoint_valid(s,t,seed),
                    required_replay_epochs_if_missing=0 if self.checkpoint_valid(s,t,seed) else meta["best_epoch"],
                    n_train=len(split["train"]),n_validation=len(split["validation"]),n_test=len(split["test"]))
                rows.append(meta)
        frame=pd.DataFrame(rows)
        csv_write(self.output/"case_audit.csv",frame)
        print(frame[["station","target","seed","checkpoint_available","required_replay_epochs_if_missing"]].to_string(index=False),flush=True)
        return frame

    def overview(self):
        rows,overall=[],[]
        for station in "ABC":
            for target in FIXED_WINDOWS:
                seq,split=self.sequence(station,target)
                p=self.predictions[(self.predictions.target_station==station)&(self.predictions.target==target)]
                a=p[p.feature_set==ARMS[0]]
                b=p[p.feature_set==FULL]
                paired=a.merge(b,on=["seed","target_start_time","horizon_step"],suffixes=("_water","_weather"),validate="one_to_one")
                if not np.array_equal(paired.y_true_water.to_numpy(dtype=np.float32),paired.y_true_weather.to_numpy(dtype=np.float32)):
                    raise ValueError("Paired observed labels differ.")
                paired["loss_water"]=abs(paired.y_pred_water-paired.y_true_water)
                paired["loss_weather"]=abs(paired.y_pred_weather-paired.y_true_weather)
                # Every stratum uses the same fixed windows, all five seeds and six leads.
                by_origin=paired.groupby(["seed","target_start_time"])[["loss_water","loss_weather"]].mean().reset_index()
                def summarize(mask,variable,band,label,low,high):
                    times=seq.target_start_times[split["test"]][mask]
                    selected=by_origin[by_origin.target_start_time.isin(times)]
                    if selected.empty:
                        return None
                    seed_means=selected.groupby("seed")[["loss_water","loss_weather"]].mean()
                    water,weather=seed_means.loss_water.mean(),seed_means.loss_weather.mean()
                    return dict(station=station,target=target,variable=variable,history_band=band,condition=label,
                        train_cut_lower=low,train_cut_upper=high,n_test_origins=len(times),n_seeds=len(seed_means),
                        MAE_water=water,MAE_weather=weather,MAE_reduction=water-weather,
                        MAE_reduction_percent=100*(1-weather/water) if water>0 else np.nan,
                        positive_seeds=int((seed_means.loss_water>seed_means.loss_weather).sum()),
                        small_stratum=len(times)<self.spec["min_stratum_origins"],
                        interpretation="observed_gain_conditioning_not_feature_importance_or_causality")
                overall.append(summarize(np.ones(len(split["test"]),bool),"overall","all","all",np.nan,np.nan))
                ages=4*np.arange(seq.x.shape[1]-1,-1,-1)
                for band,low,high in AGE_BINS:
                    tids=np.flatnonzero((ages>=low)&(ages<high))
                    if not len(tids):
                        continue
                    for variable,column,rain in [("rain","W_RAIN_4H",True),("air_temperature","W_TEM_MEAN",False),
                        ("humidity","W_RHU_MEAN",False),("pressure","W_PRS_MEAN",False),("wind_speed","W_WIND_SPEED_MAX",False)]:
                        if column not in seq.feature_columns:
                            continue
                        values=seq.x[:,tids,seq.feature_columns.index(column)]
                        values=values.sum(axis=1) if rain else values.mean(axis=1)
                        for label,mask,left,right in condition_masks(values[split["train"]],values[split["test"]],rain):
                            result=summarize(mask,variable,band,label,left,right)
                            if result:
                                rows.append(result)
                print(f"[interpretation/zero-training] {station}/{target}",flush=True)
        csv_write(self.output/"five_seed_gain_overview.csv",pd.DataFrame(overall))
        csv_write(self.output/"gain_by_weather_condition.csv",pd.DataFrame(rows))
        return pd.DataFrame(overall),pd.DataFrame(rows)

    def recover(self,station,target,seed,seq,split):
        """Explicit deterministic replay to the SAVED best epoch, no new search."""
        import torch
        from torch.nn.modules.module import register_module_forward_pre_hook
        captured={}
        def observe(module,inputs):
            if type(module).__name__=="Recurrent" and isinstance(getattr(module,"recurrent",None),torch.nn.LSTM):
                captured["model"]=module
            return None
        meta=self.case_meta(station,target,seed)
        training=dict(self.config["training"],max_epochs=meta["best_epoch"],compute_shap=False)
        print(f"[EXPLICIT REPLAY] {station}/{target}/seed{seed}: up to saved best epoch {meta['best_epoch']}",flush=True)
        handle=register_module_forward_pre_hook(observe)
        try:
            result=fit_torch_predict("lstm",seq.x,seq.y,split,seed,training)
        finally:
            handle.remove()
        expected=self.reference_predictions(station,target,seed,seq,split)
        verify_prediction(result.prediction,expected)
        if result.best_epoch!=meta["best_epoch"] or result.parameter_count!=meta["parameter_count"] or "model" not in captured:
            raise ValueError("Replay did not recover the original best checkpoint/backbone.")
        x_scaler=StandardScaler().fit(seq.x[split["train"]].reshape(-1,seq.x.shape[2]))
        y_scaler=StandardScaler().fit(seq.y[split["train"]])
        directory=self.checkpoint_dir(station,target,seed)
        directory.mkdir(parents=True,exist_ok=True)
        (directory/"metadata.json").unlink(missing_ok=True)
        torch.save({k:v.detach().cpu().clone() for k,v in captured["model"].state_dict().items()},directory/"model.tmp")
        (directory/"model.tmp").replace(directory/"model.pt")
        with (directory/"scalers.tmp").open("wb") as f:
            np.savez(f,x_mean=x_scaler.mean_,x_scale=x_scaler.scale_,y_mean=y_scaler.mean_,y_scale=y_scaler.scale_)
        (directory/"scalers.tmp").replace(directory/"scalers.npz")
        json_write(directory/"metadata.json",dict(case=meta,protocol_hash=self.protocol_hash,
            input_columns=seq.feature_columns,training=self.config["training"],
            replay_epochs=meta["best_epoch"],prediction_max_abs_difference=float(np.max(abs(result.prediction-expected))),
            files_sha256={n:digest(directory/n) for n in ["model.pt","scalers.npz"]}))
        self.replays_this_run+=1

    def explain(self,replay_missing=False,max_replays=1,max_cases=None):
        completed=[]
        missing=[]
        for case in self.spec["cases"]:
            station,target=case["station"],case["target"]
            seq,split=self.sequence(station,target)
            for seed in self.spec["interpretation_seeds"]:
                key=f"{station}_{target}_seed{seed}"
                if max_cases is not None and len(completed)>=max_cases:
                    missing.append(key);continue
                if not self.checkpoint_valid(station,target,seed):
                    if replay_missing and self.replays_this_run<max_replays:
                        try:
                            self.recover(station,target,seed,seq,split)
                        except Exception as exc:
                            self.failures[key]=f"{type(exc).__name__}: {exc}"
                            raise
                    else:
                        missing.append(key);continue
                try:
                    predict=load_predictor(self.checkpoint_dir(station,target,seed),self.config["training"]["device"])
                    x=seq.x[split["test"]]
                    verify_prediction(predict(x),self.reference_predictions(station,target,seed,seq,split))
                    directory=self.output/"attributions"/key
                    directory.mkdir(parents=True,exist_ok=True)
                    stamp=dict(protocol_hash=self.protocol_hash,checkpoint_metadata_sha256=digest(self.checkpoint_dir(station,target,seed)/"metadata.json"))
                    marker=directory/"commit.json"
                    valid=False
                    if marker.exists():
                        old=json.loads(marker.read_text(encoding="utf-8"))
                        valid=all(old.get(k)==v for k,v in stamp.items()) and all((directory/n).exists() and digest(directory/n)==v for n,v in old.get("files_sha256",{}).items()) and set(old.get("files_sha256",{}))=={"permutation_raw.csv","importance.csv"}
                    if not valid:
                        raw,summary=permutation_scores(x,seq.y[split["test"]],predict,seq.feature_columns,
                            self.spec["permutation_repeats"],self.spec["permutation_seed"])
                        for f in [raw,summary]:
                            for k,v in self.case_meta(station,target,seed).items():
                                f[k]=v
                        csv_write(directory/"permutation_raw.csv",raw)
                        csv_write(directory/"importance.csv",summary)
                        json_write(marker,dict(**stamp,files_sha256={n:digest(directory/n) for n in ["permutation_raw.csv","importance.csv"]}))
                    completed.append(key)
                    print(f"[interpretation] {key}: complete",flush=True)
                    del predict
                except Exception as exc:
                    self.failures[key]=f"{type(exc).__name__}: {exc}"
                    with (self.output/"errors.log").open("a",encoding="utf-8") as f:
                        f.write(key+"\n"+traceback.format_exc()+"\n")
                    raise
        frames=[pd.read_csv(self.output/"attributions"/k/"importance.csv") for k in completed]
        if frames:
            csv_write(self.output/"importance_summary.csv",pd.concat(frames,ignore_index=True))
        elif (self.output/"importance_summary.csv").exists():
            (self.output/"importance_summary.csv").unlink()
        return completed,missing

    def run(self,mode="overview",replay_missing=False,max_replays=1,max_cases=None):
        if replay_missing and mode!="explain":
            self._release_run_lock();raise ValueError("Replay is allowed only in explicit explain mode.")
        completed,missing=[],[]
        try:
            audit=self.audit()
            if mode=="overview":
                self.overview()
            elif mode=="explain":
                completed,missing=self.explain(replay_missing,max_replays,max_cases)
                audit["checkpoint_available"]=[self.checkpoint_valid(row.station,row.target,int(row.seed)) for row in audit.itertuples()]
                audit["required_replay_epochs_if_missing"]=np.where(audit.checkpoint_available,0,audit.best_epoch)
                csv_write(self.output/"case_audit.csv",audit)
            json_write(self.output/"run_status.json",dict(protocol=PROTOCOL,mode=mode,
                status="attribution_complete" if mode=="explain" and not missing else (
                    "waiting_for_checkpoints" if mode=="explain" else mode+"_complete"),
                completed_attribution_cases=completed,missing_or_unprocessed_cases=missing,
                replays_this_invocation=self.replays_this_run,automatic_training=False,
                planned_cases=len(audit),interpretation_seeds=self.spec["interpretation_seeds"],
                attribution_scope="fixed_seed_exploratory_model_reliance_not_causal_or_physical_rainfall_lag"))
            return self.output
        except BaseException:
            json_write(self.output/"run_status.json",dict(mode=mode,status="interrupted_or_failed",
                replays_this_invocation=self.replays_this_run,failures=self.failures,automatic_training=False))
            raise
        finally:
            self._release_run_lock()


def verify_prediction(actual,expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.allclose(actual,expected,rtol=1e-5,atol=1e-7):
        raise ValueError("Recovered/cached model does not reproduce original predictions; attribution is blocked.")


def load_predictor(directory,device):
    import torch
    from torch import nn
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    meta=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    training=meta["training"]
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent=nn.LSTM(input_size=len(meta["input_columns"]),hidden_size=training["hidden_size"],
                num_layers=training["num_layers"],dropout=training["dropout"] if training["num_layers"]>1 else 0.,batch_first=True)
            self.head=nn.Linear(training["hidden_size"],6)
        def forward(self,x):
            y,_=self.recurrent(x);return self.head(y[:,-1,:])
    model=Recurrent().to(device)
    model.load_state_dict(torch.load(directory/"model.pt",map_location=device,weights_only=True))
    model.eval()
    with np.load(directory/"scalers.npz",allow_pickle=False) as f:
        xmean,xscale,ymean,yscale=(f[k].copy() for k in ["x_mean","x_scale","y_mean","y_scale"])
    def predict(x):
        # Match StandardScaler's in-place float32 subtract/divide and inverse ops.
        scaled=x.astype(np.float32,copy=True)
        scaled-=xmean;scaled/=xscale
        batches=[]
        with torch.no_grad():
            for begin in range(0,len(x),training["batch_size"]):
                result=model(torch.from_numpy(scaled[begin:begin+training["batch_size"]]).to(device)).cpu().numpy()
                result*=yscale;result+=ymean
                batches.append(result)
        return np.concatenate(batches).astype(np.float32)
    return predict
