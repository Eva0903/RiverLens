"""Post-hoc storm phases, paired 04 gains and raw event response timing. No fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jinjiang.data import STATION_FILES, load_station_raw

ARMS = ["all_stations", "all_stations_weather", "all_stations_rain_lags", "all_stations_weather_rain_lags"]
PHASES = ["raining", "post_0_12h", "post_12_24h", "post_24_48h", "dry_over48h"]
VARIABLES = ["B_FLOW", "B_TURB", "C_TURB", "C_NH3N", "C_TP", "C_CODMN", "C_DO"]


def hourly_rain(path):
    raw = pd.read_excel(path)
    raw["DATETIME"] = pd.to_datetime(raw.DATETIME, errors="coerce")
    raw = raw.dropna(subset=["DATETIME"]).drop_duplicates("DATETIME").set_index("DATETIME").sort_index()
    core = ["PRS", "RHU", "TEM", "WIN_D_INST_MAX", "WIN_S_INST_MAX"]
    for col in core + ["PRE_1H"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    sentinel = raw[core].eq(0).all(axis=1) | raw[core].eq(999999).all(axis=1)
    rain = raw.PRE_1H.mask(sentinel | (raw.PRE_1H < 0) | (raw.PRE_1H > 500))
    return rain.reindex(pd.date_range(rain.index.min(), rain.index.max(), freq="h")).rename("rain_mm")


def catalog_events(rain, threshold=.1, interevent_dry_hours=6):
    wet = rain >= threshold - 1e-12
    group, last_wet, last_missing = 0, None, None
    ids = pd.Series(pd.NA, index=rain.index, dtype="Int64")
    for t, value in rain.items():
        if pd.isna(value):
            last_missing = t
        elif value >= threshold - 1e-12:
            # Two wet hour-end timestamps seven hours apart enclose six dry hourly bins.
            if (last_wet is None or (t-last_wet)/pd.Timedelta(hours=1) >= interevent_dry_hours+1
                    or (last_missing is not None and last_missing > last_wet)):
                group += 1
            ids.loc[t] = group
            last_wet = t
    rows = []
    for event_id, g in rain[wet].groupby(ids[wet]):
        start, end = g.index[0]-pd.Timedelta(hours=1), g.index[-1]
        total = g.sum()
        centers = (g.index-start)/pd.Timedelta(hours=1)-.5
        centroid = start + pd.Timedelta(hours=float(np.sum(centers*g.to_numpy())/total))
        rows.append(dict(event_id=int(event_id), rain_start=start, rain_end=end,
                         rain_peak=g.idxmax(), rain_centroid=centroid, rain_total_mm=total,
                         duration_hours=(end-start)/pd.Timedelta(hours=1),
                         n_wet_hours=len(g), rain_max_1h_mm=g.max()))
    events = pd.DataFrame(rows)
    events["previous_dry_hours"] = (events.rain_start-events.rain_end.shift())/pd.Timedelta(hours=1)
    events["following_dry_hours"] = (events.rain_start.shift(-1)-events.rain_end)/pd.Timedelta(hours=1)
    return events, ids


def phase_labels(rain, event_ids, threshold=.1):
    times = pd.Series(rain.index, index=rain.index)
    wet = rain >= threshold - 1e-12
    last_wet = times.where(wet).ffill()
    last_missing = times.where(rain.isna()).ffill()
    latest_event = event_ids.ffill()
    age = (times-last_wet).dt.total_seconds()/3600
    rain4 = rain.rolling(4, min_periods=4).sum()
    observed48 = rain.notna().rolling(48, min_periods=48).sum().eq(48)
    uninterrupted_after_rain = last_wet.notna() & (last_missing.isna() | (last_missing < last_wet))
    phase = pd.Series("unknown", index=rain.index)
    phase.loc[rain4 >= threshold-1e-12] = "raining"
    no_current_rain = rain4.notna() & (rain4 < threshold-1e-12)
    for low, high, label in [(0,12,"post_0_12h"),(12,24,"post_12_24h"),(24,48,"post_24_48h")]:
        phase.loc[no_current_rain & uninterrupted_after_rain & age.gt(low) & age.le(high)] = label
    phase.loc[no_current_rain & observed48 & (last_wet.isna() | age.gt(48))] = "dry_over48h"
    out = pd.DataFrame(dict(phase=phase, rainfall_4h_mm=rain4, hours_since_last_wet_hour=age,
                            last_wet_hour=last_wet, last_missing_hour=last_missing,
                            event_id=latest_event))
    # IDs link rain-associated stages to events; dry controls use independent calendar blocks.
    out.loc[~out.phase.isin(PHASES[:4]), "event_id"] = pd.NA
    return out


def basic_metrics(frame):
    e = frame.y_pred.to_numpy(float)-frame.y_true.to_numpy(float)
    y = frame.y_true.to_numpy(float)
    den = np.sum((y-y.mean())**2)
    return dict(MAE=np.mean(abs(e)), RMSE=np.sqrt(np.mean(e**2)),
                R2=1-np.sum(e**2)/den if len(y)>1 and den>0 else np.nan)


def cluster_gain_ci(paired, bootstrap=2000):
    # Seed losses have already been averaged; fits are fixed, not independent observations.
    if paired.phase.iloc[0] == "dry_over48h":
        cluster = ((paired.valid_time-paired.valid_time.min().normalize()).dt.total_seconds()/(72*3600)).astype(int)
        resampling = "observed_nonempty_72h_calendar_blocks"
    else:
        cluster = paired.event_id.astype(int)
        resampling = "rainfall_events"
    grouped=paired.assign(cluster=cluster).groupby(["cluster","horizon_hours"]).agg(
        base_error=("base_error","sum"),aug_error=("aug_error","sum"),count=("base_error","size"))
    cluster_ids=grouped.index.get_level_values("cluster").unique()
    hours=sorted(paired.horizon_hours.unique())
    arrays={key:grouped[key].unstack("horizon_hours").reindex(index=cluster_ids,columns=hours).fillna(0).to_numpy()
            for key in ["base_error","aug_error","count"]}
    n = len(cluster_ids)
    if n < 10:
        return dict(ci95_lower=np.nan, ci95_upper=np.nan, n_clusters=n,
                    resampling=resampling, ci_status="fewer_than_10_clusters")
    rng = np.random.default_rng(2026)
    ids = rng.integers(0,n,size=(bootstrap,n))
    counts=arrays["count"][ids].sum(axis=1)
    valid=(counts>0).all(axis=1)
    base=(arrays["base_error"][ids].sum(axis=1)[valid]/counts[valid]).mean(axis=1)
    aug=(arrays["aug_error"][ids].sum(axis=1)[valid]/counts[valid]).mean(axis=1)
    valid_base=base>0
    low, high = np.quantile(100*(base[valid_base]-aug[valid_base])/base[valid_base],[.025,.975])
    return dict(ci95_lower=low, ci95_upper=high, n_clusters=n,
                resampling=resampling, ci_status="conditional_on_fixed_fits_and_split")


def performance_by_phase(pred, labels, output):
    p = pred.merge(labels.reset_index(names="valid_time"), on="valid_time", how="left", validate="many_to_one")
    p["phase"] = p.phase.fillna("unknown")
    p["computed_absolute_error"] = abs(p.y_pred-p.y_true)
    obs = p[["target","target_start_time","valid_time","horizon_hours","phase","event_id"]].drop_duplicates()
    obs.to_csv(output/"prediction_phase_labels.csv",index=False,encoding="utf-8-sig")
    counts = obs.groupby(["target","phase"]).agg(n_origin_lead_pairs=("valid_time","size"),
        n_unique_target_times=("valid_time","nunique"), n_origins=("target_start_time","nunique"),
        n_rainfall_events=("event_id","nunique")).reset_index()
    counts.to_csv(output/"phase_sample_counts.csv",index=False,encoding="utf-8-sig")
    overlap=[]
    for target,g in obs.groupby("target"):
        rainy=set(g.loc[g.phase=="raining","event_id"].dropna().astype(int))
        for phase in PHASES[1:4]:
            after=set(g.loc[g.phase==phase,"event_id"].dropna().astype(int))
            shared=sorted(rainy & after)
            overlap.append(dict(target=target,post_phase=phase,n_raining_events=len(rainy),
                                n_post_events=len(after),n_shared_events=len(shared),
                                shared_event_ids="|".join(map(str,shared))))
    pd.DataFrame(overlap).to_csv(output/"rain_post_event_overlap.csv",index=False,encoding="utf-8-sig")
    rows=[]
    for (target,phase,arm,seed),g in p[p.phase.isin(PHASES)].groupby(["target","phase","feature_set","seed"]):
        rows.append(dict(target=target,phase=phase,feature_set=arm,seed=seed,horizon_hours=0,
                         n_predictions=len(g),**basic_metrics(g)))
        for hour,gg in g.groupby("horizon_hours"):
            rows.append(dict(target=target,phase=phase,feature_set=arm,seed=seed,horizon_hours=hour,
                             n_predictions=len(gg),**basic_metrics(gg)))
    scores=pd.DataFrame(rows)
    scores[scores.horizon_hours==0].to_csv(output/"pooled_phase_metrics_by_seed.csv",index=False,encoding="utf-8-sig")
    # Primary aggregation matches 04: compute each lead first, then weight six leads equally.
    leads=scores[scores.horizon_hours>0]
    equal=leads.groupby(["target","phase","feature_set","seed"]).agg(
        MAE=("MAE","mean"),RMSE=("RMSE","mean"),R2=("R2","mean"),
        n_predictions=("n_predictions","sum"),n_horizons=("horizon_hours","nunique")).reset_index()
    equal["horizon_hours"]=0
    scores=pd.concat([leads,equal],ignore_index=True)
    scores.to_csv(output/"phase_metrics_by_seed.csv",index=False,encoding="utf-8-sig")
    keys=["target","phase","seed","horizon_hours"]
    reference=scores[scores.feature_set==ARMS[0]]
    gains=[]
    for arm in ARMS[1:]:
        augmented=scores[scores.feature_set==arm]
        pairs=reference.merge(augmented,on=keys,suffixes=("_base","_aug"),validate="one_to_one")
        for (target,phase,hour),g in pairs.groupby(["target","phase","horizon_hours"]):
            if not g.n_predictions_base.equals(g.n_predictions_aug):raise ValueError("Unequal phase samples")
            base,aug=g.MAE_base.mean(),g.MAE_aug.mean()
            gains.append(dict(target=target,phase=phase,horizon_hours=hour,augmented_feature_set=arm,
                              n_seeds=len(g),n_predictions_per_seed=int(g.n_predictions_base.iloc[0]),
                              reference_MAE=base,augmented_MAE=aug,MAE_reduction=base-aug,
                              MAE_reduction_percent=100*(base-aug)/base if base>0 else np.nan,
                              n_seeds_improved=int((g.MAE_base>g.MAE_aug).sum()),
                              reference_RMSE=g.RMSE_base.mean(),augmented_RMSE=g.RMSE_aug.mean()))
    gains=pd.DataFrame(gains)
    gains.to_csv(output/"phase_gains_all_horizons.csv",index=False,encoding="utf-8-sig")
    overall=gains[gains.horizon_hours==0].copy()
    cis=[]
    prediction_keys=["target","target_start_time","valid_time","horizon_hours"]
    base=p[p.feature_set==ARMS[0]].groupby(prediction_keys).agg(base_error=("computed_absolute_error","mean"),
        phase=("phase","first"),event_id=("event_id","first")).reset_index()
    for arm in ARMS[1:]:
        aug=p[p.feature_set==arm].groupby(prediction_keys).computed_absolute_error.mean().rename("aug_error").reset_index()
        paired=base.merge(aug,on=prediction_keys,validate="one_to_one")
        for (target,phase),g in paired[paired.phase.isin(PHASES)].groupby(["target","phase"]):
            cis.append(dict(target=target,phase=phase,augmented_feature_set=arm,**cluster_gain_ci(g)))
    overall=overall.merge(pd.DataFrame(cis),on=["target","phase","augmented_feature_set"],validate="one_to_one")
    overall=overall.merge(counts,on=["target","phase"],validate="many_to_one")
    overall.to_csv(output/"phase_gain_summary.csv",index=False,encoding="utf-8-sig")
    return p,overall


def raw_event_responses(raw, rain, events, output):
    event_rows,response_rows=[],[]
    for row in events.itertuples():
        pre=rain.reindex(pd.date_range(row.rain_start-pd.Timedelta(hours=11),row.rain_start,freq="h"))
        after=rain.reindex(pd.date_range(row.rain_end+pd.Timedelta(hours=1),row.rain_end+pd.Timedelta(hours=48),freq="h"))
        isolated=(row.rain_total_mm>=1 and row.duration_hours<=24 and row.previous_dry_hours>=12
                  and row.following_dry_hours>=48 and pre.notna().all() and after.notna().all()
                  and pre.lt(.1-1e-12).all() and after.lt(.1-1e-12).all())
        er=dict(event_id=row.event_id,isolated_weather_eligible=bool(isolated))
        common=bool(isolated)
        for col in VARIABLES:
            before=raw.loc[(raw.index>=row.rain_start-pd.Timedelta(hours=12))&(raw.index<row.rain_start),col].dropna()
            window=raw.loc[(raw.index>=row.rain_start)&(raw.index<=row.rain_end+pd.Timedelta(hours=48)),col]
            valid=window.dropna()
            eligible=bool(isolated and len(before)>=2 and len(window)>0 and len(valid)/len(window)>=.75)
            er[col+"_eligible"]=eligible
            common=common and eligible
            if not eligible:continue
            baseline=before.median()
            if col in ["B_FLOW","B_TURB","C_TURB"]:
                peak=valid.idxmax();definition="maximum_observed_value"
            else:
                peak=(valid-baseline).abs().idxmax();definition="maximum_absolute_departure_from_pre_event_median"
            value=valid.loc[peak]
            rr=dict(event_id=row.event_id,variable=col,baseline=baseline,peak_value=value,
                    peak_change=value-baseline,peak_time=peak,peak_definition=definition,
                    hours_from_rain_start=(peak-row.rain_start)/pd.Timedelta(hours=1),
                    hours_from_rain_peak=(peak-row.rain_peak)/pd.Timedelta(hours=1),
                    hours_from_rain_centroid=(peak-row.rain_centroid)/pd.Timedelta(hours=1),
                    hours_from_rain_end=(peak-row.rain_end)/pd.Timedelta(hours=1),
                    n_baseline=len(before),coverage=len(valid)/len(window),
                    boundary_peak=bool(peak==valid.index[0] or peak==valid.index[-1]))
            response_rows.append(rr)
        er["common_all_variables_eligible"]=common
        event_rows.append(er)
    eligibility=pd.DataFrame(event_rows)
    catalog=events.merge(eligibility,on="event_id",validate="one_to_one")
    catalog.to_csv(output/"rain_event_catalog.csv",index=False,encoding="utf-8-sig")
    responses=pd.DataFrame(response_rows)
    if not responses.empty:
        flow=responses[responses.variable=="B_FLOW"][["event_id","peak_time","peak_change"]].rename(
            columns=dict(peak_time="B_flow_peak_time",peak_change="B_flow_peak_change"))
        responses=responses.merge(flow,on="event_id",how="left",validate="many_to_one")
        responses["hours_from_B_flow_peak"]=(responses.peak_time-responses.B_flow_peak_time)/pd.Timedelta(hours=1)
    responses.to_csv(output/"event_response_peaks.csv",index=False,encoding="utf-8-sig")
    summaries=[]
    for variable,g in responses.groupby("variable"):
        summaries.append(dict(variable=variable,n_events=len(g),
            n_positive_changes=int(g.peak_change.gt(0).sum()),n_negative_changes=int(g.peak_change.lt(0).sum()),
            median_hours_from_rain_centroid=g.hours_from_rain_centroid.median(),
            q25_hours_from_rain_centroid=g.hours_from_rain_centroid.quantile(.25),
            q75_hours_from_rain_centroid=g.hours_from_rain_centroid.quantile(.75),
            median_hours_from_rain_end=g.hours_from_rain_end.median(),
            n_peak_after_rain_end=int(g.hours_from_rain_end.gt(0).sum()),
            median_hours_from_B_flow_peak=g.hours_from_B_flow_peak.median(),
            n_boundary_peaks=int(g.boundary_peak.sum())))
    pd.DataFrame(summaries).to_csv(output/"event_response_timing_summary.csv",index=False,encoding="utf-8-sig")
    cases=catalog[catalog.common_all_variables_eligible].sort_values(["rain_total_mm","rain_start"],ascending=[False,True]).head(3)
    cases.to_csv(output/"largest_complete_physical_events.csv",index=False,encoding="utf-8-sig")
    return catalog,responses,cases


def case_prediction_gains(pred,cases,output):
    rows=[]
    for event in cases.itertuples():
        inside=pred[(pred.valid_time>=event.rain_start)&(pred.valid_time<=event.rain_end+pd.Timedelta(hours=48))]
        for target,g in inside.groupby("target"):
            ref=g[g.feature_set==ARMS[0]]
            aug=g[g.feature_set==ARMS[-1]]
            if ref.empty or aug.empty:continue
            keys=["seed","horizon_hours"]
            a=ref.assign(ae=abs(ref.y_pred-ref.y_true)).groupby(keys).ae.mean()
            b=aug.assign(ae=abs(aug.y_pred-aug.y_true)).groupby(keys).ae.mean()
            if not a.index.equals(b.index):raise ValueError("Unpaired case sample")
            per_seed_a=a.groupby("seed").mean();per_seed_b=b.groupby("seed").mean()
            rows.append(dict(event_id=event.event_id,target=target,
                n_unique_target_times=ref.valid_time.nunique(),
                n_forecast_origins=ref.target_start_time.nunique(),
                n_origin_lead_pairs_per_seed=int(len(ref)/ref.seed.nunique()),
                n_leads_available=ref.horizon_hours.nunique(),
                reference_MAE=per_seed_a.mean(),augmented_MAE=per_seed_b.mean(),
                MAE_reduction_percent=100*(1-per_seed_b.mean()/per_seed_a.mean()),
                n_seeds_improved=int((per_seed_b<per_seed_a).sum())))
    pd.DataFrame(rows).to_csv(output/"case_prediction_gains.csv",index=False,encoding="utf-8-sig")


def analyze(results_dir,data_dir,output):
    results_dir,data_dir,output=map(Path,(results_dir,data_dir,output))
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(training_runs_added=0,phase_clock="water-quality target valid_time, not forecast issue time",
        rain_threshold_mm=.1,interevent_rule="six consecutive observed dry hourly bins; unknown weather splits events",
        phases=PHASES,phase_raining="observed rainfall in preceding four hours >=0.1mm",
        phase_post="no rain in preceding 4h; hours since latest wet hourly interval ended; no intervening missing rainfall",
        dry_control="more than 48h since latest wet hour and 48 complete observed hourly bins",
        physical_event_screen=dict(minimum_total_rain_mm=1,maximum_duration_h=24,
            minimum_prior_dry_h=12,minimum_subsequent_dry_h=48,baseline_h=12,response_after_event_end_h=48,
            minimum_baseline_observations=2,minimum_response_observation_fraction=.75),
        case_selection="three largest rain totals among isolated all-seven-variable events with at least 6 unique NH3N or TP test target times; selected by coverage then rain, never gain or lag; largest physical-only events archived separately",
        overall_MAE="primary: equal mean of six lead-specific MAEs and five fixed seeds, matching 04; pooled metrics retained as descriptive sensitivity",
        uncertainty="2000 cluster bootstrap replicates: rainfall events for rain/post phases, observed 72h blocks for dry; minimum 10 clusters; conditional on fixed split/fits",
        response_peaks="raw observations only; B flow/turbidity maxima, C constituent maximum absolute departure; descriptive extrema, not identified causal travel times",
        inference="post-hoc exploratory; 4h water-quality resolution; phases use retrospective rainfall only for evaluation")
    (output/"event_analysis_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    fingerprints=json.loads((results_dir/"data_fingerprints.json").read_text())
    for name,sha in fingerprints.items():
        if hashlib.sha256((data_dir/name).read_bytes()).hexdigest()!=sha:raise ValueError("Input data changed: "+name)
    rain=hourly_rain(data_dir/"weather_hourly.xlsx")
    events,ids=catalog_events(rain)
    labels=phase_labels(rain,ids)
    labels.reset_index(names="timestamp").to_csv(output/"hourly_event_phase_labels.csv",index=False,encoding="utf-8-sig")
    pred=pd.read_csv(results_dir/"predictions.csv",parse_dates=["valid_time","target_start_time"])
    pred=pred[(pred.model=="lstm")&pred.feature_set.isin(ARMS)].copy()
    k=["target","seed","target_start_time","horizon_hours","feature_set"]
    if pred.duplicated(k).any():raise ValueError("Duplicate prediction keys")
    aligned=pred.groupby(k[:-1]).agg(n_arms=("feature_set","nunique"),n_labels=("y_true","nunique"))
    if not aligned.n_arms.eq(4).all() or not aligned.n_labels.eq(1).all():raise ValueError("Unpaired input arms")
    p,summary=performance_by_phase(pred,labels,output)
    frames=[]
    for station in ["B","C"]:
        f,_,_=load_station_raw(data_dir/STATION_FILES[station],station,"4h","2h15min")
        frames.append(f)
    raw=pd.concat(frames,axis=1)
    raw.reset_index(names="timestamp").to_csv(output/"raw_BC_event_inputs.csv",index=False,encoding="utf-8-sig")
    catalog,response,_=raw_event_responses(raw,rain,events,output)
    candidates=catalog[catalog.common_all_variables_eligible].copy()
    for target in ["DO","NH3N","CODMN","TP"]:
        candidates[target+"_test_times"]=[pred.loc[(pred.target==target)&(pred.valid_time>=r.rain_start)
            &(pred.valid_time<=r.rain_end+pd.Timedelta(hours=48)),"valid_time"].nunique()
            for r in candidates.itertuples()]
    candidates["prediction_match_eligible"]=(candidates.NH3N_test_times>=6)|(candidates.TP_test_times>=6)
    candidates.to_csv(output/"case_coverage_audit.csv",index=False,encoding="utf-8-sig")
    matched=candidates[candidates.prediction_match_eligible].sort_values(["rain_total_mm","rain_start"],ascending=[False,True])
    cases=matched.head(3)
    cases.to_csv(output/"selected_case_events.csv",index=False,encoding="utf-8-sig")
    case_prediction_gains(pred,matched,output)
    rain.rename_axis("timestamp").reset_index().to_csv(output/"hourly_observed_rain.csv",index=False,encoding="utf-8-sig")
    print("ADDED TRAINING RUNS: 0")
    print("Events",len(catalog),"isolated weather eligible",int(catalog.isolated_weather_eligible.sum()),
          "all-variable eligible",int(catalog.common_all_variables_eligible.sum()))
    print(summary[summary.augmented_feature_set==ARMS[-1]][[
        "target","phase","MAE_reduction_percent","n_seeds_improved","n_unique_target_times","n_rainfall_events","ci95_lower","ci95_upper"]].to_string(index=False))
    print(pd.read_csv(output/"event_response_timing_summary.csv").to_string(index=False))
    print(cases[["event_id","rain_start","rain_end","rain_total_mm"]].to_string(index=False))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",default="results/meteorology_random_gpu/04_meteorology")
    parser.add_argument("--data-dir",default="data/raw")
    parser.add_argument("--output",default="results/meteorology_random_gpu/04_meteorology/event_phase_analysis")
    a=parser.parse_args()
    print(analyze(a.results_dir,a.data_dir,a.output))
