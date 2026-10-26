"""Rain/dry stratification of completed 04 predictions; no fitting or GPU use."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jinjiang.data import load_weather_raw

ARMS = ["all_stations", "all_stations_weather", "all_stations_rain_lags", "all_stations_weather_rain_lags"]
RULES = {
    "issue_previous24h": "Primary: observed rainfall in (issue_time-24h, issue_time], known at issue",
    "target_4h": "Secondary: observed rainfall in (valid_time-4h, valid_time], retrospective evaluation only",
}


def classify(rain, threshold=.1):
    values = np.asarray(rain, dtype=float)
    return np.where(~np.isfinite(values), "unknown", np.where(values >= threshold - 1e-12, "rain", "dry"))


def metrics(y, pred):
    error = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    denominator = np.sum((y - y.mean()) ** 2)
    mse = np.mean(error ** 2)
    return dict(MAE=np.mean(abs(error)), RMSE=np.sqrt(mse), MSE=mse,
                R2=1-np.sum(error**2)/denominator if len(y)>1 and denominator>0 else np.nan)


def analyze(results_dir, weather_file, output, threshold=.1):
    results_dir, weather_file, output = map(Path, (results_dir, weather_file, output))
    if threshold <= 0:
        raise ValueError("Rain threshold must be positive.")
    config = json.loads((results_dir / "run_config.json").read_text(encoding="utf-8-sig"))
    status = json.loads((results_dir / "run_status.json").read_text(encoding="utf-8-sig"))
    if status.get("run_status") != "complete":
        raise ValueError("Use a completed 04 result, so all arms and seeds are paired.")
    expected_sha = json.loads((results_dir / "data_fingerprints.json").read_text()).get(weather_file.name)
    weather_sha = hashlib.sha256(weather_file.read_bytes()).hexdigest()
    if expected_sha != weather_sha:
        raise ValueError("Weather file hash does not match the completed 04 data.")
    pred = pd.read_csv(results_dir / "predictions.csv", parse_dates=["target_start_time", "valid_time"])
    pred = pred[(pred.model == "lstm") & pred.feature_set.isin(ARMS)].copy()
    keys = ["target", "seed", "horizon_hours", "target_start_time", "feature_set"]
    if pred.empty or pred.duplicated(keys).any():
        raise ValueError("LSTM prediction keys must be present and unique.")
    paired = pred.groupby(keys[:-1]).agg(n_arms=("feature_set", "nunique"), n_labels=("y_true", "nunique"))
    if not paired.n_arms.eq(4).all() or not paired.n_labels.eq(1).all():
        raise ValueError("All input arms must share the same scored targets.")
    expected_seeds = set(config["training"]["seeds"])
    for _, group in pred.groupby(["target", "feature_set", "horizon_hours"]):
        if set(group.seed) != expected_seeds:
            raise ValueError("Missing seed in an input arm.")
    pred["issue_time"] = pred.target_start_time - pd.Timedelta(hours=4)
    if not ((pred.valid_time-pred.issue_time).dt.total_seconds()/3600 == pred.horizon_hours).all():
        raise ValueError("Prediction time mapping is inconsistent.")
    weather, _ = load_weather_raw(weather_file, pred.issue_time.min()-pd.Timedelta(hours=28), pred.valid_time.max())
    rain4 = weather.W_RAIN_4H
    rain24 = rain4.rolling(6, min_periods=6).sum()
    pred["issue_previous24h_mm"] = pred.issue_time.map(rain24)
    pred["target_4h_mm"] = pred.valid_time.map(rain4)
    for rule in RULES:
        pred[rule] = classify(pred[rule+"_mm"], threshold)

    output.mkdir(parents=True, exist_ok=True)
    labels = pred[["target", "target_start_time", "issue_time", "valid_time", "horizon_hours",
                   "issue_previous24h_mm", "issue_previous24h", "target_4h_mm", "target_4h"]].drop_duplicates()
    labels.to_csv(output/"rain_labels.csv", index=False, encoding="utf-8-sig")
    count_rows, score_rows = [], []
    for rule in RULES:
        for (target, category), group in labels.groupby(["target", rule]):
            count_rows.append(dict(rule=rule, target=target, stratum=category,
                n_origins=group.target_start_time.nunique(), n_origin_lead_pairs=len(group),
                n_unique_target_times=group.valid_time.nunique(),
                n_issue_calendar_days=group.issue_time.dt.normalize().nunique()))
        for (target, category, arm, seed, hour), group in pred.groupby(["target", rule, "feature_set", "seed", "horizon_hours"]):
            if category == "unknown":
                continue
            score_rows.append(dict(rule=rule, target=target, stratum=category, feature_set=arm,
                seed=seed, horizon_hours=hour, n_predictions=len(group), n_horizons=1,
                **metrics(group.y_true, group.y_pred)))
    counts = pd.DataFrame(count_rows)
    scores = pd.DataFrame(score_rows)
    # Overall result is an equal mean of six lead-specific scores, matching 04.
    overall = scores.groupby(["rule", "target", "stratum", "feature_set", "seed"]).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), MSE=("MSE", "mean"), R2=("R2", "mean"),
        n_predictions=("n_predictions", "sum"), n_horizons=("horizon_hours", "nunique")).reset_index()
    overall["horizon_hours"] = 0
    scores = pd.concat([scores, overall], ignore_index=True)
    counts.to_csv(output/"stratum_counts.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(output/"metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary_keys = ["rule", "target", "stratum", "feature_set", "horizon_hours"]
    summary = scores.groupby(summary_keys)[["MAE", "RMSE", "MSE", "R2"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(output/"metrics_mean_sd.csv", index=False, encoding="utf-8-sig")

    gain_rows = []
    for (rule, target, category, hour), group in scores.groupby(["rule", "target", "stratum", "horizon_hours"]):
        reference = group[group.feature_set == "all_stations"].set_index("seed")
        for arm in ARMS[1:]:
            augmented = group[group.feature_set == arm].set_index("seed")
            if set(reference.index) != expected_seeds or set(augmented.index) != expected_seeds:
                raise ValueError("A stratum comparison has missing seed pairs.")
            reference, augmented = reference.sort_index(), augmented.sort_index()
            if not reference.n_predictions.equals(augmented.n_predictions):
                raise ValueError("A stratum comparison has unequal scored sample counts.")
            delta = reference.MAE - augmented.MAE
            baseline = reference.MAE.mean()
            gain_rows.append(dict(rule=rule, target=target, stratum=category, horizon_hours=hour,
                augmented_feature_set=arm, n_seeds=len(delta),
                n_predictions_per_seed=int(reference.n_predictions.iloc[0]),
                n_horizons=int(reference.n_horizons.min()),
                reference_MAE=baseline, augmented_MAE=augmented.MAE.mean(),
                MAE_reduction=delta.mean(), MAE_reduction_percent=100*delta.mean()/baseline if baseline>0 else np.nan,
                n_seeds_improved=int((delta>0).sum()),
                paired_MAE_reduction_seed_sd=delta.std(),
                reference_RMSE=reference.RMSE.mean(), augmented_RMSE=augmented.RMSE.mean(),
                reference_R2=reference.R2.mean(), augmented_R2=augmented.R2.mean()))
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(output/"paired_gains.csv", index=False, encoding="utf-8-sig")
    gains[gains.horizon_hours==0].to_csv(output/"gain_summary.csv", index=False, encoding="utf-8-sig")
    main = gains[(gains.rule=="issue_previous24h") & (gains.horizon_hours==0)]
    wet = main[main.stratum=="rain"].set_index(["target", "augmented_feature_set"])
    dry = main[main.stratum=="dry"].set_index(["target", "augmented_feature_set"])
    contrast = pd.DataFrame(dict(
        rain_MAE_reduction_percent=wet.MAE_reduction_percent,
        dry_MAE_reduction_percent=dry.MAE_reduction_percent,
        rain_minus_dry_reduction_percentage_points=wet.MAE_reduction_percent-dry.MAE_reduction_percent,
        rain_MAE_reduction=wet.MAE_reduction, dry_MAE_reduction=dry.MAE_reduction,
        rain_minus_dry_absolute_reduction=wet.MAE_reduction-dry.MAE_reduction,
    )).reset_index()
    contrast.to_csv(output/"rain_vs_dry_gain_contrast.csv", index=False, encoding="utf-8-sig")
    protocol = dict(training_runs_added=0, source_results=str(results_dir), rainfall_threshold_mm=threshold,
        rules=RULES, primary_rule="issue_previous24h", unknown_rainfall="excluded, never treated as dry",
        horizon_zero="equal mean of available per-lead metrics; n_horizons records coverage",
        seed_aggregation="five fits share each observed sample; seeds are not new rainfall observations",
        provenance="post-hoc exploratory stratification after overall 04 results; threshold not tuned to gains",
        rain_group_interpretation="recent rainfall, not necessarily rain at issue time",
        secondary_future_rain="only used for retrospective grouping, never supplied to the models",
        statistical_scope="descriptive paired comparisons on the fixed random holdout; no significance or causal claim",
        hashes=dict(weather_hourly_sha256=weather_sha,
            predictions_sha256=hashlib.sha256((results_dir/'predictions.csv').read_bytes()).hexdigest()))
    (output/"analysis_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    print("Added training runs: 0")
    print(counts[counts.rule=="issue_previous24h"].to_string(index=False))
    print(contrast.to_string(index=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/meteorology_random_gpu/04_meteorology")
    parser.add_argument("--weather", default="data/raw/weather_hourly.xlsx")
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold-mm", type=float, default=.1)
    args = parser.parse_args()
    output = args.output or str(Path(args.results_dir)/"rain_dry_analysis")
    print(analyze(args.results_dir, args.weather, output, args.threshold_mm))


if __name__ == "__main__":
    main()
