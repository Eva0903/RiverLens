from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def calculate_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict[str, object],
    mase_scale: float | None = None,
) -> list[dict[str, object]]:
    rows = []
    for horizon_index in range(y_true.shape[1]):
        observed = y_true[:, horizon_index]
        predicted = y_pred[:, horizon_index]
        error = predicted - observed
        mse = float(np.mean(error**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(error)))
        observed_sd = float(np.std(observed))
        rows.append({
            **metadata,
            "horizon_step": horizon_index + 1,
            "horizon_hours": 4 * (horizon_index + 1),
            "n_test": len(observed),
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "nRMSE_test_sd": float(rmse / observed_sd) if observed_sd > 0 else np.nan,
            "MASE": float(mae / mase_scale) if mase_scale is not None and mase_scale > 0 else np.nan,
            "MAPE_percent": float(np.mean(np.abs(error) / np.maximum(np.abs(observed), 1e-6)) * 100),
            "sMAPE_percent": float(
                np.mean(2 * np.abs(error) / np.maximum(np.abs(observed) + np.abs(predicted), 1e-6)) * 100
            ),
            "R2": float(r2_score(observed, predicted)) if len(observed) > 1 else np.nan,
        })
    return rows


def prediction_frame(
    target_start_times: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict[str, object],
) -> pd.DataFrame:
    frames = []
    for horizon_index in range(y_true.shape[1]):
        frames.append(pd.DataFrame({
            **{key: [value] * len(target_start_times) for key, value in metadata.items()},
            "target_start_time": target_start_times,
            "valid_time": target_start_times + pd.to_timedelta(4 * horizon_index, unit="h"),
            "horizon_step": horizon_index + 1,
            "horizon_hours": 4 * (horizon_index + 1),
            "y_true": y_true[:, horizon_index],
            "y_pred": y_pred[:, horizon_index],
            "error": y_pred[:, horizon_index] - y_true[:, horizon_index],
            "absolute_error": np.abs(y_pred[:, horizon_index] - y_true[:, horizon_index]),
            "squared_error": (y_pred[:, horizon_index] - y_true[:, horizon_index]) ** 2,
        }))
    return pd.concat(frames, ignore_index=True)


def mean_sd_table(metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        column for column in [
            "experiment", "subexperiment", "fold", "split_scheme", "target_station", "target",
            "feature_set", "model", "lookback_steps", "lookback_days", "horizon_step", "horizon_hours"
        ] if column in metrics.columns
    ]
    metric_columns = [
        column for column in
        ["MSE", "RMSE", "MAE", "nRMSE_test_sd", "MASE", "MAPE_percent", "sMAPE_percent", "R2"]
        if column in metrics.columns
    ]
    grouped = metrics.groupby(id_columns, dropna=False)[metric_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in grouped.columns
    ]
    return grouped


def skill_scores_table(metrics: pd.DataFrame, baselines: tuple[str, ...] = ("persistence", "seasonal_naive")) -> pd.DataFrame:
    """Calculate error-reduction skill on the exact same seed/time support."""
    if {"task_id", "horizon_step"}.issubset(metrics.columns):
        metrics = metrics.drop_duplicates(["task_id", "horizon_step"], keep="last")
    keys = [
        column for column in [
            "experiment", "subexperiment", "fold", "split_scheme", "target_station", "target",
            "seed", "lookback_steps", "lookback_days", "forecast_horizon_steps",
            "horizon_step", "horizon_hours",
        ] if column in metrics.columns
    ]
    method_columns = keys + ["feature_set", "model", "MSE", "RMSE", "MAE", "R2"]
    result = metrics[method_columns].copy()
    mae_denominators = []
    mse_denominators = []
    for baseline in baselines:
        reference = metrics[(metrics["feature_set"] == baseline) & (metrics["model"] == baseline)]
        if reference.empty:
            continue
        reference = reference[keys + ["MAE", "MSE"]].drop_duplicates(keys, keep="last").rename(columns={
            "MAE": f"MAE_{baseline}",
            "MSE": f"MSE_{baseline}",
        })
        result = result.merge(reference, on=keys, how="left", validate="many_to_one")
        result[f"MAE_skill_vs_{baseline}"] = 1 - result["MAE"] / result[f"MAE_{baseline}"]
        result[f"MSE_skill_vs_{baseline}"] = 1 - result["MSE"] / result[f"MSE_{baseline}"]
        mae_denominators.append(f"MAE_{baseline}")
        mse_denominators.append(f"MSE_{baseline}")
    if mae_denominators:
        result["MAE_best_naive"] = result[mae_denominators].min(axis=1)
        result["MSE_best_naive"] = result[mse_denominators].min(axis=1)
        result["MAE_skill_vs_best_naive"] = 1 - result["MAE"] / result["MAE_best_naive"]
        result["MSE_skill_vs_best_naive"] = 1 - result["MSE"] / result["MSE_best_naive"]
    return result


def forecastability_envelope(
    metrics: pd.DataFrame,
    skills: pd.DataFrame,
    model: str,
    r2_threshold: float = 0.0,
    skill_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics[(metrics["experiment"] == "03_window_horizon") & (metrics["model"] == model)]
    grid = selected.groupby(
        ["target_station", "target", "lookback_steps", "lookback_days", "horizon_step", "horizon_hours"],
        as_index=False,
    ).agg(
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
    )
    selected_skills = skills[(skills["experiment"] == "03_window_horizon") & (skills["model"] == model)]
    skill_grid = selected_skills.groupby(
        ["target_station", "target", "lookback_steps", "lookback_days", "horizon_step", "horizon_hours"],
        as_index=False,
    ).agg(
        MAE_skill_vs_best_naive_mean=("MAE_skill_vs_best_naive", "mean"),
        MAE_skill_vs_best_naive_sd=("MAE_skill_vs_best_naive", "std"),
    )
    grid = grid.merge(
        skill_grid,
        on=["target_station", "target", "lookback_steps", "lookback_days", "horizon_step", "horizon_hours"],
        how="left",
    )
    if grid.empty:
        return grid, grid
    grid["high_skill"] = (
        (grid["R2_mean"] >= r2_threshold)
        & (grid["MAE_skill_vs_best_naive_mean"] >= skill_threshold)
    )
    rows = []
    for keys, group in grid.groupby(["target_station", "target", "lookback_steps", "lookback_days"]):
        group = group.sort_values("horizon_hours")
        reliable = []
        for row in group.itertuples(index=False):
            if not bool(row.high_skill):
                break
            reliable.append(int(row.horizon_hours))
        rows.append({
            "target_station": keys[0],
            "target": keys[1],
            "lookback_steps": keys[2],
            "lookback_days": keys[3],
            "mean_R2_across_horizons": group["R2_mean"].mean(),
            "sd_R2_across_horizons": group["R2_mean"].std(),
            "mean_MAE_skill_vs_best_naive": group["MAE_skill_vs_best_naive_mean"].mean(),
            "high_skill_coverage": group["high_skill"].mean(),
            "reliable_lead_hours": max(reliable) if reliable else 0,
            "R2_threshold": r2_threshold,
            "MAE_skill_threshold": skill_threshold,
        })
    envelope = pd.DataFrame(rows)
    best = envelope.sort_values(
        [
            "target_station", "target", "reliable_lead_hours", "high_skill_coverage",
            "mean_MAE_skill_vs_best_naive", "mean_R2_across_horizons", "lookback_steps",
        ],
        ascending=[True, True, False, False, False, False, True],
    ).groupby(["target_station", "target"], as_index=False).head(1)
    return grid, best


def mase_scale_from_training(sequence, train_indices: np.ndarray, frequency_hours: int = 4) -> float:
    """Mean absolute one-step change using consecutive observed training targets only."""
    times = sequence.target_start_times[train_indices]
    values = sequence.y[train_indices, 0]
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    elapsed = np.diff(times.asi8) / 3_600_000_000_000
    changes = np.abs(np.diff(values))[np.isclose(elapsed, frequency_hours)]
    return float(np.mean(changes)) if len(changes) else np.nan


def distribution_shift_table(sequence, split: dict[str, np.ndarray], metadata: dict[str, object]) -> pd.DataFrame:
    rows = []
    for split_name in ("train", "validation", "test"):
        indices = split[split_name]
        values = sequence.y[indices, 0].astype(float)
        times = sequence.target_start_times[indices]
        rows.append({
            **metadata,
            "data_split": split_name,
            "n": len(values),
            "start": times.min(),
            "end": times.max(),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    frame = pd.DataFrame(rows)
    train_mean = float(frame.loc[frame["data_split"] == "train", "mean"].iloc[0])
    train_std = float(frame.loc[frame["data_split"] == "train", "std"].iloc[0])
    frame["mean_shift_from_train"] = frame["mean"] - train_mean
    frame["standardized_mean_shift"] = (
        frame["mean_shift_from_train"] / train_std if train_std > 0 else np.nan
    )
    frame["std_ratio_to_train"] = frame["std"] / train_std if train_std > 0 else np.nan
    return frame


def moving_block_bootstrap(
    values: np.ndarray,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(10, block_length * 2):
        return float(np.mean(values)), np.nan, np.nan, np.nan
    block_length = min(block_length, len(values))
    starts = np.arange(len(values) - block_length + 1)
    blocks_needed = math.ceil(len(values) / block_length)
    bootstrap_means = np.empty(replicates)
    for replicate in range(replicates):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block_length] for start in chosen])[: len(values)]
        bootstrap_means[replicate] = sample.mean()
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    # Add-one correction avoids impossible p=0 with a finite number of replicates.
    left = (np.count_nonzero(bootstrap_means <= 0) + 1) / (replicates + 1)
    right = (np.count_nonzero(bootstrap_means >= 0) + 1) / (replicates + 1)
    p_value = min(1.0, 2 * min(left, right))
    return float(values.mean()), float(lower), float(upper), float(p_value)


def paired_block_bootstrap_table(
    predictions: pd.DataFrame,
    reference_feature: str,
    reference_model: str,
    block_length: int,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_columns = [
        column for column in ["experiment", "subexperiment", "fold", "target_station", "target", "horizon_step", "horizon_hours"]
        if column in predictions.columns
    ]
    rows = []
    for keys, group in predictions.groupby(group_columns, dropna=False):
        metadata = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        reference = group[
            (group["feature_set"] == reference_feature) & (group["model"] == reference_model)
        ][["seed", "valid_time", "absolute_error", "squared_error"]].rename(
            columns={"absolute_error": "absolute_error_reference", "squared_error": "squared_error_reference"}
        )
        if reference.empty:
            continue
        methods = group[["feature_set", "model"]].drop_duplicates().itertuples(index=False, name=None)
        for feature_set, model in methods:
            if (feature_set, model) == (reference_feature, reference_model):
                continue
            competitor = group[
                (group["feature_set"] == feature_set) & (group["model"] == model)
            ][["seed", "valid_time", "absolute_error", "squared_error"]].rename(
                columns={"absolute_error": "absolute_error_competitor", "squared_error": "squared_error_competitor"}
            )
            merged = reference.merge(competitor, on=["seed", "valid_time"])
            if merged.empty:
                continue
            for loss in ("absolute_error", "squared_error"):
                delta = merged[f"{loss}_competitor"] - merged[f"{loss}_reference"]
                by_time = pd.DataFrame({"valid_time": merged["valid_time"], "delta": delta}).groupby("valid_time")["delta"].mean()
                estimate, lower, upper, p_value = moving_block_bootstrap(
                    by_time.to_numpy(), block_length, replicates, rng
                )
                rows.append({
                    **metadata,
                    "reference_feature_set": reference_feature,
                    "reference_model": reference_model,
                    "competitor_feature_set": feature_set,
                    "competitor_model": model,
                    "loss": loss,
                    "n_common_times": len(by_time),
                    "block_length_steps": block_length,
                    "bootstrap_replicates": replicates,
                    "competitor_minus_reference_loss": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "two_sided_p_value": p_value,
                    "reference_better": estimate > 0,
                })
    return pd.DataFrame(rows)
