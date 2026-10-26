from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATION_FILES = {
    "A": "station_A_Anshun.xlsx",
    "B": "station_B_Chengren.xlsx",
    "C": "station_C_Yongan.xlsx",
}
TIME_COLUMNS = {
    "A": "监测时间_安顺廊桥",
    "B": "监测时间_成仁桥",
    "C": "监测时间_永安",
}
VARIABLE_TOKENS = {
    "WT": "水温",
    "PH": "pH",
    "DO": "溶解氧",
    "EC": "电导率",
    "TURB": "浊度",
    "CODMN": "CODMn",
    "NH3N": "氨氮",
    "TP": "总磷",
    "TN": "总氮",
    "FLOW": "流量",
}
TARGETS = ("DO", "NH3N", "CODMN", "TP")


@dataclass
class PreparedDataset:
    raw_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    station_columns: dict[str, list[str]]
    target_columns: dict[str, dict[str, str]]
    weather_columns: list[str]
    split_boundaries: dict[str, pd.Timestamp]
    audit: pd.DataFrame


@dataclass
class SequenceData:
    x: np.ndarray
    y: np.ndarray
    persistence: np.ndarray
    seasonal_naive: np.ndarray
    split_labels: np.ndarray
    issue_times: pd.DatetimeIndex
    target_start_times: pd.DatetimeIndex
    target_end_times: pd.DatetimeIndex
    feature_columns: list[str]
    target_column: str


def _missing_run_sizes(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    missing = series.isna()
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    return missing, missing.groupby(groups).transform("sum")


def short_gap_linear(series: pd.Series, max_steps: int) -> pd.Series:
    """Linearly fill entire internal gaps only when the full run is short enough."""
    missing, run_size = _missing_run_sizes(series)
    eligible = missing & run_size.le(max_steps)
    candidate = series.interpolate(method="time", limit_area="inside")
    out = series.copy()
    out.loc[eligible] = candidate.loc[eligible]
    return out


def short_gap_causal_ffill(series: pd.Series, max_steps: int) -> pd.Series:
    """Operational alternative: fill a short gap using past observations only."""
    missing, run_size = _missing_run_sizes(series)
    eligible = missing & run_size.le(max_steps)
    candidate = series.ffill()
    out = series.copy()
    out.loc[eligible] = candidate.loc[eligible]
    return out


def _snap_to_grid(index: pd.DatetimeIndex, freq: str, tolerance: str) -> pd.DatetimeIndex:
    rounded = index.round(freq)
    delta = np.abs((rounded - index).asi8)
    tolerance_ns = pd.Timedelta(tolerance).value
    if np.any(delta > tolerance_ns):
        bad = index[delta > tolerance_ns]
        raise ValueError(f"Timestamps outside {tolerance} snapping tolerance: {bad[:5].tolist()}")
    return rounded


def _canonical_station_values(raw: pd.DataFrame, station: str) -> tuple[pd.DataFrame, dict[str, str]]:
    rename: dict[str, str] = {}
    for alias, token in VARIABLE_TOKENS.items():
        candidates = [c for c in raw.columns if token.lower() in str(c).lower()]
        if candidates:
            rename[candidates[0]] = f"{station}_{alias}"
    if not rename:
        raise ValueError(f"No recognized station variables for station {station}")
    values = raw[list(rename)].rename(columns=rename).apply(pd.to_numeric, errors="coerce")
    variable_map = {new.split("_", 1)[1]: new for new in rename.values()}
    return values, variable_map


def load_station_raw(
    path: Path,
    station: str,
    freq: str,
    snap_tolerance: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, object]]:
    raw = pd.read_excel(path)
    time_col = TIME_COLUMNS[station]
    if time_col not in raw:
        raise KeyError(f"{path.name} does not contain {time_col!r}")
    timestamps = pd.to_datetime(raw[time_col], errors="coerce")
    valid = timestamps.notna()
    raw = raw.loc[valid].copy()
    timestamps = pd.DatetimeIndex(timestamps.loc[valid])
    off_grid = int(((timestamps != timestamps.round(freq))).sum())
    snapped = _snap_to_grid(timestamps, freq, snap_tolerance)
    values, variable_map = _canonical_station_values(raw, station)
    values.index = snapped
    duplicate_rows = int(values.index.duplicated(keep=False).sum())
    if values.index.has_duplicates:
        values = values.groupby(level=0).mean()
    values = values.sort_index()
    expected = pd.date_range(values.index.min(), values.index.max(), freq=freq)
    observed_grid_rows = len(values)
    values = values.reindex(expected)
    audit = {
        "source": station,
        "file": path.name,
        "raw_rows": len(raw),
        "grid_rows": len(values),
        "observed_grid_rows": observed_grid_rows,
        "missing_grid_rows": len(values) - observed_grid_rows,
        "off_grid_raw_timestamps": off_grid,
        "duplicate_rows_after_snap": duplicate_rows,
        "raw_missing_cells": int(values.isna().sum().sum()),
        "start": values.index.min(),
        "end": values.index.max(),
    }
    return values, variable_map, audit


def load_weather_raw(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(path)
    raw["DATETIME"] = pd.to_datetime(raw["DATETIME"], errors="coerce")
    raw = raw.dropna(subset=["DATETIME"]).drop_duplicates("DATETIME").set_index("DATETIME").sort_index()
    expected = ["PRE_1H", "PRS", "RHU", "TEM", "WIN_D_INST_MAX", "WIN_S_INST_MAX"]
    for column in expected:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    core = ["PRS", "RHU", "TEM", "WIN_D_INST_MAX", "WIN_S_INST_MAX"]
    common_sentinel = raw[core].eq(0).all(axis=1) | raw[core].eq(999999).all(axis=1)
    raw.loc[common_sentinel, expected] = np.nan
    raw.loc[(raw["PRE_1H"] < 0) | (raw["PRE_1H"] > 500), "PRE_1H"] = np.nan
    raw.loc[(raw["PRS"] < 800) | (raw["PRS"] > 1100), "PRS"] = np.nan
    raw.loc[(raw["RHU"] <= 0) | (raw["RHU"] > 100), "RHU"] = np.nan
    raw.loc[(raw["TEM"] < -50) | (raw["TEM"] > 60), "TEM"] = np.nan
    raw.loc[(raw["WIN_D_INST_MAX"] < 0) | (raw["WIN_D_INST_MAX"] > 360), "WIN_D_INST_MAX"] = np.nan
    raw.loc[(raw["WIN_S_INST_MAX"] < 0) | (raw["WIN_S_INST_MAX"] > 80), "WIN_S_INST_MAX"] = np.nan
    raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="1h"))

    angle = np.deg2rad(raw["WIN_D_INST_MAX"])
    raw["WIND_DIR_SIN"] = np.sin(angle)
    raw["WIND_DIR_COS"] = np.cos(angle)
    resampler = raw.resample("4h", label="right", closed="right")
    index = resampler.size().index
    weather = pd.DataFrame(index=index)
    weather["W_RAIN_4H"] = resampler["PRE_1H"].sum(min_count=4)
    for source, name in [("PRS", "W_PRS_MEAN"), ("RHU", "W_RHU_MEAN"), ("TEM", "W_TEM_MEAN")]:
        weather[name] = resampler[source].mean()
        weather.loc[resampler[source].count() < 4, name] = np.nan
    weather["W_WIND_SPEED_MAX"] = resampler["WIN_S_INST_MAX"].max()
    weather.loc[resampler["WIN_S_INST_MAX"].count() < 4, "W_WIND_SPEED_MAX"] = np.nan
    for source, name in [("WIND_DIR_SIN", "W_WIND_DIR_SIN"), ("WIND_DIR_COS", "W_WIND_DIR_COS")]:
        weather[name] = resampler[source].mean()
        weather.loc[resampler[source].count() < 4, name] = np.nan

    for steps, hours in [(1, 4), (2, 8), (3, 12), (6, 24)]:
        weather[f"W_RAIN_LAG_{hours}H"] = weather["W_RAIN_4H"].shift(steps)
    weather["W_RAIN_ANTE_12H"] = weather["W_RAIN_4H"].rolling(3, min_periods=3).sum()
    weather["W_RAIN_ANTE_24H"] = weather["W_RAIN_4H"].rolling(6, min_periods=6).sum()
    weather = weather.loc[(weather.index >= start) & (weather.index <= end)]
    audit = {
        "source": "weather",
        "file": path.name,
        "raw_rows": len(raw),
        "grid_rows": len(weather),
        "observed_grid_rows": int(weather["W_RAIN_4H"].notna().sum()),
        "missing_grid_rows": int(weather["W_RAIN_4H"].isna().sum()),
        "off_grid_raw_timestamps": 0,
        "duplicate_rows_after_snap": 0,
        "raw_missing_cells": int(weather.isna().sum().sum()),
        "start": weather.index.min(),
        "end": weather.index.max(),
    }
    return weather, audit


def compute_split_boundaries(index: pd.DatetimeIndex, train_fraction: float, validation_fraction: float) -> dict[str, pd.Timestamp]:
    train_end_pos = max(0, int(len(index) * train_fraction) - 1)
    validation_end_pos = max(train_end_pos + 1, int(len(index) * (train_fraction + validation_fraction)) - 1)
    validation_end_pos = min(validation_end_pos, len(index) - 1)
    return {
        "train_end": index[train_end_pos],
        "validation_end": index[validation_end_pos],
        "test_end": index[-1],
    }


def compute_rolling_origin_boundaries(
    index: pd.DatetimeIndex,
    initial_train_fraction: float,
    validation_fraction: float,
    folds: int,
) -> list[dict[str, pd.Timestamp]]:
    """Expanding-window folds with a moving validation window and disjoint test blocks."""
    if folds < 1:
        raise ValueError("rolling folds must be >= 1")
    n = len(index)
    train_size = max(1, int(n * initial_train_fraction))
    validation_size = max(1, int(n * validation_fraction))
    first_test = train_size + validation_size
    if first_test >= n:
        raise ValueError("initial training and validation fractions leave no rolling test data")
    test_blocks = [block for block in np.array_split(np.arange(first_test, n), folds) if len(block)]
    boundaries = []
    for block in test_blocks:
        test_start = int(block[0])
        validation_start = test_start - validation_size
        train_end = validation_start - 1
        if train_end < 0:
            raise ValueError("rolling fold has no training data")
        boundaries.append({
            "train_end": index[train_end],
            "validation_end": index[test_start - 1],
            "test_end": index[int(block[-1])],
        })
    return boundaries


def _partition_slices(index: pd.DatetimeIndex, boundaries: dict[str, pd.Timestamp]) -> dict[str, np.ndarray]:
    test_end = boundaries.get("test_end", index[-1])
    return {
        "train": np.flatnonzero(index <= boundaries["train_end"]),
        "validation": np.flatnonzero((index > boundaries["train_end"]) & (index <= boundaries["validation_end"])),
        "test": np.flatnonzero((index > boundaries["validation_end"]) & (index <= test_end)),
    }


def apply_splitwise_imputation(
    raw_frame: pd.DataFrame,
    station_columns: dict[str, list[str]],
    weather_columns: list[str],
    boundaries: dict[str, pd.Timestamp],
    max_water_gap_steps: int,
    max_weather_gap_hours: int,
    policy: str,
) -> pd.DataFrame:
    if policy not in {"split_linear", "causal_ffill"}:
        raise ValueError("interpolation_policy must be 'split_linear' or 'causal_ffill'")
    fill_function = short_gap_linear if policy == "split_linear" else short_gap_causal_ffill
    out = raw_frame.copy()
    water_columns = [column for station in "ABC" for column in station_columns[station]]
    weather_continuous = [column for column in weather_columns if "RAIN" not in column]
    partitions = _partition_slices(pd.DatetimeIndex(out.index), boundaries)
    weather_steps = max(1, int(np.ceil(max_weather_gap_hours / 4)))
    for positions in partitions.values():
        if len(positions) == 0:
            continue
        segment_index = out.index[positions]
        for column in water_columns:
            out.loc[segment_index, column] = fill_function(out.loc[segment_index, column], max_water_gap_steps)
        for column in weather_continuous:
            out.loc[segment_index, column] = fill_function(out.loc[segment_index, column], weather_steps)
    return out


def prepare_dataset(data_dir: str | Path, config: dict) -> PreparedDataset:
    data_dir = Path(data_dir)
    station_frames: list[pd.DataFrame] = []
    station_columns: dict[str, list[str]] = {}
    target_columns: dict[str, dict[str, str]] = {}
    audit_rows: list[dict[str, object]] = []
    missing = config["missing_data"]
    for station, filename in STATION_FILES.items():
        frame, variable_map, audit = load_station_raw(
            data_dir / filename,
            station,
            config["frequency"],
            missing["timestamp_snap_tolerance"],
        )
        station_frames.append(frame)
        station_columns[station] = list(frame.columns)
        target_columns[station] = {name: variable_map[name] for name in TARGETS if name in variable_map}
        audit_rows.append(audit)

    start = min(frame.index.min() for frame in station_frames)
    end = max(frame.index.max() for frame in station_frames)
    full_index = pd.date_range(start, end, freq=config["frequency"])
    raw_frame = pd.concat([frame.reindex(full_index) for frame in station_frames], axis=1)
    weather, weather_audit = load_weather_raw(data_dir / "weather_hourly.xlsx", start, end)
    raw_frame = raw_frame.join(weather.reindex(full_index))
    raw_frame.index.name = "timestamp"
    weather_columns = list(weather.columns)
    audit_rows.append(weather_audit)
    boundaries = compute_split_boundaries(
        full_index,
        config["split"]["train_fraction"],
        config["split"]["validation_fraction"],
    )
    feature_frame = apply_splitwise_imputation(
        raw_frame,
        station_columns,
        weather_columns,
        boundaries,
        missing["max_water_gap_steps"],
        missing["max_weather_gap_hours"],
        missing["interpolation_policy"],
    )
    audit = pd.DataFrame(audit_rows)
    audit["splitwise_missing_cells_after_imputation"] = [
        int(feature_frame[station_columns[source]].isna().sum().sum()) if source in station_columns
        else int(feature_frame[weather_columns].isna().sum().sum())
        for source in audit["source"]
    ]
    return PreparedDataset(
        raw_frame=raw_frame,
        feature_frame=feature_frame,
        station_columns=station_columns,
        target_columns=target_columns,
        weather_columns=weather_columns,
        split_boundaries=boundaries,
        audit=audit,
    )


def reimpute_for_boundaries(
    data: PreparedDataset,
    boundaries: dict[str, pd.Timestamp],
    config: dict,
) -> PreparedDataset:
    """Rebuild features within fold-specific partitions without crossing fold boundaries."""
    missing = config["missing_data"]
    raw_frame = data.raw_frame.loc[: boundaries["test_end"]].copy()
    feature_frame = apply_splitwise_imputation(
        raw_frame,
        data.station_columns,
        data.weather_columns,
        boundaries,
        missing["max_water_gap_steps"],
        missing["max_weather_gap_hours"],
        missing["interpolation_policy"],
    )
    audit = data.audit.copy()
    audit["splitwise_missing_cells_after_imputation"] = [
        int(feature_frame[data.station_columns[source]].isna().sum().sum())
        if source in data.station_columns
        else int(feature_frame[data.weather_columns].isna().sum().sum())
        for source in audit["source"]
    ]
    return PreparedDataset(
        raw_frame=raw_frame,
        feature_frame=feature_frame,
        station_columns=data.station_columns,
        target_columns=data.target_columns,
        weather_columns=data.weather_columns,
        split_boundaries=boundaries,
        audit=audit,
    )


def rolling_origin_datasets(data: PreparedDataset, config: dict) -> list[PreparedDataset]:
    rolling = config.get("rolling", {})
    boundaries = compute_rolling_origin_boundaries(
        pd.DatetimeIndex(data.raw_frame.index),
        float(rolling.get("initial_train_fraction", 0.50)),
        float(rolling.get("validation_fraction", 0.10)),
        int(rolling.get("folds", config["split"].get("rolling_folds", 3))),
    )
    return [reimpute_for_boundaries(data, fold_boundaries, config) for fold_boundaries in boundaries]


def feature_set_columns(data: PreparedDataset, name: str, target_station: str, target: str) -> list[str]:
    other_stations = [station for station in "ABC" if station != target_station]
    synchronous_weather = [column for column in data.weather_columns if "LAG" not in column and "ANTE" not in column]
    lagged_rain = [column for column in data.weather_columns if "RAIN_LAG" in column or "RAIN_ANTE" in column]
    cross_station_target = [
        data.target_columns[station][target]
        for station in "ABC"
        if target in data.target_columns[station]
    ]
    focused_variables = {
        "DO": ["DO", "WT", "PH", "FLOW"],
        "NH3N": ["NH3N", "WT", "PH", "EC", "DO", "FLOW"],
        "CODMN": ["CODMN", "TURB", "WT", "DO", "FLOW"],
        "TP": ["TP", "TURB", "PH", "FLOW"],
    }[target]
    process_focused = [
        f"{station}_{variable}"
        for variable in focused_variables
        for station in "ABC"
        if f"{station}_{variable}" in data.station_columns[station]
    ]
    mapping = {
        "local_target": [data.target_columns[target_station][target]],
        "cross_station_target": cross_station_target,
        "process_focused": process_focused,
        "local_multivariate": data.station_columns[target_station],
        "station_A": data.station_columns["A"],
        "station_B": data.station_columns["B"],
        "upstream_AB": data.station_columns["A"] + data.station_columns["B"],
        "other_stations": sum((data.station_columns[station] for station in other_stations), []),
        "all_stations": sum((data.station_columns[station] for station in "ABC"), []),
        "all_stations_weather": sum((data.station_columns[station] for station in "ABC"), []) + synchronous_weather,
        "all_stations_rain_lags": sum((data.station_columns[station] for station in "ABC"), []) + lagged_rain,
        "all_stations_weather_rain_lags": sum((data.station_columns[station] for station in "ABC"), []) + synchronous_weather + lagged_rain,
    }
    if name not in mapping:
        raise KeyError(f"Unknown feature set {name!r}; choose from {sorted(mapping)}")
    return list(dict.fromkeys(mapping[name]))


def _split_label(start: pd.Timestamp, end: pd.Timestamp, boundaries: dict[str, pd.Timestamp]) -> str | None:
    if end <= boundaries["train_end"]:
        return "train"
    if start > boundaries["train_end"] and end <= boundaries["validation_end"]:
        return "validation"
    if start > boundaries["validation_end"] and end <= boundaries.get("test_end", end):
        return "test"
    return None


def make_sequences(
    data: PreparedDataset,
    feature_columns: Iterable[str],
    target_column: str,
    lookback: int,
    horizon: int,
    seasonal_period_steps: int = 6,
) -> SequenceData:
    feature_columns = list(feature_columns)
    x_values = data.feature_frame[feature_columns].to_numpy(dtype=np.float32)
    # Evaluation targets remain observed values; interpolated targets are never scored.
    y_values = data.raw_frame[target_column].to_numpy(dtype=np.float32)
    persistence_values = data.feature_frame[target_column].to_numpy(dtype=np.float32)
    if lookback < seasonal_period_steps:
        raise ValueError(
            f"lookback={lookback} must be at least seasonal_period_steps={seasonal_period_steps}"
        )
    if horizon > seasonal_period_steps:
        raise ValueError(
            "A 24-hour seasonal-naive baseline is only causal through 24 h; "
            f"horizon={horizon} exceeds seasonal_period_steps={seasonal_period_steps}."
        )
    times = pd.DatetimeIndex(data.raw_frame.index)
    xs, ys, pers, seasonal, labels, issue, starts, ends = [], [], [], [], [], [], [], []
    for i in range(lookback, len(times) - horizon + 1):
        x = x_values[i - lookback : i]
        y = y_values[i : i + horizon]
        last = persistence_values[i - 1]
        seasonal_indices = np.arange(i, i + horizon) - seasonal_period_steps
        seasonal_values = persistence_values[seasonal_indices]
        start, end = times[i], times[i + horizon - 1]
        label = _split_label(start, end, data.split_boundaries)
        if label is None:
            continue
        if not (
            np.isfinite(x).all()
            and np.isfinite(y).all()
            and np.isfinite(last)
            and np.isfinite(seasonal_values).all()
        ):
            continue
        xs.append(x)
        ys.append(y)
        pers.append(np.repeat(last, horizon))
        seasonal.append(seasonal_values)
        labels.append(label)
        issue.append(times[i - 1])
        starts.append(start)
        ends.append(end)
    if not xs:
        raise ValueError(
            f"No valid sequences for {target_column}, lookback={lookback}, horizon={horizon}"
        )
    return SequenceData(
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
        persistence=np.asarray(pers, dtype=np.float32),
        seasonal_naive=np.asarray(seasonal, dtype=np.float32),
        split_labels=np.asarray(labels),
        issue_times=pd.DatetimeIndex(issue),
        target_start_times=pd.DatetimeIndex(starts),
        target_end_times=pd.DatetimeIndex(ends),
        feature_columns=feature_columns,
        target_column=target_column,
    )


def official_split(sequence: SequenceData) -> dict[str, np.ndarray]:
    split = {
        "train": np.flatnonzero(sequence.split_labels == "train"),
        "validation": np.flatnonzero(sequence.split_labels == "validation"),
        "test": np.flatnonzero(sequence.split_labels == "test"),
    }
    if min(len(values) for values in split.values()) == 0:
        raise ValueError({name: len(values) for name, values in split.items()})
    return split
