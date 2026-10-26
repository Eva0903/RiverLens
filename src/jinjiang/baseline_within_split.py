"""Strict 01 protocol: partition the raw timeline, then window each partition.

This module deliberately leaves data.make_sequences and experiments 02--07 unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd

from .data import PreparedDataset, SequenceData, make_sequences, official_split


PROTOCOL = "raw_time_split_then_within_split_windows_v1"
SPLIT_NAMES = ("train", "validation", "test")


def raw_partitions(data: PreparedDataset) -> dict[str, pd.DatetimeIndex]:
    index = pd.DatetimeIndex(data.raw_frame.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Raw timeline must be sorted and unique.")
    bounds = data.split_boundaries
    return {
        "train": index[index <= bounds["train_end"]],
        "validation": index[(index > bounds["train_end"]) & (index <= bounds["validation_end"])],
        "test": index[(index > bounds["validation_end"]) & (index <= bounds["test_end"])],
    }


def make_within_split_sequences(
    data: PreparedDataset, feature_columns, target_column: str, lookback: int,
    horizon: int, seasonal_period_steps: int = 6,
) -> SequenceData:
    """Construct each partition separately; even input history cannot cross a boundary.

    Targets are raw observed values, features use the existing splitwise preprocessing.
    Both naive methods and all learned methods consume the same surviving sequences.
    """
    feature_columns = list(feature_columns)
    chunks = []
    for name, times in raw_partitions(data).items():
        if len(times) < lookback + horizon:
            raise ValueError(f"{name}: {len(times)} raw rows cannot support L={lookback}, H={horizon}.")
        part = replace(data, raw_frame=data.raw_frame.loc[times], feature_frame=data.feature_frame.loc[times])
        try:
            seq = make_sequences(part, feature_columns, target_column, lookback, horizon, seasonal_period_steps)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        if not np.all(seq.split_labels == name):
            raise AssertionError(f"Unexpected split labels in {name}.")
        if len(seq.x) < 2:
            raise ValueError(f"{name}: fewer than two valid sequences remain.")
        chunks.append(seq)

    def cat_times(field):
        result = getattr(chunks[0], field)
        return result.append([getattr(chunk, field) for chunk in chunks[1:]])

    return SequenceData(
        **{field: np.concatenate([getattr(chunk, field) for chunk in chunks], axis=0)
           for field in ("x", "y", "persistence", "seasonal_naive", "split_labels")},
        **{field: cat_times(field) for field in ("issue_times", "target_start_times", "target_end_times")},
        feature_columns=feature_columns, target_column=target_column,
    )


def training_mase_scale(sequence: SequenceData, train_indices: np.ndarray, frequency_hours=4) -> float:
    """Resolution-safe version: DatetimeIndex may hold ns or us in different pandas versions."""
    times = sequence.target_start_times[train_indices]
    values = sequence.y[train_indices, 0]
    order = np.argsort(times)
    times, values = times[order], values[order]
    elapsed = np.asarray((times[1:] - times[:-1]) / pd.Timedelta(hours=1), dtype=float)
    changes = np.abs(np.diff(values))[np.isclose(elapsed, frequency_hours)]
    return float(np.mean(changes)) if len(changes) else np.nan


def support_tables(data: PreparedDataset, sequence: SequenceData, lookback: int, horizon: int,
                   metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assert full input/target/baseline containment and emit reproducible support evidence."""
    index = pd.DatetimeIndex(data.raw_frame.index)
    start_positions = index.get_indexer(sequence.target_start_times)
    if np.any(start_positions < lookback):
        raise AssertionError("Invalid window start.")
    input_starts = index[start_positions - lookback]
    manifest = pd.DataFrame({
        **metadata,
        "sample_index": np.arange(len(sequence.x)),
        "data_split": sequence.split_labels,
        "input_start_time": input_starts,
        "issue_time": sequence.issue_times,
        "target_start_time": sequence.target_start_times,
        "target_end_time": sequence.target_end_times,
    })
    split = official_split(sequence)
    scale = training_mase_scale(sequence, split["train"])
    rows = []
    previous_end = None
    for name, times in raw_partitions(data).items():
        selected = manifest[manifest["data_split"] == name]
        contained = bool((selected["input_start_time"] >= times[0]).all()
                         and (selected["target_end_time"] <= times[-1]).all())
        if not contained:
            raise AssertionError(f"Full windows cross a {name} boundary.")
        if previous_end is not None and not previous_end < selected["input_start_time"].min():
            raise AssertionError("Raw time support overlaps between partitions.")
        previous_end = selected["target_end_time"].max()
        positions = start_positions[sequence.split_labels == name]
        if not (index[positions - 6] >= times[0]).all():
            raise AssertionError("Seasonal-naive history crosses a partition boundary.")
        signature = "\n".join(t.isoformat() for t in sequence.target_start_times[split[name]])
        candidates = len(times) - lookback - horizon + 1
        rows.append({
            **metadata, "data_split": name, "raw_rows": len(times),
            "raw_start_time": times[0], "raw_end_time": times[-1],
            "candidate_windows": candidates, "valid_windows": len(selected),
            "dropped_invalid_windows": candidates - len(selected),
            "earliest_input_time": selected["input_start_time"].min(),
            "latest_target_time": selected["target_end_time"].max(),
            "full_window_contained": contained, "cross_partition_overlap": False,
            "feature_count": len(sequence.feature_columns),
            "feature_columns": "|".join(sequence.feature_columns),
            "training_MASE_scale": scale,
            "target_start_times_sha256": hashlib.sha256(signature.encode()).hexdigest(),
        })
    return pd.DataFrame(rows), manifest
