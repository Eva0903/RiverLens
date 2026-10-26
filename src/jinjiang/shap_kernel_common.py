"""Read-only support and cache utilities for standalone Kernel SHAP.

No imports from earlier 07/Expected-Gradients/native-gradient modules.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
TARGETS = ["DO", "NH3N", "CODMN", "TP"]
FULL = "all_stations_weather_rain_lags"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, frame):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def cached_arrays(path, stamp):
    """Only complete, hash-verified shards matching current inputs are reusable."""
    path = Path(path)
    try:
        marker = read_json(path.with_suffix(".json"))
        if marker["stamp"] != stamp or marker["sha256"] != digest(path):
            return None
        with np.load(path, allow_pickle=False) as f:
            result = {k: f[k].copy() for k in f.files}
        if not np.isfinite(result["phi"]).all():
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_arrays(path, stamp, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix(".tmp").replace(path)
    write_json(path.with_suffix(".json"), {"stamp": stamp, "sha256": digest(path)})


def background_indices(train_indices, count, seed):
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if count < 1 or len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("Invalid training pool/background count.")
    return np.random.default_rng(seed).choice(train_indices, min(count, len(train_indices)), replace=False)


def feature_group(name):
    if name.startswith(("A_", "B_", "C_")):
        return "station_" + name[0]
    if name.startswith("W_RAIN"):
        return "rain"
    if name.startswith("W_WIND"):
        return "wind"
    return {"W_RHU_MEAN": "humidity", "W_PRS_MEAN": "pressure", "W_TEM_MEAN": "air_temperature"}.get(name, "other")


class ReadOnlyCaseSource:
    """Reconstruct existing 04 support directly, without invoking any old runner."""
    def __init__(self, spec, root):
        from .baseline_random import prepare_random_pool
        from .data import prepare_dataset
        self.root = Path(root)
        self.source = self.root / spec["interpretation_dir"]
        oldspec = read_json(self.source / "run_config.json")
        self.ref = self.root / oldspec["source_04_dir"]
        self.config = read_json(self.ref / "run_config.json")
        self.config["_project_root"] = str(self.root)
        self.config["data_dir"] = oldspec["data_dir"]
        if self.config.get("meteorology_protocol") != "meteorology_random_fixed_budget_v1":
            raise ValueError("Expected the completed random-window 04 run.")
        source_folder = Path(__file__).parent
        for n, h in read_json(self.source / "source_fingerprints.json").items():
            if digest(source_folder / n) != h:
                raise ValueError(f"Original source fingerprint mismatch: {n}. Do not edit old modules.")
        data_dir = self.root / oldspec["data_dir"]
        current = {p.name: digest(p) for p in sorted(data_dir.glob("*.xlsx"))}
        if not current or current != read_json(self.source / "data_fingerprints.json"):
            raise ValueError("Raw data differ from the completed 06 run.")
        if current != read_json(self.ref / "data_fingerprints.json"):
            raise ValueError("04/06 raw data fingerprint mismatch.")
        self.data = prepare_random_pool(prepare_dataset(data_dir, self.config), self.config)
        self.manifest = pd.read_csv(self.source / "sample_manifest.csv")
        self.groups = pd.read_csv(self.source / "input_groups.csv")
        self.predictions = pd.read_csv(self.ref / "predictions.csv", parse_dates=["target_start_time"])
        self.metrics = pd.read_csv(self.ref / "metrics.csv")
        self.protocol_hash = hashlib.sha256(json.dumps(oldspec, sort_keys=True).encode()).hexdigest()[:16]
        self.fingerprint = {"raw_data": current, "original_sources": read_json(self.source / "source_fingerprints.json"),
                            "reference_files": {n: digest(self.ref / n) for n in ["metrics.csv", "predictions.csv", "run_config.json"]},
                            "support_files": {n: digest(self.source / n) for n in ["sample_manifest.csv", "input_groups.csv", "run_config.json"]}}

    def case(self, target):
        from .station_rotation_random import common_station_pool
        from .interpretation_precision_fix import load_predictor_fixed, prediction_diagnostics
        _, split, arms = common_station_pool(self.data, self.config, "C", target)
        seq = arms[FULL]
        old = self.manifest[(self.manifest.target_station == "C") & (self.manifest.target == target)].sort_values("sample_index")
        if len(old) != len(seq.x) or not np.array_equal(old.data_split.to_numpy(), seq.split_labels):
            raise ValueError(f"Support split mismatch: {target}.")
        for col, now in [("target_start_time", seq.target_start_times), ("issue_time", seq.issue_times), ("target_end_time", seq.target_end_times)]:
            if not np.array_equal(pd.to_datetime(old[col]).to_numpy(dtype="datetime64[ns]"), now.to_numpy(dtype="datetime64[ns]")):
                raise ValueError(f"Support timestamp mismatch: {target}/{col}.")
        group = self.groups[(self.groups.target_station == "C") & (self.groups.target == target) & (self.groups.feature_set == FULL)]
        if len(group) != 1 or group.iloc[0].feature_columns != "|".join(seq.feature_columns):
            raise ValueError("Feature column order mismatch.")
        for label, indices in split.items():
            h = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
            if group.iloc[0][label + "_ordered_indices_sha256"] != h:
                raise ValueError("Ordered split index fingerprint mismatch.")
        directory = self.source / "checkpoints" / f"C_{target}_seed42"
        meta = read_json(directory / "metadata.json")
        if meta["protocol_hash"] != self.protocol_hash or meta["input_columns"] != seq.feature_columns:
            raise ValueError("Checkpoint protocol/feature mismatch.")
        if set(meta["files_sha256"]) != {"model.pt", "scalers.npz"}:
            raise ValueError("Incomplete checkpoint metadata.")
        for n, h in meta["files_sha256"].items():
            if digest(directory / n) != h:
                raise ValueError(f"Corrupted checkpoint {directory / n}.")
        rows = self.predictions[(self.predictions.target_station == "C") & (self.predictions.target == target) &
                                (self.predictions.seed == 42) & (self.predictions.feature_set == FULL) & (self.predictions.model == "lstm")]
        if rows.task_id.nunique() != 1 or rows.task_id.iloc[0] != meta["case"]["source_task_id"]:
            raise ValueError("Checkpoint/reference task mismatch.")
        expected_shape = (len(split["test"]), 6)
        if len(rows) != expected_shape[0] * 6 or rows.duplicated(["target_start_time", "horizon_step"]).any():
            raise ValueError("Incomplete/duplicate reference predictions.")
        aligned = lambda c: rows.pivot(index="target_start_time", columns="horizon_step", values=c).reindex(seq.target_start_times[split["test"]]).to_numpy(dtype=np.float32)
        expected = aligned("y_pred")
        if expected.shape != expected_shape or not np.array_equal(aligned("y_true"), seq.y[split["test"]]):
            raise ValueError("Reference target labels/support mismatch.")
        if seq.x.shape[2] != 39 or len(seq.feature_columns) != 39:
            raise ValueError("Expected exactly the existing 39 input columns.")
        self._predictor_loader = load_predictor_fixed
        self._prediction_diagnostics = prediction_diagnostics
        return seq, split, directory, meta, expected


def summarize_backgrounds(frame, keys):
    return frame.groupby(keys, as_index=False).agg(
        mean_abs_net_SHAP=("mean_abs_net_SHAP", "mean"),
        background_draw_sd=("mean_abs_net_SHAP", "std"),
        mean_signed_SHAP=("mean_signed_SHAP", "mean"),
        n_background_draws=("background_seed", "nunique"))
