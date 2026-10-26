"""Source-equivalent cached inference, preserving existing weights and guards."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

VERSION = "standard_scaler_source_equivalent_v2"


def restore_scaler(mean, scale):
    """Restore only the attributes consumed by transform/inverse_transform; no fit."""
    mean, scale = np.asarray(mean), np.asarray(scale)
    if mean.ndim != 1 or mean.shape != scale.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid saved scaler statistics.")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = mean.copy(), scale.copy()
    scaler.n_features_in_ = len(mean)
    return scaler


def source_scaled_test(x, scaler):
    """Match _scale_x arithmetic and full-pool test indexing memory layout.

    The original common_station_pool selects a feature list, so the indexed test
    array is contiguous in N/F/time order, not N/time/F order. This also avoids
    changing CUDA input layout between unperturbed and permuted predictions.
    """
    if x.ndim != 3 or x.shape[2] != scaler.n_features_in_:
        raise ValueError("Test inputs must have the saved number of features.")
    values = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    return np.ascontiguousarray(values.transpose(0, 2, 1)).transpose(0, 2, 1)


def source_inverse_outputs(batches, scaler):
    """The shared trainer concatenates all scaled outputs before inverse scaling."""
    if not batches:
        raise ValueError("Inference produced no batches.")
    return scaler.inverse_transform(np.concatenate(batches, axis=0)).astype(np.float32)


def load_predictor_fixed(directory, device):
    import torch
    from torch import nn
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    training = meta["training"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        for name, enabled in [("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False), ("enable_math_sdp", True)]:
            if hasattr(torch.backends.cuda, name):
                getattr(torch.backends.cuda, name)(enabled)

    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.LSTM(input_size=len(meta["input_columns"]), hidden_size=training["hidden_size"],
                num_layers=training["num_layers"], dropout=training["dropout"] if training["num_layers"] > 1 else 0., batch_first=True)
            self.head = nn.Linear(training["hidden_size"], 6)

        def forward(self, x):
            y, _ = self.recurrent(x)
            return self.head(y[:, -1, :])

    model = Recurrent().to(device)
    model.load_state_dict(torch.load(directory / "model.pt", map_location=device, weights_only=True))
    model.eval()
    if sum(p.numel() for p in model.parameters()) != meta["case"]["parameter_count"]:
        raise ValueError("Saved architecture does not match the source parameter count.")
    with np.load(directory / "scalers.npz", allow_pickle=False) as f:
        xs = restore_scaler(f["x_mean"], f["x_scale"])
        ys = restore_scaler(f["y_mean"], f["y_scale"])
    if xs.n_features_in_ != len(meta["input_columns"]) or ys.n_features_in_ != 6:
        raise ValueError("Saved scaler dimensions do not match the original model.")

    def predict(x):
        # Move the entire test tensor first, as in the unchanged shared trainer.
        test_x = torch.from_numpy(source_scaled_test(x, xs)).to(device)
        batches = []
        with torch.no_grad():
            for begin in range(0, len(test_x), training["batch_size"]):
                batches.append(model(test_x[begin:begin + training["batch_size"]]).cpu().numpy())
        return source_inverse_outputs(batches, ys)

    return predict


def prediction_diagnostics(actual, expected):
    data = dict(actual_shape=list(actual.shape), expected_shape=list(expected.shape), rtol=1e-5, atol=1e-7)
    if actual.shape != expected.shape:
        return dict(data, passed=False, reason="shape_mismatch")
    finite = np.isfinite(actual) & np.isfinite(expected)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = 1e-7 + 1e-5 * np.abs(expected.astype(np.float64))
    failed = ~finite | (difference > tolerance)
    clean = np.where(finite, difference, 0.)
    worst = np.unravel_index(np.argmax(clean), actual.shape)
    return dict(data, passed=bool(not failed.any()), n_failed=int(failed.sum()), n_values=int(actual.size),
        max_abs_difference=float(clean.max()), mean_abs_difference=float(clean.mean()),
        max_tolerance_ratio=float(np.where(finite, clean / tolerance, 0.).max()),
        worst_index=[int(i) for i in worst],
        worst_actual=float(actual[worst]) if finite[worst] else None,
        worst_expected=float(expected[worst]) if finite[worst] else None)


@contextmanager
def fixed_inference(core):
    """Patch the live loader only; never edit old source/fingerprint files."""
    original_loader, original_verify = core.load_predictor, core.verify_prediction
    original_run, original_recover = core.InterpretationRunner.run, core.InterpretationRunner.recover
    original_csv = core.csv_write
    context = {}

    def checked(actual, expected):
        data = prediction_diagnostics(actual, expected)
        if "output" in context and "key" in context:
            directory = context["output"] / "inference_precision_checks"
            directory.mkdir(exist_ok=True)
            payload = dict(data, implementation=VERSION, case=context["key"], stage=context["stage"])
            core.json_write(directory / (context["key"] + "_" + context["stage"] + ".json"), payload)
            print(f"[precision check] {context['key']} {context['stage']}: "
                  f"passed={data['passed']}, max_abs={data.get('max_abs_difference', float('nan')):.3g}, "
                  f"outside_tolerance={data.get('n_failed', 'shape mismatch')}", flush=True)
        # Crucially retain the original strict verification and exception.
        return original_verify(actual, expected)

    def loader(directory, device):
        context.update(key=Path(directory).name, stage="cached_inference")
        return load_predictor_fixed(directory, device)

    def recover(runner, station, target, seed, seq, split):
        context.update(key=f"{station}_{target}_seed{seed}", stage="replay_output")
        return original_recover(runner, station, target, seed, seq, split)

    def write_csv(path, frame):
        if Path(path).name in {"permutation_raw.csv", "importance.csv", "importance_summary.csv"}:
            frame = frame.copy()
            if Path(path).name == "importance_summary.csv":
                if "inference_implementation" not in frame:
                    frame["inference_implementation"] = "legacy_manual_scaling_v1"
                else:
                    frame["inference_implementation"] = frame.inference_implementation.fillna("legacy_manual_scaling_v1")
            else:
                frame["inference_implementation"] = VERSION
        return original_csv(path, frame)

    def run(runner, *args, **kwargs):
        context["output"] = runner.output
        runner._guard("inference_precision_fix.json", dict(implementation=VERSION,
            fix_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_interpretation_source_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            numpy_version=np.__version__, sklearn_version=sklearn.__version__,
            original_tolerance_unchanged=True, existing_weights_and_commits_preserved=True))
        return original_run(runner, *args, **kwargs)

    core.load_predictor, core.verify_prediction = loader, checked
    core.InterpretationRunner.recover, core.InterpretationRunner.run = recover, run
    core.csv_write = write_csv
    try:
        yield
    finally:
        core.load_predictor, core.verify_prediction = original_loader, original_verify
        core.InterpretationRunner.recover, core.InterpretationRunner.run = original_recover, original_run
        core.csv_write = original_csv
