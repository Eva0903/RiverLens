from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# Must be present before CUDA/cuBLAS is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class PredictionResult:
    prediction: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    parameter_count: int
    feature_importance: np.ndarray | None = None
    validation_prediction: np.ndarray | None = None
    diagnostics: dict[str, Any] | None = None


def _scale_x(
    x: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(x[train_indices].reshape(-1, x.shape[-1]))
    scaled = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    return scaled.astype(np.float32), scaler


def fit_ridge_predict(
    x: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
    alpha: float = 1.0,
) -> PredictionResult:
    x_scaled, _ = _scale_x(x, split["train"])
    x_flat = x_scaled.reshape(len(x_scaled), -1)
    y_scaler = StandardScaler().fit(y[split["train"]])
    y_scaled = y_scaler.transform(y)
    model = Ridge(alpha=alpha)
    model.fit(x_flat[split["train"]], y_scaled[split["train"]])
    scaled_prediction = np.asarray(model.predict(x_flat[split["test"]]))
    if scaled_prediction.ndim == 1:
        scaled_prediction = scaled_prediction.reshape(-1, 1)
    prediction = y_scaler.inverse_transform(scaled_prediction)
    parameter_count = int(np.size(model.coef_) + np.size(model.intercept_))
    return PredictionResult(prediction.astype(np.float32), [], 0, parameter_count, None)


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for ANN/CNN/RNN/LSTM/Transformer. "
            "In the VS Code terminal run: pip install -r requirements.txt"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _choose_attention_heads(hidden_size: int) -> int:
    for heads in (8, 4, 2, 1):
        if hidden_size % heads == 0:
            return heads
    return 1


def choose_validation_baseline(
    y: np.ndarray,
    persistence: np.ndarray,
    seasonal_naive: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Choose the lower-MAE causal baseline separately for each horizon using validation only."""
    names: list[str] = []
    selected = np.empty_like(y, dtype=np.float32)
    for horizon in range(y.shape[1]):
        observed = y[validation_indices, horizon]
        persistence_mae = float(np.mean(np.abs(observed - persistence[validation_indices, horizon])))
        seasonal_mae = float(np.mean(np.abs(observed - seasonal_naive[validation_indices, horizon])))
        if seasonal_mae < persistence_mae:
            selected[:, horizon] = seasonal_naive[:, horizon]
            names.append("seasonal_naive")
        else:
            selected[:, horizon] = persistence[:, horizon]
            names.append("persistence")
    return selected, names


def optimal_residual_gates(
    y_true: np.ndarray,
    baseline: np.ndarray,
    residual_prediction: np.ndarray,
    grid_size: int = 21,
) -> np.ndarray:
    """Validation-only shrinkage: gate=0 exactly recovers the selected naive baseline."""
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    gates = np.zeros(y_true.shape[1], dtype=np.float32)
    for horizon in range(y_true.shape[1]):
        losses = [
            np.mean(np.abs(y_true[:, horizon] - (
                baseline[:, horizon] + gate * residual_prediction[:, horizon]
            )))
            for gate in grid
        ]
        gates[horizon] = grid[int(np.argmin(losses))]
    return gates


def fit_torch_predict(
    architecture: str,
    x: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
    training: dict[str, Any],
) -> PredictionResult:
    architecture = architecture.lower()
    if architecture not in {"ann", "cnn", "rnn", "lstm", "transformer"}:
        raise ValueError(f"Unsupported torch architecture: {architecture}")
    torch, nn, DataLoader, TensorDataset = _import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic_mode = training.get("deterministic", "strict")
    if deterministic_mode:
        strict = str(deterministic_mode).lower() == "strict"
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
    device_name = training.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    x_scaled, _ = _scale_x(x, split["train"])
    y_scaler = StandardScaler().fit(y[split["train"]])
    y_scaled = y_scaler.transform(y).astype(np.float32)
    hidden_size = int(training["hidden_size"])
    num_layers = int(training["num_layers"])
    dropout = float(training["dropout"])
    input_size = x.shape[-1]
    lookback = x.shape[1]
    output_size = y.shape[-1]

    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(lookback * input_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, inputs):
            return self.network(inputs)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(hidden_size, output_size)

        def forward(self, inputs):
            encoded = self.encoder(inputs.transpose(1, 2)).squeeze(-1)
            return self.head(encoded)

    class Recurrent(nn.Module):
        def __init__(self, cell: str):
            super().__init__()
            recurrent_class = nn.RNN if cell == "rnn" else nn.LSTM
            self.recurrent = recurrent_class(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, output_size)

        def forward(self, inputs):
            output, _ = self.recurrent(inputs)
            return self.head(output[:, -1, :])

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.position = nn.Parameter(torch.zeros(1, lookback, hidden_size))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=_choose_attention_heads(hidden_size),
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, output_size)

        def forward(self, inputs):
            hidden = self.input_projection(inputs) + self.position[:, : inputs.shape[1]]
            hidden = self.encoder(hidden)
            return self.head(self.norm(hidden[:, -1, :]))

    if architecture == "ann":
        model = ANN()
    elif architecture == "cnn":
        model = CNN()
    elif architecture in {"rnn", "lstm"}:
        model = Recurrent(architecture)
    else:
        model = Transformer()
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def make_loader(indices: np.ndarray, shuffle: bool):
        dataset = TensorDataset(torch.from_numpy(x_scaled[indices]), torch.from_numpy(y_scaled[indices]))
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=shuffle,
            generator=generator,
        )

    train_loader = make_loader(split["train"], True)
    validation_loader = make_loader(split["validation"], False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(xb)
            train_count += len(xb)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for xb, yb in validation_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                validation_loss_sum += float(loss.detach().cpu()) * len(xb)
                validation_count += len(xb)
        train_loss = train_loss_sum / max(1, train_count)
        validation_loss = validation_loss_sum / max(1, validation_count)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                break

    if best_state is None:
        raise RuntimeError(f"{architecture} did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.from_numpy(x_scaled[split["test"]]).to(device)
    predictions = []
    batch_size = int(training["batch_size"])
    with torch.no_grad():
        for start in range(0, len(test_x), batch_size):
            predictions.append(model(test_x[start : start + batch_size]).cpu().numpy())
    scaled_prediction = np.concatenate(predictions, axis=0)
    prediction = y_scaler.inverse_transform(scaled_prediction)

    feature_importance = None
    if bool(training.get("compute_shap", False)):
        try:
            import shap
        except ImportError as exc:
            raise RuntimeError("SHAP attribution was requested; install shap>=0.46") from exc
        background_count = min(int(training.get("shap_background_samples", 128)), len(split["train"]))
        explanation_count = min(int(training.get("shap_explanation_samples", 256)), len(split["test"]))
        background_indices = split["train"][np.linspace(0, len(split["train"]) - 1, background_count, dtype=int)]
        explanation_indices = split["test"][np.linspace(0, len(split["test"]) - 1, explanation_count, dtype=int)]
        background = torch.from_numpy(x_scaled[background_indices]).to(device)
        explanation = torch.from_numpy(x_scaled[explanation_indices]).to(device)
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explanation)
        if isinstance(shap_values, list):
            attribution = np.stack([np.asarray(value) for value in shap_values], axis=-1)
        else:
            attribution = np.asarray(shap_values)
        feature_axes = [axis for axis, size in enumerate(attribution.shape) if size == input_size]
        if not feature_axes:
            raise RuntimeError(f"Cannot identify feature axis in SHAP array {attribution.shape}")
        feature_axis = feature_axes[-1] if attribution.shape[-1] == input_size else feature_axes[0]
        reduce_axes = tuple(axis for axis in range(attribution.ndim) if axis != feature_axis)
        feature_importance = np.mean(np.abs(attribution), axis=reduce_axes).astype(float)
    return PredictionResult(prediction.astype(np.float32), history, best_epoch, parameter_count, feature_importance)


def fit_residual_lstm_predict(
    x: np.ndarray,
    y: np.ndarray,
    persistence: np.ndarray,
    seasonal_naive: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
    training: dict[str, Any],
    refit_train_validation: bool,
) -> PredictionResult:
    """Fit a gated LSTM correction to the validation-selected causal naive forecast.

    Candidate screening returns validation predictions and never evaluates the test
    targets. The final path repeats early stopping, then reinitializes and trains for
    the selected epoch count on train+validation before predicting the test block.
    """
    torch, nn, DataLoader, TensorDataset = _import_torch()

    def reset_seed() -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    reset_seed()
    deterministic_mode = training.get("deterministic", "strict")
    if deterministic_mode:
        strict = str(deterministic_mode).lower() == "strict"
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    device_name = training.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Python environment has no CUDA-enabled PyTorch build."
        )

    selected_baseline, baseline_names = choose_validation_baseline(
        y, persistence, seasonal_naive, split["validation"]
    )
    residual_target = (y - selected_baseline).astype(np.float32)
    hidden_size = int(training["hidden_size"])
    num_layers = int(training["num_layers"])
    dropout = float(training.get("dropout", 0.0))
    bidirectional = bool(training.get("bidirectional", False))
    input_size = x.shape[-1]
    output_size = y.shape[-1]

    class ResidualLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional,
            )
            encoded_size = hidden_size * (2 if bidirectional else 1)
            self.norm = nn.LayerNorm(encoded_size)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(encoded_size, output_size)

        def forward(self, inputs):
            output, _ = self.recurrent(inputs)
            encoded = self.dropout(self.norm(output[:, -1, :]))
            return self.head(encoded)

    def build_model():
        model = ResidualLSTM().to(device)
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return model, count

    def make_loader(
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        indices: np.ndarray,
        shuffle: bool,
    ):
        dataset = TensorDataset(
            torch.from_numpy(x_scaled[indices]), torch.from_numpy(y_scaled[indices])
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=shuffle,
            generator=generator,
        )

    def loss_function():
        if str(training.get("loss", "huber")).lower() == "huber":
            return nn.SmoothL1Loss(beta=float(training.get("huber_beta", 1.0)))
        return nn.MSELoss()

    train_indices = split["train"]
    validation_indices = split["validation"]
    x_scaled, _ = _scale_x(x, train_indices)
    residual_scaler = StandardScaler().fit(residual_target[train_indices])
    residual_scaled = residual_scaler.transform(residual_target).astype(np.float32)
    model, parameter_count = build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    criterion = loss_function()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 4)),
        min_lr=float(training.get("minimum_learning_rate", 1e-5)),
    )
    train_loader = make_loader(x_scaled, residual_scaled, train_indices, True)
    validation_loader = make_loader(x_scaled, residual_scaled, validation_indices, False)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    clip_norm = float(training.get("gradient_clip_norm", 1.0))

    for epoch in range(1, int(training["max_epochs"]) + 1):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        train_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(xb)
            train_count += len(xb)
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for xb, yb in validation_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                validation_sum += float(loss.detach().cpu()) * len(xb)
                validation_count += len(xb)
        train_loss = train_sum / max(1, train_count)
        validation_loss = validation_sum / max(1, validation_count)
        scheduler.step(validation_loss)
        history.append({
            "stage": 0.0,
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate_used,
        })
        if validation_loss < best_loss - float(training.get("minimum_delta", 1e-6)):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Residual LSTM did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    def predict_residual(current_model, scaled_x: np.ndarray, indices: np.ndarray, scaler):
        batches = []
        batch_size = int(training["batch_size"])
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                xb = torch.from_numpy(scaled_x[indices[start : start + batch_size]]).to(device)
                batches.append(current_model(xb).cpu().numpy())
        return scaler.inverse_transform(np.concatenate(batches, axis=0))

    validation_residual = predict_residual(
        model, x_scaled, validation_indices, residual_scaler
    )
    gates = optimal_residual_gates(
        y[validation_indices],
        selected_baseline[validation_indices],
        validation_residual,
        int(training.get("gate_grid_size", 21)),
    )
    validation_prediction = (
        selected_baseline[validation_indices] + gates * validation_residual
    ).astype(np.float32)
    baseline_validation_mae = np.mean(
        np.abs(y[validation_indices] - selected_baseline[validation_indices]), axis=0
    )
    model_validation_mae = np.mean(
        np.abs(y[validation_indices] - validation_prediction), axis=0
    )
    validation_skill = 1 - model_validation_mae / np.maximum(baseline_validation_mae, 1e-12)

    if not refit_train_validation:
        return PredictionResult(
            validation_prediction,
            history,
            best_epoch,
            parameter_count,
            None,
            validation_prediction,
            {
                "baseline_names": baseline_names,
                "residual_gates": gates.tolist(),
                "validation_skill_by_horizon": validation_skill.tolist(),
                "mean_validation_skill": float(np.mean(validation_skill)),
                "mean_validation_mae": float(np.mean(model_validation_mae)),
            },
        )

    combined_indices = np.sort(np.concatenate([train_indices, validation_indices]))
    final_x_scaled, _ = _scale_x(x, combined_indices)
    final_residual_scaler = StandardScaler().fit(residual_target[combined_indices])
    final_residual_scaled = final_residual_scaler.transform(residual_target).astype(np.float32)
    reset_seed()
    final_model, final_parameter_count = build_model()
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    combined_loader = make_loader(
        final_x_scaled, final_residual_scaled, combined_indices, True
    )
    for epoch in range(1, best_epoch + 1):
        replayed_learning_rate = float(history[epoch - 1]["learning_rate"])
        for parameter_group in final_optimizer.param_groups:
            parameter_group["lr"] = replayed_learning_rate
        final_model.train()
        train_sum = 0.0
        train_count = 0
        for xb, yb in combined_loader:
            xb, yb = xb.to(device), yb.to(device)
            final_optimizer.zero_grad(set_to_none=True)
            loss = criterion(final_model(xb), yb)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), clip_norm)
            final_optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(xb)
            train_count += len(xb)
        history.append({
            "stage": 1.0,
            "epoch": float(epoch),
            "train_loss": train_sum / max(1, train_count),
            "validation_loss": np.nan,
            "learning_rate": float(final_optimizer.param_groups[0]["lr"]),
        })
    final_model.eval()
    test_indices = split["test"]
    test_residual = predict_residual(
        final_model, final_x_scaled, test_indices, final_residual_scaler
    )
    test_prediction = (
        selected_baseline[test_indices] + gates * test_residual
    ).astype(np.float32)
    return PredictionResult(
        test_prediction,
        history,
        best_epoch,
        final_parameter_count,
        None,
        validation_prediction,
        {
            "baseline_names": baseline_names,
            "residual_gates": gates.tolist(),
            "validation_skill_by_horizon": validation_skill.tolist(),
            "mean_validation_skill": float(np.mean(validation_skill)),
            "mean_validation_mae": float(np.mean(model_validation_mae)),
            "refit_samples": int(len(combined_indices)),
        },
    )
