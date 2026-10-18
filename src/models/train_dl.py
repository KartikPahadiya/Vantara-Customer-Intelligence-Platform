"""Deep learning models (PRD Section 8.2): ANN, LSTM, Autoencoder.

All three are implemented in PyTorch with a single fixed seed:

* **ANN** — feed-forward classifier on the same engineered feature set as
  the classical models, with BatchNorm, Dropout and early stopping on
  validation loss (used as the direct accuracy/AUC comparison point);
* **LSTM** — trained on time-ordered purchase-event sequences
  (``[log1p(amount), gap_days]``) to model sequential purchase behaviour;
* **Autoencoder** — unsupervised anomaly detection on scaled spending
  features; reconstruction error above the configured percentile threshold
  flags anomalous accounts for manual review.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.train_churn import evaluate, get_feature_columns
from src.utils.config import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class ANNClassifier(nn.Module):
    """Feed-forward network with BatchNorm + Dropout between layers."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    """Single-layer LSTM over purchase-event sequences + masked mean pooling."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, n_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # masked mean pooling over valid timesteps
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


class AutoEncoder(nn.Module):
    """Symmetric autoencoder compressing scaled features to a small code."""

    def __init__(self, in_dim: int, encoding_dim: int) -> None:
        super().__init__()
        hidden = max(encoding_dim * 2, 16)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _train_loop(
    model: nn.Module,
    loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg_ann: dict[str, Any],
    task: str = "classification",
    mask_val: torch.Tensor | None = None,
) -> tuple[nn.Module, list[float], list[float]]:
    """Generic training loop with early stopping on validation loss.

    Returns (best_model_state_loaded, train_losses, val_losses) so the loss
    curves required by the mathematical-foundations appendix can be plotted.
    """
    lr = float(cfg_ann.get("learning_rate", 1e-3))
    max_epochs = int(cfg_ann.get("max_epochs", 100))
    patience = int(cfg_ann.get("patience", 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_left = patience
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        running, n = 0.0, 0
        for batch in loader:
            xb, yb = batch[0], batch[-1]
            if len(batch) == 3:  # sequence model: (x, mask, y)
                mb = batch[1]
                logits = model(xb, mb)
            else:
                logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            n += len(xb)
        train_losses.append(running / max(n, 1))

        model.eval()
        with torch.no_grad():
            if mask_val is not None:
                val_logits = model(X_val, mask_val)
            else:
                val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            logger.info("Early stopping at epoch %d (best val loss %.4f)", epoch + 1, best_val)
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def _to_tensors(*arrays: np.ndarray) -> list[torch.Tensor]:
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


def train_ann(
    feature_table: pd.DataFrame,
    splits: tuple[np.ndarray, ...],
    cfg: Config,
) -> dict[str, Any]:
    """Train the ANN churn classifier on the same features as classical models."""
    _set_seed(int(cfg.get("project.random_seed", 42)))
    num_cols, cat_cols = get_feature_columns(feature_table)
    X_tr, X_va, X_te = splits[0][num_cols], splits[1][num_cols], splits[2][num_cols]
    y_tr, y_va, y_te = splits[3], splits[4], splits[5]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    ann_cfg = cfg.get("models.ann", {})
    model = ANNClassifier(
        in_dim=X_tr_s.shape[1],
        hidden_dims=list(ann_cfg.get("hidden_dims", [128, 64, 32])),
        dropout=float(ann_cfg.get("dropout", 0.3)),
    )

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=int(ann_cfg.get("batch_size", 64)),
        shuffle=True,
    )
    start = time.perf_counter()
    model, train_losses, val_losses = _train_loop(
        model,
        loader,
        torch.tensor(X_va_s),
        torch.tensor(y_va, dtype=torch.float32),
        ann_cfg,
    )
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        te_prob = torch.sigmoid(model(torch.tensor(X_te_s))).numpy()

    metrics = evaluate(y_te, te_prob)
    metrics["train_seconds"] = round(train_seconds, 2)
    logger.info("ANN test metrics: %s", metrics)

    artifact_dir = resolve_path(cfg.get("paths.models_dir"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, artifact_dir / "ann_scaler.joblib")
    torch.save(
        {"state_dict": model.state_dict(), "config": ann_cfg,
         "feature_columns": num_cols},
        artifact_dir / "ann_model.pt",
    )
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
        artifact_dir / "ann_loss_curve.csv", index=False
    )
    return metrics


def train_lstm(
    sequences: np.ndarray,
    mask: np.ndarray,
    feature_table: pd.DataFrame,
    splits: tuple[pd.DataFrame, ...],
    cfg: Config,
) -> dict[str, Any]:
    """Train the LSTM on time-ordered purchase-event sequences."""
    _set_seed(int(cfg.get("project.random_seed", 42)))
    lstm_cfg = cfg.get("models.lstm", {})
    X_tr_df, X_va_df, X_te_df = splits[0], splits[1], splits[2]
    y_tr = splits[3].astype(np.float32)
    y_va = splits[4].astype(np.float32)
    y_te = splits[5]

    idx_tr = X_tr_df.index.to_numpy()
    idx_va = X_va_df.index.to_numpy()
    idx_te = X_te_df.index.to_numpy()

    # Normalize sequence channels with train statistics only.
    tr_seq, tr_mask = sequences[idx_tr], mask[idx_tr]
    amt_mean, amt_std = tr_seq[..., 0][tr_mask].mean(), tr_seq[..., 0][tr_mask].std() + 1e-6
    gap_mean = tr_seq[..., 1][tr_mask].mean()
    gap_std = tr_seq[..., 1][tr_mask].std() + 1e-6
    seq = sequences.copy()
    seq[..., 0] = (seq[..., 0] - amt_mean) / amt_std
    seq[..., 1] = (seq[..., 1] - gap_mean) / gap_std

    model = LSTMClassifier(
        hidden_dim=int(lstm_cfg.get("hidden_dim", 64)),
        n_layers=int(lstm_cfg.get("n_layers", 1)),
    )
    loader = DataLoader(
        TensorDataset(
            torch.tensor(seq[idx_tr]),
            torch.tensor(mask[idx_tr]),
            torch.tensor(y_tr),
        ),
        batch_size=int(lstm_cfg.get("batch_size", 64)),
        shuffle=True,
    )
    start = time.perf_counter()
    model, train_losses, val_losses = _train_loop(
        model,
        loader,
        torch.tensor(seq[idx_va]),
        torch.tensor(y_va),
        lstm_cfg,
        mask_val=torch.tensor(mask[idx_va]),
    )
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        te_prob = torch.sigmoid(
            model(torch.tensor(seq[idx_te]), torch.tensor(mask[idx_te]))
        ).numpy()

    metrics = evaluate(y_te, te_prob)
    metrics["train_seconds"] = round(train_seconds, 2)
    logger.info("LSTM test metrics: %s", metrics)

    artifact_dir = resolve_path(cfg.get("paths.models_dir"))
    torch.save(
        {"state_dict": model.state_dict(), "config": lstm_cfg,
         "norm": {"amt_mean": amt_mean, "amt_std": amt_std,
                  "gap_mean": gap_mean, "gap_std": gap_std}},
        artifact_dir / "lstm_model.pt",
    )
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
        artifact_dir / "lstm_loss_curve.csv", index=False
    )
    return metrics


def train_autoencoder(
    feature_table: pd.DataFrame,
    splits: tuple[np.ndarray, ...],
    cfg: Config,
) -> pd.DataFrame:
    """Train the autoencoder and flag anomalous accounts.

    Returns the feature table with an ``anomaly_flag`` column added and the
    reconstruction-error threshold saved as an artifact.
    """
    _set_seed(int(cfg.get("project.random_seed", 42)))
    num_cols, _ = get_feature_columns(feature_table)
    X_tr = splits[0][num_cols].to_numpy(dtype=np.float32)
    X_all = feature_table[num_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_all_s = scaler.transform(X_all)

    ae_cfg = cfg.get("models.autoencoder", {})
    model = AutoEncoder(
        in_dim=X_tr_s.shape[1],
        encoding_dim=int(ae_cfg.get("encoding_dim", 8)),
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_s), torch.tensor(X_tr_s)),
        batch_size=int(ae_cfg.get("batch_size", 64)),
        shuffle=True,
    )
    start = time.perf_counter()
    model, train_losses, val_losses = _train_loop(
        model, loader, torch.tensor(X_tr_s[:512]), torch.tensor(X_tr_s[:512]),
        ae_cfg, task="regression",
    )
    train_seconds = time.perf_counter() - start
    logger.info("Autoencoder trained in %.1fs (best val MSE %.4f)",
                train_seconds, min(val_losses))

    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X_all_s))
        errors = ((recon - torch.tensor(X_all_s)) ** 2).mean(dim=1).numpy()

    pct = float(ae_cfg.get("anomaly_percentile", 95.0))
    threshold = float(np.percentile(errors, pct))
    feature_table = feature_table.copy()
    feature_table["reconstruction_error"] = errors
    feature_table["anomaly_flag"] = (errors > threshold).astype(int)
    logger.info(
        "Autoencoder: threshold=%.4f flagged %d anomalous accounts",
        threshold, int(feature_table["anomaly_flag"].sum()),
    )

    artifact_dir = resolve_path(cfg.get("paths.models_dir"))
    torch.save(
        {"state_dict": model.state_dict(), "config": ae_cfg,
         "threshold": threshold},
        artifact_dir / "autoencoder.pt",
    )
    joblib.dump({"scaler": scaler, "numeric_cols": num_cols},
                artifact_dir / "autoencoder_scaler.joblib")
    return feature_table
