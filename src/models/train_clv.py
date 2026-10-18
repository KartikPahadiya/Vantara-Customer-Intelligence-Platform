"""Customer lifetime value regression (PRD Section 8.1).

Trains a LightGBM regressor on repeat customers to predict the log1p of
revenue in the 180 days after the snapshot date, and reports RMSE, MAE and
R² on the held-out test set.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.train_churn import get_feature_columns
from src.utils.config import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_clv_model(
    feature_table: pd.DataFrame,
    cfg: Config,
    min_purchases: int | None = None,
) -> dict[str, float]:
    """Train the CLV regressor and persist the artifact.

    Only customers with at least ``min_purchases`` pre-cutoff orders are
    used — CLV for one-time buyers is dominated by noise.

    Returns:
        Dict of test metrics (rmse, mae, r2 in log-revenue space).
    """
    seed = int(cfg.get("project.random_seed", 42))
    min_purchases = min_purchases or int(cfg.get("features.min_purchases_for_clv", 2))
    df = feature_table[feature_table["frequency"] >= min_purchases].copy()
    logger.info("CLV training set: %d repeat customers", len(df))

    num_cols, cat_cols = get_feature_columns(df)
    X = df[num_cols + cat_cols]
    y = np.log1p(np.clip(df["clv_future"].to_numpy(), 0, None))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=float(cfg.get("models.test_size", 0.15)), random_state=seed
    )

    pre = ColumnTransformer(
        [("num", "passthrough", num_cols)]
        + ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             cat_cols)] if cat_cols else []),
        remainder="drop",
    )
    model = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    pipeline: Pipeline = Pipeline([("preprocess", pre), ("model", model)])

    start = time.perf_counter()
    pipeline.fit(X_tr, y_tr)
    train_seconds = time.perf_counter() - start

    pred = pipeline.predict(X_te)
    metrics = {
        "rmse": float(mean_squared_error(y_te, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "train_seconds": round(train_seconds, 2),
    }
    logger.info("CLV regression test metrics: %s", metrics)

    artifact_dir = resolve_path(cfg.get("paths.models_dir"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": pipeline, "feature_columns": num_cols + cat_cols},
        artifact_dir / "clv_model.joblib",
    )
    return metrics
