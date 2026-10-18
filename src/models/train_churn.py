"""Classical churn classifiers (PRD Sections 8.1 and 8.3).

Trains Logistic Regression, Decision Tree, Random Forest, KNN (distance
based), XGBoost and LightGBM under a fixed seed with a stratified
train/validation/test split. SMOTE is applied to the *training* folds
only. Every run is logged (parameters, metrics, training time) to a CSV
experiment log so the comparison table in the final report can be
regenerated from logs rather than re-typed by hand.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils.config import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

DROP_COLS = ["customer_id", "country", "churn", "clv_future"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric columns, categorical columns) for modeling."""
    feature_df = df.drop(columns=DROP_COLS, errors="ignore")
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_df.columns if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor(
    num_cols: list[str], cat_cols: list[str], scale_numeric: bool
) -> ColumnTransformer:
    """Build a preprocessing transformer; trees get raw numeric features."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if scale_numeric:
        transformers.append(("num", StandardScaler(), num_cols))
    else:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    return ColumnTransformer(transformers, remainder="drop")


def build_model_zoo(cfg: Config) -> dict[str, tuple[Any, bool]]:
    """Return {model_name: (estimator_factory, scale_numeric)} for all models.

    ``scale_numeric`` is True for scale-sensitive models (LogReg, KNN) and
    False for tree ensembles (PRD Section 6.3).
    """
    m = cfg.get("models", {})
    xgb = m.get("xgboost", {})
    lgbm = m.get("lightgbm", {})
    rf = m.get("random_forest", {})
    knn = m.get("knn", {})
    lr = m.get("logistic_regression", {})
    seed = int(cfg.get("project.random_seed", 42))

    return {
        "logistic_regression": (
            lambda: LogisticRegression(
                C=lr.get("C", 1.0),
                penalty=lr.get("penalty", "l2"),
                max_iter=int(lr.get("max_iter", 2000)),
                random_state=seed,
            ),
            True,
        ),
        "decision_tree": (
            lambda: DecisionTreeClassifier(random_state=seed), True
        ),
        "random_forest": (
            lambda: RandomForestClassifier(
                n_estimators=int(rf.get("n_estimators", 300)),
                max_depth=int(rf.get("max_depth", 12)),
                min_samples_leaf=int(rf.get("min_samples_leaf", 3)),
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        "knn": (
            lambda: KNeighborsClassifier(n_neighbors=int(knn.get("n_neighbors", 15))),
            True,
        ),
        "xgboost": (
            lambda: XGBClassifier(
                n_estimators=int(xgb.get("n_estimators", 300)),
                learning_rate=float(xgb.get("learning_rate", 0.05)),
                max_depth=int(xgb.get("max_depth", 5)),
                subsample=float(xgb.get("subsample", 0.9)),
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        "lightgbm": (
            lambda: LGBMClassifier(
                n_estimators=int(lgbm.get("n_estimators", 300)),
                learning_rate=float(lgbm.get("learning_rate", 0.05)),
                num_leaves=int(lgbm.get("num_leaves", 31)),
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            False,
        ),
    }


def _make_pipeline(
    estimator: Any, num_cols: list[str], cat_cols: list[str],
    scale_numeric: bool, use_smote: bool,
) -> Any:
    """Wrap estimator in preprocessing (+ optional SMOTE on training data only)."""
    pre = make_preprocessor(num_cols, cat_cols, scale_numeric)
    steps: list[tuple[str, Any]] = [("preprocess", pre)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute the full metric set required by PRD Section 3.2.

    Accuracy is reported but de-prioritized; recall, precision, F1 and
    ROC-AUC are the decision metrics.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def log_run(
    log_path: str | Path,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    train_seconds: float,
) -> None:
    """Append one row to the CSV experiment log (PRD Section 8.3)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [{"model": model_name, **params, **metrics,
          "train_seconds": round(train_seconds, 2)}]
    )
    header = not log_path.exists()
    row.to_csv(log_path, mode="a", header=header, index=False)


def stratified_splits(
    X: pd.DataFrame, y: np.ndarray, cfg: Config
) -> tuple[pd.DataFrame, ...]:
    """Stratified 70/15/15 split with a fixed seed (PRD Section 8.3)."""
    seed = int(cfg.get("project.random_seed", 42))
    test_size = float(cfg.get("models.test_size", 0.15))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_size = float(cfg.get("models.val_size", 0.15))
    val_relative = val_size / (1 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_relative, stratify=y_tr, random_state=seed
    )
    logger.info(
        "Split sizes: train=%d val=%d test=%d", len(X_tr), len(X_va), len(X_te)
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def train_all_classical_models(
    feature_table: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Train every classical churn model and return the comparison table.

    The test set is touched exactly once, at the end, for final reporting.
    The best model by validation ROC-AUC is saved as the production artifact
    together with its feature columns.
    """
    seed = int(cfg.get("project.random_seed", 42))
    np.random.seed(seed)
    num_cols, cat_cols = get_feature_columns(feature_table)
    logger.info("Numeric features (%d): %s", len(num_cols), num_cols)
    logger.info("Categorical features: %s", cat_cols)

    X = feature_table[num_cols + cat_cols]
    y = feature_table["churn"].to_numpy()
    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_splits(X, y, cfg)
    use_smote = bool(cfg.get("models.smote", True))

    log_path = resolve_path(cfg.get("paths.experiment_log"))
    if Path(log_path).exists():
        Path(log_path).unlink()  # fresh comparison for this pipeline run

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, (factory, scale_numeric) in build_model_zoo(cfg).items():
        pipeline = _make_pipeline(factory(), num_cols, cat_cols, scale_numeric, use_smote)
        start = time.perf_counter()
        pipeline.fit(X_tr, y_tr)
        train_seconds = time.perf_counter() - start
        fitted[name] = pipeline

        # Validation metrics (model selection happens here, never on test).
        va_prob = pipeline.predict_proba(X_va)[:, 1]
        va_metrics = evaluate(y_va, va_prob)
        va_metrics["split"] = "validation"

        # Test metrics — touched exactly once, for final reporting.
        te_prob = pipeline.predict_proba(X_te)[:, 1]
        te_metrics = evaluate(y_te, te_prob)
        te_metrics["split"] = "test"

        params = {
            k: v for k, v in pipeline.named_steps["model"].get_params().items()
            if isinstance(v, (int, float, str))
        }
        log_run(log_path, name, params, va_metrics, train_seconds)
        log_run(log_path, name, params, te_metrics, train_seconds)

        results.append(
            {"model": name, **{f"val_{k}": v for k, v in va_metrics.items()},
             **{f"test_{k}": v for k, v in te_metrics.items()},
             "train_seconds": round(train_seconds, 2)}
        )
        logger.info(
            "%-20s val AUC=%.4f recall=%.4f | test AUC=%.4f (%.1fs)",
            name, va_metrics["roc_auc"], va_metrics["recall"],
            te_metrics["roc_auc"], train_seconds,
        )

    comparison = (
        pd.DataFrame(results)
        .sort_values("val_roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    # Persist production artifact: best validation AUC model.
    best_name = comparison.iloc[0]["model"]
    artifact_dir = resolve_path(cfg.get("paths.models_dir"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": fitted[best_name],
            "model_name": best_name,
            "feature_columns": num_cols + cat_cols,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
        },
        artifact_dir / "churn_model.joblib",
    )
    comparison.to_csv(artifact_dir / "model_comparison.csv", index=False)
    logger.info("Production churn model: %s (saved to %s)", best_name, artifact_dir)
    return comparison
