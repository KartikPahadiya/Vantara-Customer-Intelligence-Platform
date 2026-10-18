"""End-to-end pipeline runner (PRD Acceptance Criterion 1).

Runs every stage from raw data to trained, persisted, explained models with
a single documented command:

    python -m src.pipeline --stage all

Stages: ``data`` → ``features`` → ``classical`` → ``clv`` → ``dl`` →
``segmentation`` → ``explainability``. Each stage loads what it needs from
disk, so the pipeline can also be resumed from any intermediate stage.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.data.clean import clean_transactions
from src.data.load import load_raw_transactions
from src.data.validate import validate_transactions
from src.explainability.explain import run_explainability
from src.features.build_features import (
    build_feature_table,
    build_lstm_sequences,
    split_by_cutoff,
)
from src.models.train_churn import get_feature_columns, stratified_splits
from src.models.train_clv import train_clv_model
from src.models.train_dl import train_ann, train_autoencoder, train_lstm
from src.segmentation.segment import run_segmentation
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def stage_data(cfg) -> pd.DataFrame:
    """Load, clean and validate raw transactions."""
    raw_path = resolve_path(cfg.get("paths.raw_data"))
    df = load_raw_transactions(raw_path)

    dc = cfg.get("data_cleaning", {})
    customer_df, product_df = clean_transactions(
        df,
        admin_stock_codes=list(dc.get("admin_stock_codes", [])),
        max_unit_price=float(dc.get("max_unit_price", 5000.0)),
        max_quantity=float(dc.get("max_quantity", 10000.0)),
        iqr_multiplier=float(dc.get("iqr_multiplier", 3.0)),
    )
    validate_transactions(customer_df)

    interim_path = resolve_path(cfg.get("paths.interim_data"))
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    customer_df.to_parquet(interim_path, index=False)
    logger.info("Cleaned transactions written to %s", interim_path)
    return customer_df


def stage_features(cfg) -> pd.DataFrame:
    """Build the leakage-safe customer-level feature table + LSTM sequences."""
    interim_path = resolve_path(cfg.get("paths.interim_data"))
    if not interim_path.exists():
        raise FileNotFoundError("Run --stage data first.")
    transactions = pd.read_parquet(interim_path)

    snapshot = str(cfg.get("features.snapshot_date"))
    table = build_feature_table(
        transactions,
        snapshot_date=snapshot,
        churn_window_days=int(cfg.get("features.churn_window_days")),
        clv_window_days=int(cfg.get("features.clv_window_days")),
        rfm_quantiles=int(cfg.get("features.rfm_quantiles", 4)),
    )
    out_path = resolve_path(cfg.get("paths.processed_features"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    logger.info("Feature table written to %s (%d rows)", out_path, len(table))

    # Persist LSTM sequences aligned with the feature-table row order.
    history, _ = split_by_cutoff(transactions, snapshot)
    seq, mask = build_lstm_sequences(
        history,
        table["customer_id"].to_numpy(),
        snapshot,
        max_len=int(cfg.get("models.lstm.max_len", 40)),
    )
    seq_path = resolve_path(cfg.get("paths.processed_sequences"))
    np.savez_compressed(seq_path, sequences=seq, mask=mask)
    logger.info("LSTM sequences written to %s", seq_path)
    return table


def stage_classical(cfg) -> pd.DataFrame:
    """Train all classical churn classifiers and select the production model."""
    from src.models.train_churn import train_all_classical_models

    table = pd.read_parquet(resolve_path(cfg.get("paths.processed_features")))
    comparison = train_all_classical_models(table, cfg)
    return comparison


def stage_clv(cfg) -> dict:
    """Train the CLV regression model."""
    table = pd.read_parquet(resolve_path(cfg.get("paths.processed_features")))
    return train_clv_model(table, cfg)


def _get_splits(table: pd.DataFrame, cfg):
    num_cols, cat_cols = get_feature_columns(table)
    X = table[num_cols + cat_cols]
    y = table["churn"].to_numpy()
    return stratified_splits(X, y, cfg)  # (X_tr, X_va, X_te, y_tr, y_va, y_te)


def stage_dl(cfg) -> dict:
    """Train ANN, LSTM and Autoencoder; persist metrics."""
    table = pd.read_parquet(resolve_path(cfg.get("paths.processed_features")))
    splits = _get_splits(table, cfg)

    ann_metrics = train_ann(table, splits, cfg)

    seq_path = resolve_path(cfg.get("paths.processed_sequences"))
    npz = np.load(seq_path)
    lstm_metrics = train_lstm(npz["sequences"], npz["mask"], table, splits, cfg)

    table = train_autoencoder(table, splits, cfg)
    out_path = resolve_path(cfg.get("paths.processed_features"))
    table.to_parquet(out_path, index=False)  # adds anomaly flags

    metrics = {"ann": ann_metrics, "lstm": lstm_metrics}
    (resolve_path(cfg.get("paths.models_dir")) / "dl_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics


def stage_segmentation(cfg):
    """Cluster customers and build business-readable segment profiles."""
    table = pd.read_parquet(resolve_path(cfg.get("paths.processed_features")))
    labeled, profiles, _diagnostics = run_segmentation(table, cfg)
    # Keep segmentation labels alongside the feature table.
    cols = ["customer_id", "segment", "gmm_segment", "pca_x", "pca_y"]
    labeled[cols].to_parquet(
        resolve_path(cfg.get("paths.processed_features")).parent
        / "customer_segments.parquet",
        index=False,
    )
    profiles.to_csv(
        resolve_path(cfg.get("paths.models_dir")) / "segment_profiles.csv", index=False
    )
    return profiles


def stage_explainability(cfg) -> dict[str, str]:
    """Run the SHAP/LIME/PDP XAI suite for the production churn model."""
    bundle = joblib.load(
        resolve_path(cfg.get("paths.models_dir")) / "churn_model.joblib"
    )
    table = pd.read_parquet(resolve_path(cfg.get("paths.processed_features")))
    return run_explainability(bundle, table, cfg)


def main() -> None:
    """CLI entrypoint for the pipeline."""
    parser = argparse.ArgumentParser(description="Vantara ML pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "features", "classical", "clv", "dl",
                 "segmentation", "explainability"],
        default="all",
    )
    args = parser.parse_args()
    cfg = load_config()

    stages = {
        "data": stage_data,
        "features": stage_features,
        "classical": stage_classical,
        "clv": stage_clv,
        "dl": stage_dl,
        "segmentation": stage_segmentation,
        "explainability": stage_explainability,
    }
    order = list(stages.keys())
    todo = order if args.stage == "all" else order[: order.index(args.stage) + 1]
    for name in todo:
        logger.info("==== stage: %s ====", name)
        stages[name](cfg)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
