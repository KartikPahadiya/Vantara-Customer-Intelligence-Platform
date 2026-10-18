"""Scoring service: loads artifacts once and exposes predict/explain logic.

Shared by the REST API and (indirectly, via HTTP) the Streamlit dashboard,
so the dashboard never loads models itself (PRD Section 14).
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.explainability.explain import plain_language_explanation
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """Holds all trained artifacts and serves predictions."""

    def __init__(self) -> None:
        cfg = load_config()
        models_dir = resolve_path(cfg.get("paths.models_dir"))

        churn_bundle = joblib.load(models_dir / "churn_model.joblib")
        self.churn_model = churn_bundle["model"]
        self.model_name = churn_bundle["model_name"]
        self.feature_columns = churn_bundle["feature_columns"]

        self.clv_bundle = joblib.load(models_dir / "clv_model.joblib")

        seg_path = models_dir / "segmentation.joblib"
        self.segmentation = joblib.load(seg_path) if seg_path.exists() else None

        profiles_path = models_dir / "segment_profiles.csv"
        self.segment_profiles = (
            pd.read_csv(profiles_path) if profiles_path.exists() else pd.DataFrame()
        )

        comparison_path = models_dir / "model_comparison.csv"
        self.comparison = (
            pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
        )

        # Feature lookup for customer-id scoring and explanations.
        features_path = resolve_path(cfg.get("paths.processed_features"))
        self.feature_table = (
            pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
        )
        seg_labels_path = features_path.parent / "customer_segments.parquet"
        self.segment_labels = (
            pd.read_parquet(seg_labels_path) if seg_labels_path.exists() else None
        )
        logger.info("ModelService loaded model=%s", self.model_name)

    # ------------------------------------------------------------------ utils
    def _frame(self, features: dict[str, Any]) -> pd.DataFrame:
        """Build a one-row feature frame with the exact training schema."""
        row = {c: features.get(c, 0) for c in self.feature_columns}
        return pd.DataFrame([row])

    def _assign_segment(self, X: pd.DataFrame) -> int:
        """Assign the K-Means segment for a feature row."""
        if self.segmentation is None:
            return -1
        seg_feats = self.segmentation["features"]
        Xs = self.segmentation["scaler"].transform(X[seg_feats].astype(float))
        return int(self.segmentation["kmeans"].predict(Xs)[0])

    def _clv(self, X: pd.DataFrame) -> float:
        """Predict future revenue (pounds) from the log1p regression output."""
        if not self.clv_bundle:
            return 0.0
        return float(np.expm1(self.clv_bundle["model"].predict(X)[0]))

    # ------------------------------------------------------------- prediction
    def predict(self, features: dict[str, Any], customer_id: int | None = None) -> dict:
        """Score one customer: churn probability, label, CLV, segment."""
        X = self._frame(features)
        prob = float(self.churn_model.predict_proba(X)[0, 1])
        segment = self._assign_segment(X)
        return {
            "customer_id": customer_id,
            "churn_probability": round(prob, 4),
            "churn_label": int(prob >= 0.5),
            "clv_estimate": round(max(self._clv(X), 0.0), 2),
            "segment": segment,
            "model_name": self.model_name,
        }

    def predict_by_id(self, customer_id: int) -> dict:
        """Score an existing customer straight from the feature table."""
        row = self.feature_table[self.feature_table["customer_id"] == customer_id]
        if row.empty:
            raise KeyError(f"customer_id {customer_id} not in feature table")
        feats = row.iloc[0][self.feature_columns].to_dict()
        result = self.predict(feats, customer_id=customer_id)
        if self.segment_labels is not None:
            seg_row = self.segment_labels[
                self.segment_labels["customer_id"] == customer_id
            ]
            if not seg_row.empty:
                result["segment"] = int(seg_row.iloc[0]["segment"])
        return result

    def explain(self, customer_id: int) -> dict:
        """SHAP-based explanation for one customer."""
        row = self.feature_table[self.feature_table["customer_id"] == customer_id]
        if row.empty:
            raise KeyError(f"customer_id {customer_id} not in feature table")
        X = self._frame(row.iloc[0][self.feature_columns].to_dict())
        Xt = self.churn_model.named_steps["preprocess"].transform(X)
        model = self.churn_model.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        sv_row = sv[0]
        cols = [
            n.split("__", 1)[-1]
            for n in self.churn_model.named_steps["preprocess"].get_feature_names_out()
        ]
        contrib = (
            pd.Series(sv_row, index=cols).sort_values(key=np.abs, ascending=False)
        )
        top = [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in contrib.head(8).items()
        ]
        return {
            "customer_id": customer_id,
            "churn_probability": self.predict_by_id(customer_id)["churn_probability"],
            "top_contributors": top,
            "plain_language": plain_language_explanation(
                row.iloc[0], sv_row, cols
            ),
        }

    def metadata(self) -> dict:
        """Model card: name, features, test metrics, segment profiles."""
        metrics: dict[str, Any] = {}
        if not self.comparison.empty:
            metrics = self.comparison.to_dict(orient="records")
        return {
            "model_name": self.model_name,
            "feature_columns": self.feature_columns,
            "metrics": metrics,
            "training_data_rows": (
                int(len(self.feature_table)) if not self.feature_table.empty else None
            ),
            "segment_profiles": (
                self.segment_profiles.to_dict(orient="records")
                if not self.segment_profiles.empty
                else []
            ),
        }
