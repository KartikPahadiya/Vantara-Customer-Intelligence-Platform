"""Explainability (PRD Section 10).

* SHAP global summary plot for the production churn model;
* SHAP force plots for three representative customers (low risk,
  high risk, borderline);
* one LIME local explanation compared against SHAP for the same customer;
* Partial Dependence Plots for the two most influential features;
* a plain-language "why this customer is flagged" template that a
  marketing analyst can read without any ML background.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay

from src.utils.config import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

NICE_NAMES = {
    "recency_days": "days since their last order",
    "frequency": "number of past orders",
    "monetary_total": "total historical spend",
    "avg_order_value": "average order value",
    "unique_products": "number of different products bought",
    "returns_ratio": "share of returned items",
    "tenure_days": "length of relationship",
    "avg_gap_days": "average gap between orders",
    "spend_last_30d": "spend in the last 30 days",
    "spend_last_90d": "spend in the last 90 days",
    "spend_slope_6m": "recent spend trend",
    "engagement_score": "engagement score",
    "month_active_ratio": "share of active months",
    "orders_per_month": "orders per month",
}


def _clean_feature_names(names: list[str]) -> list[str]:
    """Strip the ColumnTransformer ``num__``/``cat__`` prefixes."""
    return [n.split("__", 1)[-1] for n in names]


def _preprocess_for_shap(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Apply the production pipeline's preprocessing without the model step."""
    pre = bundle["model"].named_steps["preprocess"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    raw_names = list(pre.get_feature_names_out())
    return np.asarray(Xt, dtype=float), _clean_feature_names(raw_names)


def _churn_class_shap_values(explainer: Any, Xt: np.ndarray) -> np.ndarray:
    """Return SHAP values for the churn (positive) class.

    For binary classifiers ``shap_values`` has shape (n, features, 2);
    reduce to the positive class.
    """
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        return np.asarray(sv[-1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[..., -1]
    return sv


def _base_value(explainer: Any) -> float:
    """Scalar base value for the churn class."""
    ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    return float(ev[-1])


def explain_with_shap(
    bundle: dict,
    X_sample: pd.DataFrame,
    out_dir: str | Path,
) -> np.ndarray:
    """Global SHAP summary + per-customer values for the production model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Xt, cols = _preprocess_for_shap(bundle, X_sample)
    model = bundle["model"].named_steps["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = _churn_class_shap_values(explainer, Xt)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, Xt, feature_names=cols, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close()
    logger.info("SHAP summary plot saved to %s", out_dir / "shap_summary.png")
    return shap_values


def shap_force_plots(
    bundle: dict,
    X_sample: pd.DataFrame,
    y_prob: np.ndarray,
    out_dir: str | Path,
) -> dict[str, int]:
    """Force plots for one low-risk, one high-risk and one borderline customer."""
    out_dir = Path(out_dir)
    Xt, cols = _preprocess_for_shap(bundle, X_sample)
    model = bundle["model"].named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = _churn_class_shap_values(explainer, Xt)

    order = np.argsort(y_prob)
    picks = {
        "low_risk": int(order[0]),
        "high_risk": int(order[-1]),
        "borderline": int(np.argmin(np.abs(y_prob - 0.5))),
    }
    base_value = _base_value(explainer)
    for name, idx in picks.items():
        force = shap.force_plot(
            base_value, shap_values[idx], Xt[idx], feature_names=cols,
            matplotlib=False, show=False,
        )
        shap.save_html(str(out_dir / f"shap_force_{name}.html"), force)
    logger.info("SHAP force plots saved for %s", list(picks))
    return picks


def lime_explanation(
    bundle: dict,
    X_sample: pd.DataFrame,
    y_prob: np.ndarray,
    customer_pos: int,
    out_dir: str | Path,
) -> pd.DataFrame:
    """LIME local explanation for one customer, saved as CSV of contributions."""
    out_dir = Path(out_dir)
    Xt, cols = _preprocess_for_shap(bundle, X_sample)
    model = bundle["model"].named_steps["model"]

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        # LIME perturbs in the transformed space and needs full class
        # probabilities (n, 2), not a single score column.
        return model.predict_proba(arr)

    explainer = LimeTabularExplainer(
        Xt,
        feature_names=cols,
        class_names=["stay", "churn"],
        mode="classification",
        random_state=42,
        discretize_continuous=False,
    )
    exp = explainer.explain_instance(Xt[customer_pos], predict_fn, num_features=10)
    rows = [
        {"feature": f, "lime_weight": w}
        for f, w in exp.as_list()
    ]
    lime_df = pd.DataFrame(rows)
    lime_df.to_csv(out_dir / "lime_explanation.csv", index=False)
    logger.info("LIME explanation saved to %s", out_dir / "lime_explanation.csv")
    return lime_df


def partial_dependence_plots(
    bundle: dict, X_sample: pd.DataFrame, features: list[str], out_dir: str | Path
) -> None:
    """PDPs showing the direction/shape of the top features' effect."""
    out_dir = Path(out_dir)
    # PDP requires floating-point feature columns (int columns would be
    # implicitly rounded on the evaluation grid).
    X_float = X_sample.copy()
    for c in X_float.select_dtypes(include=["int"]).columns:
        X_float[c] = X_float[c].astype(float)
    fig, ax = plt.subplots(figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(
        bundle["model"], X_float, features=features, grid_resolution=30, ax=ax
    )
    plt.tight_layout()
    plt.savefig(out_dir / "pdp_top_features.png", dpi=150)
    plt.close()
    logger.info("PDP saved to %s", out_dir / "pdp_top_features.png")


def plain_language_explanation(
    feature_row: pd.Series, shap_row: np.ndarray, feature_names: list[str]
) -> str:
    """Translate SHAP values into one readable sentence for an analyst.

    Picks the three features pushing the risk **up** and the two pushing it
    **down** and phrases them in business language.
    """
    contrib = pd.Series(shap_row, index=feature_names)
    up = contrib.sort_values(ascending=False).head(3)
    down = contrib.sort_values().head(2)

    def fmt(name: str) -> str:
        nice = NICE_NAMES.get(name, name.replace("_", " "))
        val = feature_row.get(name)
        if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
            if name in ("monetary_total", "avg_order_value", "spend_last_90d",
                        "spend_last_30d"):
                return f"{nice} of £{val:,.0f}"
            if name in ("recency_days", "tenure_days", "avg_gap_days"):
                return f"{nice} ({val:,.0f} days)"
            return f"{nice} ({val:,.1f})"
        return nice

    up_txt = "; ".join(f"higher {fmt(n)} (+{v:.3f} risk)" for n, v in up.items() if v > 0)
    down_txt = "; ".join(f"lower {fmt(n)} ({v:.3f})" for n, v in down.items() if v < 0)
    parts = [p for p in (up_txt, down_txt) if p]
    return (
        "Why this customer is flagged: " + ". Protective factors: ".join(parts)
        + "."
    )


def run_explainability(
    bundle: dict, feature_table: pd.DataFrame, cfg: Config
) -> dict[str, str]:
    """Run the full XAI suite and persist every output (PRD Section 10)."""
    seed = int(cfg.get("project.random_seed", 42))
    artifact_dir = resolve_path(cfg.get("paths.models_dir")) / "explainability"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cols = bundle["feature_columns"]
    sampled = feature_table.sample(n=min(2000, len(feature_table)), random_state=seed)
    X = sampled[cols]
    y_prob = bundle["model"].predict_proba(X)[:, 1]

    shap_values = explain_with_shap(bundle, X, artifact_dir)
    picks = shap_force_plots(bundle, X, y_prob, artifact_dir)
    lime_explanation(bundle, X, y_prob, picks["borderline"], artifact_dir)

    # PDPs on the two most influential features by mean |SHAP|.
    Xt, tcols = _preprocess_for_shap(bundle, X)
    importance = np.abs(shap_values).mean(axis=0)
    top2 = [tcols[i] for i in np.argsort(importance)[::-1][:2]]
    partial_dependence_plots(bundle, X, top2, artifact_dir)

    # Save the per-customer SHAP values for the dashboard's explanation panel.
    np.savez(
        artifact_dir / "shap_values.npz",
        shap_values=shap_values,
        customer_ids=X["customer_id"].to_numpy() if "customer_id" in X else np.arange(len(X)),
        feature_names=np.array(tcols),
    )
    return {"shap_summary": str(artifact_dir / "shap_summary.png"),
            "pdp": str(artifact_dir / "pdp_top_features.png")}
