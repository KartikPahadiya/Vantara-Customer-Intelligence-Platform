"""Vantara business-facing dashboard (PRD Section 12.2).

Run with::

    streamlit run frontend/dashboard.py

The dashboard reads the served models exclusively through the FastAPI
service (``API_BASE_URL``, default ``http://localhost:8000``) and the
persisted predictions/segments store, so it never loads a model itself.

Views:
1. Segment explorer – interactive filters by segment, country, value tier
2. Churn risk leaderboard – high-value, high-risk customers first
3. Revenue trends – monthly revenue with a simple linear forecast overlay
4. Model comparison – all churn models ranked by validation AUC/recall
5. Explainability – SHAP panel for any customer, searchable by ID
6. Batch scoring – CSV upload + downloadable results
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.utils.config import load_config, resolve_path

st.set_page_config(page_title="Vantara Customer Intelligence", layout="wide")

API_BASE = __import__("os").getenv("API_BASE_URL", "http://localhost:8000")
cfg = load_config()

FEATURES_PATH = resolve_path(cfg.get("paths.processed_features"))
SEGMENTS_PATH = FEATURES_PATH.parent / "customer_segments.parquet"
INTERIM_PATH = resolve_path(cfg.get("paths.interim_data"))
MODELS_DIR = resolve_path(cfg.get("paths.models_dir"))

TITLE = "Vantara Customer Intelligence Platform"


@st.cache_data(ttl=300)
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    features = pd.read_parquet(FEATURES_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH) if SEGMENTS_PATH.exists() else None
    return features, segments


@st.cache_data(ttl=300)
def load_transactions() -> pd.DataFrame:
    return pd.read_parquet(INTERIM_PATH) if INTERIM_PATH.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def api_get(path: str) -> dict:
    r = requests.get(f"{API_BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def api_explain(customer_id: int) -> dict:
    r = requests.get(f"{API_BASE}/explain/{customer_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def api_health() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=5).ok
    except requests.RequestException:
        return False


def main() -> None:
    st.title(TITLE)

    api_ok = api_health()
    st.sidebar.caption(f"API: {API_BASE} — {'🟢 up' if api_ok else '🔴 unreachable'}")
    view = st.sidebar.radio(
        "View",
        [
            "Churn risk leaderboard",
            "Segment explorer",
            "Revenue trends & forecast",
            "Model comparison",
            "Customer explainability",
            "Batch scoring",
        ],
    )

    features, segments = load_tables()
    if segments is not None:
        features = features.merge(
            segments[["customer_id", "segment", "pca_x", "pca_y"]],
            on="customer_id",
            how="left",
        )
    profiles = pd.DataFrame()
    if (MODELS_DIR / "segment_profiles.csv").exists():
        profiles = pd.read_csv(MODELS_DIR / "segment_profiles.csv")

    if view == "Churn risk leaderboard":
        st.header("Churn risk leaderboard — high value, high risk first")
        col_a, col_b, col_c = st.columns(3)
        min_prob = col_a.slider("Min churn probability", 0.0, 1.0, 0.5)
        min_value = col_b.number_input(
            "Min monetary_total (£)", min_value=0.0, value=500.0, step=100.0
        )
        top_n = col_c.slider("Show top N", 10, 500, 100)

        # High-value × high-risk ranking: value percentile × lapse risk.
        board = (
            features[features["monetary_total"] >= min_value]
            .assign(
                value_pct=lambda d: d["monetary_total"].rank(pct=True),
                lapse_risk=lambda d: 1 - d["recency_days"].rank(pct=True),
                value_x_risk=lambda d: d["value_pct"] * d["lapse_risk"],
            )
            .pipe(lambda d: d[d["lapse_risk"] >= min_prob])
            .sort_values("value_x_risk", ascending=False)
            .head(top_n)[
                ["customer_id", "recency_days", "frequency", "monetary_total",
                 "engagement_score", "value_x_risk", "churn", "clv_future"]
            ]
        )
        st.dataframe(board, use_container_width=True)
        st.caption(
            "Note: engagement score is pre-cutoff; ``churn``/``clv_future`` are "
            "post-cutoff ground truth (not visible to the model at score time)."
        )

    elif view == "Segment explorer":
        st.header("Customer segments")
        if profiles.empty:
            st.warning("Run the segmentation stage first.")
            return
        st.subheader("Segment profiles (business-readable)")
        st.dataframe(profiles, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        seg_pick = c1.multiselect(
            "Segment", options=sorted(features["segment"].dropna().unique().tolist())
        )
        country_pick = c2.multiselect(
            "Country", options=sorted(features["country"].dropna().unique().tolist())
        )
        tier = c3.selectbox("Value tier", ["All", "Top 25% (M4)", "M3", "M2", "Bottom (M1)"])
        filtered = features.copy()
        if seg_pick:
            filtered = filtered[filtered["segment"].isin(seg_pick)]
        if country_pick:
            filtered = filtered[filtered["country"].isin(country_pick)]
        if tier != "All":
            filtered = filtered[filtered["m_score"] == int(tier[1])]

        fig = px.scatter(
            filtered, x="pca_x", y="pca_y", color="segment",
            hover_data=["customer_id", "monetary_total", "frequency", "recency_days"],
            title="Customer map (PCA projection of segment features)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(filtered):,} customers after filters")

    elif view == "Revenue trends & forecast":
        st.header("Sales & revenue trends with forecast overlay")
        tx = load_transactions()
        if tx.empty:
            st.warning("Run the data pipeline first.")
            return
        monthly = (
            tx.set_index("invoice_date")
            .groupby(pd.Grouper(freq="ME"))["line_revenue"]
            .sum()
            .reset_index()
            .rename(columns={"invoice_date": "month", "line_revenue": "revenue"})
        )
        # Simple forecast: linear trend on the last 12 observed months.
        horizon = int(cfg.get("dashboard.forecast_days", 90)) // 30
        train = monthly.dropna().tail(12).reset_index(drop=True)
        if len(train) >= 3:
            t = np.arange(len(train))  # noqa: F821
            slope, intercept = np.polyfit(t, train["revenue"], 1)  # noqa: F821
            future_t = np.arange(len(train), len(train) + horizon)  # noqa: F821
            forecast = pd.DataFrame(
                {
                    "month": pd.date_range(
                        train["month"].iloc[-1] + pd.offsets.MonthBegin(1),
                        periods=horizon,
                        freq="ME",
                    ),
                    "revenue": intercept + slope * future_t,
                    "kind": "forecast",
                }
            )
        else:
            forecast = pd.DataFrame()
        monthly_plot = monthly.assign(kind="observed")
        if not forecast.empty:
            monthly_plot = pd.concat([monthly_plot, forecast], ignore_index=True)
        fig = px.line(
            monthly_plot, x="month", y="revenue", color="kind",
            markers=True, title="Monthly revenue (observed vs. linear-trend forecast)",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Model comparison":
        st.header("Model comparison")
        comp_path = MODELS_DIR / "model_comparison.csv"
        if not comp_path.exists():
            st.warning("Train models first.")
            return
        comp = pd.read_csv(comp_path)
        show = comp[["model", "val_roc_auc", "val_recall", "val_f1",
                     "test_roc_auc", "test_recall", "train_seconds"]]
        st.dataframe(show, use_container_width=True)
        fig = px.bar(
            comp.sort_values("val_roc_auc"),
            x="val_roc_auc", y="model", orientation="h",
            color="val_recall", color_continuous_scale="Blues",
            title="Validation ROC-AUC (colour = recall)",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Customer explainability":
        st.header("Why is this customer flagged?")
        if not api_ok:
            st.error("FastAPI service is unreachable — start it with `uvicorn api.main:app`.")
            return
        cid = st.number_input(
            "Customer ID", min_value=0,
            value=int(features["customer_id"].iloc[0]),
        )
        try:
            expl = api_explain(int(cid))
        except requests.HTTPError as exc:
            st.error(f"API error: {exc}")
            return
        c1, c2 = st.columns(2)
        c1.metric("Churn probability", f"{expl['churn_probability']:.1%}")
        c2.write(expl["plain_language"])
        contrib = pd.DataFrame(expl["top_contributors"])
        fig = px.bar(
            contrib.sort_values("shap_value"),
            x="shap_value", y="feature", orientation="h",
            title="Top SHAP contributors (positive = pushes towards churn)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Global feature importance (SHAP summary)")
        img = MODELS_DIR / "explainability" / "shap_summary.png"
        if img.exists():
            st.image(str(img), use_container_width=True)

    elif view == "Batch scoring":
        st.header("Batch scoring (CSV upload)")
        if not api_ok:
            st.error("FastAPI service is unreachable — start it first.")
            return
        up = st.file_uploader("Upload CSV with feature columns", type=["csv"])
        if up is not None:
            files = {"file": (up.name, up.getvalue(), "text/csv")}
            r = requests.post(f"{API_BASE}/predict/batch", files=files, timeout=300)
            if r.ok:
                out = pd.DataFrame(r.json()["predictions"])
                st.success(f"Scored {r.json()['scored']} customers.")
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Download results CSV",
                    out.to_csv(index=False).encode(),
                    "vantara_scores.csv", "text/csv",
                )
            else:
                st.error(f"Batch scoring failed: {r.text}")


if __name__ == "__main__":
    main()
