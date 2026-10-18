"""Customer segmentation (PRD Section 9).

K-Means is the primary method; the number of clusters k is chosen by the
silhouette score over a configurable k-range (not an arbitrary fixed
number), with the elbow (inertia) curve recorded for the report. A
Gaussian Mixture Model with the same k is fitted to test whether the
spherical-cluster assumption hides overlapping/elliptical segments.
Hierarchical clustering on a sampled subset is included as an exploratory
dendrogram sanity check.

Every segment receives a **business-readable profile** (summary statistics
plus a human label like "champions" or "at-risk high-value"), because the
Marketing team — not a data scientist — is the consumer of this output.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.config import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEGMENT_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_order_value",
    "unique_products",
    "returns_ratio",
    "spend_last_90d",
    "month_active_ratio",
    "engagement_score",
]


def select_k(
    X: np.ndarray, k_range: list[int], seed: int
) -> tuple[int, pd.DataFrame]:
    """Choose k by best silhouette score; also record the elbow curve."""
    rows = []
    best_k, best_score = k_range[0], -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": score})
        logger.info("k=%d inertia=%.1f silhouette=%.4f", k, km.inertia_, score)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, pd.DataFrame(rows)


def profile_segments(feats: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    """Build a business-readable profile per segment.

    Adds a human-readable label using RFM-based heuristics on segment
    centroid medians, plus summary statistics for the Marketing team.
    """
    df = feats.copy()
    df["segment"] = labels

    def _label(g: pd.DataFrame) -> str:
        med_r = g["recency_days"].median()
        med_f = g["frequency"].median()
        med_m = g["monetary_total"].median()
        if med_r <= 30 and med_f >= 5 and med_m >= feats["monetary_total"].median():
            return "Champions (recent, frequent, high value)"
        if med_r <= 60 and med_f >= 3:
            return "Loyal regulars"
        if med_r > 120 and med_m >= feats["monetary_total"].median():
            return "At-risk high-value (lapsed big spenders)"
        if med_r > 120:
            return "Hibernating (lapsed low activity)"
        if med_f <= 2:
            return "New/one-off shoppers"
        return "Promising (moderate engagement)"

    profiles = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            pct_of_base=("customer_id", lambda s: len(s) / len(df)),
            median_recency_days=("recency_days", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary_total", "median"),
            median_aov=("avg_order_value", "median"),
            median_engagement=("engagement_score", "median"),
            churn_rate=("churn", "mean"),
            future_clv=("clv_future", "median"),
        )
        .reset_index()
    )
    labels_map = df.groupby("segment").apply(_label, include_groups=False)
    profiles["segment_label"] = profiles["segment"].map(labels_map)
    return profiles


def run_segmentation(
    feature_table: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full segmentation pipeline.

    Returns:
        (labeled feature table with ``segment`` column, segment profiles,
        k-selection diagnostics). Artifacts (scaler, KMeans, GMM, PCA) are
        persisted to the models directory.
    """
    seed = int(cfg.get("project.random_seed", 42))
    feats = feature_table[feature_table["frequency"] >= 1].copy()
    X = feats[SEGMENT_FEATURES].to_numpy(dtype=np.float64)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    k_range = [int(k) for k in cfg.get("segmentation.k_range", [2, 3, 4, 5, 6, 7, 8])]
    best_k, diagnostics = select_k(Xs, k_range, seed)
    logger.info("Selected k=%d by silhouette score", best_k)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed)
    labels = km.fit_predict(Xs)

    # GMM comparison with the same k (soft, elliptical clusters).
    gmm = GaussianMixture(n_components=best_k, random_state=seed)
    gmm_labels = gmm.fit_predict(Xs)
    gmm_silhouette = silhouette_score(Xs, gmm_labels)
    logger.info("GMM silhouette=%.4f vs K-Means %.4f", gmm_silhouette,
                silhouette_score(Xs, labels))

    # PCA projection for 2-D visualization of segments (PRD Section 16.2).
    pca = PCA(n_components=2, random_state=seed)
    proj = pca.fit_transform(Xs)
    feats["pca_x"], feats["pca_y"] = proj[:, 0], proj[:, 1]
    feats["segment"] = labels
    feats["gmm_segment"] = gmm_labels

    profiles = profile_segments(feats, labels, best_k)
    logger.info("Segment profiles:\n%s", profiles.to_string(index=False))

    # Hierarchical sanity check on a sampled subset (not persisted).
    sample_idx = np.random.default_rng(seed).choice(
        len(Xs), size=min(500, len(Xs)), replace=False
    )
    hc = AgglomerativeClustering(n_clusters=best_k)
    hc_labels = hc.fit_predict(Xs[sample_idx])
    hc_agreement = (hc_labels == labels[sample_idx]).mean()
    logger.info("Hierarchical sample agreement with K-Means: %.2f", hc_agreement)

    artifact_dir = resolve_path(cfg.get("paths.models_dir"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "kmeans": km, "gmm": gmm, "pca": pca,
         "features": SEGMENT_FEATURES, "k": best_k},
        artifact_dir / "segmentation.joblib",
    )
    diagnostics.to_csv(artifact_dir / "k_selection.csv", index=False)
    return feats, profiles, diagnostics
