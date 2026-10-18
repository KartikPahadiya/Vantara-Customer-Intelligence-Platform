"""Customer-level feature engineering (PRD Section 7).

Point-in-time cutoff discipline
-------------------------------
Every feature is computed **only** from transactions strictly before the
snapshot date, and every target **only** from transactions at/after the
snapshot date. This prevents target leakage — the single most common and
most serious error in churn modelling. The leakage guarantee is enforced
in code by construction (two disjoint transaction windows) and pinned by
the unit tests in ``tests/test_features.py``.

Feature justifications (one line each, PRD Section 7 requirement):

* ``recency_days``        – time since last purchase; the strongest single churn signal.
* ``frequency``           – number of orders; loyal habitual buyers churn less.
* ``monetary_total``      – cumulative spend; separates high-value from low-value customers.
* ``avg_order_value``     – basket size; captures buying power distinct from visit count.
* ``unique_products``     – catalogue breadth; engagement depth proxy.
* ``returns_count``/ratio – return behaviour predicts dissatisfaction and churn.
* ``tenure_days``         – relationship length; mature customers behave differently.
* ``avg_gap_days``        – natural purchase rhythm; rhythm breaks precede churn.
* ``gap_std``             – regularity of purchase timing; erratic buyers are riskier.
* ``spend_last_30d/90d``  – short-term momentum; cooling spend predicts churn.
* ``spend_slope_6m``      – linear trend of monthly spend; declining trend predicts churn.
* ``orders_per_month``    – normalized visit cadence independent of tenure.
* ``month_active_ratio``  – share of tenure months with activity; stickiness.
* ``avg_quantity``        – bulk vs. retail buying style.
* ``country_*``           – market effects (delivery cost, seasonality) on behaviour.
* ``admin_spend_ratio``   – share of spend on postage/admin; friction proxy.
* RFM scores              – quartile ranks recency/frequency/monetary into 1–4 scores.
* ``engagement_score``    – composite 0–100 RFM weighted score for tiering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_COUNTRIES = 10  # keep top-10 countries one-hot, rest -> "Other"


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division returning 0 where the denominator is 0."""
    return (num / den.replace(0, np.nan)).fillna(0.0)


def split_by_cutoff(
    transactions: pd.DataFrame, snapshot_date: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into the feature window (< snapshot) and the
    target window (>= snapshot). The two windows are disjoint by construction.
    """
    snapshot = pd.Timestamp(snapshot_date)
    history = transactions[transactions["invoice_date"] < snapshot].copy()
    future = transactions[transactions["invoice_date"] >= snapshot].copy()
    return history, future


def compute_customer_features(
    history: pd.DataFrame, snapshot_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Aggregate pre-cutoff transactions into one row per customer.

    Args:
        history: Transactions strictly before the snapshot date.
        snapshot_date: Point-in-time cutoff.

    Returns:
        Customer-level feature DataFrame indexed by ``customer_id``.
    """
    snapshot = pd.Timestamp(snapshot_date)
    g = history.groupby("customer_id")

    feats = pd.DataFrame(
        {
            "recency_days": (
                snapshot - g["invoice_date"].max()
            ).dt.total_seconds()
            / 86400.0,
            "frequency": g["invoice"].nunique(),
            "monetary_total": g["line_revenue"].sum(),
            "avg_order_value": g.apply(
                lambda d: d.groupby("invoice")["line_revenue"].sum().mean(),
                include_groups=False,
            ),
            "unique_products": g["stock_code"].nunique(),
            "returns_count": g["is_return"].sum(),
            "tenure_days": (
                g["invoice_date"].max() - g["invoice_date"].min()
            ).dt.total_seconds()
            / 86400.0,
            "avg_quantity": g["quantity"].mean(),
            "admin_spend": g.apply(
                lambda d: d.loc[d["is_admin_code"], "line_revenue"].sum(),
                include_groups=False,
            ),
        }
    ).fillna(0.0)

    # Purchase-gap statistics from successive order timestamps.
    order_dates = (
        history.groupby(["customer_id", "invoice"])["invoice_date"].min().reset_index()
    )
    gaps = order_dates.sort_values("invoice_date").groupby("customer_id")[
        "invoice_date"
    ].diff().dt.total_seconds() / 86400.0
    order_dates["gap_days"] = gaps
    gap_stats = order_dates.groupby("customer_id")["gap_days"].agg(
        ["mean", "std"]
    )
    feats["avg_gap_days"] = gap_stats["mean"]
    feats["gap_std"] = gap_stats["std"]

    # Short-window momentum.
    for days in (30, 90):
        lo = snapshot - pd.Timedelta(days=days)
        spend = history[history["invoice_date"] >= lo].groupby("customer_id")[
            "line_revenue"
        ].sum()
        feats[f"spend_last_{days}d"] = spend

    # Monthly spend slope over the last 6 months (linear trend coefficient).
    lo6m = snapshot - pd.DateOffset(months=6)
    monthly = (
        history[history["invoice_date"] >= lo6m]
        .set_index("invoice_date")
        .groupby("customer_id")["line_revenue"]
        .resample("ME")
        .sum()
        .reset_index()
    )
    monthly["t"] = (
        monthly["invoice_date"] - monthly["invoice_date"].min()
    ).dt.days / 30.0

    def _slope(d: pd.DataFrame) -> float:
        if len(d) < 2 or d["t"].std() == 0:
            return 0.0
        return float(np.polyfit(d["t"], d["line_revenue"], 1)[0])

    slopes = monthly.groupby("customer_id").apply(_slope, include_groups=False)
    feats["spend_slope_6m"] = slopes

    # Cadence / stickiness.
    tenure_months = np.maximum(feats["tenure_days"] / 30.44, 1.0)
    active_months = (
        history.groupby("customer_id")["invoice_date"]
        .apply(lambda s: s.dt.to_period("M").nunique())
        .astype(float)
    )
    feats["orders_per_month"] = feats["frequency"] / tenure_months
    feats["month_active_ratio"] = active_months / tenure_months

    # Derived ratios.
    feats["avg_order_value"] = feats["avg_order_value"].fillna(0.0)
    feats["returns_ratio"] = _safe_div(
        feats["returns_count"].astype(float),
        history.groupby("customer_id").size().astype(float),
    )
    feats["admin_spend_ratio"] = _safe_div(
        feats["admin_spend"], feats["monetary_total"]
    )
    feats["avg_gap_days"] = feats["avg_gap_days"].fillna(0.0)
    feats["gap_std"] = feats["gap_std"].fillna(0.0)
    feats["spend_last_30d"] = feats["spend_last_30d"].fillna(0.0)
    feats["spend_last_90d"] = feats["spend_last_90d"].fillna(0.0)
    feats["spend_slope_6m"] = feats["spend_slope_6m"].fillna(0.0)

    # Mode country (most frequent invoice country).
    country = (
        history.groupby("customer_id")["country"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "Unknown")
        .rename("country")
    )
    feats = feats.join(country)
    feats["is_uk"] = (feats["country"] == "United Kingdom").astype(int)

    top = feats["country"].value_counts().head(TOP_COUNTRIES).index
    country_oh = pd.get_dummies(
        feats["country"].where(feats["country"].isin(top), "Other")
    ).astype(int)
    country_oh.columns = [f"country_{c}" for c in country_oh.columns]
    feats = feats.join(country_oh)

    return feats.reset_index()


def compute_targets(
    future: pd.DataFrame,
    snapshot_date: str | pd.Timestamp,
    churn_window_days: int,
    clv_window_days: int,
) -> pd.DataFrame:
    """Compute churn label and CLV target from post-cutoff transactions only.

    * ``churn`` = 1 if the customer made **no** purchase within
      ``churn_window_days`` after the snapshot.
    * ``clv_future`` = total revenue in the ``clv_window_days`` after snapshot.
    """
    snapshot = pd.Timestamp(snapshot_date)
    churn_hi = snapshot + pd.Timedelta(days=churn_window_days)
    clv_hi = snapshot + pd.Timedelta(days=clv_window_days)

    first_next = future.groupby("customer_id")["invoice_date"].min()
    churned = pd.Series(0, index=first_next.index, dtype=int)
    churned[first_next >= churn_hi] = 1
    churned = churned.rename("churn")

    clv = (
        future[future["invoice_date"] < clv_hi]
        .groupby("customer_id")["line_revenue"]
        .sum()
        .rename("clv_future")
    )

    targets = pd.concat([churned, clv], axis=1).fillna({"churn": 1, "clv_future": 0.0})
    targets["churn"] = targets["churn"].astype(int)
    return targets.reset_index()


def add_rfm_scores(feats: pd.DataFrame, n_quantiles: int = 4) -> pd.DataFrame:
    """Add R (recency), F (frequency), M (monetary) quartile scores (1..n).

    Recency is reverse-scored: the most recent buyer gets the top score.
    Also adds a composite ``engagement_score`` in 0–100 (25% R + 25% F +
    50% M — monetary weighted double because it drives business value).
    """
    feats = feats.copy()
    n = n_quantiles
    feats["r_score"] = (
        pd.qcut(feats["recency_days"].rank(method="first", ascending=False), n, labels=False) + 1
    )
    feats["f_score"] = (
        pd.qcut(feats["frequency"].rank(method="first"), n, labels=False) + 1
    )
    feats["m_score"] = (
        pd.qcut(feats["monetary_total"].rank(method="first"), n, labels=False) + 1
    )
    feats["engagement_score"] = (
        100.0
        * (0.25 * feats["r_score"] + 0.25 * feats["f_score"] + 0.5 * feats["m_score"])
        / n
    ).round(1)
    return feats


def build_feature_table(
    transactions: pd.DataFrame,
    snapshot_date: str,
    churn_window_days: int,
    clv_window_days: int,
    rfm_quantiles: int = 4,
) -> pd.DataFrame:
    """End-to-end: split by cutoff, compute features and targets, join.

    Args:
        transactions: Cleaned customer-level transaction table.
        snapshot_date: Point-in-time cutoff (features strictly before,
            targets at/after).
        churn_window_days: Inactivity horizon defining churn.
        clv_window_days: Revenue horizon defining the CLV regression target.
        rfm_quantiles: Number of quantiles for RFM scores.

    Returns:
        One row per customer with features + ``churn`` + ``clv_future``.
    """
    history, future = split_by_cutoff(transactions, snapshot_date)
    features = compute_customer_features(history, snapshot_date)
    targets = compute_targets(future, snapshot_date, churn_window_days, clv_window_days)
    table = features.merge(targets, on="customer_id", how="left")
    table["churn"] = table["churn"].fillna(1).astype(int)
    table["clv_future"] = table["clv_future"].fillna(0.0)
    table = add_rfm_scores(table, rfm_quantiles)

    logger.info(
        "Feature table built: %d customers | churn rate %.1f%% | mean future CLV %.2f",
        len(table),
        100 * table["churn"].mean(),
        table["clv_future"].mean(),
    )
    return table


def build_lstm_sequences(
    history: pd.DataFrame,
    customer_ids: np.ndarray,
    snapshot_date: str,
    max_len: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded time-ordered purchase-event sequences per customer.

    Each event is ``[log1p(amount), gap_days]`` where ``amount`` is the
    order total for that invoice and ``gap_days`` the days since the
    previous order. Sequences are right-padded with zeros and returned
    together with a boolean padding mask.

    Returns:
        Tuple ``(sequences, mask)`` with shapes ``(n_customers, max_len, 2)``
        and ``(n_customers, max_len)``.
    """
    snapshot = pd.Timestamp(snapshot_date)
    order_totals = (
        history.groupby(["customer_id", "invoice"])
        .agg(amount=("line_revenue", "sum"), date=("invoice_date", "min"))
        .reset_index()
    )
    order_totals = order_totals.sort_values(["customer_id", "date"])
    order_totals["gap_days"] = (
        order_totals.groupby("customer_id")["date"].diff().dt.total_seconds() / 86400.0
    )
    order_totals["gap_days"] = order_totals["gap_days"].fillna(0.0)
    order_totals["gap_days"] = order_totals["gap_days"].clip(
        upper=(snapshot - order_totals["date"]).dt.total_seconds() / 86400.0
    )

    n = len(customer_ids)
    seq = np.zeros((n, max_len, 2), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=bool)
    index = {cid: i for i, cid in enumerate(customer_ids)}

    for cid, grp in order_totals.groupby("customer_id"):
        i = index.get(cid)
        if i is None:
            continue
        events = grp.tail(max_len)
        amt = np.log1p(np.clip(events["amount"].to_numpy(), 0, None))
        gap = events["gap_days"].to_numpy(dtype=np.float32)
        L = len(events)
        seq[i, -L:, 0] = amt
        seq[i, -L:, 1] = gap
        mask[i, -L:] = True

    return seq, mask


def fit_feature_scaler(
    X_train: pd.DataFrame, artifact_dir: str | Path, numeric_cols: list[str]
) -> StandardScaler:
    """Fit and persist a ``StandardScaler`` for scale-sensitive models.

    The same fitted scaler is reloaded at inference time so training and
    serving apply the identical transformation (PRD Section 6.3).
    """
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(
        {"scaler": scaler, "numeric_cols": numeric_cols},
        artifact_dir / "feature_scaler.joblib",
    )
    logger.info("Feature scaler saved to %s", artifact_dir / "feature_scaler.joblib")
    return scaler
