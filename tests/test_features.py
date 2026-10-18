"""Unit tests for feature engineering — including target-leakage prevention.

The leakage tests (PRD Section 7) construct synthetic transactions where a
customer's behaviour *changes completely* after the snapshot date. If any
feature or target leaks across the cutoff, these tests fail loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_transactions
from src.data.validate import ValidationError, check_schema
from src.features.build_features import (
    build_feature_table,
    build_lstm_sequences,
    compute_customer_features,
    compute_targets,
    split_by_cutoff,
)

SNAPSHOT = "2011-06-01"


def _txn(customer_id: int, date: str, quantity: int = 1, price: float = 10.0,
         invoice: str | None = None, country: str = "United Kingdom") -> dict:
    return {
        "invoice": invoice or f"{customer_id}-{date}",
        "stock_code": "85099B",
        "description": "JUMBO BAG RED RETROSPOT",
        "quantity": quantity,
        "invoice_date": pd.Timestamp(date),
        "price": price,
        "customer_id": float(customer_id),
        "country": country,
    }


@pytest.fixture()
def transactions() -> pd.DataFrame:
    """Two customers: A is active before and after the cutoff; B is active
    only before (churner); C is active only after (new — must not leak into
    pre-cutoff features)."""
    rows = [
        _txn(1, "2011-05-01", quantity=2, price=5.0, invoice="A1"),
        _txn(1, "2011-05-15", quantity=1, price=20.0, invoice="A2"),
        _txn(1, "2011-07-10", quantity=1, price=30.0, invoice="A3"),
        _txn(2, "2011-04-20", quantity=1, price=10.0, invoice="B1"),
        _txn(3, "2011-07-05", quantity=5, price=100.0, invoice="C1"),
    ]
    df = pd.DataFrame(rows)
    df["line_revenue"] = df["quantity"] * df["price"]
    df["is_admin_code"] = False
    df["is_return"] = df["quantity"] < 0
    df["is_price_error"] = False
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    return df


# --------------------------------------------------------------------- split
def test_split_by_cutoff_is_disjoint(transactions: pd.DataFrame) -> None:
    history, future = split_by_cutoff(transactions, SNAPSHOT)
    assert set(history["invoice"]).isdisjoint(set(future["invoice"]))
    assert (history["invoice_date"] < pd.Timestamp(SNAPSHOT)).all()
    assert (future["invoice_date"] >= pd.Timestamp(SNAPSHOT)).all()


def test_features_ignore_future_activity(transactions: pd.DataFrame) -> None:
    """Customer 3 buys £500 *after* the cutoff — must not appear in features."""
    history, _ = split_by_cutoff(transactions, SNAPSHOT)
    feats = compute_customer_features(history, SNAPSHOT).set_index("customer_id")
    assert 3 not in feats.index, "future-only customer leaked into features"
    assert 1 in feats.index and 2 in feats.index
    # Customer 1's monetary must only count pre-cutoff spend (10 + 20 = 30).
    assert feats.loc[1, "monetary_total"] == pytest.approx(30.0)


def test_recency_measured_to_snapshot(transactions: pd.DataFrame) -> None:
    history, _ = split_by_cutoff(transactions, SNAPSHOT)
    feats = compute_customer_features(history, SNAPSHOT).set_index("customer_id")
    # Last pre-cutoff order 2011-05-15 -> 17 days before 2011-06-01.
    assert feats.loc[1, "recency_days"] == pytest.approx(17.0, abs=1e-6)


# -------------------------------------------------------------------- targets
def test_targets_use_future_window_only(transactions: pd.DataFrame) -> None:
    _, future = split_by_cutoff(transactions, SNAPSHOT)
    targets = compute_targets(future, SNAPSHOT, churn_window_days=90,
                              clv_window_days=180).set_index("customer_id")
    assert targets.loc[1, "churn"] == 0          # ordered again within 90 days
    assert targets.loc[1, "clv_future"] == pytest.approx(30.0)
    # Customer 2 never appears in the future window -> churn=1 via the
    # left-merge fill in build_feature_table.
    table = build_feature_table(
        transactions, SNAPSHOT, churn_window_days=90, clv_window_days=180
    ).set_index("customer_id")
    assert table.loc[2, "churn"] == 1
    assert table.loc[2, "clv_future"] == 0.0


def test_no_leakage_full_table(transactions: pd.DataFrame) -> None:
    """End-to-end: per-customer feature rows contain no post-cutoff signal."""
    table = build_feature_table(
        transactions, SNAPSHOT, churn_window_days=90, clv_window_days=180
    ).set_index("customer_id")
    assert table.loc[1, "monetary_total"] == pytest.approx(30.0)
    assert 3 not in table.index
    assert table.loc[1, "churn"] == 0
    assert table.loc[2, "churn"] == 1
    assert {"r_score", "f_score", "m_score", "engagement_score"} <= set(table.columns)


# ---------------------------------------------------------------------- LSTM
def test_lstm_sequences_padded_and_ordered(transactions: pd.DataFrame) -> None:
    history, _ = split_by_cutoff(transactions, SNAPSHOT)
    ids = np.array([1, 2])
    seq, mask = build_lstm_sequences(history, ids, SNAPSHOT, max_len=5)
    assert seq.shape == (2, 5, 2)
    assert mask.shape == (2, 5)
    # Customer 1 has 2 events -> they occupy the last two timesteps.
    assert mask[0].sum() == 2
    assert not mask[0, 0]
    # Customer 2 has 1 event.
    assert mask[1].sum() == 1


# ------------------------------------------------------------------ cleaning
def test_cleaning_keeps_legitimate_repeats() -> None:
    rows = [_txn(1, "2011-05-01", invoice="X", quantity=1),
            _txn(1, "2011-05-01", invoice="X", quantity=2)]  # same SKU, different qty
    df = pd.DataFrame(rows)
    customer_df, _ = clean_transactions(
        df, admin_stock_codes=["POST"], max_unit_price=5000, max_quantity=10000,
        iqr_multiplier=3.0,
    )
    assert len(customer_df) == 2, "legitimate repeats must be preserved"


def test_cleaning_drops_exact_duplicates() -> None:
    rows = [_txn(1, "2011-05-01", invoice="X", quantity=1)] * 2
    df = pd.DataFrame(rows)
    customer_df, _ = clean_transactions(
        df, admin_stock_codes=["POST"], max_unit_price=5000, max_quantity=10000,
        iqr_multiplier=3.0,
    )
    assert len(customer_df) == 1, "exact duplicates must be removed"


def test_missing_customer_id_split() -> None:
    rows = [_txn(1, "2011-05-01", invoice="X"),
            {**_txn(np.nan, "2011-05-02", invoice="Y"), "customer_id": np.nan}]
    df = pd.DataFrame(rows)
    customer_df, product_df = clean_transactions(
        df, admin_stock_codes=["POST"], max_unit_price=5000, max_quantity=10000,
        iqr_multiplier=3.0,
    )
    assert len(customer_df) == 1 and len(product_df) == 1


# ---------------------------------------------------------------- validation
def test_schema_check_raises_on_missing_column(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        check_schema(transactions.drop(columns=["price"]))


def test_returns_flagged_not_dropped(transactions: pd.DataFrame) -> None:
    df = pd.DataFrame([_txn(1, "2011-05-01", quantity=-2, invoice="R1")])
    customer_df, _ = clean_transactions(
        df, admin_stock_codes=["POST"], max_unit_price=5000, max_quantity=10000,
        iqr_multiplier=3.0,
    )
    assert customer_df["is_return"].sum() == 1, "returns must be kept as signal"
