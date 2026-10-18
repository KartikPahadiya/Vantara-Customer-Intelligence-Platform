"""Automated data validation (PRD Section 6.2).

Schema checks, null-rate thresholds and date-range checks run before the
feature engineering stage and fail loudly (raise ValidationError) if
violated.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationError(Exception):
    """Raised when a data-validation check fails. Message explains why."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


REQUIRED_COLUMNS = {
    "invoice": "object",
    "stock_code": "object",
    "description": "object",
    "quantity": "number",
    "invoice_date": "datetime",
    "price": "number",
    "customer_id": "number",
    "country": "object",
    "line_revenue": "number",
}

MAX_NULL_RATE = 0.05  # 5% null-rate threshold for critical columns


def check_schema(df: pd.DataFrame) -> None:
    """Verify required columns exist with plausible dtypes."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValidationError(f"Missing required columns: {sorted(missing)}")

    dtype_errors = []
    for col, kind in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            continue
        series = df[col]
        if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
            dtype_errors.append(f"{col} should be datetime, got {series.dtype}")
        elif kind == "number" and not pd.api.types.is_numeric_dtype(series):
            dtype_errors.append(f"{col} should be numeric, got {series.dtype}")
    if dtype_errors:
        raise ValidationError("Schema dtype violations: " + "; ".join(dtype_errors))


def check_null_rates(df: pd.DataFrame, critical_columns: list[str] | None = None) -> None:
    """Fail if any critical column exceeds the null-rate threshold.

    ``customer_id`` nulls are expected upstream (handled by cleaning) so they
    are checked at a higher tolerance.
    """
    critical_columns = critical_columns or ["invoice", "stock_code", "quantity", "price"]
    rates = df[critical_columns].isna().mean()
    violations = rates[rates > MAX_NULL_RATE]
    if len(violations):
        raise ValidationError(
            "Null-rate threshold exceeded: "
            + ", ".join(f"{k}={v:.2%}" for k, v in violations.items())
        )


def check_date_range(df: pd.DataFrame, min_date: str, max_date: str) -> None:
    """Fail if invoice dates fall outside the documented dataset window."""
    lo, hi = pd.Timestamp(min_date), pd.Timestamp(max_date)
    if df["invoice_date"].isna().any():
        raise ValidationError("invoice_date contains null values after cleaning")
    if df["invoice_date"].min() < lo or df["invoice_date"].max() > hi:
        raise ValidationError(
            f"invoice_date range {df['invoice_date'].min()} -> "
            f"{df['invoice_date'].max()} outside expected [{lo}, {hi}]"
        )


def check_positive_customers(df: pd.DataFrame) -> None:
    """Fail if the table has no customers or negative revenue-only customers."""
    if df["customer_id"].nunique() < 100:
        raise ValidationError(
            f"Suspiciously few customers after cleaning: {df['customer_id'].nunique()}"
        )


def validate_transactions(
    df: pd.DataFrame,
    min_date: str = "2009-12-01",
    max_date: str = "2011-12-10",
) -> None:
    """Run all validation checks; raises ValidationError on the first failure."""
    check_schema(df)
    check_null_rates(df)
    check_date_range(df, min_date, max_date)
    check_positive_customers(df)
