"""Data cleaning rules (PRD Section 6.2: Data Cleaning).

Every rule is implemented as a small, unit-testable pure function so the
same logic runs in both the training pipeline and the live scoring path:

* rows with missing ``customer_id`` are removed from the customer-level
  table (kept aside for product-level analysis only);
* exact duplicate invoice lines are dropped, while legitimate repeats of
  the same SKU inside one order (different line attributes) are preserved;
* returns/cancellations (negative quantity) and zero/negative prices are
  flagged, not dropped — return behaviour is a predictive signal;
* administrative ``stock_code`` values (postage, bank charges, manual
  entries) are flagged so they can be excluded from product-affinity work;
* IQR-based plus domain-rule outlier bounds for ``quantity`` and ``price``;
* product descriptions standardized (case/whitespace) via a lookup keyed
  on ``stock_code``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def drop_missing_customer_id(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off rows without a Customer ID.

    Returns:
        Tuple of (customer-level table, product-level-only table).
    """
    has_id = df["customer_id"].notna()
    return df[has_id].copy(), df[~has_id].copy()


def deduplicate_invoice_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Drop *exact* duplicate invoice lines only.

    A row is a duplicate only if every column matches; legitimate repeated
    purchases of the same SKU in one order with different quantities or
    prices are preserved.
    """
    before = len(df)
    df = df.drop_duplicates(subset=None, keep="first")
    logger.info("Deduplication removed %d exact duplicate lines", before - len(df))
    return df.reset_index(drop=True)


def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize product descriptions using a lookup keyed on StockCode.

    Trims whitespace, collapses repeated spaces and normalizes case using the
    modal description per StockCode — a documented, deterministic rule.
    """
    lookup = (
        df.dropna(subset=["description"])
        .assign(description=lambda d: d["description"].astype(str).str.strip())
        .groupby("stock_code")["description"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0])
    )
    df["description"] = (
        df["stock_code"].map(lookup).fillna(df["description"].astype(str)).str.strip()
    )
    return df


def flag_admin_stock_codes(df: pd.DataFrame, admin_codes: list[str]) -> pd.DataFrame:
    """Flag administrative (non-product) stock codes such as POST / BANK CHARGES."""
    admin_upper = {c.upper() for c in admin_codes}
    df["is_admin_code"] = df["stock_code"].astype(str).str.upper().isin(admin_upper)
    return df


def add_line_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Compute signed line revenue (negative for returns)."""
    df = df.copy()
    df["line_revenue"] = df["quantity"] * df["price"]
    return df


def flag_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Flag returns/cancellations explicitly instead of dropping them."""
    df = df.copy()
    df["is_return"] = df["quantity"] < 0
    return df


def cap_data_entry_errors(
    df: pd.DataFrame,
    max_unit_price: float,
    max_quantity: float,
) -> pd.DataFrame:
    """Flag implausible values from data-entry errors.

    Unit prices of exactly zero or negative are adjustments/manual
    corrections; extreme magnitudes above the domain ceilings are treated as
    data-entry errors and excluded from revenue features (documented rule).
    """
    df = df.copy()
    df["is_price_error"] = (
        (df["price"] <= 0)
        | (df["price"] > max_unit_price)
        | (df["quantity"].abs() > max_quantity)
    )
    return df


def add_iqr_outlier_flags(df: pd.DataFrame, iqr_multiplier: float) -> pd.DataFrame:
    """Add IQR-based outlier flags for quantity and price.

    A generous multiplier (3x IQR, configurable) is used so that legitimate
    bulk wholesale orders are kept — only extreme tails are flagged.
    """
    df = df.copy()
    for col in ("quantity", "price"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        df[f"{col}_iqr_outlier"] = (df[col] < lower) | (df[col] > upper)
    n_flagged = int((df["quantity_iqr_outlier"] | df["price_iqr_outlier"]).sum())
    logger.info("IQR rules flagged %d extreme lines (kept, flagged only)", n_flagged)
    return df


def clean_transactions(
    df: pd.DataFrame,
    admin_stock_codes: list[str],
    max_unit_price: float,
    max_quantity: float,
    iqr_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning chain and return (customer-level, product-level).

    The cleaned customer-level table keeps returns and flags rather than
    silently dropping behavioural signal, per PRD Section 6.2.
    """
    # Parquet requires homogeneous column types; return invoices ("C…")
    # live in the same column as numeric invoice ids.
    df = df.copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df = deduplicate_invoice_lines(df)
    df = standardize_descriptions(df)
    df = flag_admin_stock_codes(df, admin_stock_codes)
    df = add_line_revenue(df)
    df = flag_returns(df)
    df = cap_data_entry_errors(df, max_unit_price, max_quantity)
    df = add_iqr_outlier_flags(df, iqr_multiplier)

    customer_df, product_df = drop_missing_customer_id(df)
    # Exclude confirmed data-entry errors from customer-level modelling.
    customer_df = customer_df[~customer_df["is_price_error"]].reset_index(drop=True)
    customer_df["customer_id"] = customer_df["customer_id"].astype(np.int64)

    logger.info(
        "Cleaned customer-level table: %d rows, %d unique customers",
        len(customer_df),
        customer_df["customer_id"].nunique(),
    )
    return customer_df, product_df
