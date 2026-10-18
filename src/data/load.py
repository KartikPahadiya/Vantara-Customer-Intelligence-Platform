"""Raw data loading (PRD Section 6.1: Data Collection).

Loads both sheets of the Online Retail II workbook, standardizes column
names and types, and combines them into a single chronologically ordered
transaction table before any cleaning logic is applied.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

COLUMN_RENAME_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_raw_transactions(path: str | Path) -> pd.DataFrame:
    """Load the raw Online Retail II transactions from the saved pickle.

    The pickle is produced once from the source workbook (both sheets,
    ``Year 2009-2010`` and ``Year 2010-2011`` concatenated) — see the
    README "Data" section for the download instructions.

    Args:
        path: Path to ``online_retail_raw.pkl``.

    Returns:
        Combined, chronologically sorted transactions DataFrame with
        standardized snake_case columns.

    Raises:
        FileNotFoundError: If the raw pickle does not exist.
        ValueError: If expected columns are missing from the source data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download Online Retail II per README."
        )

    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:  # pragma: no cover - convenience fallback
        df = pd.read_excel(path, sheet_name=None)
        df = pd.concat(df.values(), ignore_index=True)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")

    df = df.rename(columns=COLUMN_RENAME_MAP)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)

    logger.info(
        "Loaded %d raw transactions from %s (date range %s -> %s)",
        len(df),
        path.name,
        df["invoice_date"].min(),
        df["invoice_date"].max(),
    )
    return df
