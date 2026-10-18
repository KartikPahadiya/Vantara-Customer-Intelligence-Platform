"""Download the Online Retail II dataset from UCI (PRD Section 5.5).

The ``ucimlrepo`` programmatic import is not available for this dataset
(the workbook is not importable through the package), so this script uses the
authoritative UCI direct-download URL as the fallback Option A flow.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    pkl_path = RAW / "online_retail_raw.pkl"
    if pkl_path.exists():
        print(f"Dataset already present at {pkl_path}")
        return

    print(f"Downloading {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL, timeout=300) as resp:
        blob = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    xlsx_name = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
    zf.extract(xlsx_name, RAW)

    df1 = pd.read_excel(RAW / xlsx_name, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(RAW / xlsx_name, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_pickle(pkl_path)
    # Remove the extracted workbook to keep the raw folder lean.
    (RAW / xlsx_name).unlink()
    (RAW / "online_retail_ii.zip").unlink(missing_ok=True)
    print(f"Saved {len(df):,} transactions to {pkl_path}")


if __name__ == "__main__":
    sys.exit(main())
