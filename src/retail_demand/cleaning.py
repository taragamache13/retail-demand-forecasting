from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}

ADMINISTRATIVE_STOCK_CODES = {
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "B",
    "BANK CHARGES",
    "C2",
    "D",
    "DOT",
    "M",
    "POST",
}

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Online Retail II transactions for demand forecasting.

    Cleaning rules:
    - Remove exact duplicate rows.
    - Remove cancelled invoices.
    - Keep only positive quantities.
    - Keep only positive prices.
    - Remove administrative and non-merchandise StockCodes.
    - Keep rows with missing Customer ID.
    - Keep rows with missing Description when StockCode is available.
    """

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cleaned = df.copy()

    # Exact duplicates would double-count demand.
    cleaned = cleaned.drop_duplicates()

    # Normalize identifier columns as strings.
    cleaned["Invoice"] = (
        cleaned["Invoice"]
        .astype("string")
        .str.strip()
    )

    cleaned["StockCode"] = (
        cleaned["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Cancelled invoices begin with "C".
    cancelled_invoices = (
        cleaned["Invoice"]
        .str.upper()
        .str.startswith("C", na=False)
    )

    cleaned = cleaned.loc[
        ~cancelled_invoices
    ].copy()

    # Demand must represent positive purchased units.
    cleaned = cleaned.loc[
        cleaned["Quantity"] > 0
    ].copy()

    # Restrict demand to paid product transactions.
    cleaned = cleaned.loc[
        cleaned["Price"] > 0
    ].copy()

    # Remove known administrative/non-merchandise transaction codes.
    non_merchandise = (
        cleaned["StockCode"].isin(ADMINISTRATIVE_STOCK_CODES)
        | cleaned["StockCode"].str.startswith("GIFT_", na=False)
    )

    cleaned = cleaned.loc[
        ~non_merchandise
    ].copy()

    return cleaned.reset_index(drop=True)