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

    Clean rules:
    - Remove exact duplicate rows.
    - Remove cancelled invoices.
    - Keep only positive quantities.
    - Keep onlu positive prices.
    - Remove administrative and non-merchandise StockCodes.
    - Keep rows with missing Customer ID.
    - Keep rows with missing Description when StockCode is available.

    Parameters
    ----------
    df:
        Raw transaction data.

    Returns
    -------
    pd.DataFrame
        Cleaned merchandise transactions suitable for demand analysis.
    """

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cleaned = df.copy()

    # Exact duplicates would double-count demand.
    cleaned = cleaned.drop_duplicates()

    # Cancelled invoices begin with "C".
    invoice_codes = (
        cleaned["Invoice"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    cleaned = cleaned.loc[
        ~invoice_codes.str.startswith("C", na=False)
    ].copy()

    # Demand must represent positive purchased units.
    cleaned = cleaned.loc[
        cleaned["Quantity"] > 0
    ].copy()

    # Restrict demand to paid product transactions.
    cleaned = cleaned.loc[
        cleaned["Price"] > 0
    ].copy()

    # Restrict known administrative/non-merchandise transaction codes.
    stock_codes = (
        cleaned["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    non_merchandise = (
        stock_codes.isin(ADMINISTRATIVE_STOCK_CODES)
        | stock_codes.str.startswith("GIFT_", na=False)
    )

    cleaned= cleaned.loc[
        ~non_merchandise
    ].copy()

    return cleaned.reset_index(drop=True)
