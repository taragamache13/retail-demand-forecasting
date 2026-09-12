from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "InvoiceDate",
    "StockCode",
    "Quantity",
}

def aggregate_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cleaned retail transactions into daily product demand.

    Each output row represents the total number of units sold for one
    StockCode on one calendar date.
    """

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    daily = df.copy()

    daily["InvoiceDate"] = pd.to_datetime(
        daily["InvoiceDate"],
        errors="raise",
    )

    daily["Date"] = daily["InvoiceDate"].dt.normalize()

    daily = (
        daily.groupby(
            ["Date", "StockCode"],
            as_index=False,
        )["Quantity"]
        .sum()
        .rename(
            columns={
                "Quantity": "UnitsSold",
            }
        )
    )

    return daily.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)