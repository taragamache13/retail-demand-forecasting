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

def fill_zero_demand_days(
        daily_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    
    Create a continuous daily time series for each product.
    
    Missing dates between a product's first and last observec sale are filled with zero demand.
    
    Parameters
    ----------
    daily_demand:
        Aggregated daily demand containing:
        Date, StockCode, UnitsSold

    Returns
    -------
    pd.DataFrame
        Continuous daily product demand with zero-demand days included.
    """

    required_columns = {
        "Date",
        "StockCode",
        "UnitsSold",
    }

    missing_columns = required_columns.difference(
        daily_demand.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    demand = daily_demand.copy()

    demand["Date"] = pd.to_datetime(
        demand["Date"],
        errors="raise",
    )

    completed_products = []

    for stock_code, product_data in demand.groupby("StockCode"):
        product_data = product_data.sort_values("Date")

        full_date_range = pd.date_range(
            start=product_data["Date"].min(),
            end=product_data["Date"].max(),
            freq="D",
        )

        product_data = (
            product_data
            .set_index("Date")
            .reindex(full_date_range)
        )

        product_data.index.name = "Date"

        product_data["StockCode"] = stock_code

        product_data["UnitsSold"] = (
            product_data["UnitsSold"]
            .fillna(0)
        )

        completed_products.append(
            product_data.reset_index()
        )

    completed = pd.concat(
        completed_products,
        ignore_index=True
    )  

    completed["UnitsSold"] = (
        completed["UnitsSold"]
        .astype(int)
    )

    return completed.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)
      