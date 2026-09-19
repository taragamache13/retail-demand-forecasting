from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "Date",
    "StockCode",
    "UnitsSold",
}


def calculate_product_metrics(
        daily_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate forecasting history metrics for each product.
    """

    missing_columns = REQUIRED_COLUMNS.difference(
        daily_demand.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    metrics = (
        daily_demand.groupby("StockCode")
        .agg(
            history_days=("Date", "count"),
            active_days=(
                "UnitsSold",
                lambda values: (values > 0).sum(),
            ),
            total_units=("UnitsSold", "sum"),
            avg_daily_demand=("UnitsSold", "mean"),
        )
        .reset_index()
    )

    return metrics

def select_eligible_products(
        metrics: pd.DataFrame,
        min_history_days: int = 365,
        min_active_days: int = 50,
) -> pd.DataFrame:
    """
    Select products with enough history for initial forecasting.
    """

    eligible = metrics.loc[
        (metrics["history_days"] >= min_history_days)
        & (metrics["active_days"] >= min_active_days)
    ].copy()

    return eligible.reset_index(drop=True)
def select_active_products(
    daily_demand: pd.DataFrame,
    eligible_stock_codes: list[str],
    cutoff_date: pd.Timestamp,
    activity_days: int = 28,
) -> list[str]:
    """
    Select forecast-eligible products that sold recently.

    A product is considered active if it recorded positive demand
    during the activity window immediately before the forecast cutoff.
    """

    if activity_days <= 0:
        raise ValueError(
            "activity_days must be greater than zero."
        )

    demand = daily_demand.copy()

    demand["Date"] = pd.to_datetime(
        demand["Date"],
        errors="raise",
    )

    window_start = (
        cutoff_date
        - pd.Timedelta(days=activity_days - 1)
    )

    recent = demand.loc[
        demand["StockCode"].isin(eligible_stock_codes)
        & (demand["Date"] >= window_start)
        & (demand["Date"] <= cutoff_date)
        & (demand["UnitsSold"] > 0)
    ]

    return sorted(
        recent["StockCode"].unique().tolist()
    )
