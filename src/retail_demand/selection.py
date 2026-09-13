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