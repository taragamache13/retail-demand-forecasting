from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "StockCode",
    "UnitsSold",
}


def create_demand_features(
    daily_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create time-series features for daily demand forecasting.

    All lag and rolling features use only information available
    before the prediction date to prevent target leakage.
    """

    missing_columns = REQUIRED_COLUMNS.difference(
        daily_demand.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    features = daily_demand.copy()

    features["Date"] = pd.to_datetime(
        features["Date"],
        errors="raise",
    )

    features = features.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    grouped = features.groupby("StockCode")["UnitsSold"]

    # Lag features
    features["lag_1"] = grouped.shift(1)
    features["lag_7"] = grouped.shift(7)
    features["lag_14"] = grouped.shift(14)
    features["lag_28"] = grouped.shift(28)

    # Rolling averages must be shifted first so today's demand
    # is not accidentally included in today's predictors.
    features["rolling_mean_7"] = (
        grouped
        .shift(1)
        .groupby(features["StockCode"])
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    features["rolling_mean_28"] = (
        grouped
        .shift(1)
        .groupby(features["StockCode"])
        .rolling(28)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Calendar features
    features["day_of_week"] = features["Date"].dt.dayofweek
    features["day_of_month"] = features["Date"].dt.day
    features["month"] = features["Date"].dt.month

    return features