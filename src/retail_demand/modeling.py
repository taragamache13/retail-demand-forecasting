from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "day_of_month",
    "month",
]


def create_time_split(
    feature_data: pd.DataFrame,
    holdout_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature data into training and future holdout periods.

    A global calendar cutoff is used so no product's future dates
    appear in the training dataset.
    """

    if holdout_days <= 0:
        raise ValueError(
            "holdout_days must be greater than zero."
        )

    data = feature_data.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise",
    )

    required_columns = {
        "Date",
        "StockCode",
        "UnitsSold",
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Rows at the beginning of each product history do not yet
    # have enough past information to create 28-day features.
    data = data.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    final_date = data["Date"].max()

    cutoff_date = (
        final_date
        - pd.Timedelta(days=holdout_days)
    )

    train = data.loc[
        data["Date"] <= cutoff_date
    ].copy()

    test = data.loc[
        data["Date"] > cutoff_date
    ].copy()

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
    )