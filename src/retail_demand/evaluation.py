from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from retail_demand.baseline import trailing_mean_forecast


REQUIRED_COLUMNS = {
    "Date",
    "StockCode",
    "UnitsSold",
}


def evaluate_trailing_mean_baseline(
    daily_demand: pd.DataFrame,
    eligible_stock_codes: Iterable[str],
    horizon_days: int = 28,
    window: int = 28,
) -> pd.DataFrame:
    """
    Evaluate the trailing-mean baseline using a time-based holdout.

    For each eligible product:
    - Reserve the final horizon_days as test data.
    - Use earlier observations as training history.
    - Generate a forecast using training data only.
    - Compare the forecast against the unseen test period.
    """

    missing_columns = REQUIRED_COLUMNS.difference(
        daily_demand.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be greater than zero."
        )

    if window <= 0:
        raise ValueError(
            "window must be greater than zero."
        )

    eligible_codes = set(eligible_stock_codes)

    demand = daily_demand.loc[
        daily_demand["StockCode"].isin(eligible_codes)
    ].copy()

    demand["Date"] = pd.to_datetime(
        demand["Date"],
        errors="raise",
    )

    demand = demand.sort_values(
        ["StockCode", "Date"]
    )

    results = []

    for stock_code, product_data in demand.groupby("StockCode"):
        if len(product_data) <= horizon_days:
            continue

        train = product_data.iloc[:-horizon_days]
        test = product_data.iloc[-horizon_days:]

        prediction = trailing_mean_forecast(
            train["UnitsSold"],
            window=window,
        )

        absolute_errors = (
            test["UnitsSold"] - prediction
        ).abs()

        actual_total = test["UnitsSold"].sum()

        wape = (
            absolute_errors.sum() / actual_total
            if actual_total > 0
            else float("nan")
        )

        results.append(
            {
                "StockCode": stock_code,
                "Forecast": prediction,
                "MAE": absolute_errors.mean(),
                "AbsoluteErrorSum": absolute_errors.sum(),
                "ActualMean": test["UnitsSold"].mean(),
                "ActualTotal": actual_total,
                "WAPE": wape,
                "TrainEndDate": train["Date"].max(),
                "TestStartDate": test["Date"].min(),
                "TestEndDate": test["Date"].max(),
            }
        )

    return pd.DataFrame(results)