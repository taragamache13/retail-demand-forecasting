from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retail_demand.artifacts import (
    load_metadata,
    load_model_artifact,
)
from retail_demand.croston import croston_sba_forecast
from retail_demand.features import create_demand_features
from retail_demand.modeling import FEATURE_COLUMNS


DAILY_DEMAND_PATH = Path(
    "data/processed/daily_product_demand.parquet"
)

ARTIFACT_DIR = Path("artifacts")

MODEL_PATH = ARTIFACT_DIR / "poisson_model.joblib"
SEGMENTS_PATH = ARTIFACT_DIR / "product_segments.parquet"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

FORECAST_OUTPUT_PATH = Path(
    "reports/next_day_forecast.csv"
)


def create_next_day_features(
    demand: pd.DataFrame,
    stock_codes: list[str],
    forecast_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Create leakage-safe features for the next forecast date.
    """

    history = demand.loc[
        demand["StockCode"].isin(stock_codes)
    ].copy()

    future_rows = pd.DataFrame(
        {
            "Date": [forecast_date] * len(stock_codes),
            "StockCode": stock_codes,
            "UnitsSold": [np.nan] * len(stock_codes),
        }
    )

    combined = pd.concat(
        [
            history[
                ["Date", "StockCode", "UnitsSold"]
            ],
            future_rows,
        ],
        ignore_index=True,
    )

    features = create_demand_features(
        combined
    )

    next_day = features.loc[
        features["Date"] == forecast_date
    ].copy()

    return next_day.reset_index(drop=True)


def generate_next_day_forecasts() -> pd.DataFrame:
    """
    Load production artifacts and generate next-day forecasts
    without retraining the forecasting model.
    """

    demand = pd.read_parquet(
        DAILY_DEMAND_PATH
    )

    demand["Date"] = pd.to_datetime(
        demand["Date"],
        errors="raise",
    )

    model = load_model_artifact(
        MODEL_PATH
    )

    metadata = load_metadata(
        METADATA_PATH
    )

    segments = pd.read_parquet(
        SEGMENTS_PATH
    )

    final_date = demand["Date"].max()

    forecast_date = (
        final_date
        + pd.Timedelta(days=1)
    )

    stock_codes = (
        segments["StockCode"]
        .astype(str)
        .tolist()
    )

    next_day_features = create_next_day_features(
        demand=demand,
        stock_codes=stock_codes,
        forecast_date=forecast_date,
    )

    missing_features = next_day_features[
        FEATURE_COLUMNS
    ].isna().any(axis=1)

    if missing_features.any():
        raise ValueError(
            "Some products do not have enough history "
            "to create next-day features."
        )

    ml_predictions = model.predict(
        next_day_features[FEATURE_COLUMNS]
    )

    ml_predictions = np.clip(
        ml_predictions,
        a_min=0,
        a_max=None,
    )

    forecasts = next_day_features[
        ["Date", "StockCode"]
    ].copy()

    forecasts["Forecast"] = ml_predictions

    forecasts = forecasts.merge(
        segments[
            ["StockCode", "DemandSegment"]
        ],
        on="StockCode",
        how="left",
    )

    croston_alpha = metadata[
        "croston_alpha"
    ]

    intermittent_mask = (
        forecasts["DemandSegment"]
        == "intermittent"
    )

    for index in forecasts.loc[
        intermittent_mask
    ].index:
        stock_code = forecasts.loc[
            index,
            "StockCode",
        ]

        history = demand.loc[
            demand["StockCode"] == stock_code,
            "UnitsSold",
        ].reset_index(drop=True)

        forecasts.loc[
            index,
            "Forecast",
        ] = croston_sba_forecast(
            history,
            alpha=croston_alpha,
        )

    forecasts["Forecast"] = (
        forecasts["Forecast"]
        .clip(lower=0)
    )

    FORECAST_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        FORECAST_OUTPUT_PATH,
        index=False,
    )

    print("NEXT-DAY FORECAST COMPLETE")
    print(
        f"Historical data through: "
        f"{final_date.date()}"
    )
    print(
        f"Forecast date: "
        f"{forecast_date.date()}"
    )
    print(
        f"Products forecasted: "
        f"{len(forecasts):,}"
    )
    print(
        f"Forecast saved to: "
        f"{FORECAST_OUTPUT_PATH}"
    )

    return forecasts


if __name__ == "__main__":
    generate_next_day_forecasts()