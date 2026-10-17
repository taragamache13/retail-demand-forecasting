import pandas as pd
import pytest

from retail_demand.croston import (
    croston_sba_forecast,
    evaluate_croston_sba,
)


def test_croston_sba_returns_zero_for_zero_history():
    history = pd.Series(
        [0, 0, 0, 0, 0]
    )

    forecast = croston_sba_forecast(
        history
    )

    assert forecast == 0.0


def test_croston_sba_forecasts_sparse_demand():
    history = pd.Series(
        [
            5,
            0,
            0,
            5,
            0,
            0,
            5,
        ]
    )

    forecast = croston_sba_forecast(
        history,
        alpha=0.2,
    )

    assert forecast == pytest.approx(
        2.6163,
        abs=0.0001,
    )

def test_evaluate_croston_sba_creates_complete_holdout():
    daily_demand = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2010-01-01",
                periods=8,
            ),
            "StockCode": ["ABC1"] * 8,
            "UnitsSold": [
                5,
                0,
                0,
                5,
                0,
                0,
                5,
                0,
            ],
        }
    )

    results = evaluate_croston_sba(
        daily_demand=daily_demand,
        stock_codes=["ABC1"],
        cutoff_date=pd.Timestamp("2010-01-05"),
        horizon_days=3,
        alpha=0.2,
    )

    assert len(results) == 3

    assert results["Date"].min() == pd.Timestamp(
        "2010-01-06"
    )

    assert results["Date"].max() == pd.Timestamp(
        "2010-01-08"
    )

    assert (
        results["CrostonPrediction"] >= 0
    ).all()