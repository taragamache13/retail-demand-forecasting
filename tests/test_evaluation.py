import pandas as pd

from retail_demand.evaluation import (
    evaluate_trailing_mean_baseline,
)

def test_evaluation_uses_future_holdout_data():
    daily_demand = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2010-01-01",
                periods=8,
            ),
            "StockCode": ["ABC1"] * 8,
            "UnitsSold": [
                1,
                2,
                3,
                4,
                5,
                6,
                10,
                12,
            ],
        }
    )

    results = evaluate_trailing_mean_baseline(
        daily_demand=daily_demand,
        eligible_stock_codes=["ABC1"],
        horizon_days=2,
        window=3,
    )

    assert len(results) == 1

    result = results.iloc[0]

    assert result["Forecast"] == 5.0
    assert result["MAE"] == 6.0
    assert result["TrainEndDate"] == pd.Timestamp(
        "2010-01-06"
    )
    assert result["TestStartDate"] == pd.Timestamp(
        "2010-01-07"
    )