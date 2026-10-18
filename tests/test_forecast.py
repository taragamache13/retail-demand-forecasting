import pandas as pd

from retail_demand.forecast import (
    create_next_day_features,
)


def test_create_next_day_features_uses_history_only():
    demand = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2010-01-01",
                periods=35,
            ),
            "StockCode": ["ABC1"] * 35,
            "UnitsSold": list(range(1, 36)),
        }
    )

    forecast_date = pd.Timestamp(
        "2010-02-05"
    )

    features = create_next_day_features(
        demand=demand,
        stock_codes=["ABC1"],
        forecast_date=forecast_date,
    )

    assert len(features) == 1

    row = features.iloc[0]

    assert row["Date"] == forecast_date
    assert row["lag_1"] == 35
    assert row["lag_7"] == 29
    assert row["lag_28"] == 8
    assert row["UnitsSold"] != row["UnitsSold"]