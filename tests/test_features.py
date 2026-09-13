import pandas as pd

from retail_demand.features import create_demand_features


def test_features_use_only_past_demand():
    daily_demand = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2010-01-01",
                periods=30,
            ),
            "StockCode": ["ABC1"] * 30,
            "UnitsSold": list(range(1, 31)),
        }
    )

    features = create_demand_features(daily_demand)

    row = features.iloc[28]

    assert row["UnitsSold"] == 29

    assert row["lag_1"] == 28
    assert row["lag_7"] == 22
    assert row["lag_28"] == 1

    assert row["rolling_mean_7"] == 25.0
    assert row["rolling_mean_28"] == 14.5