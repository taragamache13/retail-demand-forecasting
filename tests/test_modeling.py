import pandas as pd

from retail_demand.features import create_demand_features
from retail_demand.modeling import create_time_split


def test_time_split_keeps_future_out_of_training():
    demand = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2010-01-01",
                periods=60,
            ),
            "StockCode": ["ABC1"] * 60,
            "UnitsSold": [1] * 60,
        }
    )

    features = create_demand_features(demand)

    train, test = create_time_split(
        features,
        holdout_days=10,
    )

    assert train["Date"].max() < test["Date"].min()

    assert train["Date"].max() == pd.Timestamp(
        "2010-02-19"
    )

    assert test["Date"].min() == pd.Timestamp(
        "2010-02-20"
    )

    assert test["Date"].max() == pd.Timestamp(
        "2010-03-01"
    )

