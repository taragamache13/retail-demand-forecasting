import pandas as pd

from retail_demand.selection import (
    calculate_product_metrics,
    select_eligible_products,
)

def test_select_eligible_products_filters_sparse_history():
    daily_demand = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2010-01-01",
                periods=400,
            ).tolist()
            + pd.date_range(
                "2010-01-01",
                periods=100,
            ).tolist(),
            "StockCode": (
                ["ABC1"] * 400
                + ["XYZ2"] * 100
            ),
            "UnitsSold": (
                [1] * 60
                + [0] * 340
                + [1] * 20
                + [0] *80
            ),
        }
    )

    metrics = calculate_product_metrics(
        daily_demand
    )

    eligible = select_eligible_products(
        metrics,
        min_history_days=365,
        min_active_days=50,
    )

    assert len(eligible) == 1
    assert eligible["StockCode"].iloc[0] == "ABC1"
    assert eligible["history_days"].iloc[0] == 400
    assert eligible["active_days"].iloc[0] == 60