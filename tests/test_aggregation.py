import pandas as pd

from retail_demand.aggregation import aggregate_daily_demand


def test_aggregate_daily_demand_sums_product_sales_by_day():
    transactions = pd.DataFrame(
        [
            {
                "InvoiceDate": "2010-01-01 09:00:00",
                "StockCode": "ABC1",
                "Quantity": 2,
            },
            {
                "InvoiceDate": "2010-01-01 11:30:00",
                "StockCode": "ABC1",
                "Quantity": 3,
            },
            {
                "InvoiceDate": "2010-01-01 14:00:00",
                "StockCode": "XYZ2",
                "Quantity": 4,
            },
            {
                "InvoiceDate": "2010-01-02 10:00:00",
                "StockCode": "ABC1",
                "Quantity": 1,
            },
        ]
    )

    daily = aggregate_daily_demand(transactions)

    abc1_day_one = daily[
        (daily["Date"] == pd.Timestamp("2010-01-01"))
        & (daily["StockCode"] == "ABC1")
    ]

    assert len(daily) == 3
    assert abc1_day_one["UnitsSold"].iloc[0] == 5
    assert list(daily.columns) == [
        "Date",
        "StockCode",
        "UnitsSold",
    ]