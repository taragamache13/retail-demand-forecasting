import pandas as pd

from retail_demand.aggregation import (
    aggregate_daily_demand,
    fill_zero_demand_days,
    extend_products_to_end_date,
)


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

def test_fill_zero_demand_days_adds_missing_dates():
    daily_demand = pd.DataFrame(
        [
            {
                "Date": "2010-01-01",
                "StockCode": "ABC1",
                "UnitsSold": 5,
            },
            {
                "Date": "2010-01-03",
                "StockCode": "ABC1",
                "UnitsSold": 8,
            },
        ]
    )

    completed = fill_zero_demand_days(daily_demand)

    assert len(completed) == 3

    jan_second = completed[
        completed["Date"] == pd.Timestamp("2010-01-02")
        ]

    assert len(jan_second) == 1
    assert jan_second["UnitsSold"].iloc[0] == 0

def test_extend_products_to_end_date_adds_trailing_zeros():
    daily_demand = pd.DataFrame(
        [
            {
                "Date": "2010-01-01",
                "StockCode": "ABC1",
                "UnitsSold": 5,
            },
            {
                "Date": "2010-01-02",
                "StockCode": "ABC1",
                "UnitsSold": 3,
            },
        ]
    )

    completed = extend_products_to_end_date(
        daily_demand,
        stock_codes=["ABC1"],
        end_date=pd.Timestamp("2010-01-04")
    )

    assert len(completed) == 4

    assert completed["UnitsSold"].tolist() == [
        5,
        3,
        0,
        0,
    ]