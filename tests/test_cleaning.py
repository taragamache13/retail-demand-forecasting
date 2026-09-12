import pandas as pd

from retail_demand.cleaning import clean_transactions


def test_clean_transactions_removes_invalid_demand():
    raw_data = pd.DataFrame(
        [
            {
                "Invoice": "10001",
                "StockCode": "ABC1",
                "Description": "Test Product",
                "Quantity": 5,
                "InvoiceDate": "2010-01-01",
                "Price": 2.50,
                "Customer ID": 12345,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10001",
                "StockCode": "ABC1",
                "Description": "Test Product",
                "Quantity": 5,
                "InvoiceDate": "2010-01-01",
                "Price": 2.50,
                "Customer ID": 12345,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "C10002",
                "StockCode": "ABC2",
                "Description": "Cancelled Product",
                "Quantity": -2,
                "InvoiceDate": "2010-01-02",
                "Price": 3.00,
                "Customer ID": 12345,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10003",
                "StockCode": "ABC3",
                "Description": "Zero Quantity",
                "Quantity": 0,
                "InvoiceDate": "2010-01-03",
                "Price": 4.00,
                "Customer ID": 12345,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10004",
                "StockCode": "ABC4",
                "Description": "Free Product",
                "Quantity": 3,
                "InvoiceDate": "2010-01-04",
                "Price": 0,
                "Customer ID": 12345,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10005",
                "StockCode": "POST",
                "Description": "POSTAGE",
                "Quantity": 1,
                "InvoiceDate": "2010-01-05",
                "Price": 5.00,
                "Customer ID": 12345,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10006",
                "StockCode": "ABC6",
                "Description": "Anonymous Sale",
                "Quantity": 4,
                "InvoiceDate": "2010-01-06",
                "Price": 6.00,
                "Customer ID": None,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10007",
                "StockCode": "ABC7",
                "Description": None,
                "Quantity": 2,
                "InvoiceDate": "2010-01-07",
                "Price": 7.00,
                "Customer ID": 54321,
                "Country": "United Kingdom",
            },
        ]
    )

    cleaned = clean_transactions(raw_data)

    assert len(cleaned) == 3
    assert set(cleaned["Invoice"]) == {"10001", "10006", "10007"}
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["Price"] > 0).all()