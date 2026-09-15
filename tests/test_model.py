import pandas as pd

from retail_demand.model import (
    evaluate_demand_model,
    train_demand_model,
)

from retail_demand.modeling import FEATURE_COLUMNS


def test_model_trains_and_evaluates():
    rows = []

    for day in range(100):
        row = {
            "Date": pd.Timestamp("2010-01-01")
            + pd.Timedelta(days=day),
            "StockCode": "ABC1",
            "UnitsSold": float(day % 7)
        }

        for feature in FEATURE_COLUMNS:
            row[feature] = float(day % 7)

        rows.append(row)

    data = pd.DataFrame(rows)

    train = data.iloc[:80].copy()
    test = data.iloc[80:].copy()

    model = train_demand_model(train)

    metrics = evaluate_demand_model(
        model,
        test,
    )

    assert "ml_mae" in metrics
    assert "baseline_mae" in metrics
    assert "ml_wape" in metrics
    assert "baseline_wape" in metrics

    assert metrics["ml_mae"] >= 0
    assert metrics["baseline_mae"] >= 0