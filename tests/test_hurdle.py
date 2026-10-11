import numpy as np
import pandas as pd

from retail_demand.hurdle import (
    evaluate_hurdle_model,
    predict_hurdle_demand,
    train_hurdle_model,
)
from retail_demand.modeling import FEATURE_COLUMNS


class DummyClassifier:
    def predict_proba(self, features):
        return np.array(
            [
                [0.75, 0.25],
                [0.20, 0.80],
            ]
        )


class DummyRegressor:
    def predict(self, features):
        return np.array(
            [8.0, 10.0]
        )


def test_hurdle_prediction_combines_probability_and_quantity():
    data = pd.DataFrame(
        {
            "UnitsSold": [2.0, 8.0],
            "rolling_mean_28": [3.0, 6.0],
        }
    )

    for feature in FEATURE_COLUMNS:
        if feature not in data.columns:
            data[feature] = 1.0

    predictions = predict_hurdle_demand(
        DummyClassifier(),
        DummyRegressor(),
        data,
    )

    assert predictions.tolist() == [
        2.0,
        8.0,
    ]


def test_hurdle_model_trains_and_evaluates():
    rows = []

    for day in range(120):
        demand = (
            float((day % 5) + 1)
            if day % 4 == 0
            else 0.0
        )

        row = {
            "Date": pd.Timestamp("2010-01-01")
            + pd.Timedelta(days=day),
            "StockCode": "ABC1",
            "UnitsSold": demand,
        }

        for feature in FEATURE_COLUMNS:
            row[feature] = float(
                (day % 7) + 1
            )

        rows.append(row)

    data = pd.DataFrame(rows)

    train = data.iloc[:90].copy()
    test = data.iloc[90:].copy()

    classifier, quantity_model = (
        train_hurdle_model(train)
    )

    metrics = evaluate_hurdle_model(
        classifier,
        quantity_model,
        test,
    )

    assert metrics["hurdle_mae"] >= 0
    assert metrics["hurdle_wape"] >= 0
    assert metrics["prediction_mean"] >= 0
    assert metrics["prediction_max"] >= 0