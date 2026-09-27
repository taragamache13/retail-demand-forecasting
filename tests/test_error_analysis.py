import numpy as np
import pandas as pd

from retail_demand.error_analysis import (
    create_prediction_results,
    summarize_product_errors,
)
from retail_demand.modeling import FEATURE_COLUMNS


class DummyModel:
    def predict(self, features):
        return np.array([-2.0, 4.0])


def test_create_prediction_results_clips_negative_predictions():
    test_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2010-01-01", "2010-01-02"]
            ),
            "StockCode": ["ABC1", "ABC1"],
            "UnitsSold": [3, 5],
        }
    )

    for feature in FEATURE_COLUMNS:
        test_data[feature] = 1.0

    results = create_prediction_results(
        DummyModel(),
        test_data,
    )

    assert results["Prediction"].tolist() == [0.0, 4.0]
    assert results["AbsoluteError"].tolist() == [3.0, 1.0]


def test_summarize_product_errors_calculates_metrics():
    prediction_results = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2010-01-01", "2010-01-02"]
            ),
            "StockCode": ["ABC1", "ABC1"],
            "UnitsSold": [3, 5],
            "Prediction": [2.0, 7.0],
            "AbsoluteError": [1.0, 2.0],
        }
    )

    summary = summarize_product_errors(
        prediction_results
    )

    result = summary.iloc[0]

    assert result["ActualTotal"] == 8
    assert result["PredictedTotal"] == 9
    assert result["MAE"] == 1.5
    assert result["AbsoluteErrorSum"] == 3.0
    assert result["MaxActual"] == 5
    assert result["MaxPrediction"] == 7.0
    assert result["WAPE"] == 0.375