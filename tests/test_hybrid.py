import pandas as pd

from retail_demand.hybrid import (
    create_hybrid_predictions,
    evaluate_hybrid_predictions,
)


def test_hybrid_routes_intermittent_products_to_baseline():
    prediction_results = pd.DataFrame(
        {
            "StockCode": [
                "INTER",
                "LUMPY",
            ],
            "UnitsSold": [
                10,
                10,
            ],
            "Prediction": [
                4.0,
                8.0,
            ],
            "BaselinePrediction": [
                9.0,
                6.0,
            ],
        }
    )

    segments = pd.DataFrame(
        {
            "StockCode": [
                "INTER",
                "LUMPY",
            ],
            "DemandSegment": [
                "intermittent",
                "lumpy",
            ],
        }
    )

    hybrid = create_hybrid_predictions(
        prediction_results,
        segments,
    )

    intermittent = hybrid.loc[
        hybrid["StockCode"] == "INTER"
    ].iloc[0]

    lumpy = hybrid.loc[
        hybrid["StockCode"] == "LUMPY"
    ].iloc[0]

    assert intermittent["HybridPrediction"] == 9.0
    assert lumpy["HybridPrediction"] == 8.0


def test_hybrid_evaluation_calculates_metrics():
    results = pd.DataFrame(
        {
            "UnitsSold": [10, 10],
            "HybridPrediction": [9.0, 8.0],
            "HybridAbsoluteError": [1.0, 2.0],
        }
    )

    metrics = evaluate_hybrid_predictions(results)

    assert metrics["hybrid_mae"] == 1.5
    assert metrics["hybrid_wape"] == 0.15