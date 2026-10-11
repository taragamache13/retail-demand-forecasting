from __future__ import annotations

import pandas as pd


def create_hybrid_predictions(
    prediction_results: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Route forecasts according to historical demand segment.

    Current routing policy:
    - intermittent: trailing-mean baseline
    - erratic: ML model
    - lumpy: ML model
    - smooth: ML model
    """

    results = prediction_results.merge(
        segments[
            ["StockCode", "DemandSegment"]
        ],
        on="StockCode",
        how="left",
    )

    results["HybridPrediction"] = results["Prediction"]

    intermittent_mask = (
        results["DemandSegment"] == "intermittent"
    )

    results.loc[
        intermittent_mask,
        "HybridPrediction",
    ] = results.loc[
        intermittent_mask,
        "BaselinePrediction",
    ]

    results["HybridAbsoluteError"] = (
        results["UnitsSold"]
        - results["HybridPrediction"]
    ).abs()

    return results


def evaluate_hybrid_predictions(
    hybrid_results: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate overall MAE and WAPE for the hybrid forecasting strategy.
    """

    mae = hybrid_results[
        "HybridAbsoluteError"
    ].mean()

    absolute_error_sum = hybrid_results[
        "HybridAbsoluteError"
    ].sum()

    actual_total = hybrid_results[
        "UnitsSold"
    ].sum()

    wape = (
        absolute_error_sum / actual_total
        if actual_total > 0
        else float("nan")
    )

    return {
        "hybrid_mae": float(mae),
        "hybrid_wape": float(wape),
    }