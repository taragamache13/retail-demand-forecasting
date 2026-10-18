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

def create_enhanced_hybrid_predictions(
    prediction_results: pd.DataFrame,
    segments: pd.DataFrame,
    croston_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Route forecasts using the best observed method by demand segment.

    Routing policy:
    - erratic: Poisson ML
    - lumpy: Poisson ML
    - smooth: Poisson ML
    - intermittent: Croston/SBA
    """

    results = prediction_results.merge(
        segments[
            ["StockCode", "DemandSegment"]
        ],
        on="StockCode",
        how="left",
    )

    croston_lookup = croston_results[
        [
            "Date",
            "StockCode",
            "CrostonPrediction",
        ]
    ].copy()

    results = results.merge(
        croston_lookup,
        on=["Date", "StockCode"],
        how="left",
    )

    results["EnhancedHybridPrediction"] = (
        results["Prediction"]
    )

    intermittent_mask = (
        results["DemandSegment"] == "intermittent"
    )

    if results.loc[
        intermittent_mask,
        "CrostonPrediction",
    ].isna().any():
        raise ValueError(
            "Missing Croston predictions for "
            "intermittent products."
        )

    results.loc[
        intermittent_mask,
        "EnhancedHybridPrediction",
    ] = results.loc[
        intermittent_mask,
        "CrostonPrediction",
    ]

    results["EnhancedHybridAbsoluteError"] = (
        results["UnitsSold"]
        - results["EnhancedHybridPrediction"]
    ).abs()

    return results


def evaluate_enhanced_hybrid_predictions(
    enhanced_results: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate MAE and WAPE for the enhanced hybrid strategy.
    """

    mae = enhanced_results[
        "EnhancedHybridAbsoluteError"
    ].mean()

    absolute_error_sum = enhanced_results[
        "EnhancedHybridAbsoluteError"
    ].sum()

    actual_total = enhanced_results[
        "UnitsSold"
    ].sum()

    wape = (
        absolute_error_sum / actual_total
        if actual_total > 0
        else float("nan")
    )

    return {
        "enhanced_hybrid_mae": float(mae),
        "enhanced_hybrid_wape": float(wape),
    }