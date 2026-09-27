from __future__ import annotations

import numpy as np
import pandas as pd

from retail_demand.modeling import FEATURE_COLUMNS


def create_prediction_results(
    model,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create row-level predictions and errors for model analysis.
    """

    results = test[
        ["Date", "StockCode", "UnitsSold"]
    ].copy()

    predictions = model.predict(
        test[FEATURE_COLUMNS]
    )

    results["Prediction"] = np.clip(
        predictions,
        a_min=0,
        a_max=None,
    )

    results["AbsoluteError"] = (
        results["UnitsSold"]
        - results["Prediction"]
    ).abs()

    return results


def summarize_product_errors(
    prediction_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize forecast accuracy for each product.
    """

    summary = (
        prediction_results
        .groupby("StockCode")
        .agg(
            ActualTotal=("UnitsSold", "sum"),
            PredictedTotal=("Prediction", "sum"),
            MAE=("AbsoluteError", "mean"),
            AbsoluteErrorSum=("AbsoluteError", "sum"),
            MaxActual=("UnitsSold", "max"),
            MaxPrediction=("Prediction", "max"),
        )
        .reset_index()
    )

    summary["WAPE"] = (
        summary["AbsoluteErrorSum"]
        / summary["ActualTotal"].replace(0, np.nan)
    )

    return summary