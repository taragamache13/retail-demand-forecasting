from pathlib import Path

import pandas as pd

from retail_demand.aggregation import extend_products_to_end_date
from retail_demand.features import create_demand_features
from retail_demand.model import (
    evaluate_demand_model,
    train_demand_model,
)
from retail_demand.modeling import create_time_split
from retail_demand.selection import (
    calculate_product_metrics,
    select_active_products,
    select_eligible_products,
)
from retail_demand.error_analysis import (
    create_prediction_results,
    summarize_product_errors,
    summarize_segment_performance,
)
from retail_demand.segmentation import calculate_demand_segments


DAILY_DEMAND_PATH = Path(
    "data/processed/daily_product_demand.parquet"
)

HOLDOUT_DAYS = 28


def run_experiment() -> dict[str, float]:
    """
    Run the corrected demand-forecasting experiment.

    Product eligibility and activity are determined using training
    data only. Active products are extended through the common
    forecast end date so every product has a complete holdout period.
    """

    demand = pd.read_parquet(
        DAILY_DEMAND_PATH
    )

    demand["Date"] = pd.to_datetime(
        demand["Date"],
        errors="raise",
    )

    final_date = demand["Date"].max()

    cutoff_date = (
        final_date
        - pd.Timedelta(days=HOLDOUT_DAYS)
    )

    # Only pre-cutoff data may determine product eligibility.
    selection_history = demand.loc[
        demand["Date"] <= cutoff_date
    ].copy()

    product_metrics = calculate_product_metrics(
        selection_history
    )

    eligible = select_eligible_products(
        product_metrics
    )

    active_stock_codes = select_active_products(
        daily_demand=selection_history,
        eligible_stock_codes=eligible["StockCode"].tolist(),
        cutoff_date=cutoff_date,
        activity_days=HOLDOUT_DAYS,
    )

    segmentation_history = extend_products_to_end_date(
        daily_demand=selection_history,
        stock_codes=active_stock_codes,
        end_date=cutoff_date,
    )

    segments = calculate_demand_segments(
        segmentation_history
    )

    print("\nDEMAND SEGMENTS")
    print(
        segments["DemandSegment"]
        .value_counts()
        .to_string()
    )

    # Extend every selected product through the same final date.
    completed_demand = extend_products_to_end_date(
        daily_demand=demand,
        stock_codes=active_stock_codes,
        end_date=final_date,
    )

    features = create_demand_features(
        completed_demand
    )

    train, test = create_time_split(
        features,
        holdout_days=HOLDOUT_DAYS,
    )

    expected_test_rows = (
        len(active_stock_codes)
        * HOLDOUT_DAYS
    )

    if len(test) != expected_test_rows:
        raise ValueError(
            "Incomplete holdout detected: "
            f"expected {expected_test_rows:,} rows "
            f"but found {len(test):,}."
        )

    print(f"Final date: {final_date.date()}")
    print(f"Training cutoff: {cutoff_date.date()}")
    print(f"Eligible products: {len(eligible):,}")
    print(f"Active products: {len(active_stock_codes):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(
        f"Expected test rows: "
        f"{expected_test_rows:,}"
    )

    model = train_demand_model(
        train
    )

    results = evaluate_demand_model(
        model,
        test,
    )

    
    prediction_results = create_prediction_results(
    model,
    test,
    )

    product_errors = summarize_product_errors(
        prediction_results
    )

    segment_performance = summarize_segment_performance(
        prediction_results,
        segments,
    )

    product_errors["ForecastBias"] = (
        product_errors["PredictedTotal"]
        - product_errors["ActualTotal"]
    )

    total_absolute_error = (
        product_errors["AbsoluteErrorSum"].sum()
    )

    product_errors["ErrorShare"] = (
        product_errors["AbsoluteErrorSum"]
        / total_absolute_error
    )

    mae_improvement = (
        (
            results["baseline_mae"]
            - results["ml_mae"]
        )
            / results["baseline_mae"]
            * 100
        )

    wape_improvement = (
        (
            results["baseline_wape"]
            - results["ml_wape"]
        )
        / results["baseline_wape"]
        * 100
    )

    print("\nMODEL RESULTS")

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    print(
        f"MAE improvement vs baseline: "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"WAPE improvement vs baseline: "
        f"{wape_improvement:.2f}%"
    )

    print("\nWORST PRODUCTS BY TOTAL ABSOLUTE ERROR")

    worst_products = (
    product_errors
    .sort_values(
        "AbsoluteErrorSum",
        ascending=False,
    )
    .head(10)
)

    print(
        worst_products[
            [
                "StockCode",
                "ActualTotal",
                "PredictedTotal",
                "MAE",
                "WAPE",
                "ForecastBias",
                "MaxActual",
                "MaxPrediction",
                "ErrorShare",
            ]
        ].to_string(index=False)
    )

    print("/LARGEST MODEL OVERPREDICTIONS")

    largest_overpredictions = (
        product_errors.sort_values(
            "ForecastBias",
            ascending=False,
        )
        .head(10)
    )

    print(
        largest_overpredictions[
            [
                "StockCode",
                "ActualTotal",
                "PredictedTotal",
                "ForecastBias",
                "MaxActual",
                "MaxPrediction",
            ]
        ].to_string(index=False)
    )

    error_output_path = Path(
    "data/processed/product_error_summary.parquet"
    )

    product_errors.to_parquet(
        error_output_path,
        index=False,
    )

    print(
        f"\nSaved product error summary to: "
        f"{error_output_path}"
    )

    print("\nPERFORMANCE BY DEMAND SEGMENT")

    segment_display = segment_performance[
        [
            "DemandSegment",
            "Products",
            "TestRows",
            "ModelMAE",
            "BaselineMAE",
            "ModelWAPE",
            "BaselineWAPE",
            "WAPEImprovementPct",
        ]
    ].copy()

    print(
        segment_display.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    segment_output_path = Path(
        "data/processed/segment_performance.parquet"
    )

    segment_performance.to_parquet(
        segment_output_path,
        index=False,
    )

    print(
        f"\nSaved segment performance to: "
        f"{segment_output_path}"
    )

    return results


if __name__ == "__main__":
    run_experiment()