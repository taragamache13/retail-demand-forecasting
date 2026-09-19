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

    return results


if __name__ == "__main__":
    run_experiment()