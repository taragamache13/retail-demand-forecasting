from pathlib import Path

from retail_demand.artifacts import (
    load_metadata,
    load_model_artifact,
    save_metadata,
    save_model_artifact,
)


def test_model_artifact_can_be_saved_and_loaded(
    tmp_path: Path,
):
    model = {
        "name": "test-model",
        "version": 1,
    }

    model_path = (
        tmp_path / "model.joblib"
    )

    save_model_artifact(
        model,
        model_path,
    )

    loaded = load_model_artifact(
        model_path
    )

    assert loaded == model


def test_metadata_can_be_saved_and_loaded(
    tmp_path: Path,
):
    metadata = {
        "model": "Poisson gradient boosting",
        "croston_alpha": 0.1,
        "forecast_horizon_days": 28,
    }

    metadata_path = (
        tmp_path / "metadata.json"
    )

    save_metadata(
        metadata,
        metadata_path,
    )

    loaded = load_metadata(
        metadata_path
    )

    assert loaded == metadata