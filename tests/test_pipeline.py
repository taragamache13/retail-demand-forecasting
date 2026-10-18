from pathlib import Path

import pandas as pd

from retail_demand.artifacts import save_metadata
from retail_demand.pipeline import (
    validate_production_artifacts,
)


def test_validate_production_artifacts_accepts_consistent_files(
    tmp_path: Path,
):
    model_path = (
        tmp_path / "model.joblib"
    )

    segments_path = (
        tmp_path / "segments.parquet"
    )

    metadata_path = (
        tmp_path / "metadata.json"
    )

    model_path.write_bytes(
        b"test-model"
    )

    segments = pd.DataFrame(
        {
            "StockCode": [
                "ABC1",
                "XYZ2",
            ],
            "DemandSegment": [
                "lumpy",
                "intermittent",
            ],
        }
    )

    segments.to_parquet(
        segments_path,
        index=False,
    )

    save_metadata(
        {
            "active_products": 2,
        },
        metadata_path,
    )

    validate_production_artifacts(
        model_path=model_path,
        segments_path=segments_path,
        metadata_path=metadata_path,
    )


def test_validate_production_artifacts_rejects_mismatch(
    tmp_path: Path,
):
    model_path = (
        tmp_path / "model.joblib"
    )

    segments_path = (
        tmp_path / "segments.parquet"
    )

    metadata_path = (
        tmp_path / "metadata.json"
    )

    model_path.write_bytes(
        b"test-model"
    )

    pd.DataFrame(
        {
            "StockCode": ["ABC1"],
            "DemandSegment": ["lumpy"],
        }
    ).to_parquet(
        segments_path,
        index=False,
    )

    save_metadata(
        {
            "active_products": 2,
        },
        metadata_path,
    )

    try:
        validate_production_artifacts(
            model_path=model_path,
            segments_path=segments_path,
            metadata_path=metadata_path,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected artifact validation to fail."
        )