import pandas as pd

from retail_demand.segmentation import (
    calculate_demand_segments,
)


def test_calculate_demand_segments_identifies_patterns():
    smooth = pd.DataFrame(
        {
            "StockCode": ["SMOOTH"] * 10,
            "UnitsSold": [5] * 10,
        }
    )

    intermittent = pd.DataFrame(
        {
            "StockCode": ["INTERMITTENT"] * 10,
            "UnitsSold": [
                5, 0, 0, 5, 0,
                0, 5, 0, 0, 5,
            ],
        }
    )

    lumpy = pd.DataFrame(
        {
            "StockCode": ["LUMPY"] * 10,
            "UnitsSold": [
                1, 0, 0, 20, 0,
                0, 1, 0, 0, 20,
            ],
        }
    )

    demand = pd.concat(
        [
            smooth,
            intermittent,
            lumpy,
        ],
        ignore_index=True,
    )

    segments = calculate_demand_segments(demand)

    segment_map = dict(
        zip(
            segments["StockCode"],
            segments["DemandSegment"],
        )
    )

    assert segment_map["SMOOTH"] == "smooth"
    assert segment_map["INTERMITTENT"] == "intermittent"
    assert segment_map["LUMPY"] == "lumpy"