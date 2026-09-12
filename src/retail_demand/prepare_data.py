from pathlib import Path

import pandas as pd

from retail_demand.cleaning import clean_transactions


RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")
PROCESSED_DATA_PATH = Path(
    "data/processed/retail_transactions_clean.paqruet"
)

def load_raw_transactions(path: Path) -> pd.DataFrame:
    """
    Load and combine all sheets from the Online Retail II workbook.
    """

    sheets = pd.read_excel(
        path,
        sheet_name=None,
    )

    frames = []

    for sheet_name, dataframe in sheets.items():
        dataframe = dataframe.copy()
        dataframe["SourceSheet"] = sheet_name
        frames.append(dataframe)

    return pd.concat(
        frames,
        ignore_index=True,
    )

def prepare_dataset(
        raw_path: Path = RAW_DATA_PATH,
        processed_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """
    Load, clean, validate, and save the retail transaction datatset.
    """

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}"
        )

    raw = load_raw_transactions(raw_path)

    print(f"Raw rows: {len(raw):,}")

    cleaned = clean_transactions(raw)

    print(f"Cleaned rows: {len(cleaned):,}")
    print(
        f"Rows removed: "
        f"{len(raw) - len(cleaned):,}"
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_parquet(
        processed_path,
        index=False,
    )

    print(f"Saved cleaned data to: {processed_path}")

    return cleaned


if __name__=="__main__":
    prepare_dataset()