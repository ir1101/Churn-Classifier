"""Download, load and split the IBM Telco Customer Churn dataset."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import requests

from churn import config


class DataError(RuntimeError):
    """The dataset is missing, unreachable or not in the expected shape."""


def validate_raw(df: pd.DataFrame) -> None:
    """Check that a raw DataFrame has the columns and target values we expect."""
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Dataset is missing expected columns: {', '.join(missing)}")
    if df.empty:
        raise DataError("Dataset has no rows.")
    bad_target = set(df[config.TARGET_COL].dropna().unique()) - {"Yes", "No"}
    if bad_target or df[config.TARGET_COL].isna().any():
        raise DataError(
            f"Column '{config.TARGET_COL}' must contain only 'Yes' or 'No' "
            f"(found: {sorted(map(str, bad_target)) or 'missing values'})."
        )


def download_raw(
    path: Path = config.RAW_CSV, url: str = config.DATA_URL, force: bool = False
) -> Path:
    """Download the raw CSV unless it is already on disk. Returns its path."""
    path = Path(path)
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataError(
            f"Could not download the dataset ({exc}).\n"
            f"Download Telco-Customer-Churn.csv by hand (it is also on Kaggle as "
            f"'blastchar/telco-customer-churn') and save it as {path}"
        ) from exc

    partial = path.with_name(path.name + ".part")
    partial.write_bytes(response.content)
    try:
        df = pd.read_csv(partial)
        validate_raw(df)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if len(df) != config.EXPECTED_ROWS:
        warnings.warn(
            f"Expected {config.EXPECTED_ROWS} rows but the download has {len(df)}; "
            "the source file may have changed.",
            stacklevel=2,
        )
    partial.replace(path)
    print(f"Saved {len(df):,} rows to {path}")
    return path


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Read and validate the raw CSV."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"{path} not found. Run `python -m churn.train`, which downloads it."
        )
    df = pd.read_csv(path)
    validate_raw(df)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, ids).

    X holds only the model's input columns (no ID, target or excluded columns),
    y is 1 for churn and 0 otherwise, and ids is the customerID column.
    """
    validate_raw(df)
    X = df[config.REQUIRED_INPUT_COLS].copy()
    y = (df[config.TARGET_COL] == "Yes").astype(int).rename("churn")
    ids = df[config.ID_COL].copy()
    return X, y, ids
