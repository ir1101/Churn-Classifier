"""Score customers with the saved churn model.

Usage (from the project folder):
    python -m churn.predict --input customers.csv --output predictions.csv
    python -m churn.predict --input customers.csv --output predictions.csv --threshold 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

from churn import config


class MissingColumnsError(ValueError):
    """The input is missing columns the model needs."""


def default_model_path() -> Path:
    """CHURN_MODEL_PATH if set, otherwise models/churn_model.joblib."""
    return Path(os.environ.get("CHURN_MODEL_PATH", config.DEFAULT_MODEL_PATH))


def load_model(path: Path | str | None = None) -> dict:
    """Load the saved bundle: {"pipeline", "threshold", "metadata"}."""
    path = Path(path) if path is not None else default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run `python -m churn.train` first."
        )
    bundle = joblib.load(path)
    missing = {"pipeline", "threshold", "metadata"} - set(bundle)
    if missing:
        raise ValueError(f"{path} is not a churn model bundle (missing {sorted(missing)}).")
    return bundle


def required_columns(bundle: dict) -> list[str]:
    return list(bundle["metadata"].get("required_columns", config.REQUIRED_INPUT_COLS))


def predict_df(bundle: dict, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Return customerID (when present), churn_probability and churn_prediction.

    Extra columns (Churn, gender, ...) are ignored; missing required columns
    raise MissingColumnsError naming them.
    """
    needed = required_columns(bundle)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise MissingColumnsError(f"Input is missing required columns: {', '.join(missing)}")

    cutoff = float(bundle["threshold"] if threshold is None else threshold)
    # XGBoost returns float32; widen before rounding so 0.7868 stays 0.7868 in JSON.
    proba = bundle["pipeline"].predict_proba(df[needed])[:, 1].astype("float64")

    out = pd.DataFrame(index=df.index)
    if config.ID_COL in df.columns:
        out[config.ID_COL] = df[config.ID_COL]
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = (proba >= cutoff).astype(int)
    return out


def _probability(text: str) -> float:
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score customers with the churn model.")
    parser.add_argument("--input", required=True, type=Path,
                        help="CSV with the Telco input columns")
    parser.add_argument("--output", required=True, type=Path,
                        help="where to write the predictions CSV")
    parser.add_argument("--model", type=Path, default=None,
                        help="model bundle (default: CHURN_MODEL_PATH or models/churn_model.joblib)")
    parser.add_argument("--threshold", type=_probability, default=None,
                        help="override the saved probability cutoff")
    args = parser.parse_args(argv)

    try:
        bundle = load_model(args.model)
        customers = pd.read_csv(args.input)
        predictions = predict_df(bundle, customers, args.threshold)
    except (FileNotFoundError, MissingColumnsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    cutoff = bundle["threshold"] if args.threshold is None else args.threshold
    flagged = int(predictions["churn_prediction"].sum())
    print(
        f"Scored {len(predictions):,} customers; {flagged:,} flagged as likely to churn "
        f"(cutoff {cutoff:.3f}). Wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
