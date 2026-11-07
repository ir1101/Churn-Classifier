"""Feature engineering and preprocessing shared by training, the CLI and the API.

Everything here runs inside the saved scikit-learn pipeline, so a CSV row or an
API request goes through exactly the same cleaning as the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config

TENURE_BINS = [-np.inf, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw columns and add engineered features. Returns a new DataFrame."""
    X = X.copy()
    tenure = pd.to_numeric(X["tenure"], errors="coerce")
    monthly = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(X["TotalCharges"], errors="coerce")
    # Blank TotalCharges belong to brand-new customers (tenure 0) who have not
    # been billed yet, so tenure * MonthlyCharges (= 0 for them) is the right fill.
    total = total.fillna(tenure * monthly)

    X["SeniorCitizen"] = pd.to_numeric(X["SeniorCitizen"], errors="coerce")
    X["tenure"] = tenure
    X["MonthlyCharges"] = monthly
    X["TotalCharges"] = total

    X["num_addon_services"] = (X[config.ADDON_SERVICE_COLS] == "Yes").sum(axis=1)
    X["avg_monthly_spend"] = (total / tenure).where(tenure > 0, monthly)
    X["tenure_group"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(object)
    return X


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale numeric columns and one-hot encode categorical ones."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, config.MODEL_NUMERIC_COLS),
            ("cat", categorical, config.MODEL_CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    """Full pipeline: raw input columns -> features -> preprocessing -> model."""
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
