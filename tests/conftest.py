"""Shared fixtures: a synthetic Telco-shaped dataset and a tiny trained model bundle."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn import config
from churn.data import split_features_target
from churn.features import build_pipeline


def make_synthetic_telco(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Raw-format rows with the real schema and a learnable churn signal."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(config.CATEGORY_LEVELS["Contract"], n, p=[0.55, 0.2, 0.25])
    internet = rng.choice(config.CATEGORY_LEVELS["InternetService"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 118, n).round(2)

    def addon() -> np.ndarray:
        return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))

    df = pd.DataFrame({
        "customerID": [f"TEST-{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "InternetService": internet,
        **{col: addon() for col in config.ADDON_SERVICE_COLS},
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(config.CATEGORY_LEVELS["PaymentMethod"], n),
        "MonthlyCharges": monthly,
        # Stored as text with blanks for tenure 0, exactly like the source file.
        "TotalCharges": np.where(tenure == 0, " ", (tenure * monthly).round(2).astype(str)),
    })
    logit = -1.0 + 1.6 * (contract == "Month-to-month") + 0.8 * (internet == "Fiber optic") - 0.04 * tenure
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df[config.RAW_COLUMNS]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_telco()


@pytest.fixture(scope="session")
def tiny_bundle_path(tmp_path_factory) -> str:
    """A small logistic-regression bundle in the same format train.py saves."""
    X, y, _ = split_features_target(make_synthetic_telco(n=400, seed=1))
    pipeline = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)
    bundle = {
        "pipeline": pipeline,
        "threshold": 0.4,
        "metadata": {
            "model_name": "logistic_regression",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "required_columns": list(config.REQUIRED_INPUT_COLS),
            "test_metrics": {"roc_auc": 0.5},
        },
    }
    path = tmp_path_factory.mktemp("model") / "tiny_model.joblib"
    joblib.dump(bundle, path)
    return str(path)


@pytest.fixture
def tiny_bundle(tiny_bundle_path) -> dict:
    return joblib.load(tiny_bundle_path)
