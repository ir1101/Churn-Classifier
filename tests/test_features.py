import numpy as np
import pandas as pd

from churn import config
from churn.data import split_features_target
from churn.features import add_features, build_preprocessor


def _one_customer(**overrides) -> pd.DataFrame:
    row = {
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 10,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": "800.0",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_blank_total_charges_for_new_customer_becomes_zero():
    out = add_features(_one_customer(tenure=0, TotalCharges=" "))
    assert out.loc[0, "TotalCharges"] == 0.0
    # No billing history yet, so average spend falls back to the monthly charge.
    assert out.loc[0, "avg_monthly_spend"] == 80.0


def test_missing_total_charges_estimated_from_tenure():
    out = add_features(_one_customer(tenure=5, TotalCharges=None))
    assert out.loc[0, "TotalCharges"] == 400.0


def test_text_total_charges_parsed_and_average_computed():
    out = add_features(_one_customer(tenure=10, TotalCharges="850.0"))
    assert out.loc[0, "TotalCharges"] == 850.0
    assert out.loc[0, "avg_monthly_spend"] == 85.0


def test_addon_count_and_tenure_group():
    out = add_features(_one_customer())
    assert out.loc[0, "num_addon_services"] == 3  # security, device protection, TV
    assert out.loc[0, "tenure_group"] == "0-12"
    assert add_features(_one_customer(tenure=60)).loc[0, "tenure_group"] == "49+"


def test_input_is_not_mutated():
    original = _one_customer(TotalCharges=" ", tenure=0)
    snapshot = original.copy()
    add_features(original)
    pd.testing.assert_frame_equal(original, snapshot)


def test_preprocessor_output_is_numeric_and_complete(synthetic_df):
    X, _, _ = split_features_target(synthetic_df)
    X.loc[X.index[:3], "Contract"] = np.nan  # missing values get imputed
    matrix = build_preprocessor().fit_transform(add_features(X))
    assert matrix.shape[0] == len(X)
    assert not np.isnan(matrix).any()
    expected_min_width = len(config.MODEL_NUMERIC_COLS) + len(config.MODEL_CATEGORICAL_COLS)
    assert matrix.shape[1] > expected_min_width
