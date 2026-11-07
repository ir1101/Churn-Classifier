"""Project-wide settings: paths, data source, column groups and modelling constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Data source -------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
EXPECTED_ROWS = 7043

# --- Output locations (relative to an output root, normally PROJECT_ROOT) ----
MODEL_FILENAME = "churn_model.joblib"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME

# --- Columns -----------------------------------------------------------------
ID_COL = "customerID"
TARGET_COL = "Churn"

# Every column of the source file, in file order.
RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

# gender carries almost no churn signal in this data, and leaving it out keeps
# retention targeting from depending on it.
EXCLUDED_COLS = ["gender"]

# Raw columns the model reads (file order).
REQUIRED_INPUT_COLS = [
    c for c in RAW_COLUMNS if c not in (ID_COL, TARGET_COL, *EXCLUDED_COLS)
]

ADDON_SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Allowed values of each categorical input, as they appear in the source file.
_ADDON_LEVELS = ["Yes", "No", "No internet service"]
CATEGORY_LEVELS = {
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{col: _ADDON_LEVELS for col in ADDON_SERVICE_COLS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

NUMERIC_INPUT_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_INPUT_COLS = list(CATEGORY_LEVELS)

# Columns added by churn.features.add_features.
ENGINEERED_NUMERIC_COLS = ["num_addon_services", "avg_monthly_spend"]
ENGINEERED_CATEGORICAL_COLS = ["tenure_group"]

MODEL_NUMERIC_COLS = NUMERIC_INPUT_COLS + ENGINEERED_NUMERIC_COLS
MODEL_CATEGORICAL_COLS = CATEGORICAL_INPUT_COLS + ENGINEERED_CATEGORICAL_COLS

# Readable names for reports and the dashboard.
FEATURE_LABELS = {
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "tenure": "Tenure (months)",
    "tenure_group": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}

# --- Modelling ---------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# Prefer logistic regression when its CV PR-AUC is within this margin of the best.
SIMPLICITY_MARGIN = 0.005
