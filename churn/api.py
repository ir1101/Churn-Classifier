"""REST API for churn predictions.

Run from the project folder:
    python -m uvicorn churn.api:app --reload
then open http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from churn import __version__
from churn.predict import load_model, predict_df

YesNo = Literal["Yes", "No"]
PhoneLines = Literal["Yes", "No", "No phone service"]
InternetAddon = Literal["Yes", "No", "No internet service"]
InternetType = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentType = Literal[
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
]

EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class Customer(BaseModel):
    """One customer, using the column names and values of the Telco dataset."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customerID: str | None = Field(None, description="Optional; echoed back in the response")
    SeniorCitizen: int = Field(ge=0, le=1, description="1 if the customer is 65 or older")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=1200, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: PhoneLines
    InternetService: InternetType
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentType
    MonthlyCharges: float = Field(ge=0, description="Current monthly bill")
    TotalCharges: float | None = Field(
        None, ge=0,
        description="Billed to date. Leave empty for new customers; "
                    "it is then estimated as tenure x MonthlyCharges.",
    )


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float = Field(description="Estimated probability of churning, 0 to 1")
    churn_prediction: bool = Field(description="True when the probability reaches the threshold")
    threshold: float
    model: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fails fast with "Run `python -m churn.train` first" if there is no model yet.
    app.state.bundle = load_model()
    yield


app = FastAPI(
    title="Telco Churn Classifier",
    version=__version__,
    description="Predicts the probability that a telecom customer will churn.",
    lifespan=lifespan,
)


def _score(bundle: dict, customers: list[Customer]) -> list[Prediction]:
    frame = pd.DataFrame([c.model_dump() for c in customers])
    result = predict_df(bundle, frame)
    threshold = float(bundle["threshold"])
    model = bundle["metadata"]["model_name"]
    return [
        Prediction(
            customerID=customer.customerID,
            churn_probability=float(row.churn_probability),
            churn_prediction=bool(row.churn_prediction),
            threshold=threshold,
            model=model,
        )
        for customer, row in zip(customers, result.itertuples(index=False))
    ]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(request: Request) -> dict:
    """Confirm the service is up and show which model it is serving."""
    meta = request.app.state.bundle["metadata"]
    return {
        "status": "ok",
        "model": meta["model_name"],
        "threshold": float(request.app.state.bundle["threshold"]),
        "trained_at": meta["trained_at"],
        "test_roc_auc": round(meta["test_metrics"]["roc_auc"], 4),
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request) -> Prediction:
    """Churn probability for one customer."""
    return _score(request.app.state.bundle, [customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    """Churn probabilities for up to 1,000 customers at once."""
    return BatchResponse(predictions=_score(request.app.state.bundle, batch.customers))
