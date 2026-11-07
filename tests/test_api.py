import typing

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn import config
from churn.api import EXAMPLE_CUSTOMER, Customer, app
from churn.predict import predict_df


@pytest.fixture
def client(tiny_bundle_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", tiny_bundle_path)
    with TestClient(app) as test_client:  # runs the lifespan, which loads the model
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "logistic_regression"
    assert body["threshold"] == 0.4


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_predict_one_customer(client, tiny_bundle):
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["customerID"] == EXAMPLE_CUSTOMER["customerID"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])
    # Same answer as the batch scoring path used by the CLI.
    expected = predict_df(tiny_bundle, pd.DataFrame([EXAMPLE_CUSTOMER]))
    assert body["churn_probability"] == pytest.approx(expected["churn_probability"].iloc[0])


def test_total_charges_is_optional(client):
    customer = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "TotalCharges"}
    assert client.post("/predict", json=customer).status_code == 200


@pytest.mark.parametrize("field, value", [
    ("Contract", "Weekly"),
    ("InternetService", "Satellite"),
    ("tenure", -3),
    ("SeniorCitizen", 2),
    ("MonthlyCharges", -10),
])
def test_invalid_values_are_rejected(client, field, value):
    response = client.post("/predict", json={**EXAMPLE_CUSTOMER, field: value})
    assert response.status_code == 422
    assert field in response.text


def test_missing_field_is_rejected(client):
    customer = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "Contract"}
    assert client.post("/predict", json=customer).status_code == 422


def test_batch_keeps_order(client):
    customers = [
        {**EXAMPLE_CUSTOMER, "customerID": "A", "tenure": 2, "Contract": "Month-to-month"},
        {**EXAMPLE_CUSTOMER, "customerID": "B", "tenure": 70, "Contract": "Two year"},
        {**EXAMPLE_CUSTOMER, "customerID": "C"},
    ]
    response = client.post("/predict/batch", json={"customers": customers})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [p["customerID"] for p in predictions] == ["A", "B", "C"]


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_allowed_values_match_the_dataset():
    for column, levels in config.CATEGORY_LEVELS.items():
        allowed = typing.get_args(Customer.model_fields[column].annotation)
        assert set(allowed) == set(levels), column


def test_request_fields_cover_every_model_input():
    assert set(config.REQUIRED_INPUT_COLS) <= set(Customer.model_fields)
