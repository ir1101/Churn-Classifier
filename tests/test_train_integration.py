"""End-to-end check on the real dataset (skipped until it has been downloaded)."""

import json

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from churn import config
from churn.predict import load_model, predict_df
from churn.train import run, select_model

pytestmark = pytest.mark.skipif(
    not config.RAW_CSV.exists(),
    reason="real dataset not downloaded; run `python -m churn.train` once",
)

DASHBOARD_FILES = [
    "headline_metrics", "model_comparison", "curves",
    "thresholds", "feature_importance", "segment_churn",
]
FIGURES = ["roc_pr_curves", "confusion_matrix", "gains_curve", "calibration", "feature_importance"]


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("quick_run")
    return out_dir, run(quick=True, out_dir=out_dir)


def test_quick_run_performs_sensibly(quick_run):
    _, meta = quick_run
    metrics = meta["test_metrics"]
    assert 0.80 < metrics["roc_auc"] < 0.95  # far above 0.95 would suggest leakage
    assert metrics["pr_auc"] > 0.55
    assert 0.05 <= meta["threshold"] <= 0.95
    assert meta["dataset"]["customers"] == config.EXPECTED_ROWS


def test_quick_run_writes_all_outputs(quick_run):
    out_dir, meta = quick_run
    assert (out_dir / "models" / config.MODEL_FILENAME).exists()
    assert json.loads((out_dir / "reports" / "metrics.json").read_text())["model_name"] == meta["model_name"]
    assert (out_dir / "reports" / "model_comparison.csv").exists()
    for name in FIGURES:
        assert (out_dir / "reports" / "figures" / f"{name}.png").stat().st_size > 10_000
    for name in DASHBOARD_FILES:
        table = pd.read_csv(out_dir / "reports" / "dashboard" / f"{name}.csv")
        assert not table.empty
        assert config.ID_COL not in table.columns  # aggregates only


def test_saved_model_reproduces_reported_test_score(quick_run):
    out_dir, meta = quick_run
    holdout = pd.read_csv(out_dir / "data" / "processed" / "holdout.csv")
    assert len(holdout) == meta["dataset"]["test_customers"]

    bundle = load_model(out_dir / "models" / config.MODEL_FILENAME)
    scored = predict_df(bundle, holdout)
    auc = roc_auc_score(holdout["Churn"] == "Yes", scored["churn_probability"])
    assert auc == pytest.approx(meta["test_metrics"]["roc_auc"], abs=1e-3)


def test_selection_prefers_logistic_regression_when_close():
    summaries = [
        {"model": "logistic_regression", "label": "Logistic regression", "cv_pr_auc_mean": 0.660},
        {"model": "xgboost", "label": "XGBoost", "cv_pr_auc_mean": 0.663},
    ]
    assert select_model(summaries)[0] == "logistic_regression"
    summaries[1]["cv_pr_auc_mean"] = 0.700
    assert select_model(summaries)[0] == "xgboost"
