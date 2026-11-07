import pandas as pd
import pytest

from churn import config
from churn.predict import MissingColumnsError, load_model, main, predict_df


def _raw_proba(bundle, df):
    return bundle["pipeline"].predict_proba(df[config.REQUIRED_INPUT_COLS])[:, 1]


def test_outputs_probability_and_thresholded_label(tiny_bundle, synthetic_df):
    out = predict_df(tiny_bundle, synthetic_df)

    assert list(out.columns) == ["customerID", "churn_probability", "churn_prediction"]
    assert out["customerID"].tolist() == synthetic_df["customerID"].tolist()
    assert out["churn_probability"].between(0, 1).all()
    expected = (_raw_proba(tiny_bundle, synthetic_df) >= tiny_bundle["threshold"]).astype(int)
    assert out["churn_prediction"].tolist() == expected.tolist()


def test_gender_and_other_extra_columns_do_not_change_predictions(tiny_bundle, synthetic_df):
    flipped = synthetic_df.assign(
        gender=synthetic_df["gender"].map({"Male": "Female", "Female": "Male"}),
        Churn="No",
        unrelated_column=1,
    )
    pd.testing.assert_frame_equal(predict_df(tiny_bundle, synthetic_df),
                                  predict_df(tiny_bundle, flipped))


def test_missing_columns_are_listed(tiny_bundle, synthetic_df):
    with pytest.raises(MissingColumnsError, match="tenure, .*Contract"):
        predict_df(tiny_bundle, synthetic_df.drop(columns=["tenure", "Contract"]))


def test_threshold_override(tiny_bundle, synthetic_df):
    strict = predict_df(tiny_bundle, synthetic_df, threshold=0.95)
    lenient = predict_df(tiny_bundle, synthetic_df, threshold=0.05)
    assert strict["churn_prediction"].sum() < lenient["churn_prediction"].sum()
    assert (strict["churn_probability"] == lenient["churn_probability"]).all()


def test_works_without_customer_id(tiny_bundle, synthetic_df):
    out = predict_df(tiny_bundle, synthetic_df.drop(columns="customerID"))
    assert "customerID" not in out.columns
    assert len(out) == len(synthetic_df)


def test_load_model_uses_env_var(tiny_bundle_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", tiny_bundle_path)
    assert load_model()["threshold"] == 0.4


def test_load_model_missing_file_explains_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn.train"):
        load_model(tmp_path / "missing.joblib")


def test_cli_round_trip(tmp_path, tiny_bundle_path, synthetic_df, capsys):
    source, target = tmp_path / "customers.csv", tmp_path / "out" / "predictions.csv"
    synthetic_df.to_csv(source, index=False)

    code = main(["--input", str(source), "--output", str(target), "--model", tiny_bundle_path])

    assert code == 0
    written = pd.read_csv(target)
    assert len(written) == len(synthetic_df)
    assert set(written.columns) == {"customerID", "churn_probability", "churn_prediction"}
    assert "Scored 400 customers" in capsys.readouterr().out


def test_cli_reports_missing_columns(tmp_path, tiny_bundle_path, synthetic_df, capsys):
    source = tmp_path / "customers.csv"
    synthetic_df.drop(columns="Contract").to_csv(source, index=False)

    code = main(["--input", str(source), "--output", str(tmp_path / "o.csv"),
                 "--model", tiny_bundle_path])

    assert code == 1
    assert "Contract" in capsys.readouterr().err
