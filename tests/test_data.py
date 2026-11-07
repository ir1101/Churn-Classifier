import pandas as pd
import pytest

from churn import config
from churn.data import DataError, download_raw, load_raw, split_features_target


def test_split_maps_target_and_drops_non_features(synthetic_df):
    X, y, ids = split_features_target(synthetic_df)

    assert list(X.columns) == config.REQUIRED_INPUT_COLS
    for col in ("customerID", "gender", "Churn"):
        assert col not in X.columns
    assert y.tolist() == (synthetic_df["Churn"] == "Yes").astype(int).tolist()
    assert ids.tolist() == synthetic_df["customerID"].tolist()


def test_missing_column_is_named(synthetic_df):
    with pytest.raises(DataError, match="Contract"):
        split_features_target(synthetic_df.drop(columns="Contract"))


def test_unexpected_target_values_are_rejected(synthetic_df):
    synthetic_df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataError, match="Yes"):
        split_features_target(synthetic_df)


def test_load_raw_reads_and_validates(tmp_path, synthetic_df):
    path = tmp_path / "telco.csv"
    synthetic_df.to_csv(path, index=False)
    loaded = load_raw(path)
    assert len(loaded) == len(synthetic_df)


def test_load_raw_missing_file_explains_next_step(tmp_path):
    with pytest.raises(DataError, match="churn.train"):
        load_raw(tmp_path / "nope.csv")


def test_download_skips_existing_file(tmp_path):
    path = tmp_path / "already_here.csv"
    path.write_text("placeholder")
    # An unreachable URL proves no request is made when the file exists.
    assert download_raw(path, url="http://127.0.0.1:9/unreachable.csv") == path
    assert path.read_text() == "placeholder"


def test_download_failure_suggests_manual_fallback(tmp_path):
    with pytest.raises(DataError, match="Kaggle"):
        download_raw(tmp_path / "telco.csv", url="http://127.0.0.1:9/unreachable.csv")
    assert not (tmp_path / "telco.csv").exists()
