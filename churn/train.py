"""Train, compare and select a churn model, then save the model and reports.

Usage (from the project folder):
    python -m churn.train            # full search, a few minutes
    python -m churn.train --quick    # small search, for smoke tests
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from xgboost import XGBClassifier

from churn import config, evaluate, report
from churn.data import download_raw, load_raw, split_features_target
from churn.features import build_pipeline

# Metrics recorded during cross-validation. The search refits on PR-AUC, which
# focuses on how well the model ranks the minority (churn) class.
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
    "log_loss": "neg_log_loss",
}


def candidate_searches(quick: bool = False) -> list[dict]:
    """Model families to compare, each with its hyperparameter search space.

    Each model runs single-threaded; the search itself runs folds in parallel.
    """
    rs = config.RANDOM_STATE
    logreg = {
        "name": "logistic_regression",
        "label": "Logistic regression",
        "model": LogisticRegression(max_iter=5000),
        "params": {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    }
    forest = {
        "name": "random_forest",
        "label": "Random forest",
        # The slowest family to fit (~3-5 s per fit), so its grid is kept small.
        "model": RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=1),
        "params": {
            "model__min_samples_leaf": [5, 10, 20],
            "model__max_features": ["sqrt", 0.3],
        },
    }
    xgb = {
        "name": "xgboost",
        "label": "XGBoost",
        "model": XGBClassifier(
            tree_method="hist", eval_metric="logloss", random_state=rs, n_jobs=1
        ),
        "params": {
            "model__n_estimators": [200, 400, 600],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [1.0, 3.0, 10.0],
        },
        "n_iter": 25,
    }
    if quick:
        logreg["params"] = {"model__C": [0.1, 1.0]}
        forest["model"].set_params(n_estimators=100)
        forest["params"] = {"model__min_samples_leaf": [10], "model__max_features": ["sqrt"]}
        xgb["params"] = {
            "model__n_estimators": [150],
            "model__learning_rate": [0.05],
            "model__max_depth": [3],
        }
        del xgb["n_iter"]
    return [logreg, forest, xgb]


def _plain(value):
    """numpy scalar -> plain Python value (for JSON)."""
    return value.item() if hasattr(value, "item") else value


def tune(spec: dict, X: pd.DataFrame, y: pd.Series, cv, n_jobs: int):
    """Cross-validated hyperparameter search for one model family."""
    pipe = build_pipeline(spec["model"])
    common = dict(scoring=SCORING, refit="pr_auc", cv=cv, n_jobs=n_jobs, error_score="raise")
    if "n_iter" in spec:
        search = RandomizedSearchCV(
            pipe, spec["params"], n_iter=spec["n_iter"],
            random_state=config.RANDOM_STATE, **common,
        )
    else:
        search = GridSearchCV(pipe, spec["params"], **common)

    start = time.perf_counter()
    search.fit(X, y)
    res, i = search.cv_results_, search.best_index_
    summary = {
        "model": spec["name"],
        "label": spec["label"],
        "cv_pr_auc_mean": float(res["mean_test_pr_auc"][i]),
        "cv_pr_auc_std": float(res["std_test_pr_auc"][i]),
        "cv_roc_auc_mean": float(res["mean_test_roc_auc"][i]),
        "cv_roc_auc_std": float(res["std_test_roc_auc"][i]),
        "cv_brier_mean": float(-res["mean_test_brier"][i]),
        "cv_log_loss_mean": float(-res["mean_test_log_loss"][i]),
        "best_params": {
            k.removeprefix("model__"): _plain(v) for k, v in search.best_params_.items()
        },
        "settings_tried": len(res["params"]),
        "search_seconds": round(time.perf_counter() - start, 1),
    }
    return search.best_estimator_, summary


def select_model(summaries: list[dict]) -> tuple[str, str]:
    """Pick the best CV PR-AUC, preferring logistic regression when it is close."""
    best = max(summaries, key=lambda s: s["cv_pr_auc_mean"])
    logreg = next((s for s in summaries if s["model"] == "logistic_regression"), None)
    if logreg is not None and best is not logreg:
        gap = best["cv_pr_auc_mean"] - logreg["cv_pr_auc_mean"]
        if gap <= config.SIMPLICITY_MARGIN:
            return logreg["model"], (
                f"Within {gap:.3f} PR-AUC of the top scorer ({best['label']}), "
                "and simpler to explain, so logistic regression was preferred."
            )
    return best["model"], "Highest cross-validated PR-AUC."


def run(
    quick: bool = False,
    out_dir: Path = config.PROJECT_ROOT,
    n_jobs: int | None = None,
    raw_csv: Path = config.RAW_CSV,
) -> dict:
    """Run the whole training workflow and return the saved model's metadata.

    n_jobs defaults to all cores for a full run and 1 for a quick run: starting
    worker processes costs several seconds on Windows, more than a quick run saves.
    """
    out_dir = Path(out_dir)
    if n_jobs is None:
        n_jobs = 1 if quick else -1
    started = time.perf_counter()

    raw = load_raw(download_raw(raw_csv))
    X, y, _ = split_features_target(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    folds = 3 if quick else config.CV_FOLDS
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    print(
        f"Training on {len(X_train):,} customers; holding out {len(X_test):,} "
        f"for the final test.{' (quick mode)' if quick else ''}"
    )

    fitted, summaries = {}, []
    for spec in candidate_searches(quick):
        print(f"  Tuning {spec['label']}...", flush=True)
        pipe, summary = tune(spec, X_train, y_train, cv, n_jobs)
        fitted[spec["name"]] = pipe
        summaries.append(summary)

    chosen, reason = select_model(summaries)
    for s in summaries:
        s["selected"] = s["model"] == chosen
    chosen_summary = next(s for s in summaries if s["selected"])
    final = fitted[chosen]

    # Decision threshold from out-of-fold probabilities (training data only).
    oof = cross_val_predict(
        clone(final), X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs
    )[:, 1]
    threshold, oof_f1 = evaluate.best_f1_threshold(y_train, oof)

    # The one and only look at the held-out test set.
    p_test = final.predict_proba(X_test)[:, 1]
    test_metrics = evaluate.classification_metrics(y_test, p_test, threshold)
    gains = evaluate.gains_curve(y_test, p_test)
    print("  Measuring feature importance...", flush=True)
    importances = evaluate.permutation_importances(
        final, X_test, y_test, n_repeats=3 if quick else 10, n_jobs=1
    )
    segments = evaluate.segment_churn(X, y)

    metadata = {
        "model_name": chosen,
        "model_label": chosen_summary["label"],
        "best_params": chosen_summary["best_params"],
        "selection_reason": reason,
        "threshold": threshold,
        "threshold_rule": "Maximizes F1 on out-of-fold predictions for the training set",
        "oof_f1": oof_f1,
        "cv_folds": folds,
        "cv": summaries,
        "test_metrics": test_metrics,
        "gains": {
            f"top_{k}pct": evaluate.captured_at(gains, k / 100) for k in (10, 20, 30)
        },
        "dataset": {
            "source": config.DATA_URL,
            "customers": int(len(raw)),
            "churners": int(y.sum()),
            "churn_rate": float(y.mean()),
            "train_customers": int(len(X_train)),
            "test_customers": int(len(X_test)),
        },
        "required_columns": list(config.REQUIRED_INPUT_COLS),
        "excluded_columns": list(config.EXCLUDED_COLS),
        "quick_mode": quick,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    model_path = out_dir / "models" / config.MODEL_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "threshold": threshold, "metadata": metadata},
                model_path, compress=3)

    holdout_path = out_dir / "data" / "processed" / "holdout.csv"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    raw.loc[X_test.index].to_csv(holdout_path, index=False)

    reports_dir = out_dir / "reports"
    report.write_reports(
        reports_dir,
        metadata=metadata,
        y_test=y_test.to_numpy(),
        p_test=p_test,
        importances=importances,
        segments=segments,
    )

    print_summary(metadata)
    print(f"\nSaved model:    {model_path}")
    print(f"Holdout rows:   {holdout_path}")
    print(f"Reports:        {reports_dir}")
    minutes, seconds = divmod(int(time.perf_counter() - started), 60)
    print(f"Done in {minutes}m {seconds:02d}s.")
    return metadata


def print_summary(meta: dict) -> None:
    """Plain-ASCII summary for the console."""
    print(f"\nCross-validated scores on the training set ({meta['cv_folds']} folds, mean +/- std):")
    print(f"  {'Model':<22}{'PR-AUC':<18}{'ROC-AUC':<18}{'Brier':<8}")
    for s in meta["cv"]:
        flag = "  <- selected" if s["selected"] else ""
        print(
            f"  {s['label']:<22}"
            f"{s['cv_pr_auc_mean']:.3f} +/- {s['cv_pr_auc_std']:.3f}   "
            f"{s['cv_roc_auc_mean']:.3f} +/- {s['cv_roc_auc_std']:.3f}   "
            f"{s['cv_brier_mean']:.3f}{flag}"
        )
    print(f"\nSelected: {meta['model_label']}. {meta['selection_reason']}")
    print(f"Best settings: {meta['best_params']}")
    print(f"Decision threshold: {meta['threshold']:.3f} (out-of-fold F1 {meta['oof_f1']:.3f})")

    m, ds = meta["test_metrics"], meta["dataset"]
    print(f"\nHeld-out test set ({m['n']:,} customers, {m['churn_rate']:.1%} churn):")
    print(f"  ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | Brier {m['brier']:.3f}")
    print(
        f"  At threshold {m['threshold']:.3f}: precision {m['precision']:.3f}, "
        f"recall {m['recall']:.3f}, F1 {m['f1']:.3f}, accuracy {m['accuracy']:.3f} "
        f"(always predicting 'no churn' scores {m['baseline_accuracy']:.3f})"
    )
    print(
        f"  Contacting the riskiest 20% of customers catches "
        f"{meta['gains']['top_20pct']:.0%} of churners."
    )
    print(f"  Dataset: {ds['customers']:,} customers, {ds['churn_rate']:.1%} churned.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Telco churn classifier.")
    parser.add_argument("--quick", action="store_true",
                        help="small hyperparameter search, for smoke tests")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="parallel workers for the searches "
                             "(default: all cores, or 1 with --quick)")
    parser.add_argument("--out-dir", type=Path, default=config.PROJECT_ROOT,
                        help="folder that receives models/, reports/ and data/processed/")
    args = parser.parse_args(argv)
    run(quick=args.quick, out_dir=args.out_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
