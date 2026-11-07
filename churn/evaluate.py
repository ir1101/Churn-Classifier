"""Metrics and analysis tables computed from predicted churn probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn import config
from churn.features import TENURE_LABELS, add_features

# Segments shown in the "churn rate by segment" view, in display order.
SEGMENT_COLS = [
    "Contract", "tenure_group", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling", "SeniorCitizen",
]


def best_f1_threshold(y_true, proba, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Return the probability cutoff (on a 0.005 grid) that maximizes F1, and that F1."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.951, 0.005), 3)
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(proba)[None, :] >= grid[:, None]
    tp = (pred & y).sum(axis=1)
    fp = (pred & ~y).sum(axis=1)
    fn = (~pred & y).sum(axis=1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best = int(np.argmax(f1))
    return float(grid[best]), float(f1[best])


def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Ranking, probability and thresholded metrics for one set of predictions."""
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    churn_rate = float(y.mean())
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "flagged_share": float(pred.mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(y)),
        "churn_rate": churn_rate,
        "baseline_accuracy": max(churn_rate, 1 - churn_rate),
    }


def threshold_table(y_true, proba, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Precision, recall, F1 and share of customers flagged at each cutoff."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "flagged_share": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def gains_curve(y_true, proba, points: int = 100) -> pd.DataFrame:
    """Cumulative share of churners caught when contacting the riskiest customers first."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(proba, dtype=float), kind="stable")
    captured = np.concatenate([[0.0], np.cumsum(y[order]) / y.sum()])
    contacted = np.arange(len(y) + 1) / len(y)
    grid = np.linspace(0, 1, points + 1)
    return pd.DataFrame({
        "share_contacted": grid,
        "share_churners_captured": np.interp(grid, contacted, captured),
    })


def captured_at(gains: pd.DataFrame, share: float) -> float:
    """Share of churners caught when contacting the top `share` of customers."""
    return float(np.interp(share, gains["share_contacted"], gains["share_churners_captured"]))


def _thin(df: pd.DataFrame, max_points: int = 300) -> pd.DataFrame:
    """Keep at most `max_points` evenly spaced rows (always the first and last)."""
    if len(df) <= max_points:
        return df.reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(df) - 1, max_points).round().astype(int))
    return df.iloc[idx].reset_index(drop=True)


def roc_points(y_true, proba) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return _thin(pd.DataFrame({"fpr": fpr, "tpr": tpr}))


def pr_points(y_true, proba) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    # precision_recall_curve runs from high to low recall; flip to ascending.
    return _thin(pd.DataFrame({"recall": recall[::-1], "precision": precision[::-1]}))


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed})


def permutation_importances(
    pipeline, X: pd.DataFrame, y, n_repeats: int = 10, n_jobs: int | None = None
) -> pd.DataFrame:
    """Drop in test PR-AUC when each original input column is shuffled."""
    result = permutation_importance(
        pipeline, X, y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=n_jobs,
    )
    table = pd.DataFrame({
        "feature": X.columns,
        "label": [config.FEATURE_LABELS.get(c, c) for c in X.columns],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def segment_churn(X: pd.DataFrame, y) -> pd.DataFrame:
    """Churn rate for each value of a few key customer attributes."""
    feats = add_features(X)
    feats["SeniorCitizen"] = feats["SeniorCitizen"].map({0: "No", 1: "Yes"})
    churn = np.asarray(y).astype(int)
    level_order = {**config.CATEGORY_LEVELS, "tenure_group": TENURE_LABELS, "SeniorCitizen": ["No", "Yes"]}

    rows = []
    for col in SEGMENT_COLS:
        stats = (
            pd.DataFrame({"segment": feats[col].astype(str).to_numpy(), "churn": churn})
            .groupby("segment")["churn"]
            .agg(customers="size", churners="sum", churn_rate="mean")
        )
        order = [s for s in level_order.get(col, []) if s in stats.index]
        order += [s for s in stats.index if s not in order]
        for rank, segment in enumerate(order):
            row = stats.loc[segment]
            rows.append({
                "feature": col,
                "feature_label": config.FEATURE_LABELS.get(col, col),
                "segment": segment,
                "segment_order": rank,
                "customers": int(row["customers"]),
                "churners": int(row["churners"]),
                "churn_rate": float(row["churn_rate"]),
            })
    return pd.DataFrame(rows)
