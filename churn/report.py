"""Write evaluation reports: metrics.json, CSV tables, PNG figures and dashboard datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from churn import evaluate  # noqa: E402

# Chart colors for a light surface: recessive chrome, one data hue.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

DPI = 160


def _rc() -> dict:
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans"]
    family = [f for f in preferred if f in available][:1] + ["DejaVu Sans"]
    return {
        "font.family": family,
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
    }


def _end_dot(ax, x, y) -> None:
    """Emphasized point: series fill with a surface-colored ring."""
    ax.plot(x, y, "o", markersize=9, color=SERIES, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)


def _reference_line(ax, label: str, label_xy: tuple[float, float]) -> None:
    """Diagonal y = x drawn as a muted hairline with a direct label above it."""
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, zorder=1)
    ax.text(*label_xy, label, color=MUTED, fontsize=9, rotation=45,
            rotation_mode="anchor", ha="center", va="bottom")


def plot_roc_pr(path: Path, y, p, m: dict) -> None:
    roc, pr = evaluate.roc_points(y, p), evaluate.pr_points(y, p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.9), layout="constrained")

    _reference_line(ax1, "Random guess", (0.62, 0.63))
    ax1.plot(roc["fpr"], roc["tpr"], color=SERIES, zorder=3)
    ax1.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
            xlabel="False positive rate", ylabel="True positive rate (recall)")
    ax1.set_title(f"ROC curve, AUC {m['roc_auc']:.3f}")

    base = m["churn_rate"]
    ax2.axhline(base, color=MUTED, linewidth=1, zorder=1)
    ax2.text(0.02, base + 0.015, f"Random guess ({base:.1%} churn rate)",
             color=MUTED, fontsize=9, ha="left", va="bottom")
    ax2.plot(pr["recall"], pr["precision"], color=SERIES, zorder=3)
    _end_dot(ax2, m["recall"], m["precision"])
    ax2.annotate(
        f"Cutoff {m['threshold']:.3f}: precision {m['precision']:.2f}, "
        f"recall {m['recall']:.2f}",
        xy=(m["recall"], m["precision"]), xytext=(-10, -12),
        textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=9,
    )
    ax2.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
            xlabel="Recall (share of churners flagged)",
            ylabel="Precision (share of flagged who churn)")
    ax2.set_title(f"Precision-recall curve, AP {m['pr_auc']:.3f}")

    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_confusion(path: Path, m: dict) -> None:
    cells = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    names = [["Stayed, not flagged", "Stayed, flagged\n(false alarm)"],
             ["Churned, missed", "Churned, flagged\n(caught)"]]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    norm = Normalize(vmin=0, vmax=cells.max())

    fig, ax = plt.subplots(figsize=(5.8, 5), layout="constrained")
    ax.imshow(cells, cmap=cmap, norm=norm)
    for row in range(2):
        for col in range(2):
            r, g, b, _ = cmap(norm(cells[row, col]))
            text_color = "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK
            ax.text(col, row - 0.08, f"{cells[row, col]:,}", ha="center", va="center",
                    fontsize=18, fontweight="semibold", color=text_color)
            ax.text(col, row + 0.2, names[row][col], ha="center", va="center",
                    fontsize=9, color=text_color)
    ax.axvline(0.5, color=SURFACE, linewidth=4)
    ax.axhline(0.5, color=SURFACE, linewidth=4)
    ax.set_xticks([0, 1], ["Predicted: stays", "Predicted: churns"])
    ax.set_yticks([0, 1], ["Actually stayed", "Actually churned"])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Confusion matrix, test set (cutoff {m['threshold']:.3f})")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_gains(path: Path, gains: pd.DataFrame, top20: float) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
    _reference_line(ax, "Random targeting", (0.66, 0.67))
    ax.plot(gains["share_contacted"], gains["share_churners_captured"], color=SERIES, zorder=3)
    _end_dot(ax, 0.2, top20)
    # Label sits in the empty space above the curve, so its leader crosses nothing.
    ax.annotate(
        f"Contacting the riskiest 20%\ncatches {top20:.0%} of churners",
        xy=(0.2, top20), xytext=(0.03, 0.85), textcoords="data",
        color=INK_2, fontsize=9, ha="left", va="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1,
                    "shrinkA": 4, "shrinkB": 6},
    )
    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
           xlabel="Customers contacted, riskiest first",
           ylabel="Churners caught")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Cumulative gains, test set")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_calibration(path: Path, calibration: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
    # Label the diagonal past the last bin, where the model line never reaches.
    _reference_line(ax, "Perfectly calibrated", (0.87, 0.875))
    ax.plot(calibration["mean_predicted"], calibration["observed_rate"], "-o", color=SERIES,
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
           xlabel="Predicted churn probability", ylabel="Observed churn rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Calibration, test set (10 equal-size bins)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_importance(path: Path, importances: pd.DataFrame) -> None:
    imp = importances.sort_values("importance_mean").reset_index(drop=True)
    values = imp["importance_mean"].clip(lower=0)
    rows = np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(imp) + 1.3), layout="constrained")
    ax.barh(rows, values, height=0.55, color=SERIES, zorder=3)
    ax.errorbar(values, rows, xerr=imp["importance_std"], fmt="none",
                ecolor=MUTED, elinewidth=1, capsize=0, zorder=4)
    pad = values.max() * 0.015
    for row, (v, sd) in enumerate(zip(imp["importance_mean"], imp["importance_std"])):
        label = f"{v:.3f}".replace("-0.000", "0.000")  # tiny negatives are noise
        ax.text(max(v, 0) + sd + pad, row, label, va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(rows, imp["label"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, (values + imp["importance_std"]).max() * 1.15)
    ax.set_xlabel("Drop in test PR-AUC when the column is shuffled (mean of repeats)")
    ax.set_title("What drives the predictions")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def dashboard_tables(
    metadata: dict, y_test, p_test, importances: pd.DataFrame, segments: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Tidy, aggregate-only tables for the results dashboard (no customer rows)."""
    m, ds, g = metadata["test_metrics"], metadata["dataset"], metadata["gains"]
    headline = pd.DataFrame(
        [
            ("customers", ds["customers"], "Customers in the dataset"),
            ("churners", ds["churners"], "Customers who churned"),
            ("churn_rate", ds["churn_rate"], "Share of customers who churned"),
            ("train_customers", ds["train_customers"], "Customers used for training"),
            ("test_customers", ds["test_customers"], "Customers held out for the final test"),
            ("threshold", metadata["threshold"], "Probability cutoff for flagging a customer"),
            ("roc_auc", m["roc_auc"], "Test ROC-AUC"),
            ("pr_auc", m["pr_auc"], "Test PR-AUC (average precision)"),
            ("precision", m["precision"], "Share of flagged test customers who churned"),
            ("recall", m["recall"], "Share of test churners who were flagged"),
            ("f1", m["f1"], "F1 score at the cutoff"),
            ("accuracy", m["accuracy"], "Accuracy at the cutoff"),
            ("baseline_accuracy", m["baseline_accuracy"], "Accuracy of always predicting no churn"),
            ("brier", m["brier"], "Brier score (lower is better)"),
            ("flagged_share", m["flagged_share"], "Share of test customers flagged"),
            ("true_negatives", m["tn"], "Stayed and not flagged"),
            ("false_positives", m["fp"], "Stayed but flagged (false alarm)"),
            ("false_negatives", m["fn"], "Churned but not flagged (missed)"),
            ("true_positives", m["tp"], "Churned and flagged (caught)"),
            ("captured_top_10pct", g["top_10pct"], "Churners caught by contacting the riskiest 10%"),
            ("captured_top_20pct", g["top_20pct"], "Churners caught by contacting the riskiest 20%"),
            ("captured_top_30pct", g["top_30pct"], "Churners caught by contacting the riskiest 30%"),
        ],
        columns=["metric", "value", "description"],
    )
    comparison = pd.DataFrame([
        {
            "model": s["model"],
            "label": s["label"],
            "cv_pr_auc_mean": s["cv_pr_auc_mean"],
            "cv_pr_auc_std": s["cv_pr_auc_std"],
            "cv_roc_auc_mean": s["cv_roc_auc_mean"],
            "cv_roc_auc_std": s["cv_roc_auc_std"],
            "cv_brier_mean": s["cv_brier_mean"],
            "selected": "yes" if s["selected"] else "no",
            "best_params": ", ".join(f"{k}={v}" for k, v in s["best_params"].items()),
        }
        for s in metadata["cv"]
    ])
    # Long format: one row per point, `curve` says which chart it belongs to.
    curve_tables = {
        "roc": evaluate.roc_points(y_test, p_test),                # x = FPR, y = TPR
        "pr": evaluate.pr_points(y_test, p_test),                  # x = recall, y = precision
        "gains": evaluate.gains_curve(y_test, p_test),             # x = contacted, y = caught
        "calibration": evaluate.calibration_table(y_test, p_test), # x = predicted, y = observed
    }
    curves = pd.concat(
        [t.set_axis(["x", "y"], axis=1).assign(curve=name) for name, t in curve_tables.items()],
        ignore_index=True,
    )[["curve", "x", "y"]]
    return {
        "headline_metrics": headline,
        "model_comparison": comparison,
        "curves": curves,
        "thresholds": evaluate.threshold_table(y_test, p_test),
        "feature_importance": importances,
        "segment_churn": segments,
    }


def write_reports(
    reports_dir: Path,
    metadata: dict,
    y_test,
    p_test,
    importances: pd.DataFrame,
    segments: pd.DataFrame,
) -> dict[str, Path]:
    """Write every report file and return their paths."""
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    dashboard_dir = reports_dir / "dashboard"
    for folder in (reports_dir, figures_dir, dashboard_dir):
        folder.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written["metrics"] = metrics_path

    tables = dashboard_tables(metadata, y_test, p_test, importances, segments)
    for name, table in tables.items():
        path = dashboard_dir / f"{name}.csv"
        table.round(4).to_csv(path, index=False)
        written[f"dashboard_{name}"] = path
    comparison_path = reports_dir / "model_comparison.csv"
    tables["model_comparison"].round(4).to_csv(comparison_path, index=False)
    written["model_comparison"] = comparison_path

    m = metadata["test_metrics"]
    gains = evaluate.gains_curve(y_test, p_test)
    with plt.rc_context(_rc()):
        figures = {
            "roc_pr_curves": lambda path: plot_roc_pr(path, y_test, p_test, m),
            "confusion_matrix": lambda path: plot_confusion(path, m),
            "gains_curve": lambda path: plot_gains(path, gains, metadata["gains"]["top_20pct"]),
            "calibration": lambda path: plot_calibration(
                path, evaluate.calibration_table(y_test, p_test)),
            "feature_importance": lambda path: plot_importance(path, importances),
        }
        for name, draw in figures.items():
            path = figures_dir / f"{name}.png"
            draw(path)
            written[f"figure_{name}"] = path
    return written
