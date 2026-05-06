"""Evaluate the saved credit risk model and tune alert thresholds."""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


FEATURES_PATH = PROCESSED_DATA_DIR / "application_train_features.csv"
MODEL_PATH = MODELS_DIR / "credit_risk_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
FIGURES_DIR = REPORTS_DIR / "figures"
THRESHOLD_REPORT_PATH = REPORTS_DIR / "threshold_tuning_report.csv"
RISK_BAND_REPORT_PATH = REPORTS_DIR / "risk_band_report.csv"
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


def load_artifacts():
    """Load the trained model and expected feature columns."""
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


def prepare_test_data(feature_columns):
    """Prepare the same held-out test split used during model training."""
    df = pd.read_csv(FEATURES_PATH)
    y = df["TARGET"].copy()
    drop_columns = [column for column in ["TARGET", "SK_ID_CURR"] if column in df]
    X = df.drop(columns=drop_columns)

    X = X.replace([np.inf, -np.inf], np.nan)

    categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_columns:
        X[categorical_columns] = X[categorical_columns].fillna("Unknown")
        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False,
            dtype=np.uint8,
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    X = X.fillna(0)

    missing_columns = [column for column in feature_columns if column not in X]
    for column in missing_columns:
        X[column] = 0

    X = X[feature_columns]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Prepared test data shape: {X_test.shape}")
    return X_test, y_test


def plot_roc_curve(y_test, y_proba):
    """Plot and save the ROC curve."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()


def plot_precision_recall_curve(y_test, y_proba):
    """Plot and save the precision-recall curve."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"Average Precision = {average_precision:.4f}",
        linewidth=2,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, threshold):
    """Plot and save a confusion matrix for a selected threshold."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Non-Default", "Predicted Default"],
        yticklabels=["Actual Non-Default", "Actual Default"],
    )
    plt.title(f"Confusion Matrix at Threshold {threshold:.2f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"confusion_matrix_threshold_{threshold:.2f}.png",
        dpi=150,
    )
    plt.close()


def evaluate_thresholds(y_test, y_proba):
    """Evaluate classification performance across business alert thresholds."""
    threshold_rows = []

    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        flagged_rate = float(y_pred.mean() * 100)

        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": float(
                    precision_score(y_test, y_pred, zero_division=0)
                ),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "false_positive_count": int(fp),
                "false_negative_count": int(fn),
                "true_positive_count": int(tp),
                "true_negative_count": int(tn),
                "flagged_rate_percentage": flagged_rate,
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(THRESHOLD_REPORT_PATH, index=False)
    return threshold_df


def assign_risk_band(prob):
    """Assign a readable risk band from default probability."""
    if prob < 0.30:
        return "Low Risk"
    if prob < 0.60:
        return "Medium Risk"
    return "High Risk"


def create_risk_band_report(y_test, y_proba):
    """Create and save a risk-band summary report."""
    risk_df = pd.DataFrame(
        {
            "actual_target": y_test.to_numpy(),
            "predicted_probability": y_proba,
        }
    )
    risk_df["risk_band"] = risk_df["predicted_probability"].apply(assign_risk_band)

    risk_band_report = (
        risk_df.groupby("risk_band")
        .agg(
            customer_count=("actual_target", "size"),
            average_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            actual_default_rate=("actual_target", "mean"),
        )
        .reset_index()
    )

    band_order = {
        "Low Risk": 0,
        "Medium Risk": 1,
        "High Risk": 2,
    }
    risk_band_report["risk_band_order"] = risk_band_report["risk_band"].map(
        band_order
    )
    risk_band_report = (
        risk_band_report.sort_values("risk_band_order")
        .drop(columns=["risk_band_order"])
        .reset_index(drop=True)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    risk_band_report.to_csv(RISK_BAND_REPORT_PATH, index=False)
    return risk_band_report


def save_evaluation_summary(roc_auc, average_precision):
    """Save a compact JSON summary of model evaluation."""
    summary = {
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "threshold_report_path": str(THRESHOLD_REPORT_PATH),
        "risk_band_report_path": str(RISK_BAND_REPORT_PATH),
        "figures": {
            "roc_curve": str(FIGURES_DIR / "roc_curve.png"),
            "precision_recall_curve": str(
                FIGURES_DIR / "precision_recall_curve.png"
            ),
            "confusion_matrix_threshold_0.30": str(
                FIGURES_DIR / "confusion_matrix_threshold_0.30.png"
            ),
            "confusion_matrix_threshold_0.50": str(
                FIGURES_DIR / "confusion_matrix_threshold_0.50.png"
            ),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with EVALUATION_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def evaluate_saved_model():
    """Run saved-model evaluation, threshold tuning, and reporting."""
    model, feature_columns = load_artifacts()
    X_test, y_test = prepare_test_data(feature_columns)

    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)

    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)

    threshold_df = evaluate_thresholds(y_test, y_proba)
    for threshold in [0.30, 0.50]:
        y_pred = (y_proba >= threshold).astype(int)
        plot_confusion_matrix(y_test, y_pred, threshold)

    risk_band_report = create_risk_band_report(y_test, y_proba)
    save_evaluation_summary(roc_auc, average_precision)

    print("\nSaved model evaluation summary")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print("\nThreshold tuning:")
    print(
        threshold_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "flagged_rate_percentage",
            ]
        ].to_string(index=False)
    )
    print("\nRisk band report:")
    print(risk_band_report.to_string(index=False))
    print(f"\nSaved threshold report to: {THRESHOLD_REPORT_PATH}")
    print(f"Saved risk band report to: {RISK_BAND_REPORT_PATH}")
    print(f"Saved evaluation summary to: {EVALUATION_SUMMARY_PATH}")
    print(f"Saved charts to: {FIGURES_DIR}")


if __name__ == "__main__":
    evaluate_saved_model()
