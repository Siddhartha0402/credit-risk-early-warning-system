"""Train credit risk models from engineered application features."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


FEATURES_PATH = PROCESSED_DATA_DIR / "application_train_features.csv"
MODEL_PATH = MODELS_DIR / "credit_risk_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
LOGISTIC_REGRESSION_MAX_TRAIN_ROWS = 100_000


def load_features():
    """Load feature-engineered application training data."""
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature data shape: {df.shape}")
    return df


def prepare_train_test_data(df):
    """Clean, encode, and split engineered features for model training."""
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

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Feature columns: {len(feature_columns)}")

    return X_train, X_test, y_train, y_test, feature_columns


def calculate_scale_pos_weight(y_train):
    """Calculate XGBoost class imbalance weight."""
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0:
        raise ValueError("Cannot calculate scale_pos_weight without positive labels.")

    scale_pos_weight = negative_count / positive_count
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    return scale_pos_weight


def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate a fitted classifier and return serializable metrics."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report,
    }

    print(f"\n{model_name} results")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    return metrics


def create_logistic_regression_model():
    """Create the scaled Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def create_xgboost_model(scale_pos_weight):
    """Create the XGBoost credit risk classifier."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def sample_logistic_regression_training_data(X_train, y_train):
    """Use a stratified baseline sample if the full training split is large."""
    if len(X_train) <= LOGISTIC_REGRESSION_MAX_TRAIN_ROWS:
        return X_train, y_train

    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=LOGISTIC_REGRESSION_MAX_TRAIN_ROWS,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    print(
        "Logistic Regression baseline sample: "
        f"{len(X_sample)} of {len(X_train)} training rows"
    )
    return X_sample, y_sample


def save_training_artifacts(best_model, feature_columns, metrics):
    """Save model artifacts and evaluation reports."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    with MODEL_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved best model to: {MODEL_PATH}")
    print(f"Saved feature columns to: {FEATURE_COLUMNS_PATH}")
    print(f"Saved model metrics to: {MODEL_METRICS_PATH}")


def train_models():
    """Train, evaluate, compare, and save credit risk models."""
    df = load_features()
    X_train, X_test, y_train, y_test, feature_columns = prepare_train_test_data(df)

    logistic_model = create_logistic_regression_model()
    X_lr_train, y_lr_train = sample_logistic_regression_training_data(
        X_train,
        y_train,
    )
    print("\nTraining Logistic Regression baseline...")
    logistic_model.fit(X_lr_train, y_lr_train)
    logistic_metrics = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test,
    )

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    xgboost_model = create_xgboost_model(scale_pos_weight)
    print("\nTraining XGBoost...")
    xgboost_model.fit(X_train, y_train)
    xgboost_metrics = evaluate_model("XGBoost", xgboost_model, X_test, y_test)

    model_metrics = {
        "logistic_regression": logistic_metrics,
        "xgboost": xgboost_metrics,
    }

    if xgboost_metrics["roc_auc"] >= logistic_metrics["roc_auc"]:
        best_model_name = "xgboost"
        best_model = xgboost_model
    else:
        best_model_name = "logistic_regression"
        best_model = logistic_model

    model_metrics["best_model"] = best_model_name
    print(f"\nBest model by ROC-AUC: {best_model_name}")

    save_training_artifacts(best_model, feature_columns, model_metrics)


if __name__ == "__main__":
    train_models()
