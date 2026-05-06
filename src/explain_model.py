"""Generate SHAP explainability artifacts for the saved credit risk model."""

import json
import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
(PROJECT_ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "application_train_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_GLOBAL_IMPORTANCE_PATH = REPORTS_DIR / "shap_global_importance.csv"
HIGH_RISK_CUSTOMER_PATH = REPORTS_DIR / "example_high_risk_customer.json"
SINGLE_CUSTOMER_DRIVERS_PATH = REPORTS_DIR / "shap_single_customer_drivers.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 2_000


def load_artifacts():
    """Load the trained model and expected feature columns."""
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


def prepare_explainability_data(sample_size=DEFAULT_SAMPLE_SIZE):
    """Prepare a representative held-out sample for SHAP explainability."""
    _, feature_columns = load_artifacts()
    df = pd.read_csv(FEATURES_PATH)

    y = df["TARGET"].copy()
    drop_columns = [column for column in ["TARGET", "SK_ID_CURR"] if column in df]
    X = df.drop(columns=drop_columns)

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

    extra_columns = [column for column in X.columns if column not in feature_columns]
    if extra_columns:
        X = X.drop(columns=extra_columns)

    X = X[feature_columns]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if len(X_test) > sample_size:
        X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
        y_sample = y_test.loc[X_sample.index]
    else:
        X_sample = X_test.copy()
        y_sample = y_test.copy()

    return X_sample, y_sample


def _get_tree_model(model):
    """Return a tree estimator when the saved model wraps one in a pipeline."""
    if hasattr(model, "named_steps"):
        for step_name in ("model", "classifier", "estimator"):
            if step_name in model.named_steps:
                candidate = model.named_steps[step_name]
                if hasattr(candidate, "get_booster") or hasattr(
                    candidate,
                    "estimators_",
                ):
                    return candidate

    return model


def generate_shap_explanations(model, X_sample):
    """Generate SHAP values using TreeExplainer when the model supports it."""
    tree_model = _get_tree_model(model)

    try:
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)

    return explainer, shap_values


def _positive_class_shap_values(shap_values):
    """Normalize SHAP outputs to a 2D positive-class contribution matrix."""
    if isinstance(shap_values, shap.Explanation):
        values = shap_values.values
    else:
        values = shap_values

    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])

    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

    return values


def _positive_class_expected_value(explainer, shap_values=None, row_position=None):
    """Return the expected value aligned to the positive-class SHAP values."""
    if isinstance(shap_values, shap.Explanation) and row_position is not None:
        base_values = np.asarray(shap_values.base_values)
        if base_values.ndim == 2:
            return float(base_values[row_position, 1])
        if base_values.ndim == 1:
            return float(base_values[row_position])

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_values = np.asarray(expected_value).reshape(-1)
        if len(expected_values) > 1:
            return float(expected_values[1])
        return float(expected_values[0])

    return float(expected_value)


def save_global_importance(shap_values, X_sample):
    """Save the top 30 mean absolute SHAP feature importances."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shap_matrix = _positive_class_shap_values(shap_values)
    importance = np.abs(shap_matrix).mean(axis=0)

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap_value": importance,
            }
        )
        .sort_values("mean_abs_shap_value", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )

    importance_df.to_csv(SHAP_GLOBAL_IMPORTANCE_PATH, index=False)
    return importance_df


def plot_shap_bar(shap_values, X_sample):
    """Create and save the global SHAP bar plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap_matrix = _positive_class_shap_values(shap_values)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_matrix,
        X_sample,
        plot_type="bar",
        max_display=30,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_summary(shap_values, X_sample):
    """Create and save the SHAP beeswarm summary plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap_matrix = _positive_class_shap_values(shap_values)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_matrix,
        X_sample,
        max_display=30,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_summary_beeswarm.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def explain_single_customer(model, explainer, shap_values, X_sample):
    """Explain the highest-risk sampled customer and save local SHAP artifacts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predicted_probabilities = model.predict_proba(X_sample)[:, 1]
    high_risk_position = int(np.argmax(predicted_probabilities))
    customer_index = X_sample.index[high_risk_position]
    customer_features = X_sample.iloc[high_risk_position]
    shap_matrix = _positive_class_shap_values(shap_values)
    local_values = shap_matrix[high_risk_position]

    customer_payload = {
        "sample_index": int(customer_index),
        "predicted_default_probability": float(
            predicted_probabilities[high_risk_position]
        ),
        "features": {
            feature: _to_json_value(value)
            for feature, value in customer_features.to_dict().items()
        },
    }
    with HIGH_RISK_CUSTOMER_PATH.open("w", encoding="utf-8") as file:
        json.dump(customer_payload, file, indent=4)

    local_driver_df = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "feature_value": customer_features.to_numpy(),
                "shap_value": local_values,
                "abs_shap_value": np.abs(local_values),
            }
        )
        .sort_values("abs_shap_value", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    local_driver_df.to_csv(SINGLE_CUSTOMER_DRIVERS_PATH, index=False)

    explanation = shap.Explanation(
        values=local_values,
        base_values=_positive_class_expected_value(
            explainer,
            shap_values=shap_values,
            row_position=high_risk_position,
        ),
        data=customer_features.to_numpy(),
        feature_names=X_sample.columns.tolist(),
    )
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_single_customer_waterfall.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    return customer_payload, local_driver_df


def _to_json_value(value):
    """Convert numpy scalar values to JSON-serializable Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_explainability():
    """Run end-to-end SHAP explainability and save all report artifacts."""
    model, _ = load_artifacts()
    X_sample, y_sample = prepare_explainability_data(sample_size=DEFAULT_SAMPLE_SIZE)
    explainer, shap_values = generate_shap_explanations(model, X_sample)

    global_importance_df = save_global_importance(shap_values, X_sample)
    plot_shap_bar(shap_values, X_sample)
    plot_shap_summary(shap_values, X_sample)
    customer_payload, _ = explain_single_customer(
        model,
        explainer,
        shap_values,
        X_sample,
    )

    print("\nSaved SHAP explainability artifacts")
    print(f"Explainability sample rows: {len(X_sample)}")
    print(f"Sample default rate: {float(y_sample.mean()):.4f}")
    print(f"Top global driver: {global_importance_df.iloc[0]['feature']}")
    print(
        "Selected customer probability: "
        f"{customer_payload['predicted_default_probability']:.4f}"
    )
    print(f"Saved global importance to: {SHAP_GLOBAL_IMPORTANCE_PATH}")
    print(f"Saved high-risk customer JSON to: {HIGH_RISK_CUSTOMER_PATH}")
    print(f"Saved local drivers to: {SINGLE_CUSTOMER_DRIVERS_PATH}")
    print(f"Saved SHAP charts to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_explainability()
