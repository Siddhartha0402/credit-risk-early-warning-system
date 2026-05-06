"""Prediction utilities for trained credit risk models."""

import pathlib
import sys
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import create_application_features


MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
DAYS_EMPLOYED_ANOMALY_VALUE = 365243


def load_prediction_artifacts():
    """Load the trained model and saved feature-column contract."""
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


def assign_risk_band(probability):
    """Assign a business risk band from default probability."""
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    return "High Risk"


def recommend_action(risk_band):
    """Map a risk band to the recommended operational action."""
    recommendations = {
        "Low Risk": "Standard approval / normal monitoring",
        "Medium Risk": "Manual review recommended",
        "High Risk": "Escalate to risk analyst / early intervention",
    }
    return recommendations[risk_band]


def prepare_single_customer(input_data: Dict[str, Any], feature_columns):
    """Transform one customer payload into the trained model feature schema."""
    X = pd.DataFrame([input_data])

    if "DAYS_EMPLOYED" in X:
        X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_ANOMALY_VALUE,
            np.nan,
        )

    X = create_application_features(X)

    categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_columns:
        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False,
            dtype=np.uint8,
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(0)
    X = X.fillna(0)

    missing_columns = [column for column in feature_columns if column not in X]
    if missing_columns:
        missing_feature_df = pd.DataFrame(
            0,
            index=X.index,
            columns=missing_columns,
        )
        X = pd.concat([X, missing_feature_df], axis=1)

    extra_columns = [column for column in X.columns if column not in feature_columns]
    if extra_columns:
        X = X.drop(columns=extra_columns)

    X = X[feature_columns].copy()
    return X


def predict_customer_risk(input_data: Dict[str, Any]):
    """Predict default risk for one customer payload."""
    model, feature_columns = load_prediction_artifacts()
    X = prepare_single_customer(input_data, feature_columns)

    default_probability = float(model.predict_proba(X)[:, 1][0])
    risk_band = assign_risk_band(default_probability)
    recommended_action = recommend_action(risk_band)

    return {
        "default_probability": default_probability,
        "risk_band": risk_band,
        "recommended_action": recommended_action,
    }


if __name__ == "__main__":
    sample_customer = {
        "AMT_INCOME_TOTAL": 157500.0,
        "AMT_CREDIT": 497520.0,
        "AMT_ANNUITY": 27000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "DAYS_BIRTH": -14500,
        "DAYS_EMPLOYED": -2400,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3.0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
    }
    print(predict_customer_risk(sample_customer))
