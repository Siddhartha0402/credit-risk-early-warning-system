"""Data cleaning and preprocessing utilities for application_train.csv."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import APPLICATION_TRAIN_PATH, PROCESSED_DATA_DIR, REPORTS_DIR


DAYS_EMPLOYED_ANOMALY_VALUE = 365243
PROCESSED_APPLICATION_TRAIN_PATH = (
    PROCESSED_DATA_DIR / "application_train_processed.csv"
)
FEATURE_COLUMNS_PATH = PROCESSED_DATA_DIR / "application_feature_columns.json"
CLEANING_REPORT_PATH = REPORTS_DIR / "application_cleaning_report.json"


def load_application_data():
    """Load the raw application training dataset."""
    df = pd.read_csv(APPLICATION_TRAIN_PATH)
    print(f"Loaded application_train.csv shape: {df.shape}")
    return df


def fix_abnormal_values(df):
    """Replace known abnormal sentinel values and preserve anomaly flags."""
    cleaned_df = df.copy()
    anomaly_mask = cleaned_df["DAYS_EMPLOYED"].eq(DAYS_EMPLOYED_ANOMALY_VALUE)
    cleaned_df["DAYS_EMPLOYED_ANOMALY"] = anomaly_mask.astype(int)
    cleaned_df.loc[anomaly_mask, "DAYS_EMPLOYED"] = np.nan
    return cleaned_df


def create_basic_cleaning_report(df, output_path):
    """Create a JSON report summarizing core cleaning inputs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_counts = df["TARGET"].value_counts(dropna=False).sort_index()
    target_percentages = (
        df["TARGET"].value_counts(normalize=True, dropna=False).sort_index() * 100
    )

    missing_counts = df.isna().sum().sort_values(ascending=False).head(30)
    missing_percentages = (missing_counts / len(df)) * 100

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    report = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {
            str(target): {
                "count": int(target_counts.loc[target]),
                "percentage": float(round(target_percentages.loc[target], 4)),
            }
            for target in target_counts.index
        },
        "missing_values_top_30": {
            column: {
                "count": int(missing_counts.loc[column]),
                "percentage": float(round(missing_percentages.loc[column], 4)),
            }
            for column in missing_counts.index
        },
        "numeric_column_count": int(len(numeric_cols)),
        "categorical_column_count": int(len(categorical_cols)),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def split_features_target(df):
    """Split TARGET from features and remove the row identifier from X."""
    y = df["TARGET"].copy()
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    return X, y


def identify_column_types(X):
    """Identify numeric and categorical feature columns."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def clean_missing_values(X, numeric_cols, categorical_cols):
    """Fill missing numeric and categorical values."""
    cleaned_X = X.copy()

    for column in numeric_cols:
        cleaned_X[column] = cleaned_X[column].fillna(cleaned_X[column].median())

    for column in categorical_cols:
        mode_values = cleaned_X[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        cleaned_X[column] = cleaned_X[column].fillna(fill_value)

    return cleaned_X


def encode_categoricals(X, categorical_cols):
    """One-hot encode categorical feature columns."""
    return pd.get_dummies(X, columns=categorical_cols, drop_first=False)


def preprocess_application_train():
    """Run the full application_train.csv preprocessing pipeline."""
    df = load_application_data()
    df = fix_abnormal_values(df)

    create_basic_cleaning_report(df, CLEANING_REPORT_PATH)

    X, y = split_features_target(df)
    numeric_cols, categorical_cols = identify_column_types(X)
    X = clean_missing_values(X, numeric_cols, categorical_cols)
    X = encode_categoricals(X, categorical_cols)

    processed_df = X.copy()
    processed_df["TARGET"] = y.values

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_APPLICATION_TRAIN_PATH, index=False)

    with FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as file:
        json.dump(list(X.columns), file, indent=4)

    print(f"Final processed shape: {processed_df.shape}")
    print("Application training preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_application_train()
