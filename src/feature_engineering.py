"""Feature engineering utilities for application_train.csv."""

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
APPLICATION_TRAIN_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "application_train_features.csv"
)
FEATURE_ENGINEERING_SUMMARY_PATH = (
    REPORTS_DIR / "feature_engineering_summary.json"
)
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def safe_divide(numerator, denominator):
    """Divide values and replace divide-by-zero or infinite results with NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)

    if isinstance(result, (pd.Series, pd.DataFrame)):
        return result.replace([np.inf, -np.inf], np.nan)

    if np.isscalar(result):
        return np.nan if not np.isfinite(result) else result

    result = np.asarray(result, dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result


def add_financial_ratio_features(df):
    """Add finance-domain credit, income, annuity, and family income ratios."""
    featured_df = df.copy()
    featured_df["credit_to_income_ratio"] = safe_divide(
        featured_df["AMT_CREDIT"], featured_df["AMT_INCOME_TOTAL"]
    )
    featured_df["annuity_to_income_ratio"] = safe_divide(
        featured_df["AMT_ANNUITY"], featured_df["AMT_INCOME_TOTAL"]
    )
    featured_df["goods_to_credit_ratio"] = safe_divide(
        featured_df["AMT_GOODS_PRICE"], featured_df["AMT_CREDIT"]
    )
    featured_df["credit_to_goods_ratio"] = safe_divide(
        featured_df["AMT_CREDIT"], featured_df["AMT_GOODS_PRICE"]
    )
    featured_df["income_per_child"] = safe_divide(
        featured_df["AMT_INCOME_TOTAL"], featured_df["CNT_CHILDREN"] + 1
    )
    featured_df["income_per_family_member"] = safe_divide(
        featured_df["AMT_INCOME_TOTAL"], featured_df["CNT_FAM_MEMBERS"]
    )
    return featured_df


def add_age_employment_features(df):
    """Add age, employment tenure, and employment-to-age ratio features."""
    featured_df = df.copy()
    featured_df["age_years"] = featured_df["DAYS_BIRTH"].abs() / 365
    featured_df["employment_years"] = featured_df["DAYS_EMPLOYED"].abs() / 365
    featured_df["employment_to_age_ratio"] = safe_divide(
        featured_df["employment_years"], featured_df["age_years"]
    )
    return featured_df


def add_external_score_features(df):
    """Add aggregate external source score features when all inputs exist."""
    featured_df = df.copy()
    if all(column in featured_df.columns for column in EXT_SOURCE_COLUMNS):
        ext_source_scores = featured_df[EXT_SOURCE_COLUMNS]
        featured_df["ext_source_mean"] = ext_source_scores.mean(axis=1)
        featured_df["ext_source_std"] = ext_source_scores.std(axis=1)
        featured_df["ext_source_min"] = ext_source_scores.min(axis=1)
        featured_df["ext_source_max"] = ext_source_scores.max(axis=1)
    return featured_df


def add_family_features(df):
    """Add family composition features."""
    featured_df = df.copy()
    featured_df["children_ratio"] = safe_divide(
        featured_df["CNT_CHILDREN"], featured_df["CNT_FAM_MEMBERS"]
    )
    featured_df["has_children"] = (featured_df["CNT_CHILDREN"] > 0).astype(int)
    return featured_df


def create_application_features(df):
    """Apply the full application feature engineering pipeline."""
    featured_df = add_financial_ratio_features(df)
    featured_df = add_age_employment_features(featured_df)
    featured_df = add_external_score_features(featured_df)
    featured_df = add_family_features(featured_df)
    return featured_df


def save_feature_engineering_summary(
    original_shape,
    final_shape,
    new_feature_names,
    days_employed_replaced_count,
):
    """Save a JSON summary of the application feature engineering run."""
    FEATURE_ENGINEERING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_path": str(APPLICATION_TRAIN_PATH),
        "output_path": str(APPLICATION_TRAIN_FEATURES_PATH),
        "original_shape": list(original_shape),
        "final_shape": list(final_shape),
        "new_feature_count": int(len(new_feature_names)),
        "new_feature_names": new_feature_names,
        "days_employed_anomaly_value": DAYS_EMPLOYED_ANOMALY_VALUE,
        "days_employed_replaced_with_nan": int(days_employed_replaced_count),
    }

    with FEATURE_ENGINEERING_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def engineer_application_train():
    """Load raw application_train.csv, engineer features, and save artifacts."""
    df = pd.read_csv(APPLICATION_TRAIN_PATH)
    original_shape = df.shape
    original_columns = set(df.columns)

    days_employed_mask = df["DAYS_EMPLOYED"].eq(DAYS_EMPLOYED_ANOMALY_VALUE)
    days_employed_replaced_count = int(days_employed_mask.sum())
    df.loc[days_employed_mask, "DAYS_EMPLOYED"] = np.nan

    featured_df = create_application_features(df)
    final_shape = featured_df.shape
    new_feature_names = [
        column for column in featured_df.columns if column not in original_columns
    ]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(APPLICATION_TRAIN_FEATURES_PATH, index=False)
    save_feature_engineering_summary(
        original_shape,
        final_shape,
        new_feature_names,
        days_employed_replaced_count,
    )

    print(f"Original shape: {original_shape}")
    print(f"Final shape: {final_shape}")
    print(f"New features created: {len(new_feature_names)}")


if __name__ == "__main__":
    engineer_application_train()
