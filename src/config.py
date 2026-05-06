"""Project-wide configuration and dataset path constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

APPLICATION_TRAIN_PATH = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST_PATH = RAW_DATA_DIR / "application_test.csv"
BUREAU_PATH = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE_PATH = RAW_DATA_DIR / "bureau_balance.csv"
PREVIOUS_APPLICATION_PATH = RAW_DATA_DIR / "previous_application.csv"
POS_CASH_BALANCE_PATH = RAW_DATA_DIR / "POS_CASH_balance.csv"
CREDIT_CARD_BALANCE_PATH = RAW_DATA_DIR / "credit_card_balance.csv"
INSTALLMENTS_PAYMENTS_PATH = RAW_DATA_DIR / "installments_payments.csv"
COLUMNS_DESCRIPTION_PATH = RAW_DATA_DIR / "HomeCredit_columns_description.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"


REQUIRED_RAW_FILES = {
    "application_train.csv": APPLICATION_TRAIN_PATH,
    "application_test.csv": APPLICATION_TEST_PATH,
    "bureau.csv": BUREAU_PATH,
    "bureau_balance.csv": BUREAU_BALANCE_PATH,
    "previous_application.csv": PREVIOUS_APPLICATION_PATH,
    "POS_CASH_balance.csv": POS_CASH_BALANCE_PATH,
    "credit_card_balance.csv": CREDIT_CARD_BALANCE_PATH,
    "installments_payments.csv": INSTALLMENTS_PAYMENTS_PATH,
    "HomeCredit_columns_description.csv": COLUMNS_DESCRIPTION_PATH,
    "sample_submission.csv": SAMPLE_SUBMISSION_PATH,
}
