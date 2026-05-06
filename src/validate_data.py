"""Validate availability and readability of required raw Home Credit CSV files.

Run from the project root with:
    python src/validate_data.py
"""

from __future__ import annotations

import sys
import csv
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import APPLICATION_TRAIN_PATH, REQUIRED_RAW_FILES  # noqa: E402


def format_file_size(size_bytes: int) -> str:
    """Return a human-readable file size."""
    units = ("B", "KB", "MB", "GB")
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def validate_required_files() -> bool:
    """Check required raw files exist and can be sampled."""
    print("Raw CSV file validation")
    print("-" * 80)

    all_files_available = True

    for file_name, file_path in REQUIRED_RAW_FILES.items():
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        status = "FOUND" if exists else "MISSING"

        print(f"{file_name:<35} {status:<8} {format_file_size(size)}")

        if not exists:
            all_files_available = False
            continue

        try:
            sample_row_count, sample_column_count = read_csv_sample_shape(file_path, nrows=5)
            print(f"  Read check: OK, sample shape=({sample_row_count}, {sample_column_count})")
        except Exception as exc:
            all_files_available = False
            print(f"  Read check: FAILED ({exc})")

    print("-" * 80)
    return all_files_available


def read_csv_sample_shape(file_path: Path, nrows: int) -> tuple[int, int]:
    """Read up to nrows from a CSV and return a pandas-like sample shape."""
    with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, [])
        row_count = 0

        for row_count, _ in enumerate(reader, start=1):
            if row_count >= nrows:
                break

    return row_count, len(header)


def print_application_train_summary() -> None:
    """Print shape and target distribution for application_train.csv."""
    row_count, column_count, target_distribution = summarize_application_train(APPLICATION_TRAIN_PATH)

    print("application_train.csv shape:")
    print((row_count, column_count))
    print()

    print("TARGET distribution:")
    for target_value, count in sorted(target_distribution.items()):
        print(f"{target_value}: {count}")
    print()

    print("TARGET distribution (%):")
    for target_value, count in sorted(target_distribution.items()):
        percentage = (count / row_count) * 100 if row_count else 0
        print(f"{target_value}: {percentage:.2f}%")


def summarize_application_train(file_path: Path) -> tuple[int, int, Counter[str]]:
    """Stream application_train.csv and return shape plus TARGET counts."""
    target_distribution: Counter[str] = Counter()

    with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, [])

        if "TARGET" not in header:
            raise ValueError("TARGET column not found in application_train.csv")

        target_index = header.index("TARGET")
        row_count = 0

        for row_count, row in enumerate(reader, start=1):
            target_value = row[target_index] if target_index < len(row) else ""
            target_distribution[target_value] += 1

    return row_count, len(header), target_distribution


def main() -> None:
    """Run raw data validation checks."""
    all_files_available = validate_required_files()

    if not all_files_available:
        raise FileNotFoundError(
            "One or more required raw CSV files are missing or unreadable. "
            "Place the original Home Credit files in data/raw and rerun validation."
        )

    print()
    print_application_train_summary()
    print()
    print("SUCCESS: All required raw CSV files are available and readable.")


if __name__ == "__main__":
    main()
