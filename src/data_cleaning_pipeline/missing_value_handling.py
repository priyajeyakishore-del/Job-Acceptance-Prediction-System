
# ============================================================
# FILE: missing_value_handling.py
# PROJECT: Job Acceptance Prediction System
# PURPOSE: Handle missing values using selected imputation
#          methods and save the cleaned dataset and reports.
# ============================================================

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

# Project root:
# Job Acceptance Prediction System/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step3_missing_indicators.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_cleaning"
    / "missing_value_handling"
)

OUTPUT_FILE = OUTPUT_DIR / "step4_missing_values_handled.csv"

NUMERICAL_REPORT = REPORT_DIR / "numerical_imputation_report.csv"
CATEGORICAL_REPORT = REPORT_DIR / "categorical_imputation_report.csv"
SUMMARY_REPORT = REPORT_DIR / "missing_value_handling_summary.csv"


# ============================================================
# 2. DEFINE IMPUTATION STRATEGIES
# ============================================================

# Numerical columns with missing values
MEAN_COLUMNS = [
    "ssc_percentage",
    "hsc_percentage",
]

MEDIAN_COLUMNS = [
    "notice_period_days",
    "employment_gap_months",
]

# Categorical columns with missing values
CATEGORICAL_COLUMNS = [
    "career_switch_willingness",
    "relevant_experience",
    "job_role_match",
    "layoff_history",
    "relocation_willingness",
]

TARGET_COLUMN = "status"


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():
    """Load the dataset produced by the missing-indicator step."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}\n"
            "Run missing_indicators.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("MISSING VALUE HANDLING")
    print("=" * 70)
    print(f"Input file: {INPUT_FILE}")
    print(f"Dataset shape: {df.shape}")
    print(f"Total missing values: {df.isna().sum().sum()}")

    return df


# ============================================================
# 4. VALIDATE EXPECTED COLUMNS
# ============================================================

def validate_columns(df):
    """Check that required columns exist before imputation."""

    required_columns = (
        MEAN_COLUMNS
        + MEDIAN_COLUMNS
        + CATEGORICAL_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following expected columns are missing from the dataset: "
            f"{missing_columns}"
        )

    # The target should not be imputed.
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains missing values. "
            "Investigate or remove those rows before model training."
        )


# ============================================================
# 5. IMPUTE NUMERICAL COLUMNS
# ============================================================

def impute_numerical_columns(df):
    """
    Fill numerical missing values using the selected strategies.

    Note:
    This script is intended for the data-cleaning pipeline.
    For final model evaluation, fit imputers on training data only.
    """

    numerical_report = []

    # --------------------------------------------------------
    # 5A. MEAN IMPUTATION
    # --------------------------------------------------------

    mean_columns_present = [
        column for column in MEAN_COLUMNS
        if df[column].isna().any()
    ]

    if mean_columns_present:
        imputer = SimpleImputer(strategy="mean")

        df[mean_columns_present] = imputer.fit_transform(
            df[mean_columns_present]
        )

        for column, statistic in zip(
            mean_columns_present,
            imputer.statistics_,
        ):
            numerical_report.append({
                "column": column,
                "method": "Mean",
                "imputation_value": statistic,
            })

    # --------------------------------------------------------
    # 5B. MEDIAN IMPUTATION
    # --------------------------------------------------------

    median_columns_present = [
        column for column in MEDIAN_COLUMNS
        if df[column].isna().any()
    ]

    if median_columns_present:
        imputer = SimpleImputer(strategy="median")

        df[median_columns_present] = imputer.fit_transform(
            df[median_columns_present]
        )

        for column, statistic in zip(
            median_columns_present,
            imputer.statistics_,
        ):
            numerical_report.append({
                "column": column,
                "method": "Median",
                "imputation_value": statistic,
            })

    numerical_report_df = pd.DataFrame(
        numerical_report,
        columns=["column", "method", "imputation_value"],
    )

    return df, numerical_report_df


# ============================================================
# 6. IMPUTE CATEGORICAL COLUMNS
# ============================================================

def impute_categorical_columns(df):
    """
    Replace missing categorical values with 'Unknown'.

    'Unknown' preserves the distinction between a missing response
    and an explicit answer such as 'no' or 'not willing'.
    """

    categorical_report = []

    for column in CATEGORICAL_COLUMNS:

        missing_before = int(df[column].isna().sum())

        if missing_before > 0:
            df[column] = df[column].fillna("Unknown")

        categorical_report.append({
            "column": column,
            "method": "Constant: Unknown",
            "missing_before": missing_before,
            "missing_after": int(df[column].isna().sum()),
        })

    categorical_report_df = pd.DataFrame(
        categorical_report,
        columns=[
            "column",
            "method",
            "missing_before",
            "missing_after",
        ],
    )

    return df, categorical_report_df


# ============================================================
# 7. CHECK REMAINING MISSING VALUES
# ============================================================

def create_missing_value_summary(df):
    """Create a report showing missing values after imputation."""

    missing_counts = df.isna().sum()

    summary_df = pd.DataFrame({
        "column": df.columns,
        "missing_before_imputation": [
            # The original before-imputation counts are recorded
            # separately in main() to preserve the correct baseline.
            0
        ] * len(df.columns),
        "missing_after_imputation": [
            int(missing_counts[column])
            for column in df.columns
        ],
    })

    return summary_df


# ============================================================
# 8. SAVE DATASET AND REPORTS
# ============================================================

def save_outputs(df, numerical_report, categorical_report,
                 original_missing_counts):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the processed dataset.
    df.to_csv(OUTPUT_FILE, index=False)

    # Save numerical imputation details.
    numerical_report.to_csv(NUMERICAL_REPORT, index=False)

    # Save categorical imputation details.
    categorical_report.to_csv(CATEGORICAL_REPORT, index=False)

    # Create a before-and-after missing-value summary.
    summary_df = pd.DataFrame({
        "column": df.columns,
        "missing_before_imputation": [
            int(original_missing_counts[column])
            for column in df.columns
        ],
        "missing_after_imputation": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
    })

    summary_df.to_csv(SUMMARY_REPORT, index=False)

    print("\nOUTPUT FILES")
    print("-" * 70)
    print(f"Processed dataset: {OUTPUT_FILE}")
    print(f"Numerical report:  {NUMERICAL_REPORT}")
    print(f"Categorical report: {CATEGORICAL_REPORT}")
    print(f"Summary report:    {SUMMARY_REPORT}")


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    """Run the missing-value handling pipeline."""

    df = load_dataset()

    # Validate the dataset before modifying it.
    validate_columns(df)

    # Record the original missing counts.
    original_missing_counts = df.isna().sum().copy()

    print("\nMissing values before imputation:")
    print(original_missing_counts[original_missing_counts > 0])

    # Impute numerical features.
    df, numerical_report = impute_numerical_columns(df)

    # Impute categorical features.
    df, categorical_report = impute_categorical_columns(df)

    # Confirm that no missing values remain.
    remaining_missing = int(df.isna().sum().sum())

    print("\nMissing values after imputation:")
    print(f"Total remaining missing values: {remaining_missing}")

    # Do not silently continue if missing values remain.
    if remaining_missing > 0:
        remaining_columns = df.columns[df.isna().any()].tolist()

        raise ValueError(
            "Missing values remain in these columns: "
            f"{remaining_columns}. Investigate before proceeding."
        )

    # Save dataset and reports.
    save_outputs(
        df,
        numerical_report,
        categorical_report,
        original_missing_counts,
    )

    print("\n" + "=" * 70)
    print("MISSING VALUE HANDLING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Final dataset shape: {df.shape}")
    print("Target column 'status' was not imputed.")
    print("Missingness indicator columns were preserved.")


# ============================================================
# 10. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
