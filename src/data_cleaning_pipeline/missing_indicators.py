
# ============================================================
# STEP 3: CREATE MISSING-VALUE INDICATORS
# Project: Job Acceptance Prediction System
#
# Purpose:
# 1. Load the categorical-cleaned dataset.
# 2. Identify columns containing missing values.
# 3. Create binary missing-value flags.
# 4. Generate a missing-value summary report.
# 5. Validate and save the updated dataset.
#
# Flag values:
# 1 = Original value was missing
# 0 = Original value was available
# ============================================================

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\Priya Jeyakishore\OneDrive\Desktop\Job Acceptance Prediction System"
)

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "step2_categorical_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "data_cleaning"

OUTPUT_FILE = OUTPUT_DIR / "step3_missing_indicators.csv"
SUMMARY_FILE = REPORT_DIR / "missing_indicators_summary.csv"


# ------------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORIES
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

def load_dataset():
    """Load the dataset produced by Step 2."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please run categorical_cleaning.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("STEP 3: MISSING-VALUE INDICATORS")
    print("=" * 70)

    print(f"Input file : {INPUT_FILE.name}")
    print(f"Rows       : {df.shape[0]:,}")
    print(f"Columns    : {df.shape[1]}")

    return df


# ------------------------------------------------------------
# 4. ANALYZE MISSING VALUES
# ------------------------------------------------------------

def analyze_missing_values(df):
    """Calculate missing counts and percentages before imputation."""

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df) * 100
    )

    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing_count.values,
        "Missing_Percentage": missing_percentage.values,
        "Non_Missing_Count": df.notnull().sum().values
    })

    # Keep columns that actually contain missing values.
    summary_df = summary_df[
        summary_df["Missing_Count"] > 0
    ].sort_values(
        by="Missing_Count",
        ascending=False
    ).reset_index(drop=True)

    print("\nMISSING-VALUE SUMMARY")
    print("-" * 70)

    if summary_df.empty:
        print("No missing values found.")
    else:
        print(summary_df.to_string(index=False))

    return summary_df


# ------------------------------------------------------------
# 5. CREATE MISSING-VALUE FLAGS
# ------------------------------------------------------------

def create_missing_flags(df, summary_df):
    """
    Create one binary flag for each column with missing values.

    Example:
        ssc_percentage_missing_flag

    1 = Original SSC percentage was missing
    0 = Original SSC percentage was present
    """

    result_df = df.copy()
    flag_columns = []

    print("\nCREATING MISSING-VALUE FLAGS")
    print("-" * 70)

    for column in summary_df["Column"]:

        # Do not create a flag for the prediction target.
        if column == "status":
            print("Skipped target column: status")
            continue

        flag_name = f"{column}_missing_flag"

        # Avoid accidentally overwriting an existing column.
        if flag_name in result_df.columns:
            raise ValueError(
                f"Flag column already exists: {flag_name}"
            )

        # Create the flag BEFORE imputation.
        result_df[flag_name] = (
            result_df[column].isnull().astype("int8")
        )

        flag_columns.append(flag_name)

        missing_records = int(result_df[flag_name].sum())

        print(
            f"{flag_name}: "
            f"{missing_records:,} missing records flagged"
        )

    print(f"\nTotal flags created: {len(flag_columns)}")

    return result_df, flag_columns


# ------------------------------------------------------------
# 6. VALIDATE MISSING-VALUE FLAGS
# ------------------------------------------------------------

def validate_missing_flags(
    original_df,
    result_df,
    summary_df,
    flag_columns
):
    """Verify that every flag matches the original missing values."""

    print("\nVALIDATION")
    print("-" * 70)

    # Check that original rows and columns remain unchanged.
    if len(original_df) != len(result_df):
        raise ValueError("Validation failed: row count changed.")

    if result_df.columns[:len(original_df.columns)].tolist() != (
        original_df.columns.tolist()
    ):
        raise ValueError(
            "Validation failed: original columns changed."
        )

    # Confirm original values have not been modified.
    pd.testing.assert_frame_equal(
        original_df,
        result_df[original_df.columns],
        check_dtype=False
    )

    # Verify each flag against the original missing values.
    for column in summary_df["Column"]:

        if column == "status":
            continue

        flag_name = f"{column}_missing_flag"

        expected = original_df[column].isnull().astype("int8")

        if not result_df[flag_name].equals(expected):
            raise ValueError(
                f"Validation failed for flag: {flag_name}"
            )

    print(f"Rows preserved: {len(original_df):,}")
    print(f"Original columns preserved: {len(original_df.columns)}")
    print(f"Flags validated: {len(flag_columns)}")
    print("Original missing values preserved: Yes")
    print("Validation passed.")


# ------------------------------------------------------------
# 7. SAVE DATASET AND REPORT
# ------------------------------------------------------------

def save_outputs(result_df, summary_df):
    """Save the dataset with flags and the missing-value report."""

    result_df.to_csv(OUTPUT_FILE, index=False)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\nOUTPUT FILES")
    print("-" * 70)

    print(f"Updated dataset : {OUTPUT_FILE}")
    print(f"Summary report  : {SUMMARY_FILE}")


# ------------------------------------------------------------
# 8. MAIN PIPELINE
# ------------------------------------------------------------

def main():

    # Load output from Step 2.
    df = load_dataset()

    # Analyze missing values before imputation.
    summary_df = analyze_missing_values(df)

    # Create flags for columns with missing values.
    result_df, flag_columns = create_missing_flags(
        df,
        summary_df
    )

    # Validate the flags.
    validate_missing_flags(
        df,
        result_df,
        summary_df,
        flag_columns
    )

    # Save results for the next cleaning step.
    save_outputs(result_df, summary_df)

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


# ------------------------------------------------------------
# 9. EXECUTION ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
