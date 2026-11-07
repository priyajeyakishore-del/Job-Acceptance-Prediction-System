
# ============================================================
# STEP 2: CATEGORICAL DATA CLEANING
# Project: Job Acceptance Prediction System
# Purpose:
# 1. Load the dataset after duplicate removal
# 2. Identify categorical columns
# 3. Inspect unique categorical values
# 4. Standardize text formatting
# 5. Validate and save the cleaned dataset
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
    BASE_DIR / "data" / "processed" / "step1_deduplicated.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "data_cleaning"

OUTPUT_FILE = OUTPUT_DIR / "step2_categorical_cleaned.csv"
SUMMARY_FILE = REPORT_DIR / "categorical_cleaning_summary.csv"
UNIQUE_VALUES_FILE = REPORT_DIR / "categorical_unique_values.csv"


# ------------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORIES
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

def load_dataset():
    """Load the dataset produced by Step 1."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run duplicate_removal.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("STEP 2: CATEGORICAL DATA CLEANING")
    print("=" * 70)

    print(f"Input file : {INPUT_FILE.name}")
    print(f"Rows       : {df.shape[0]:,}")
    print(f"Columns    : {df.shape[1]}")

    return df


# ------------------------------------------------------------
# 4. IDENTIFY CATEGORICAL COLUMNS
# ------------------------------------------------------------

def identify_categorical_columns(df):
    """
    Identify columns stored as text or category data types.
    The target column 'status' is included for inspection,
    but its category labels are not changed semantically.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    print("\nCATEGORICAL COLUMNS")
    print("-" * 70)

    for column in categorical_columns:
        print(f"{column}: {df[column].nunique(dropna=True)} unique values")

    return categorical_columns


# ------------------------------------------------------------
# 5. INSPECT ORIGINAL CATEGORICAL VALUES
# ------------------------------------------------------------

def inspect_categorical_values(df, categorical_columns):
    """
    Save the unique values and their frequencies before cleaning.
    This helps identify inconsistent labels.
    """

    records = []

    print("\nORIGINAL CATEGORICAL VALUE INSPECTION")
    print("-" * 70)

    for column in categorical_columns:
        print(f"\nColumn: {column}")

        value_counts = df[column].value_counts(
            dropna=False
        )

        for value, count in value_counts.items():
            records.append({
                "Column": column,
                "Value": "<MISSING>" if pd.isna(value) else str(value),
                "Count": int(count)
            })

        print(value_counts.head(10).to_string())

    report_df = pd.DataFrame(records)
    report_df.to_csv(UNIQUE_VALUES_FILE, index=False)

    print(f"\nUnique-value report saved: {UNIQUE_VALUES_FILE}")

    return report_df


# ------------------------------------------------------------
# 6. STANDARDIZE TEXT FORMATTING
# ------------------------------------------------------------

def standardize_categorical_values(df, categorical_columns):
    """
    Remove leading/trailing spaces and standardize letter case
    for ordinary text categories.

    This step does not fill missing values or invent categories.
    """

    cleaned_df = df.copy()
    summary_records = []

    print("\nSTANDARDIZING CATEGORICAL VALUES")
    print("-" * 70)

    for column in categorical_columns:
        # Do not alter the target labels used for prediction.
        # Inspect status separately if its labels appear inconsistent.
        if column == "status":
            print(f"Skipped target column: {column}")
            continue

        original = cleaned_df[column].copy()

        # Preserve missing values while cleaning text.
        cleaned = original.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

        # Convert text to lowercase for consistent formatting.
        cleaned = cleaned.map(
            lambda value: value.lower() if isinstance(value, str) else value
        )

        changed_count = int(
            (original.fillna("<MISSING>") !=
             cleaned.fillna("<MISSING>")).sum()
        )

        cleaned_df[column] = cleaned

        summary_records.append({
            "Column": column,
            "Non_Missing_Before": int(original.notna().sum()),
            "Non_Missing_After": int(cleaned.notna().sum()),
            "Unique_Values_Before": int(original.nunique(dropna=True)),
            "Unique_Values_After": int(cleaned.nunique(dropna=True)),
            "Values_Changed": changed_count
        })

        print(
            f"{column}: {changed_count:,} values standardized"
        )

    # Normalize the target only if its values are already recognized.
    # Do not change the meaning of Placed or Not Placed.
    if "status" in cleaned_df.columns:
        target_values = cleaned_df["status"].dropna().unique().tolist()
        print(f"\nTarget column 'status' values: {target_values}")

    summary_df = pd.DataFrame(summary_records)

    return cleaned_df, summary_df


# ------------------------------------------------------------
# 7. VALIDATE CLEANED DATASET
# ------------------------------------------------------------

def validate_cleaned_data(df_before, df_after):
    """Check that cleaning did not change the dataset structure."""

    print("\nVALIDATION")
    print("-" * 70)

    structure_valid = (
        df_before.shape == df_after.shape
        and df_before.columns.tolist() == df_after.columns.tolist()
    )

    print(f"Original shape : {df_before.shape}")
    print(f"Cleaned shape  : {df_after.shape}")
    print(f"Structure preserved: {structure_valid}")

    if not structure_valid:
        raise ValueError(
            "Validation failed: row count, column count, "
            "or column names changed unexpectedly."
        )

    # Missing values must not be unintentionally filled or removed.
    missing_counts_valid = (
        df_before.isna().sum().equals(df_after.isna().sum())
    )

    print(f"Missing-value counts preserved: {missing_counts_valid}")

    if not missing_counts_valid:
        raise ValueError(
            "Validation failed: missing-value counts changed."
        )

    print("Validation passed.")


# ------------------------------------------------------------
# 8. SAVE CLEANED DATASET AND REPORTS
# ------------------------------------------------------------

def save_outputs(df, summary_df):
    """Save the cleaned dataset and categorical cleaning summary."""

    df.to_csv(OUTPUT_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\nOUTPUT FILES")
    print("-" * 70)
    print(f"Cleaned dataset : {OUTPUT_FILE}")
    print(f"Summary report  : {SUMMARY_FILE}")


# ------------------------------------------------------------
# 9. MAIN PIPELINE
# ------------------------------------------------------------

def main():
    # Load data from Step 1
    df = load_dataset()

    # Identify categorical columns
    categorical_columns = identify_categorical_columns(df)

    # Save a report of values before cleaning
    inspect_categorical_values(df, categorical_columns)

    # Standardize category formatting
    cleaned_df, summary_df = standardize_categorical_values(
        df,
        categorical_columns
    )

    # Validate the changes
    validate_cleaned_data(df, cleaned_df)

    # Save output for Step 3
    save_outputs(cleaned_df, summary_df)

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
