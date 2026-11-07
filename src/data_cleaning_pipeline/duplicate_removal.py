
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STEP 1: REMOVE DUPLICATE RECORDS
# File: src/data_cleaning_pipeline/duplicate_removal.py
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\Priya Jeyakishore\OneDrive\Desktop\Job Acceptance Prediction System"
)

DATASET_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "data_cleaning"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. FIND AND LOAD THE RAW DATASET
# ------------------------------------------------------------

csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {DATASET_DIR}"
    )

if len(csv_files) > 1:
    raise ValueError(
        "Multiple CSV files found in the raw folder. "
        "Keep the intended dataset selected explicitly "
        "before running duplicate removal."
    )

DATASET_PATH = csv_files[0]

df = pd.read_csv(DATASET_PATH)

# ------------------------------------------------------------
# 3. INITIAL DATASET SUMMARY
# ------------------------------------------------------------

print("=" * 70)
print("STEP 1: DUPLICATE RECORD REMOVAL")
print("=" * 70)

rows_before = len(df)
columns_before = len(df.columns)

print(f"Input file       : {DATASET_PATH.name}")
print(f"Rows before      : {rows_before:,}")
print(f"Columns          : {columns_before}")

# ------------------------------------------------------------
# 4. IDENTIFY EXACT DUPLICATE ROWS
# ------------------------------------------------------------

duplicate_mask = df.duplicated(keep="first")

duplicate_count = int(duplicate_mask.sum())

duplicate_percentage = (
    round((duplicate_count / rows_before) * 100, 2)
    if rows_before > 0
    else 0.0
)

print("\nDUPLICATE ANALYSIS")
print("-" * 70)
print(f"Exact duplicates found : {duplicate_count:,}")
print(f"Duplicate percentage   : {duplicate_percentage}%")

# ------------------------------------------------------------
# 5. SAVE SAMPLE DUPLICATE RECORDS FOR REVIEW
# ------------------------------------------------------------

duplicate_samples = df.loc[
    df.duplicated(keep=False)
].head(100)

duplicate_sample_path = REPORT_DIR / "duplicate_samples.csv"

duplicate_samples.to_csv(
    duplicate_sample_path,
    index=False
)

print(f"Duplicate samples saved: {duplicate_sample_path}")

# ------------------------------------------------------------
# 6. REMOVE EXACT DUPLICATES
# ------------------------------------------------------------

df_cleaned = df.drop_duplicates(keep="first").copy()

rows_after = len(df_cleaned)
rows_removed = rows_before - rows_after

print("\nDUPLICATE REMOVAL RESULTS")
print("-" * 70)
print(f"Rows before removal : {rows_before:,}")
print(f"Rows removed        : {rows_removed:,}")
print(f"Rows after removal  : {rows_after:,}")

# ------------------------------------------------------------
# 7. VALIDATE THE CLEANED DATASET
# ------------------------------------------------------------

remaining_duplicates = int(
    df_cleaned.duplicated().sum()
)

print("\nVALIDATION")
print("-" * 70)
print(f"Remaining duplicates: {remaining_duplicates:,}")

if remaining_duplicates == 0:
    print("Validation passed: no exact duplicate rows remain.")
else:
    print("Warning: exact duplicate rows remain.")

# ------------------------------------------------------------
# 8. SAVE THE PROCESSED DATASET
# ------------------------------------------------------------

OUTPUT_PATH = PROCESSED_DIR / "step1_deduplicated.csv"

df_cleaned.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nCleaned dataset saved: {OUTPUT_PATH}")

# ------------------------------------------------------------
# 9. SAVE A SUMMARY REPORT
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "Input_File": DATASET_PATH.name,
    "Rows_Before": rows_before,
    "Columns": columns_before,
    "Exact_Duplicates_Found": duplicate_count,
    "Duplicate_Percentage": duplicate_percentage,
    "Rows_Removed": rows_removed,
    "Rows_After": rows_after,
    "Remaining_Duplicates": remaining_duplicates
}])

SUMMARY_PATH = REPORT_DIR / "duplicate_removal_summary.csv"

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

print(f"Summary report saved : {SUMMARY_PATH}")

print("\nStep 1 completed successfully!")
print("=" * 70)
