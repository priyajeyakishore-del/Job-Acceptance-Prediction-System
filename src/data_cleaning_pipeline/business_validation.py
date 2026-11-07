
# ============================================================
# FILE: business_validation.py
# PROJECT: Job Acceptance Prediction System
#
# PURPOSE:
# 1. Validate numerical values using business rules.
# 2. Validate categorical values.
# 3. Identify duplicates before and after imputation.
# 4. Remove duplicates that already existed before imputation.
# 5. Retain and report possible duplicates created after imputation.
# 6. Generate duplicate investigation reports.
# 7. Generate the final business validation report.
#
# INPUT:
# step3_missing_indicators.csv
# step4_missing_values_handled.csv
#
# OUTPUT:
# step5_business_validated.csv
# business_validation_report.csv
# duplicate_investigation_report.csv
# duplicate_comparison_report.csv
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BEFORE_IMPUTATION_FILE = (
    PROCESSED_DIR / "step3_missing_indicators.csv"
)

INPUT_FILE = (
    PROCESSED_DIR / "step4_missing_values_handled.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "step5_business_validated.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_cleaning"
    / "business_validation"
)

VALIDATION_REPORT = REPORT_DIR / "business_validation_report.csv"

DUPLICATE_REPORT = REPORT_DIR / "duplicate_investigation_report.csv"

DUPLICATE_COMPARISON_REPORT = REPORT_DIR / "duplicate_comparison_report.csv"


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def load_datasets():

    if not BEFORE_IMPUTATION_FILE.exists():
        raise FileNotFoundError(
            f"Before-imputation file not found: {BEFORE_IMPUTATION_FILE}"
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"After-imputation file not found: {INPUT_FILE}"
        )

    before_df = pd.read_csv(BEFORE_IMPUTATION_FILE)
    after_df = pd.read_csv(INPUT_FILE)

    print("=" * 75)
    print("BUSINESS VALIDATION")
    print("=" * 75)

    print(f"Before imputation shape: {before_df.shape}")
    print(f"After imputation shape:  {after_df.shape}")

    print(
        f"Missing values after imputation: "
        f"{after_df.isna().sum().sum()}"
    )

    # Row-wise comparison requires the same original row order.
    if len(before_df) != len(after_df):
        raise ValueError(
            "Row counts differ between before and after imputation. "
            "Cannot safely compare records by row position."
        )

    if list(before_df.columns) != list(after_df.columns):
        raise ValueError(
            "Column names or order differ between the two datasets."
        )

    return before_df, after_df


# ============================================================
# 3. HANDLE DUPLICATES
# ============================================================

def handle_duplicates(before_df, after_df):

    print("\nDUPLICATE HANDLING")
    print("=" * 75)

    # Work on a copy so the original loaded dataset is not modified.
    cleaned_df = after_df.copy()

    # --------------------------------------------------------
    # 3A. IDENTIFY DUPLICATES BEFORE IMPUTATION
    # --------------------------------------------------------

    before_duplicate_mask = before_df.duplicated(keep="first")

    before_duplicate_count = int(
        before_duplicate_mask.sum()
    )

    print(
        f"Duplicates already present before imputation: "
        f"{before_duplicate_count}"
    )

    # --------------------------------------------------------
    # 3B. REMOVE ONLY PRE-EXISTING DUPLICATE ROWS
    # --------------------------------------------------------

    # Use the original row positions to remove the corresponding
    # repeated records from the after-imputation dataset.
    rows_to_remove = before_duplicate_mask.to_numpy()

    cleaned_df = cleaned_df.iloc[
        ~rows_to_remove
    ].copy()

    removed_count = int(rows_to_remove.sum())

    print(
        f"Pre-existing duplicate rows removed: {removed_count}"
    )

    # --------------------------------------------------------
    # 3C. CHECK DUPLICATES AFTER REMOVAL
    # --------------------------------------------------------

    remaining_duplicate_mask = cleaned_df.duplicated(
        keep=False
    )

    remaining_duplicate_count = int(
        cleaned_df.duplicated(keep="first").sum()
    )

    remaining_duplicate_rows = int(
        remaining_duplicate_mask.sum()
    )

    print(
        f"Remaining duplicate occurrences: "
        f"{remaining_duplicate_count}"
    )

    print(
        f"Rows involved in remaining duplicate groups: "
        f"{remaining_duplicate_rows}"
    )

    # --------------------------------------------------------
    # 3D. CREATE DUPLICATE INVESTIGATION REPORT
    # --------------------------------------------------------

    audit_records = []

    # Record duplicates already present before imputation.
    for position in range(len(before_df)):

        if before_duplicate_mask.iloc[position]:

            audit_records.append({
                "original_row_number": position + 2,
                "duplicate_type": "Pre-existing duplicate",
                "action": "Removed",
                "reason": (
                    "Exact duplicate existed before imputation. "
                    "Repeated copy removed; first occurrence retained."
                )
            })

    # Record remaining duplicate groups after removal.
    remaining_positions = cleaned_df.index[
        remaining_duplicate_mask
    ]

    for original_index in remaining_positions:

        audit_records.append({
            "original_row_number": int(original_index) + 2,
            "duplicate_type": "Post-imputation duplicate candidate",
            "action": "Retained for review",
            "reason": (
                "Identical record found after imputation. "
                "Retained because different original records may "
                "become identical after missing values are filled."
            )
        })

    duplicate_report_df = pd.DataFrame(
        audit_records,
        columns=[
            "original_row_number",
            "duplicate_type",
            "action",
            "reason"
        ]
    )

    # --------------------------------------------------------
    # 3E. DISPLAY DUPLICATE SUMMARY
    # --------------------------------------------------------

    print("\nDUPLICATE HANDLING SUMMARY")
    print("-" * 75)

    print(f"Original rows: {len(after_df)}")
    print(f"Pre-existing duplicates removed: {removed_count}")
    print(f"Final rows: {len(cleaned_df)}")
    print(
        f"Post-imputation duplicate occurrences retained: "
        f"{remaining_duplicate_count}"
    )

    return cleaned_df, duplicate_report_df, {
        "before_duplicates": before_duplicate_count,
        "removed_duplicates": removed_count,
        "after_duplicates": remaining_duplicate_count,
        "remaining_duplicate_rows": remaining_duplicate_rows
    }


# ============================================================
# 4. VALIDATE NUMERICAL RANGES
# ============================================================

def validate_numerical_ranges(df):

    validation_results = []

    # Percentage and score validation.
    percentage_columns = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "technical_score",
        "aptitude_score",
        "communication_score",
        "skills_match_percentage",
    ]

    for column in percentage_columns:

        if column not in df.columns:
            continue

        invalid_count = int(
            ((df[column] < 0) | (df[column] > 100)).sum()
        )

        validation_results.append({
            "column": column,
            "business_rule": "Value must be between 0 and 100",
            "invalid_count": invalid_count,
            "status": "PASS" if invalid_count == 0 else "FAIL"
        })

    # Age validation.
    if "age_years" in df.columns:

        invalid_count = int(
            ((df["age_years"] < 18) | (df["age_years"] > 60)).sum()
        )

        validation_results.append({
            "column": "age_years",
            "business_rule": "Age must be between 18 and 60",
            "invalid_count": invalid_count,
            "status": "PASS" if invalid_count == 0 else "FAIL"
        })

    # Non-negative validation.
    non_negative_columns = [
        "certifications_count",
        "years_of_experience",
        "previous_ctc_lpa",
        "notice_period_days",
        "employment_gap_months",
    ]

    for column in non_negative_columns:

        if column not in df.columns:
            continue

        invalid_count = int(
            (df[column] < 0).sum()
        )

        validation_results.append({
            "column": column,
            "business_rule": "Value must not be negative",
            "invalid_count": invalid_count,
            "status": "PASS" if invalid_count == 0 else "FAIL"
        })

    # Expected CTC validation.
    if "expected_ctc_lpa" in df.columns:

        invalid_count = int(
            (df["expected_ctc_lpa"] <= 0).sum()
        )

        validation_results.append({
            "column": "expected_ctc_lpa",
            "business_rule": "Expected CTC must be greater than zero",
            "invalid_count": invalid_count,
            "status": "PASS" if invalid_count == 0 else "FAIL"
        })

    return validation_results


# ============================================================
# 5. VALIDATE CATEGORICAL VALUES
# ============================================================

def validate_categorical_values(df):

    validation_results = []

    if "status" in df.columns:

        allowed_values = ["Placed", "Not Placed"]

        invalid_count = int(
            (~df["status"].isin(allowed_values)).sum()
        )

        validation_results.append({
            "column": "status",
            "business_rule": "Value must be Placed or Not Placed",
            "invalid_count": invalid_count,
            "status": "PASS" if invalid_count == 0 else "FAIL"
        })

    return validation_results


# ============================================================
# 6. GENERATE DUPLICATE COMPARISON REPORT
# ============================================================

def generate_duplicate_comparison_report(duplicate_summary):

    before_count = duplicate_summary["before_duplicates"]
    removed_count = duplicate_summary["removed_duplicates"]
    after_count = duplicate_summary["after_duplicates"]

    comparison_df = pd.DataFrame([{
        "before_imputation_duplicates": before_count,
        "pre_existing_duplicates_removed": removed_count,
        "after_imputation_duplicates_after_removal": after_count,
        "observation": (
            "Post-imputation duplicate candidates remain"
            if after_count > 0
            else "No remaining exact duplicate records"
        )
    }])

    print("\nDUPLICATE COMPARISON")
    print("-" * 75)

    print(f"Before imputation: {before_count}")
    print(f"Pre-existing duplicates removed: {removed_count}")
    print(f"Remaining after removal: {after_count}")

    return comparison_df


# ============================================================
# 7. GENERATE BUSINESS VALIDATION REPORT
# ============================================================

def generate_validation_report(df, duplicate_summary):

    validation_results = []

    validation_results.extend(
        validate_numerical_ranges(df)
    )

    validation_results.extend(
        validate_categorical_values(df)
    )

    # Duplicate validation.
    remaining_count = duplicate_summary["after_duplicates"]

    validation_results.append({
        "column": "Entire dataset",
        "business_rule": (
            "Remove pre-existing exact duplicates and investigate "
            "post-imputation duplicate candidates"
        ),
        "invalid_count": remaining_count,
        "status": "PASS" if remaining_count == 0 else "REVIEW"
    })

    return pd.DataFrame(validation_results)


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

def save_outputs(
    df,
    report_df,
    duplicate_report_df,
    comparison_df
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save final validated dataset.
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Save business validation report.
    report_df.to_csv(
        VALIDATION_REPORT,
        index=False
    )

    # Save duplicate investigation report.
    duplicate_report_df.to_csv(
        DUPLICATE_REPORT,
        index=False
    )

    # Save before/after duplicate comparison.
    comparison_df.to_csv(
        DUPLICATE_COMPARISON_REPORT,
        index=False
    )

    print("\nOUTPUT FILES")
    print("-" * 75)

    print(f"Validated dataset: {OUTPUT_FILE}")
    print(f"Validation report: {VALIDATION_REPORT}")
    print(f"Duplicate report: {DUPLICATE_REPORT}")
    print(f"Comparison report: {DUPLICATE_COMPARISON_REPORT}")


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():

    # STEP 1: Load both datasets.
    before_df, after_df = load_datasets()

    # STEP 2: Handle duplicates.
    cleaned_df, duplicate_report_df, duplicate_summary = (
        handle_duplicates(before_df, after_df)
    )

    # STEP 3: Generate duplicate comparison report.
    comparison_df = generate_duplicate_comparison_report(
        duplicate_summary
    )

    # STEP 4: Run business validation.
    report_df = generate_validation_report(
        cleaned_df,
        duplicate_summary
    )

    # STEP 5: Display validation results.
    print("\nBUSINESS VALIDATION RESULTS")
    print("=" * 75)

    print(
        report_df.to_string(index=False)
    )

    # STEP 6: Display summary.
    passed = int(
        (report_df["status"] == "PASS").sum()
    )

    failed = int(
        (report_df["status"] == "FAIL").sum()
    )

    review = int(
        (report_df["status"] == "REVIEW").sum()
    )

    print("\nVALIDATION SUMMARY")
    print("-" * 75)

    print(f"Passed checks: {passed}")
    print(f"Failed checks: {failed}")
    print(f"Checks requiring review: {review}")

    # STEP 7: Save all outputs.
    save_outputs(
        cleaned_df,
        report_df,
        duplicate_report_df,
        comparison_df
    )

    # STEP 8: Final status.
    print("\n" + "=" * 75)
    print("BUSINESS VALIDATION COMPLETED")
    print("=" * 75)

    if failed > 0:
        print("WARNING: Invalid business values detected.")

    elif review > 0:
        print(
            "REVIEW REQUIRED: Post-imputation duplicate "
            "candidates need investigation."
        )

    else:
        print("All business validation checks passed.")


# ============================================================
# 10. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
