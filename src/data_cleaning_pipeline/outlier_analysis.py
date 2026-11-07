
# ============================================================
# FILE: outlier_analysis.py
# PROJECT: Job Acceptance Prediction System
#
# PURPOSE:
# 1. Load the business-validated dataset.
# 2. Identify outliers using the IQR method.
# 3. Apply feature-wise outlier treatment decisions.
# 4. Retain genuine outliers.
# 5. Avoid unnecessary deletion or capping.
# 6. Generate an outlier treatment decision report.
#
# INPUT:
# step5_business_validated.csv
#
# OUTPUT:
# step6_outlier_treated.csv
# outlier_treatment_report.csv
#
# TREATMENT STRATEGY:
# Retain all valid observations.
# No automatic deletion or capping.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR / "step5_business_validated.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "step6_outlier_treated.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_cleaning"
    / "outlier_treatment"
)

TREATMENT_REPORT = (
    REPORT_DIR / "outlier_treatment_report.csv"
)


# ============================================================
# 3. DEFINE NUMERICAL FEATURES
# ============================================================

NUMERICAL_COLUMNS = [
    "age_years",
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "skills_match_percentage",
    "certifications_count",
    "years_of_experience",
    "previous_ctc_lpa",
    "expected_ctc_lpa",
    "notice_period_days",
    "employment_gap_months",
]


# ============================================================
# 4. DEFINE OUTLIER TREATMENT DECISIONS
# ============================================================

TREATMENT_DECISIONS = {

    "age_years": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "No IQR outliers detected."
    },

    "ssc_percentage": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Valid academic percentage values."
    },

    "hsc_percentage": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Valid academic percentage values."
    },

    "degree_percentage": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Valid academic percentage values."
    },

    "technical_score": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "No IQR outliers detected."
    },

    "aptitude_score": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Valid aptitude scores within business limits."
    },

    "communication_score": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Valid communication scores within business limits."
    },

    "skills_match_percentage": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "No IQR outliers detected."
    },

    "certifications_count": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Higher certification counts may be genuine."
    },

    "years_of_experience": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "No IQR outliers detected."
    },

    "previous_ctc_lpa": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Higher previous salaries may be genuine."
    },

    "expected_ctc_lpa": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Higher salary expectations may be genuine."
    },

    "notice_period_days": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Longer notice periods may be valid."
    },

    "employment_gap_months": {
        "decision": "Retain",
        "method": "No treatment",
        "reason": "Longer employment gaps may be genuine."
    }
}


# ============================================================
# 5. LOAD DATASET
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 75)
    print("OUTLIER TREATMENT")
    print("=" * 75)

    print(f"Input file: {INPUT_FILE}")
    print(f"Dataset shape: {df.shape}")

    print(
        f"Total missing values: "
        f"{df.isna().sum().sum()}"
    )

    return df


# ============================================================
# 6. IDENTIFY AVAILABLE NUMERICAL FEATURES
# ============================================================

def get_available_numerical_columns(df):

    available_columns = []

    for column in NUMERICAL_COLUMNS:

        if column not in df.columns:

            print(
                f"WARNING: Missing feature skipped: {column}"
            )

            continue

        if not pd.api.types.is_numeric_dtype(df[column]):

            print(
                f"WARNING: Non-numerical feature skipped: {column}"
            )

            continue

        available_columns.append(column)

    return available_columns


# ============================================================
# 7. APPLY OUTLIER TREATMENT DECISIONS
# ============================================================

def apply_outlier_treatment(df, numerical_columns):

    # Work on a copy to preserve the input dataframe.
    treated_df = df.copy()

    treatment_results = []

    print("\nOUTLIER TREATMENT DECISIONS")
    print("-" * 75)

    for column in numerical_columns:

        # Get feature-specific decision.
        decision_info = TREATMENT_DECISIONS[column]

        # Ignore missing values for IQR calculations.
        values = treated_df[column].dropna()

        if values.empty:
            continue

        # Calculate quartiles.
        q1 = values.quantile(0.25)

        q3 = values.quantile(0.75)

        # Calculate IQR.
        iqr = q3 - q1

        # Calculate boundaries.
        lower_bound = q1 - (1.5 * iqr)

        upper_bound = q3 + (1.5 * iqr)

        # Identify outliers.
        outlier_mask = (
            (values < lower_bound) |
            (values > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        # ----------------------------------------------------
        # APPLY TREATMENT DECISION
        # ----------------------------------------------------

        if decision_info["decision"] == "Retain":

            # Do not modify, cap or delete any values.
            treated_df[column] = df[column].copy()

            treated_count = 0

        else:

            raise ValueError(
                f"Unsupported treatment decision for {column}"
            )

        # ----------------------------------------------------
        # SAVE TREATMENT DETAILS
        # ----------------------------------------------------

        treatment_results.append({

            "feature": column,

            "q1": q1,

            "q3": q3,

            "iqr": iqr,

            "lower_bound": lower_bound,

            "upper_bound": upper_bound,

            "outlier_count": outlier_count,

            "outlier_percentage": round(
                (outlier_count / len(values)) * 100,
                2
            ),

            "treatment_decision": decision_info["decision"],

            "treatment_method": decision_info["method"],

            "values_modified": treated_count,

            "reason": decision_info["reason"],

            "status": "Completed"

        })

        print(f"\nFeature: {column}")

        print(
            f"Outlier count: {outlier_count}"
        )

        print(
            f"Decision: {decision_info['decision']}"
        )

        print(
            f"Method: {decision_info['method']}"
        )

        print(
            f"Values modified: {treated_count}"
        )

    treatment_report = pd.DataFrame(
        treatment_results
    )

    return treated_df, treatment_report


# ============================================================
# 8. VERIFY DATA PRESERVATION
# ============================================================

def verify_data_preservation(original_df, treated_df):

    print("\nDATA PRESERVATION VALIDATION")
    print("-" * 75)

    # Check shape.
    shape_matches = (
        original_df.shape == treated_df.shape
    )

    print(
        f"Dataset shape unchanged: {shape_matches}"
    )

    # Check column names and order.
    columns_match = (
        list(original_df.columns) ==
        list(treated_df.columns)
    )

    print(
        f"Columns unchanged: {columns_match}"
    )

    # Check every value is unchanged.
    values_match = original_df.equals(treated_df)

    print(
        f"All original values retained: {values_match}"
    )

    if not shape_matches or not columns_match or not values_match:

        raise ValueError(
            "Data preservation validation failed."
        )

    print(
        "PASS: No rows or values were modified."
    )


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(treated_df, treatment_report):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save treated dataset.
    treated_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Save treatment report.
    treatment_report.to_csv(
        TREATMENT_REPORT,
        index=False
    )

    print("\nOUTPUT FILES")
    print("-" * 75)

    print(f"Processed dataset: {OUTPUT_FILE}")

    print(f"Treatment report: {TREATMENT_REPORT}")


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():

    # STEP 1: Load dataset.
    df = load_dataset()

    # STEP 2: Identify numerical columns.
    numerical_columns = (
        get_available_numerical_columns(df)
    )

    print(
        f"\nNumerical features selected: "
        f"{len(numerical_columns)}"
    )

    # STEP 3: Apply treatment decisions.
    treated_df, treatment_report = (
        apply_outlier_treatment(
            df,
            numerical_columns
        )
    )

    # STEP 4: Verify data preservation.
    verify_data_preservation(
        df,
        treated_df
    )

    # STEP 5: Display treatment report.
    print("\nOUTLIER TREATMENT REPORT")
    print("=" * 75)

    print(
        treatment_report.to_string(index=False)
    )

    # STEP 6: Save outputs.
    save_outputs(
        treated_df,
        treatment_report
    )

    # STEP 7: Final status.
    print("\n" + "=" * 75)
    print("OUTLIER TREATMENT COMPLETED")
    print("=" * 75)

    print(
        "All identified outliers were retained."
    )

    print(
        "No rows were deleted."
    )

    print(
        "No numerical values were capped or modified."
    )


# ============================================================
# 11. RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()
