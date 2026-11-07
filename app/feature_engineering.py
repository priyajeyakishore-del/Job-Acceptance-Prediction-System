
import pandas as pd

from app.config import (
    MODEL_FEATURES,
    MISSING_FLAG_FEATURES,
    PROJECT_ROOT
)


# ============================================================
# 1. DEFINE IMPUTATION REPORT PATH
# ============================================================

IMPUTATION_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_cleaning"
    / "missing_value_handling"
    / "numerical_imputation_report.csv"
)


# ============================================================
# 2. DEFINE IMPUTATION COLUMNS
# ============================================================

MEAN_COLUMNS = [
    "ssc_percentage",
    "hsc_percentage"
]

MEDIAN_COLUMNS = [
    "notice_period_days",
    "employment_gap_months"
]

CATEGORICAL_IMPUTATION_COLUMNS = [
    "career_switch_willingness",
    "relevant_experience",
    "job_role_match",
    "layoff_history",
    "relocation_willingness"
]


# ============================================================
# 3. LOAD TRAINING IMPUTATION VALUES
# ============================================================

def load_imputation_values():
    """
    Load the numerical imputation values generated
    by the Step 4 missing-value handling pipeline.
    """

    if not IMPUTATION_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Imputation report not found: "
            f"{IMPUTATION_REPORT_PATH}"
        )

    report_df = pd.read_csv(IMPUTATION_REPORT_PATH)

    required_columns = {
        "column",
        "method",
        "imputation_value"
    }

    if not required_columns.issubset(report_df.columns):
        raise ValueError(
            "Numerical imputation report has an unexpected format."
        )

    imputation_values = dict(
        zip(
            report_df["column"],
            report_df["imputation_value"]
        )
    )

    required_features = MEAN_COLUMNS + MEDIAN_COLUMNS

    missing_statistics = [
        column
        for column in required_features
        if column not in imputation_values
    ]

    if missing_statistics:
        raise ValueError(
            "Missing imputation statistics for: "
            f"{missing_statistics}"
        )

    return imputation_values


# ============================================================
# 4. CREATE MISSING-VALUE FLAGS
# ============================================================

def create_missing_flags(candidate_data):
    """
    Create nine missing-value indicators.

    1 = Original candidate value was missing.
    0 = Original candidate value was available.
    """

    missing_flag_mapping = {
        "layoff_history_missing_flag": "layoff_history",
        "relocation_willingness_missing_flag": "relocation_willingness",
        "career_switch_willingness_missing_flag": "career_switch_willingness",
        "relevant_experience_missing_flag": "relevant_experience",
        "job_role_match_missing_flag": "job_role_match",
        "ssc_percentage_missing_flag": "ssc_percentage",
        "hsc_percentage_missing_flag": "hsc_percentage",
        "notice_period_days_missing_flag": "notice_period_days",
        "employment_gap_months_missing_flag": "employment_gap_months"
    }

    for flag_name in MISSING_FLAG_FEATURES:

        original_column = missing_flag_mapping[flag_name]

        original_value = candidate_data[original_column]

        candidate_data[flag_name] = (
            1 if pd.isna(original_value) else 0
        )

    return candidate_data


# ============================================================
# 5. IMPUTE MISSING NUMERICAL VALUES
# ============================================================

def impute_numerical_values(candidate_data, imputation_values):
    """
    Apply the same Step 4 numerical imputation statistics.

    SSC and HSC: Mean
    Notice period and employment gap: Median
    """

    for column in MEAN_COLUMNS + MEDIAN_COLUMNS:

        if pd.isna(candidate_data[column]):

            candidate_data[column] = float(
                imputation_values[column]
            )

    return candidate_data


# ============================================================
# 6. IMPUTE MISSING CATEGORICAL VALUES
# ============================================================

def impute_categorical_values(candidate_data):
    """
    Replace missing categorical values with Unknown,
    following the Step 4 training pipeline.
    """

    for column in CATEGORICAL_IMPUTATION_COLUMNS:

        if pd.isna(candidate_data[column]):
            candidate_data[column] = "Unknown"

    return candidate_data


# ============================================================
# 7. CREATE ENGINEERED CATEGORICAL FEATURES
# ============================================================

def create_engineered_features(candidate_data):
    """
    Generate the same four engineered features
    used during model training.
    """

    # --------------------------------------------------------
    # 7A. EXPERIENCE CATEGORY
    # --------------------------------------------------------

    experience = candidate_data["years_of_experience"]

    if experience <= 0:
        experience_category = "Fresher"

    elif experience <= 3:
        experience_category = "Junior"

    else:
        experience_category = "Senior"

    candidate_data["experience_category"] = (
        experience_category
    )

    # --------------------------------------------------------
    # 7B. ACADEMIC PERFORMANCE BAND
    # --------------------------------------------------------

    academic_average = (
        candidate_data["ssc_percentage"]
        + candidate_data["hsc_percentage"]
        + candidate_data["degree_percentage"]
    ) / 3

    if academic_average < 60:
        academic_band = "Low"

    elif academic_average < 70:
        academic_band = "Average"

    elif academic_average < 80:
        academic_band = "Good"

    else:
        academic_band = "Excellent"

    candidate_data["academic_performance_band"] = (
        academic_band
    )

    # --------------------------------------------------------
    # 7C. SKILLS MATCH LEVEL
    # --------------------------------------------------------

    skills_match = candidate_data["skills_match_percentage"]

    if skills_match < 50:
        skills_level = "Low"

    elif skills_match < 75:
        skills_level = "Medium"

    else:
        skills_level = "High"

    candidate_data["skills_match_level"] = skills_level

    # --------------------------------------------------------
    # 7D. INTERVIEW PERFORMANCE CATEGORY
    # --------------------------------------------------------

    interview_average = (
        candidate_data["technical_score"]
        + candidate_data["aptitude_score"]
        + candidate_data["communication_score"]
    ) / 3

    if interview_average < 50:
        interview_category = "Low"

    elif interview_average < 75:
        interview_category = "Medium"

    else:
        interview_category = "High"

    candidate_data["interview_performance_category"] = (
        interview_category
    )

    return candidate_data


# ============================================================
# 8. PREPARE FINAL MODEL INPUT
# ============================================================

def prepare_model_input(candidate_data):
    """
    Generate all required features and return
    a DataFrame with exactly 38 model columns.
    """

    # Check that all original candidate inputs are present.
    required_input_columns = [
        "age_years",
        "gender",
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "degree_specialization",
        "technical_score",
        "aptitude_score",
        "communication_score",
        "skills_match_percentage",
        "certifications_count",
        "internship_experience",
        "years_of_experience",
        "career_switch_willingness",
        "relevant_experience",
        "previous_ctc_lpa",
        "expected_ctc_lpa",
        "company_tier",
        "job_role_match",
        "competition_level",
        "bond_requirement",
        "notice_period_days",
        "layoff_history",
        "employment_gap_months",
        "relocation_willingness"
    ]

    missing_columns = [
        column
        for column in required_input_columns
        if column not in candidate_data
    ]

    if missing_columns:
        raise ValueError(
            f"Missing candidate input fields: {missing_columns}"
        )

    # Work on a copy.
    candidate_data = candidate_data.copy()

    # Step 1: Create missing flags BEFORE imputation.
    candidate_data = create_missing_flags(candidate_data)

    # Step 2: Load the original training imputation values.
    imputation_values = load_imputation_values()

    # Step 3: Impute numerical missing values.
    candidate_data = impute_numerical_values(
        candidate_data,
        imputation_values
    )

    # Step 4: Impute categorical missing values.
    candidate_data = impute_categorical_values(
        candidate_data
    )

    # Step 5: Generate four engineered features.
    candidate_data = create_engineered_features(
        candidate_data
    )

    # Step 6: Arrange exactly 38 features in training order.
    model_input = pd.DataFrame(
        [candidate_data],
        columns=MODEL_FEATURES
    )

    # Step 7: Validate final model input.
    if model_input.shape[1] != 38:
        raise ValueError(
            f"Expected 38 features, received {model_input.shape[1]}"
        )

    if model_input.isnull().any().any():
        missing = model_input.columns[
            model_input.isnull().any()
        ].tolist()

        raise ValueError(
            f"Missing values remain in model input: {missing}"
        )

    return model_input
