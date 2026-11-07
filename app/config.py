
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Trained model
MODEL_PATH = MODEL_DIR / "final_model.pkl"

# Fitted preprocessing pipeline
PREPROCESSOR_PATH = MODEL_DIR / "final_preprocessing_pipeline.pkl"


# ============================================================
# 2. STREAMLIT APPLICATION SETTINGS
# ============================================================

APP_TITLE = "Job Acceptance Prediction System"

APP_ICON = "💼"

APP_LAYOUT = "wide"


# ============================================================
# 3. TARGET COLUMN
# ============================================================

TARGET_COLUMN = "status"

TARGET_LABELS = {
    0: "Not Placed",
    1: "Placed"
}


# ============================================================
# 4. ORIGINAL NUMERICAL FEATURES (14)
# ============================================================

NUMERICAL_FEATURES = [
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
    "employment_gap_months"
]


# ============================================================
# 5. ORIGINAL CATEGORICAL FEATURES (11)
# ============================================================

CATEGORICAL_FEATURES = [
    "gender",
    "degree_specialization",
    "internship_experience",
    "career_switch_willingness",
    "relevant_experience",
    "company_tier",
    "job_role_match",
    "competition_level",
    "bond_requirement",
    "layoff_history",
    "relocation_willingness"
]


# ============================================================
# 6. MISSING-VALUE INDICATOR FEATURES (9)
# ============================================================

MISSING_FLAG_FEATURES = [
    "layoff_history_missing_flag",
    "relocation_willingness_missing_flag",
    "career_switch_willingness_missing_flag",
    "relevant_experience_missing_flag",
    "job_role_match_missing_flag",
    "ssc_percentage_missing_flag",
    "hsc_percentage_missing_flag",
    "notice_period_days_missing_flag",
    "employment_gap_months_missing_flag"
]


# ============================================================
# 7. ENGINEERED CATEGORICAL FEATURES (4)
# ============================================================

ENGINEERED_FEATURES = [
    "experience_category",
    "academic_performance_band",
    "skills_match_level",
    "interview_performance_category"
]


# ============================================================
# 8. EXACT MODEL INPUT COLUMN ORDER (38)
# ============================================================

MODEL_FEATURES = [
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
    "relocation_willingness",
    "layoff_history_missing_flag",
    "relocation_willingness_missing_flag",
    "career_switch_willingness_missing_flag",
    "relevant_experience_missing_flag",
    "job_role_match_missing_flag",
    "ssc_percentage_missing_flag",
    "hsc_percentage_missing_flag",
    "notice_period_days_missing_flag",
    "employment_gap_months_missing_flag",
    "experience_category",
    "academic_performance_band",
    "skills_match_level",
    "interview_performance_category"
]


# ============================================================
# 9. VALIDATE FEATURE CONFIGURATION
# ============================================================

if len(MODEL_FEATURES) != 38:
    raise ValueError(
        f"Expected 38 model features, found {len(MODEL_FEATURES)}"
    )

if len(set(MODEL_FEATURES)) != len(MODEL_FEATURES):
    raise ValueError("Duplicate feature names found.")

if set(MODEL_FEATURES) != (
    set(NUMERICAL_FEATURES)
    | set(CATEGORICAL_FEATURES)
    | set(MISSING_FLAG_FEATURES)
    | set(ENGINEERED_FEATURES)
):
    raise ValueError(
        "Model feature groups do not match MODEL_FEATURES."
    )
