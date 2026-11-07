
import streamlit as st
import pandas as pd

from app.config import PROJECT_ROOT


# ============================================================
# 1. LOAD CATEGORICAL OPTIONS FROM TRAINING DATA
# ============================================================

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "X_train.csv"
)

@st.cache_data
def load_categorical_options():
    """
    Load the exact categorical values used during model training.

    This prevents users from entering unsupported category values.
    """

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_DATA_PATH}"
        )

    df = pd.read_csv(TRAINING_DATA_PATH)

    categorical_columns = [
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

    options = {}

    for column in categorical_columns:
        options[column] = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    return options


# ============================================================
# 2. NUMERICAL INPUT HELPER
# ============================================================

def required_number_input(
    label,
    minimum=0.0,
    maximum=None,
    default=0.0,
    step=1.0
):
    """Collect a required numerical value."""

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step
    )


def optional_number_input(
    label,
    minimum=0.0,
    maximum=None,
    default=0.0,
    step=1.0
):
    """
    Collect an optional numerical value.

    Returns None when the candidate indicates that the
    information is not available.
    """

    not_available = st.checkbox(
        f"{label} - Not provided",
        key=f"{label}_missing"
    )

    value = st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        disabled=not_available
    )

    if not_available:
        return None

    return value


# ============================================================
# 3. CATEGORICAL INPUT HELPER
# ============================================================

def categorical_input(
    label,
    column_name,
    options,
    optional=False
):
    """Display a dropdown using the training dataset categories."""

    choices = options[column_name].copy()

    if optional:
        choices.insert(0, "Not provided")

    selected_value = st.selectbox(
        label,
        options=choices,
        key=f"select_{column_name}"
    )

    if optional and selected_value == "Not provided":
        return None

    return selected_value


# ============================================================
# 4. MAIN CANDIDATE INPUT FORM
# ============================================================

def render_input_form():
    """
    Display the candidate registration form.

    Returns:
        dict: Candidate input values after successful submission.
        None: If the form has not been submitted.
    """

    st.subheader("Candidate Information")

    st.write(
        "Enter the candidate details below to predict "
        "the job placement outcome."
    )

    # Load exact category choices from X_train.csv.
    try:
        options = load_categorical_options()

    except Exception as error:
        st.error(f"Unable to load category options: {error}")
        return None

    candidate_data = {}

    with st.form("candidate_input_form"):

        # ----------------------------------------------------
        # SECTION 1: PERSONAL DETAILS
        # ----------------------------------------------------

        st.markdown("### 1. Personal Details")

        col1, col2 = st.columns(2)

        with col1:
            candidate_data["age_years"] = required_number_input(
                "Age",
                minimum=18,
                maximum=100,
                default=25,
                step=1
            )

        with col2:
            candidate_data["gender"] = categorical_input(
                "Gender",
                "gender",
                options
            )

        # ----------------------------------------------------
        # SECTION 2: ACADEMIC DETAILS
        # ----------------------------------------------------

        st.markdown("### 2. Academic Details")

        col1, col2 = st.columns(2)

        with col1:
            candidate_data["ssc_percentage"] = optional_number_input(
                "SSC Percentage",
                minimum=0.0,
                maximum=100.0,
                default=70.0,
                step=0.1
            )

            candidate_data["hsc_percentage"] = optional_number_input(
                "HSC Percentage",
                minimum=0.0,
                maximum=100.0,
                default=70.0,
                step=0.1
            )

        with col2:
            candidate_data["degree_percentage"] = required_number_input(
                "Degree Percentage",
                minimum=0.0,
                maximum=100.0,
                default=70.0,
                step=0.1
            )

            candidate_data["degree_specialization"] = categorical_input(
                "Degree Specialization",
                "degree_specialization",
                options
            )

        # ----------------------------------------------------
        # SECTION 3: SKILLS AND ASSESSMENT
        # ----------------------------------------------------

        st.markdown("### 3. Skills and Assessment")

        col1, col2 = st.columns(2)

        with col1:
            candidate_data["technical_score"] = required_number_input(
                "Technical Score",
                minimum=0.0,
                maximum=100.0,
                default=70.0,
                step=0.1
            )

            candidate_data["aptitude_score"] = required_number_input(
                "Aptitude Score",
                minimum=0.0,
                maximum=100.0,
                default=70.0,
                step=0.1
            )

            candidate_data["communication_score"] = required_number_input(
                "Communication Score",
                minimum=0.0,
                maximum=100.0,
                default=70.0,
                step=0.1
            )

        with col2:
            candidate_data["skills_match_percentage"] = required_number_input(
                "Skills Match Percentage",
                minimum=0.0,
                maximum=100.0,
                default=70.0,
                step=0.1
            )

            candidate_data["certifications_count"] = required_number_input(
                "Number of Certifications",
                minimum=0,
                default=0,
                step=1
            )

            candidate_data["internship_experience"] = categorical_input(
                "Internship Experience",
                "internship_experience",
                options
            )

        # ----------------------------------------------------
        # SECTION 4: EXPERIENCE DETAILS
        # ----------------------------------------------------

        st.markdown("### 4. Experience Details")

        col1, col2 = st.columns(2)

        with col1:
            candidate_data["years_of_experience"] = required_number_input(
                "Years of Experience",
                minimum=0,
                default=0,
                step=1
            )

            candidate_data["career_switch_willingness"] = categorical_input(
                "Career Switch Willingness",
                "career_switch_willingness",
                options,
                optional=True
            )

            candidate_data["relevant_experience"] = categorical_input(
                "Relevant Experience",
                "relevant_experience",
                options,
                optional=True
            )

        with col2:
            candidate_data["previous_ctc_lpa"] = required_number_input(
                "Previous CTC (LPA)",
                minimum=0.0,
                default=0.0,
                step=0.1
            )

            candidate_data["expected_ctc_lpa"] = required_number_input(
                "Expected CTC (LPA)",
                minimum=0.0,
                default=3.0,
                step=0.1
            )

            candidate_data["company_tier"] = categorical_input(
                "Company Tier",
                "company_tier",
                options
            )

        # ----------------------------------------------------
        # SECTION 5: JOB PREFERENCES
        # ----------------------------------------------------

        st.markdown("### 5. Job Preferences")

        col1, col2 = st.columns(2)

        with col1:
            candidate_data["job_role_match"] = categorical_input(
                "Job Role Match",
                "job_role_match",
                options,
                optional=True
            )

            candidate_data["competition_level"] = categorical_input(
                "Competition Level",
                "competition_level",
                options
            )

            candidate_data["bond_requirement"] = categorical_input(
                "Bond Requirement",
                "bond_requirement",
                options
            )

        with col2:
            candidate_data["notice_period_days"] = optional_number_input(
                "Notice Period (Days)",
                minimum=0,
                default=30,
                step=1
            )

            candidate_data["layoff_history"] = categorical_input(
                "Layoff History",
                "layoff_history",
                options,
                optional=True
            )

            candidate_data["employment_gap_months"] = optional_number_input(
                "Employment Gap (Months)",
                minimum=0,
                default=0,
                step=1
            )

        # ----------------------------------------------------
        # SECTION 6: RELOCATION
        # ----------------------------------------------------

        st.markdown("### 6. Relocation Details")

        candidate_data["relocation_willingness"] = categorical_input(
            "Relocation Willingness",
            "relocation_willingness",
            options,
            optional=True
        )

        # ----------------------------------------------------
        # FORM SUBMISSION
        # ----------------------------------------------------

        submitted = st.form_submit_button(
            "Predict Job Acceptance",
            type="primary",
            use_container_width=True
        )

    if submitted:

        # Check all required fields.
        missing_required = [
            key
            for key, value in candidate_data.items()
            if value is None and key not in [
                "ssc_percentage",
                "hsc_percentage",
                "notice_period_days",
                "employment_gap_months",
                "career_switch_willingness",
                "relevant_experience",
                "job_role_match",
                "layoff_history",
                "relocation_willingness"
            ]
        ]

        if missing_required:
            st.error(
                "Please provide all required candidate details."
            )
            return None

        return candidate_data

    return None
