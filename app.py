
import streamlit as st

from app.config import APP_TITLE, APP_ICON, APP_LAYOUT

from app.input_form import render_input_form

from app.prediction import predict_candidate

from app.display_results import (
    display_prediction_result,
    display_candidate_summary
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

    .main-title {
        font-size: 34px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 16px;
        color: #808495;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .small-note {
        font-size: 13px;
        color: #808495;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        min-height: 45px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💼 Job Predictor")

    st.caption("Machine Learning Application")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        """
        This application predicts job placement outcomes
        using candidate academic, technical, professional
        and job-related information.
        """
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** XGBoost")

    st.write("**Model Features:** 38")

    st.write("**Processed Features:** 68")

    st.write("**Training Records:** 40,000")

    st.write("**Testing Records:** 10,000")

    st.divider()

    st.markdown("### Model Performance")

    st.metric("Accuracy", "91.79%")

    st.metric("F1 Score", "86.34%")

    st.metric("ROC-AUC", "97.62%")

    st.divider()

    st.caption("GUVI Data Science Project")

    st.caption("Job Acceptance Prediction System")


# ============================================================
# 4. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Job Acceptance Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Data-Driven Job Placement Prediction using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Evaluate candidate profiles and estimate their job "
    "placement outcome using the trained XGBoost model."
)


# ============================================================
# 5. CANDIDATE INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">📝 Candidate Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the candidate details below and submit the form "
    "to generate a placement prediction."
)

candidate_data = render_input_form()


# ============================================================
# 6. PREDICTION PROCESS
# ============================================================

if candidate_data is not None:

    try:

        with st.spinner(
            "Analyzing candidate profile and generating prediction..."
        ):

            result = predict_candidate(candidate_data)

        st.success("Candidate prediction generated successfully!")

        st.divider()

        # ====================================================
        # 7. PREDICTION RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-heading">📊 Prediction Dashboard</div>',
            unsafe_allow_html=True
        )

        display_prediction_result(result)

        st.divider()

        # ====================================================
        # 8. CANDIDATE SUMMARY
        # ====================================================

        with st.expander(
            "👤 View Candidate Profile Summary",
            expanded=False
        ):

            display_candidate_summary(candidate_data)

    except FileNotFoundError as error:

        st.error(f"Required file not found: {error}")

    except ValueError as error:

        st.error(f"Prediction error: {error}")

    except Exception as error:

        st.error(
            f"An unexpected error occurred: {error}"
        )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:

    st.caption(
        "Job Acceptance Prediction System | GUVI Data Science"
    )

with footer_col2:

    st.caption(
        "Powered by Python, Streamlit and XGBoost"
    )

st.caption(
    "Disclaimer: Predictions are machine learning estimates. "
    "They do not guarantee actual hiring outcomes."
)
