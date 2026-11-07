
import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# 1. DISPLAY PREDICTION RESULT
# ============================================================

def display_prediction_result(result):

    prediction_label = result["prediction_label"]

    probabilities = result.get("probabilities", {})

    # --------------------------------------------------------
    # PREDICTION STATUS
    # --------------------------------------------------------

    st.markdown("### 🎯 Prediction Result")

    if prediction_label == "Placed":

        st.success("✅ Predicted Outcome: PLACED")

        st.markdown(
            "The model predicts a positive job placement outcome."
        )

    else:

        st.error("⚠️ Predicted Outcome: NOT PLACED")

        st.markdown(
            "The model predicts a negative job placement outcome."
        )

    st.write("")

    # --------------------------------------------------------
    # PROBABILITY METRICS
    # --------------------------------------------------------

    st.markdown("### 📊 Placement Probability Estimates")

    if probabilities:

        not_placed = probabilities.get("Not Placed", 0)

        placed = probabilities.get("Placed", 0)

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Not Placed",
                value=f"{not_placed:.2f}%"
            )

        with col2:

            st.metric(
                label="Placed",
                value=f"{placed:.2f}%"
            )

        st.divider()

        # ----------------------------------------------------
        # PROBABILITY PROGRESS BARS
        # ----------------------------------------------------

        st.markdown("#### Probability Distribution")

        st.write(f"**Placed — {placed:.2f}%**")

        st.progress(
            max(0.0, min(1.0, placed / 100))
        )

        st.write(f"**Not Placed — {not_placed:.2f}%**")

        st.progress(
            max(0.0, min(1.0, not_placed / 100))
        )

        st.divider()

        # ----------------------------------------------------
        # INTERACTIVE PLOTLY CHART
        # ----------------------------------------------------

        st.markdown("#### Class Probability Comparison")

        probability_df = pd.DataFrame({
            "Class": ["Not Placed", "Placed"],
            "Probability": [
                not_placed,
                placed
            ]
        })

        fig = px.bar(
            probability_df,
            x="Class",
            y="Probability",
            text="Probability",
            color="Class",
            color_discrete_map={
                "Not Placed": "#F59E0B",
                "Placed": "#16A34A"
            },
            title="Candidate Placement Probability"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="Prediction Category",
            yaxis_range=[0, 110],
            showlegend=False,
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Probability information is not available."
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL PROCESSING SUMMARY
    # --------------------------------------------------------

    st.markdown("### ⚙️ Model Processing Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Original Input Fields",
            value="25"
        )

    with col2:

        st.metric(
            label="Final Model Features",
            value="38"
        )

    with col3:

        st.metric(
            label="Processed Features",
            value=result.get("processed_features", "N/A")
        )

    st.caption(
        "Note: Model probabilities are estimates and may not "
        "represent calibrated real-world placement chances."
    )


# ============================================================
# 2. DISPLAY CANDIDATE SUMMARY
# ============================================================

def display_candidate_summary(candidate_data):

    st.markdown("### 👤 Candidate Profile")

    summary_fields = {

        "age_years": "Age",

        "gender": "Gender",

        "degree_specialization": "Degree Specialization",

        "ssc_percentage": "SSC Percentage",

        "hsc_percentage": "HSC Percentage",

        "degree_percentage": "Degree Percentage",

        "years_of_experience": "Years of Experience",

        "technical_score": "Technical Score",

        "aptitude_score": "Aptitude Score",

        "communication_score": "Communication Score",

        "skills_match_percentage": "Skills Match Percentage",

        "certifications_count": "Certifications",

        "internship_experience": "Internship Experience",

        "company_tier": "Company Tier",

        "job_role_match": "Job Role Match",

        "competition_level": "Competition Level"
    }

    summary_data = []

    for field, label in summary_fields.items():

        value = candidate_data.get(field)

        if value is None:

            value = "Not provided"

        summary_data.append({
            "Candidate Detail": label,
            "Value": value
        })

    summary_df = pd.DataFrame(summary_data)

    # --------------------------------------------------------
    # DISPLAY SUMMARY IN TWO COLUMNS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    midpoint = (len(summary_df) + 1) // 2

    with col1:

        st.dataframe(
            summary_df.iloc[:midpoint],
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.dataframe(
            summary_df.iloc[midpoint:],
            use_container_width=True,
            hide_index=True
        )
