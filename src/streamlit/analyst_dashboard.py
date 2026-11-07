
# ==============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STREAMLIT EDA & ANALYST DASHBOARD
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# ==============================================================
# 1. PAGE CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="Job Placement Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================
# 2. CUSTOM CSS
# ==============================================================

st.markdown("""
<style>

.main {
    background-color: #F5F7FB;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 32px;
    font-weight: 750;
    color: #142D4E;
    margin-bottom: 4px;
}

.dashboard-subtitle {
    font-size: 15px;
    color: #64748B;
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #142D4E;
    padding-bottom: 8px;
    border-bottom: 2px solid #DCE5F0;
    margin-top: 15px;
}

.kpi-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0px 2px 8px rgba(15, 35, 60, 0.05);
    min-height: 115px;
}

.kpi-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
}

.kpi-value {
    color: #142D4E;
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
}

.kpi-caption {
    color: #94A3B8;
    font-size: 11px;
    margin-top: 4px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

</style>
""", unsafe_allow_html=True)


# ==============================================================
# 3. PROJECT CONFIGURATION
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_cleaned_data.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "analyst_dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# 4. DATA LOADING
# ==============================================================

@st.cache_data
def load_uploaded_data(file_bytes):

    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data
def load_local_data(file_path):

    return pd.read_csv(file_path)


def prepare_data(data):

    data.columns = data.columns.str.strip()

    # Standardize target labels
    data["status"] = (
        data["status"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Convert relevant columns to numeric
    numeric_columns = [
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
        "placement_probability_score"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


# ==============================================================
# 5. SIDEBAR - DATA SOURCE
# ==============================================================

st.sidebar.title("📊 Dashboard Controls")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Cleaned Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = load_uploaded_data(uploaded_file.getvalue())
    except Exception as error:
        st.error(f"Unable to read uploaded CSV: {error}")
        st.stop()

elif DEFAULT_DATA_PATH.exists():

    df = load_local_data(str(DEFAULT_DATA_PATH))

else:

    st.warning(
        "Cleaned dataset not found. "
        "Upload your cleaned CSV using the sidebar."
    )

    st.stop()


df = prepare_data(df)


# ==============================================================
# 6. VALIDATE REQUIRED COLUMNS
# ==============================================================

required_columns = [
    "status",
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "skills_match_percentage",
    "certifications_count",
    "company_tier",
    "experience_category",
    "interview_performance_category",
    "technical_score",
    "aptitude_score",
    "communication_score"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ==============================================================
# 7. SIDEBAR FILTERS
# ==============================================================

st.sidebar.subheader("Filter Candidates")

filtered_df = df.copy()


def add_category_filter(data, column, label):

    if column not in data.columns:
        return data

    values = sorted(
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected = st.sidebar.multiselect(
        label,
        options=values,
        default=values
    )

    if selected:

        data = data[
            data[column].astype(str).isin(selected)
        ]

    return data


filtered_df = add_category_filter(
    filtered_df,
    "company_tier",
    "Company Tier"
)

filtered_df = add_category_filter(
    filtered_df,
    "experience_category",
    "Experience Category"
)

filtered_df = add_category_filter(
    filtered_df,
    "status",
    "Placement Status"
)

# Certification count filter
cert_min = int(
    filtered_df["certifications_count"].min()
    if filtered_df["certifications_count"].notna().any()
    else 0
)

cert_max = int(
    filtered_df["certifications_count"].max()
    if filtered_df["certifications_count"].notna().any()
    else 0
)

if cert_min < cert_max:

    cert_range = st.sidebar.slider(
        "Certification Count",
        min_value=cert_min,
        max_value=cert_max,
        value=(cert_min, cert_max)
    )

    filtered_df = filtered_df[
        filtered_df["certifications_count"].between(
            cert_range[0],
            cert_range[1]
        )
    ]


st.sidebar.markdown("---")

st.sidebar.write(
    f"**Filtered Records:** {len(filtered_df):,}"
)

if filtered_df.empty:

    st.warning(
        "No candidates match the selected filters. "
        "Please change your filters."
    )

    st.stop()


# ==============================================================
# 8. HEADER
# ==============================================================

st.markdown(
    '<div class="dashboard-title">'
    'Job Placement Analytics Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Candidate Performance • Placement Trends • '
    'Interview Analytics • Employability Insights'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ==============================================================
# 9. HELPER FUNCTIONS
# ==============================================================

def safe_mean(data, column):

    if column not in data.columns:
        return np.nan

    return data[column].mean()


def placement_rate(data):

    total = len(data)

    if total == 0:
        return 0

    placed = (data["status"] == "Placed").sum()

    return placed / total * 100


def show_kpi(label, value, caption=""):

    st.markdown(
        f"""
        <div class="kpi-card">

            <div class="kpi-label">
                {label}
            </div>

            <div class="kpi-value">
                {value}
            </div>

            <div class="kpi-caption">
                {caption}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def section_heading(title):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def placement_summary(data, group_column):

    result = (
        data.groupby(group_column, dropna=False)
        .agg(
            Total_Candidates=("status", "size"),
            Placed_Candidates=(
                "status",
                lambda x: (x == "Placed").sum()
            )
        )
        .reset_index()
    )

    result["Not_Placed_Candidates"] = (
        result["Total_Candidates"]
        - result["Placed_Candidates"]
    )

    result["Placement_Rate"] = (
        result["Placed_Candidates"]
        / result["Total_Candidates"]
        * 100
    )

    return result


# ==============================================================
# 10. KEY KPIs
# ==============================================================

section_heading("📌 Key Performance Indicators")

total_candidates = len(filtered_df)

placed_candidates = (
    filtered_df["status"] == "Placed"
).sum()

not_placed_candidates = (
    filtered_df["status"] == "Not Placed"
).sum()

overall_placement_rate = placement_rate(filtered_df)

avg_academic_score = filtered_df[
    [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage"
    ]
].mean(axis=1).mean()

avg_skills_match = safe_mean(
    filtered_df,
    "skills_match_percentage"
)

avg_experience = safe_mean(
    filtered_df,
    "years_of_experience"
)

avg_certifications = safe_mean(
    filtered_df,
    "certifications_count"
)

avg_employability_score = filtered_df[
    [
        "technical_score",
        "aptitude_score",
        "communication_score"
    ]
].mean(axis=1).mean()

kpi_rows = [
    [
        ("Total Candidates", f"{total_candidates:,}", "Candidates in current filter"),
        ("Placed Candidates", f"{placed_candidates:,}", "Status = Placed"),
        ("Not Placed", f"{not_placed_candidates:,}", "Status = Not Placed"),
        ("Placement Rate", f"{overall_placement_rate:.2f}%", "Placed / Total × 100")
    ],
    [
        ("Average Academic Score", f"{avg_academic_score:.2f}" if pd.notna(avg_academic_score) else "N/A", "Mean of SSC, HSC and degree percentages"),
        ("Average Skills Match", f"{avg_skills_match:.2f}%" if pd.notna(avg_skills_match) else "N/A", "Mean skills-match percentage"),
        ("Average Experience", f"{avg_experience:.2f}" if pd.notna(avg_experience) else "N/A", "Years of experience"),
        ("Average Certifications", f"{avg_certifications:.2f}" if pd.notna(avg_certifications) else "N/A", "Certifications per candidate")
    ],
    [
        ("Employability Score", f"{avg_employability_score:.2f}" if pd.notna(avg_employability_score) else "N/A", "Mean of technical, aptitude and communication scores"),
        ("Company Tiers", f"{filtered_df['company_tier'].nunique()}", "Distinct tiers in current filter"),
        ("Experience Categories", f"{filtered_df['experience_category'].nunique()}", "Distinct categories in current filter"),
        ("Average Age", f"{safe_mean(filtered_df, 'age_years'):.1f}" if pd.notna(safe_mean(filtered_df, 'age_years')) else "N/A", "Years")
    ]
]

for row in kpi_rows:

    cols = st.columns(4)

    for col, (label, value, caption) in zip(cols, row):

        with col:
            show_kpi(label, value, caption)

    st.write("")


st.markdown("---")


# ==============================================================
# 11. OVERVIEW CHARTS
# ==============================================================

section_heading("📈 Placement Overview")

col1, col2 = st.columns(2)

with col1:

    status_counts = (
        filtered_df["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Candidates")
    )

    fig = px.pie(
        status_counts,
        names="Status",
        values="Candidates",
        hole=0.55,
        title="Candidate Placement Distribution",
        color="Status",
        color_discrete_map={
            "Placed": "#10B981",
            "Not Placed": "#EF4444"
        }
    )

    fig.update_layout(
        legend_title="Placement Status"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    tier_result = placement_summary(
        filtered_df,
        "company_tier"
    )

    fig = px.bar(
        tier_result,
        x="company_tier",
        y="Placement_Rate",
        text="Placement_Rate",
        color="Placement_Rate",
        color_continuous_scale="Teal",
        title="Placement Rate by Company Tier"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Placement Rate (%)",
        xaxis_title="Company Tier"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==============================================================
# 12. CANDIDATE PERFORMANCE ANALYSIS
# ==============================================================

section_heading("🎓 Candidate Performance Analysis")

# --------------------------------------------------------------
# TASK 1: ACADEMIC SCORES VS PLACEMENT OUTCOME
# --------------------------------------------------------------

st.subheader("1. Academic Scores vs Placement Outcome")

academic_columns = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage"
]

academic_data = filtered_df[
    academic_columns + ["status"]
].melt(
    id_vars="status",
    var_name="Academic_Level",
    value_name="Percentage"
)

fig = px.box(
    academic_data,
    x="Academic_Level",
    y="Percentage",
    color="status",
    color_discrete_map={
        "Placed": "#10B981",
        "Not Placed": "#EF4444"
    },
    title="Academic Performance Distribution by Placement Status"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

academic_summary = (
    filtered_df.groupby("status")[academic_columns]
    .mean()
    .round(2)
)

st.write("**Average Academic Scores by Placement Status**")

st.dataframe(
    academic_summary,
    use_container_width=True
)


# --------------------------------------------------------------
# TASK 2: SKILLS MATCH VS INTERVIEW PERFORMANCE
# --------------------------------------------------------------

st.subheader("2. Skills Match vs Interview Performance")

skills_interview = pd.crosstab(
    filtered_df["skills_match_level"],
    filtered_df["interview_performance_category"],
    normalize="index"
) * 100

fig = px.imshow(
    skills_interview,
    text_auto=".1f",
    color_continuous_scale="Teal",
    aspect="auto",
    title="Skills Match Level vs Interview Performance (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.write("**Interview Performance Distribution (%)**")

st.dataframe(
    skills_interview.round(2),
    use_container_width=True
)


# --------------------------------------------------------------
# TASK 3: CERTIFICATION IMPACT
# --------------------------------------------------------------

st.subheader("3. Certification Count vs Placement Rate")

cert_result = placement_summary(
    filtered_df,
    "certifications_count"
).sort_values("certifications_count")

fig = px.bar(
    cert_result,
    x="certifications_count",
    y="Placement_Rate",
    text="Placement_Rate",
    title="Certification Count vs Placement Rate",
    color="Placement_Rate",
    color_continuous_scale="Teal"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Number of Certifications",
    yaxis_title="Placement Rate (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    cert_result.round(2),
    use_container_width=True
)


# ==============================================================
# 13. PLACEMENT ANALYSIS
# ==============================================================

section_heading("💼 Placement Analysis")


# --------------------------------------------------------------
# TASK 4: COMPANY TIER VS PLACEMENT
# --------------------------------------------------------------

st.subheader("4. Placement Rate by Company Tier")

company_result = placement_summary(
    filtered_df,
    "company_tier"
)

fig = px.bar(
    company_result,
    x="company_tier",
    y="Placement_Rate",
    color="company_tier",
    text="Placement_Rate",
    title="Company Tier vs Placement Rate"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Company Tier",
    yaxis_title="Placement Rate (%)",
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    company_result.round(2),
    use_container_width=True
)


# --------------------------------------------------------------
# TASK 5: EXPERIENCE VS PLACEMENT
# --------------------------------------------------------------

st.subheader("5. Experience Category vs Placement Success")

experience_result = placement_summary(
    filtered_df,
    "experience_category"
)

fig = px.bar(
    experience_result,
    x="experience_category",
    y="Placement_Rate",
    color="experience_category",
    text="Placement_Rate",
    title="Experience Category vs Placement Rate"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Experience Category",
    yaxis_title="Placement Rate (%)",
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    experience_result.round(2),
    use_container_width=True
)


# ==============================================================
# 14. INTERVIEW & EVALUATION ANALYSIS
# ==============================================================

section_heading("🧠 Interview & Evaluation Analysis")


# --------------------------------------------------------------
# TASK 6: INTERVIEW SCORES VS PLACEMENT PROBABILITY
# --------------------------------------------------------------

st.subheader("6. Interview Scores vs Placement Probability")

score_columns = [
    "technical_score",
    "aptitude_score",
    "communication_score"
]

if "placement_probability_score" in filtered_df.columns:

    correlation = filtered_df[
        score_columns + ["placement_probability_score"]
    ].corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Interview Scores vs Placement Probability Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    selected_score = st.selectbox(
        "Select Interview Score",
        score_columns
    )

    fig = px.scatter(
        filtered_df,
        x=selected_score,
        y="placement_probability_score",
        color="status",
        opacity=0.65,
        color_discrete_map={
            "Placed": "#10B981",
            "Not Placed": "#EF4444"
        },
        title=f"{selected_score} vs Placement Probability Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Interpretation caution: placement_probability_score is an "
        "engineered feature. Check its calculation before treating "
        "this relationship as independent evidence. It is not "
        "necessarily a calibrated model probability."
    )

else:

    st.warning(
        "placement_probability_score is not available."
    )


# --------------------------------------------------------------
# TASK 7: EMPLOYABILITY TEST SCORE ANALYSIS
# --------------------------------------------------------------

st.subheader("7. Employability Test Score Analysis")

employability_columns = [
    "technical_score",
    "aptitude_score",
    "communication_score"
]

employability_data = filtered_df[
    employability_columns
].melt(
    var_name="Score_Type",
    value_name="Score"
)

fig = px.histogram(
    employability_data,
    x="Score",
    color="Score_Type",
    barmode="overlay",
    opacity=0.65,
    marginal="box",
    title="Employability Score Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

score_comparison = filtered_df[
    employability_columns + ["status"]
].melt(
    id_vars="status",
    var_name="Score_Type",
    value_name="Score"
)

fig = px.box(
    score_comparison,
    x="Score_Type",
    y="Score",
    color="status",
    title="Employability Scores by Placement Status",
    color_discrete_map={
        "Placed": "#10B981",
        "Not Placed": "#EF4444"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.write("**Employability Score Summary**")

score_summary = (
    filtered_df[employability_columns]
    .describe()
    .round(2)
)

st.dataframe(
    score_summary,
    use_container_width=True
)


# ==============================================================
# 15. DOWNLOADABLE ANALYSIS
# ==============================================================

section_heading("📥 Download Analysis Results")

download_options = {
    "Academic Scores": academic_summary.reset_index(),
    "Skills vs Interview": skills_interview.reset_index(),
    "Certification Analysis": cert_result,
    "Company Tier Analysis": company_result,
    "Experience Analysis": experience_result,
    "Employability Statistics": score_summary.reset_index()
}

for label, result in download_options.items():

    st.download_button(
        label=f"Download {label} CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=label.lower().replace(" ", "_") + ".csv",
        mime="text/csv"
    )


# ==============================================================
# 16. FILTERED DATA PREVIEW
# ==============================================================

with st.expander("View Filtered Candidate Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.download_button(
        label="Download Filtered Dataset",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_candidate_data.csv",
        mime="text/csv"
    )


# ==============================================================
# 17. FOOTER
# ==============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748B;font-size:12px;">

        Job Placement Analytics Dashboard

        <br>

        GUVI Data Science Project

        <br>

        Candidate Performance | Placement Analytics |
        Interview Evaluation

    </div>
    """,
    unsafe_allow_html=True
)
