
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STEP 1: DATA UNDERSTANDING DOCUMENTATION
# ============================================================

import pandas as pd
import streamlit as st
from pathlib import Path
from io import StringIO

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="wide"
)

# ------------------------------------------------------------
# 2. PROJECT PATH
# ------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\Priya Jeyakishore\OneDrive\Desktop\Job Acceptance Prediction System"
)

DATASET_DIR = BASE_DIR / "data" / "raw"

# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

@st.cache_data
def load_dataset(file_path, modified_time):
    return pd.read_csv(file_path)


st.title("🎯 Job Acceptance Prediction System")
st.subheader("Step 1: Data Understanding and Quality Assessment")

st.markdown("""
This documentation presents the initial analysis of the raw candidate
dataset, including dataset dimensions, column definitions, missing values,
duplicates, categorical distributions, numerical ranges, and business
logic checks.
""")

csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:
    st.error(f"No CSV files found in: {DATASET_DIR}")
    st.stop()

if len(csv_files) > 1:
    st.warning(
        "Multiple CSV files were found. Select the correct raw dataset below."
    )
    selected_file = st.selectbox(
        "Select raw dataset",
        csv_files,
        format_func=lambda file: file.name
    )
else:
    selected_file = csv_files[0]

try:
    df = load_dataset(
        str(selected_file),
        selected_file.stat().st_mtime
    )
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

if df.empty:
    st.error("The selected dataset contains no records.")
    st.stop()

st.success(f"Dataset loaded successfully: {selected_file.name}")

# ------------------------------------------------------------
# 4. COLUMN DEFINITIONS
# ------------------------------------------------------------

column_details = {
    "age_years": ("Candidate age in years", "27"),
    "gender": ("Candidate gender", "Male, Female"),
    "ssc_percentage": ("10th standard percentage", "65.06"),
    "hsc_percentage": ("12th standard percentage", "83.84"),
    "degree_percentage": ("Degree percentage", "75.85"),
    "degree_specialization": ("Degree specialization", "Computer Science"),
    "technical_score": ("Technical assessment score", "58.22"),
    "aptitude_score": ("Aptitude assessment score", "89.56"),
    "communication_score": ("Communication assessment score", "64.47"),
    "skills_match_percentage": ("Skills matching job requirements", "79.55"),
    "certifications_count": ("Number of certifications", "2"),
    "internship_experience": ("Internship experience status", "Yes, No"),
    "years_of_experience": ("Years of work experience", "1"),
    "career_switch_willingness": ("Willingness to change career", "Not Willing"),
    "relevant_experience": ("Experience relevant to the job", "Relevant"),
    "previous_ctc_lpa": ("Previous CTC in lakhs per annum", "4.84"),
    "expected_ctc_lpa": ("Expected CTC in lakhs per annum", "8.00"),
    "company_tier": ("Company category or tier", "Tier 1"),
    "job_role_match": ("Match between candidate and job role", "Matched"),
    "competition_level": ("Level of competition", "Low, Medium, High"),
    "bond_requirement": ("Whether a job bond is required", "Required"),
    "notice_period_days": ("Notice period in days", "15"),
    "layoff_history": ("Previous layoff history", "Yes, No"),
    "employment_gap_months": ("Employment gap in months", "6"),
    "relocation_willingness": ("Willingness to relocate", "Willing"),
    "status": ("Placement outcome (target variable)", "Placed, Not Placed")
}

column_rows = []

for column in df.columns:
    meaning, example = column_details.get(
        column,
        ("Meaning not documented", "-")
    )

    column_rows.append({
        "Column Name": column,
        "Data Type": str(df[column].dtype),
        "Meaning": meaning,
        "Example": example,
        "Non-Null Count": int(df[column].notna().sum()),
        "Missing Count": int(df[column].isna().sum())
    })

column_documentation = pd.DataFrame(column_rows)

# ------------------------------------------------------------
# 5. DUPLICATE ANALYSIS
# ------------------------------------------------------------

duplicate_count = int(df.duplicated().sum())

duplicate_percentage = round(
    duplicate_count / len(df) * 100, 2
)

duplicate_samples = df[df.duplicated(keep=False)].head(10)

# ------------------------------------------------------------
# 6. TARGET VARIABLE ANALYSIS
# ------------------------------------------------------------

if "status" in df.columns:
    target_counts = (
        df["status"]
        .value_counts(dropna=False)
        .rename_axis("Status")
        .reset_index(name="Candidate Count")
    )

    target_counts["Percentage"] = (
        target_counts["Candidate Count"] / len(df) * 100
    ).round(2)
else:
    target_counts = pd.DataFrame(
        columns=["Status", "Candidate Count", "Percentage"]
    )

# ------------------------------------------------------------
# 7. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

missing_summary = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing Percentage": (df.isna().mean() * 100).round(2)
})

missing_summary = (
    missing_summary[missing_summary["Missing Count"] > 0]
    .sort_values("Missing Percentage", ascending=False)
    .reset_index(names="Column")
)

# ------------------------------------------------------------
# 8. CATEGORICAL ANALYSIS
# ------------------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# ------------------------------------------------------------
# 9. NUMERICAL RANGE ANALYSIS
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

numeric_summary = df[numeric_columns].agg(
    ["count", "min", "max", "mean", "median"]
).T.reset_index(names="Column")

# ------------------------------------------------------------
# 10. BUSINESS LOGIC CHECKS
# ------------------------------------------------------------

business_checks = []

if {"previous_ctc_lpa", "expected_ctc_lpa"}.issubset(df.columns):
    invalid_ctc = (
        (df["previous_ctc_lpa"] <= 0)
        | (df["expected_ctc_lpa"] <= 0)
    )

    business_checks.append({
        "Check": "CTC values must be positive",
        "Records Flagged": int(invalid_ctc.sum())
    })

if "notice_period_days" in df.columns:
    invalid_notice = (
        (df["notice_period_days"] < 0)
        | (df["notice_period_days"] > 90)
    )

    business_checks.append({
        "Check": "Notice period outside 0–90 days",
        "Records Flagged": int(invalid_notice.sum())
    })

if "employment_gap_months" in df.columns:
    invalid_gap = df["employment_gap_months"] < 0

    business_checks.append({
        "Check": "Negative employment gap",
        "Records Flagged": int(invalid_gap.sum())
    })

if "status" in df.columns:
    unexpected_labels = (
        set(df["status"].dropna().unique())
        - {"Placed", "Not Placed"}
    )

    business_checks.append({
        "Check": "Unexpected target labels",
        "Records Flagged": int(
            df["status"].isin(unexpected_labels).sum()
        )
    })

business_summary = pd.DataFrame(business_checks)

# ------------------------------------------------------------
# 11. DOCUMENTATION SECTIONS
# ------------------------------------------------------------

st.divider()
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Candidates", f"{len(df):,}")
col2.metric("Total Columns", len(df.columns))
col3.metric("Duplicate Rows", f"{duplicate_count:,}")
col4.metric("Columns With Missing Values", len(missing_summary))

st.caption(
    f"Duplicate percentage: {duplicate_percentage}% of all rows. "
    "Duplicates are flagged for investigation, not automatically removed."
)

st.write("**Dataset preview**")
st.dataframe(df.head(10), use_container_width=True)

st.download_button(
    "Download raw dataset preview",
    df.head(100).to_csv(index=False).encode("utf-8"),
    file_name="dataset_preview.csv",
    mime="text/csv"
)

st.divider()
st.header("2. Data Types and Column Documentation")

st.dataframe(
    column_documentation,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download column documentation",
    column_documentation.to_csv(index=False).encode("utf-8"),
    file_name="column_documentation.csv",
    mime="text/csv"
)

st.divider()
st.header("3. Duplicate Analysis")

st.write(f"**Exact duplicate rows:** {duplicate_count:,}")
st.write(f"**Duplicate percentage:** {duplicate_percentage}%")

with st.expander("View sample duplicate records"):
    if duplicate_samples.empty:
        st.info("No duplicate records found.")
    else:
        st.dataframe(duplicate_samples, use_container_width=True)

st.divider()
st.header("4. Target Variable Analysis")

if target_counts.empty:
    st.warning("The target column 'status' was not found.")
else:
    st.dataframe(
        target_counts,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        target_counts.set_index("Status")["Candidate Count"]
    )

st.divider()
st.header("5. Categorical Value Analysis")

if not categorical_columns:
    st.info("No categorical columns found.")
else:
    selected_category = st.selectbox(
        "Select a categorical column",
        categorical_columns
    )

    category_counts = (
        df[selected_category]
        .value_counts(dropna=False)
        .rename_axis("Category")
        .reset_index(name="Count")
    )

    category_counts["Percentage"] = (
        category_counts["Count"] / len(df) * 100
    ).round(2)

    st.write(
        f"Unique non-missing values: "
        f"{df[selected_category].nunique(dropna=True)}"
    )

    st.dataframe(
        category_counts,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        category_counts.head(20).set_index("Category")["Count"]
    )

st.divider()
st.header("6. Numerical Range Checks")

st.markdown(
    "The following table shows descriptive statistics. "
    "Extreme values should be investigated before deciding whether "
    "they are invalid."
)

st.dataframe(
    numeric_summary.round(2),
    use_container_width=True,
    hide_index=True
)

st.divider()
st.header("7. Missing Value Report")

if missing_summary.empty:
    st.success("No missing values found.")
else:
    st.dataframe(
        missing_summary,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        missing_summary.set_index("Column")["Missing Percentage"]
    )

    st.download_button(
        "Download missing value report",
        missing_summary.to_csv(index=False).encode("utf-8"),
        file_name="missing_value_report.csv",
        mime="text/csv"
    )

st.divider()
st.header("8. Business Logic Checks")

if business_summary.empty:
    st.info("No business logic checks could be performed.")
else:
    st.dataframe(
        business_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These are preliminary checks. Confirm the business rules "
        "against the dataset documentation before treating flagged "
        "records as invalid."
    )

# ------------------------------------------------------------
# 12. DOWNLOAD COMPLETE HTML DOCUMENTATION
# ------------------------------------------------------------

st.divider()
st.header("9. Export Documentation")

html_report = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Job Acceptance Prediction System - Data Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 35px; color: #222; }}
h1, h2 {{ color: #174A7E; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background-color: #eaf1f8; }}
p {{ line-height: 1.5; }}
</style>
</head>
<body>
<h1>Job Acceptance Prediction System</h1>
<h2>Step 1: Data Understanding and Quality Assessment</h2>
<p>Dataset: {selected_file.name}</p>
<p>Total records: {len(df):,}</p>
<p>Total columns: {len(df.columns)}</p>
<p>Exact duplicate rows: {duplicate_count:,}</p>

<h2>Column Documentation</h2>
{column_documentation.to_html(index=False, border=0)}

<h2>Target Variable Analysis</h2>
{target_counts.to_html(index=False, border=0)}

<h2>Missing Value Report</h2>
{missing_summary.to_html(index=False, border=0)}

<h2>Numerical Summary</h2>
{numeric_summary.round(2).to_html(index=False, border=0)}

<h2>Business Logic Checks</h2>
{business_summary.to_html(index=False, border=0)}

<p>Note: Preliminary checks require validation against the dataset
documentation. No raw data has been modified by this report.</p>
</body>
</html>
"""

st.download_button(
    "📄 Download Complete Documentation (HTML)",
    data=html_report.encode("utf-8"),
    file_name="job_acceptance_data_documentation.html",
    mime="text/html"
)

st.success("Data understanding documentation generated successfully.")
