# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STEP 1: LOAD AND UNDERSTAND THE RAW DATASET
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. DEFINE PROJECT AND DATASET PATH
# ------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\Priya Jeyakishore\OneDrive\Desktop\Job Acceptance Prediction System"
)

DATASET_DIR = BASE_DIR / "data" / "raw"

# ------------------------------------------------------------
# 2. FIND CSV FILES
# ------------------------------------------------------------

csv_files = list(DATASET_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found in the raw dataset folder.")
    print(f"Please check: {DATASET_DIR}")
    raise SystemExit(1)

print("CSV files found:")
for index, file in enumerate(csv_files, start=1):
    print(f"{index}. {file.name}")

# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

# If there is only one CSV file, load it automatically.
# If there are multiple CSV files, select the first one for now.

DATASET_PATH = csv_files[0]

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("File name:", DATASET_PATH.name)

# ------------------------------------------------------------
# 4. DATASET OVERVIEW
# ------------------------------------------------------------

# DATASET DIMENSIONS
print("\n1. DATASET DIMENSIONS")
print("-" * 70)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# DATA TYPES AND MEANINGS OF ALL COLUMNS
print("\n2. DATA TYPES AND MEANINGS OF ALL COLUMNS")

# Dictionary containing column meanings and examples
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
    "career_switch_willingness": ("Willingness to change career", "Willing, Not Willing"),
    "relevant_experience": ("Experience relevant to the job", "Relevant, Not Relevant"),
    "previous_ctc_lpa": ("Previous CTC in lakhs per annum", "4.84"),
    "expected_ctc_lpa": ("Expected CTC in lakhs per annum", "8.00"),
    "company_tier": ("Company category or tier", "Tier 1, Tier 2, Tier 3"),
    "job_role_match": ("Match between candidate and job role", "Matched, Not Matched"),
    "competition_level": ("Level of competition", "Low, Medium, High"),
    "bond_requirement": ("Whether a job bond is required", "Required, Not Required"),
    "notice_period_days": ("Notice period in days", "15"),
    "layoff_history": ("Previous layoff history", "Yes, No"),
    "employment_gap_months": ("Employment gap in months", "6"),
    "relocation_willingness": ("Willingness to relocate", "Willing, Not Willing"),
    "status": ("Placement outcome (target variable)", "Placed, Not Placed")
}

# Print table header
print(
    f"{'Column Name':<30}"
    f"{'Data Type':<12}"
    f"{'Meaning':<43}"
    f"{'Example':<25}"
)
print("-" * 110)

# Print details for every column in the dataset
for column in df.columns:
    data_type = str(df[column].dtype)
    meaning, example = column_details.get(
        column, ("Meaning not documented", "-")
    )

    print(
        f"{column:<30}"
        f"{data_type:<12}"
        f"{meaning:<43}"
        f"{example:<25}"
    )

print("=" * 110)
print("Total columns:", len(df.columns))

# DUPLICATE ANALYSIS
print("\n3. DUPLICATE ANALYSIS")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print("Exact duplicate rows:", duplicate_count)
print("Duplicate percentage:", round(
    duplicate_count / len(df) * 100, 2
), "%")

print("\nSample duplicate records:")
print(df[df.duplicated(keep=False)].head(10))

# TARGET VARIABLE ANALYSIS
print("\n4. TARGET VARIABLE ANALYSIS")
print("-" * 70)

target = "status"

print("\nTarget value counts:")
print(df[target].value_counts(dropna=False))

print("\nTarget percentage:")
print(
    (df[target].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)

# 4. CATEGORICAL VALUE ANALYSIS
print("\n5. CATEGORICAL VALUE ANALYSIS")
print("-" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for column in categorical_columns:
    print(f"\nColumn: {column}")
    print("Unique values:", df[column].nunique(dropna=True))
    print(df[column].value_counts(dropna=False).head(20))


# 5. NUMERICAL RANGE CHECKS
print("\n6. NUMERICAL RANGE CHECKS")
print("-" * 70)

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    print(f"\nColumn: {column}")
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())


# 6. MISSING VALUE SUMMARY
print("\n7. MISSING VALUE SUMMARY")
print("-" * 70)

missing_summary = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

missing_summary = missing_summary[
    missing_summary["Missing_Count"] > 0
].sort_values("Missing_Percentage", ascending=False)

print(missing_summary)


# 7. BUSINESS LOGIC CHECKS
print("\n8. BUSINESS LOGIC CHECKS")
print("-" * 70)

if {"previous_ctc_lpa", "expected_ctc_lpa"}.issubset(df.columns):
    invalid_ctc = (
        (df["previous_ctc_lpa"] <= 0)
        | (df["expected_ctc_lpa"] <= 0)
    )
    print("Invalid CTC records:", invalid_ctc.sum())

if "notice_period_days" in df.columns:
    invalid_notice = (
        (df["notice_period_days"] < 0)
        | (df["notice_period_days"] > 90)
    )
    print("Notice period outside 0-90 days:", invalid_notice.sum())

if "employment_gap_months" in df.columns:
    invalid_gap = df["employment_gap_months"] < 0
    print("Negative employment gaps:", invalid_gap.sum())

if "status" in df.columns:
    print("Unexpected target labels:",
          sorted(set(df["status"].dropna().unique())
                 - {"Placed", "Not Placed"}))

print("\nData quality assessment completed!")

