
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# FEATURE ENGINEERING
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step6_outlier_treated.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step7_feature_engineered.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_cleaning"
    / "feature_engineering"
)

REPORT_FILE = REPORT_DIR / "feature_engineering_report.csv"


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "years_of_experience",
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "skills_match_percentage",
    "status"
]


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_data():

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nOriginal Dataset Shape:", df.shape)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values. "
            "Please check the previous cleaning steps."
        )

    print("Dataset loaded successfully.")

    return df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_features(df):

    print("\nCreating 5 engineered features...")

    # --------------------------------------------------------
    # FEATURE 1: EXPERIENCE CATEGORY
    # --------------------------------------------------------
    # 0 years       = Fresher
    # Above 0 to 3  = Junior
    # Above 3       = Senior

    df["experience_category"] = pd.cut(
        df["years_of_experience"],
        bins=[
            -float("inf"),
            0,
            3,
            float("inf")
        ],
        labels=[
            "Fresher",
            "Junior",
            "Senior"
        ]
    )

    # --------------------------------------------------------
    # FEATURE 2: ACADEMIC PERFORMANCE BAND
    # --------------------------------------------------------
    # First calculate average academic percentage.

    academic_average = df[
        [
            "ssc_percentage",
            "hsc_percentage",
            "degree_percentage"
        ]
    ].mean(axis=1)

    # Convert academic average into performance bands.

    df["academic_performance_band"] = pd.cut(
        academic_average,
        bins=[
            -float("inf"),
            60,
            70,
            80,
            float("inf")
        ],
        labels=[
            "Low",
            "Average",
            "Good",
            "Excellent"
        ],
        right=False
    )

    # --------------------------------------------------------
    # FEATURE 3: SKILLS MATCH LEVEL
    # --------------------------------------------------------
    # Below 50       = Low
    # 50 to below 75 = Medium
    # 75 and above   = High

    df["skills_match_level"] = pd.cut(
        df["skills_match_percentage"],
        bins=[
            -float("inf"),
            50,
            75,
            float("inf")
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        right=False
    )

    # --------------------------------------------------------
    # FEATURE 4: INTERVIEW PERFORMANCE CATEGORY
    # --------------------------------------------------------
    # Interview performance is calculated using:
    # Technical score
    # Aptitude score
    # Communication score

    interview_average = df[
        [
            "technical_score",
            "aptitude_score",
            "communication_score"
        ]
    ].mean(axis=1)

    # Below 50       = Low
    # 50 to below 75 = Medium
    # 75 and above   = High

    df["interview_performance_category"] = pd.cut(
        interview_average,
        bins=[
            -float("inf"),
            50,
            75,
            float("inf")
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        right=False
    )

    # --------------------------------------------------------
    # FEATURE 5: PLACEMENT PROBABILITY SCORE
    # --------------------------------------------------------
    # Weighted readiness score:
    # Academic performance  = 30%
    # Assessment performance = 40%
    # Skills match           = 30%
    #
    # NOTE:
    # This is a manually designed readiness score.
    # It is NOT a statistically calibrated probability.

    assessment_average = interview_average

    df["placement_probability_score"] = (
        (academic_average * 0.30)
        + (assessment_average * 0.40)
        + (df["skills_match_percentage"] * 0.30)
    ).round(2)

    print("5 features created successfully.")

    return df


# ============================================================
# 5. FEATURE ENGINEERING REPORT
# ============================================================

def create_report(df):

    new_features = [
        "experience_category",
        "academic_performance_band",
        "skills_match_level",
        "interview_performance_category",
        "placement_probability_score"
    ]

    report_rows = []

    for feature in new_features:

        report_rows.append({
            "feature_name": feature,
            "data_type": str(df[feature].dtype),
            "unique_values": df[feature].nunique(),
            "missing_values": df[feature].isnull().sum(),
            "status": "Created"
        })

    report_df = pd.DataFrame(report_rows)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_df.to_csv(
        REPORT_FILE,
        index=False
    )

    print("\nFeature Engineering Report:")
    print(report_df.to_string(index=False))

    return report_df


# ============================================================
# 6. VALIDATION
# ============================================================

def validate_data(
    original_df,
    engineered_df,
    original_columns
):

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING VALIDATION")
    print("=" * 65)

    new_features = [
        "experience_category",
        "academic_performance_band",
        "skills_match_level",
        "interview_performance_category",
        "placement_probability_score"
    ]

    checks = {
        "Row count unchanged":
            len(original_df) == len(engineered_df),

        "Original columns preserved":
            all(
                col in engineered_df.columns
                for col in original_columns
            ),

        "Target column unchanged":
            original_df["status"].equals(
                engineered_df["status"]
            ),

        "Original values unchanged":
            original_df[original_columns].equals(
                engineered_df[original_columns]
            ),

        "Exactly 5 new features created":
            len(
                set(engineered_df.columns)
                - set(original_columns)
            ) == 5,

        "No missing values in new features":
            engineered_df[new_features]
            .isnull()
            .sum()
            .sum() == 0
    }

    for check, result in checks.items():

        print(
            f"{check}: "
            f"{'PASS' if result else 'FAIL'}"
        )

    if not all(checks.values()):
        raise ValueError(
            "Feature engineering validation failed."
        )

    print("\nAll validation checks passed.")


# ============================================================
# 7. SAVE DATASET
# ============================================================

def save_data(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset saved successfully:")
    print(OUTPUT_FILE)

    print("\nFinal Dataset Shape:", df.shape)


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():

    # Load dataset
    df = load_data()

    # Preserve original data for validation
    original_df = df.copy()

    original_columns = df.columns.tolist()

    # Create five features
    df = create_features(df)

    # Generate report
    create_report(df)

    # Validate dataset
    validate_data(
        original_df,
        df,
        original_columns
    )

    # Save output
    save_data(df)

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 65)


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
