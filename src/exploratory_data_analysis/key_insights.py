
# ============================================================
# KEY INSIGHTS GENERATION
# Project: Job Acceptance Prediction System
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step7_feature_engineered.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "reports"
    / "key_insights"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORT_DIR / "key_insights.txt"


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "status" not in df.columns:
        raise ValueError("Target column 'status' is missing.")

    return df


# ------------------------------------------------------------
# 3. TARGET ANALYSIS
# ------------------------------------------------------------

def analyze_target(df):

    insights = []

    total_candidates = len(df)

    target_counts = df["status"].value_counts()
    target_percentages = (
        df["status"].value_counts(normalize=True) * 100
    )

    placed = target_counts.get("Placed", 0)
    not_placed = target_counts.get("Not Placed", 0)

    placement_rate = (
        placed / total_candidates * 100
        if total_candidates else 0
    )

    insights.append("TARGET DISTRIBUTION")
    insights.append("-" * 60)
    insights.append(f"Total candidates: {total_candidates}")
    insights.append(f"Placed candidates: {placed}")
    insights.append(f"Not placed candidates: {not_placed}")
    insights.append(
        f"Overall placement rate: {placement_rate:.2f}%"
    )

    for category, count in target_counts.items():
        percentage = target_percentages[category]

        insights.append(
            f"{category}: {count} ({percentage:.2f}%)"
        )

    insights.append("")

    return insights


# ------------------------------------------------------------
# 4. CATEGORICAL PLACEMENT ANALYSIS
# ------------------------------------------------------------

def analyze_categorical_features(df):

    insights = []

    categorical_features = [
        "experience_category",
        "academic_performance_band",
        "skills_match_level",
        "interview_performance_category",
        "gender",
        "degree_specialization",
        "internship_experience",
        "relevant_experience",
        "company_tier",
        "job_role_match",
        "competition_level",
        "relocation_willingness"
    ]

    insights.append("CATEGORICAL FEATURE INSIGHTS")
    insights.append("-" * 60)

    for column in categorical_features:

        if column not in df.columns:
            continue

        grouped = df.groupby(
            column,
            dropna=False
        )["status"].agg(
            total_candidates="count",
            placed=lambda x: (x == "Placed").sum()
        )

        grouped["placement_rate"] = (
            grouped["placed"]
            / grouped["total_candidates"]
            * 100
        )

        insights.append(f"\nFeature: {column}")

        for category, row in grouped.iterrows():

            insights.append(
                f"{category}: "
                f"Candidates={int(row['total_candidates'])}, "
                f"Placed={int(row['placed'])}, "
                f"Placement Rate={row['placement_rate']:.2f}%"
            )

        if not grouped.empty:

            highest = grouped["placement_rate"].idxmax()
            lowest = grouped["placement_rate"].idxmin()

            insights.append(
                f"Highest placement rate: {highest} "
                f"({grouped.loc[highest, 'placement_rate']:.2f}%)"
            )

            insights.append(
                f"Lowest placement rate: {lowest} "
                f"({grouped.loc[lowest, 'placement_rate']:.2f}%)"
            )

    insights.append("")

    return insights


# ------------------------------------------------------------
# 5. NUMERICAL PLACEMENT ANALYSIS
# ------------------------------------------------------------

def analyze_numerical_features(df):

    insights = []

    numerical_features = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Exclude binary missing indicators from the main comparison.
    numerical_features = [
        col for col in numerical_features
        if not col.endswith("_missing_flag")
    ]

    insights.append("NUMERICAL FEATURE INSIGHTS")
    insights.append("-" * 60)

    for column in numerical_features:

        placed_values = df.loc[
            df["status"] == "Placed", column
        ].dropna()

        not_placed_values = df.loc[
            df["status"] == "Not Placed", column
        ].dropna()

        if placed_values.empty or not_placed_values.empty:
            continue

        placed_mean = placed_values.mean()
        not_placed_mean = not_placed_values.mean()

        difference = placed_mean - not_placed_mean

        insights.append(
            f"\nFeature: {column}"
        )

        insights.append(
            f"Placed average: {placed_mean:.2f}"
        )

        insights.append(
            f"Not placed average: {not_placed_mean:.2f}"
        )

        insights.append(
            f"Mean difference: {difference:.2f}"
        )

        if difference > 0:
            insights.append(
                "Observation: Higher average among placed candidates."
            )

        elif difference < 0:
            insights.append(
                "Observation: Higher average among not placed candidates."
            )

        else:
            insights.append(
                "Observation: Both groups have equal averages."
            )

    insights.append("")

    return insights


# ------------------------------------------------------------
# 6. CORRELATION INSIGHTS
# ------------------------------------------------------------

def analyze_correlations(df):

    insights = []

    numerical_df = df.select_dtypes(
        include=["number"]
    )

    # Remove engineered score to avoid highlighting
    # a score derived from other predictors as independent evidence.
    numerical_df = numerical_df.drop(
        columns=["placement_probability_score"],
        errors="ignore"
    )

    numerical_df = numerical_df.drop(
        columns=[
            col for col in numerical_df.columns
            if col.endswith("_missing_flag")
        ],
        errors="ignore"
    )

    correlation = numerical_df.corr()

    insights.append("HIGH NUMERICAL CORRELATIONS")
    insights.append("-" * 60)

    found_pair = False

    for i in range(len(correlation.columns)):

        for j in range(i + 1, len(correlation.columns)):

            corr_value = correlation.iloc[i, j]

            if abs(corr_value) >= 0.80:

                found_pair = True

                insights.append(
                    f"{correlation.columns[i]} and "
                    f"{correlation.columns[j]}: "
                    f"{corr_value:.3f}"
                )

    if not found_pair:
        insights.append(
            "No numerical feature pairs with |correlation| >= 0.80."
        )

    insights.append("")

    return insights


# ------------------------------------------------------------
# 7. SAVE KEY INSIGHTS
# ------------------------------------------------------------

def save_insights(insights):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "JOB ACCEPTANCE PREDICTION SYSTEM\n"
        )

        file.write(
            "EXPLORATORY DATA ANALYSIS - KEY INSIGHTS\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write("\n".join(insights))

    print("\nKey insights saved successfully!")
    print(f"Report location: {OUTPUT_PATH}")


# ------------------------------------------------------------
# 8. MAIN FUNCTION
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("KEY INSIGHTS GENERATION")
    print("=" * 70)

    df = load_data()

    print(f"Dataset loaded: {df.shape}")

    insights = []

    insights.extend(analyze_target(df))

    insights.extend(
        analyze_categorical_features(df)
    )

    insights.extend(
        analyze_numerical_features(df)
    )

    insights.extend(
        analyze_correlations(df)
    )

    save_insights(insights)

    print("\nKey insights generation completed!")


if __name__ == "__main__":
    main()
