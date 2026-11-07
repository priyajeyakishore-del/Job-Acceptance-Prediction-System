
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# ANALYST TASKS - EDA & ML ANALYTICS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step7_feature_engineered.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "analyst_tasks"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

# Standardize target labels
df["status"] = df["status"].astype(str).str.strip().str.title()

print("=" * 70)
print("JOB ACCEPTANCE PREDICTION SYSTEM - ANALYST TASKS")
print("=" * 70)

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(df["status"].value_counts())


# ------------------------------------------------------------
# COMMON FUNCTION: PLACEMENT RATE
# ------------------------------------------------------------

def calculate_placement_rate(data, group_column):

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

    result["Placement_Rate_Percentage"] = (
        result["Placed_Candidates"]
        / result["Total_Candidates"]
        * 100
    )

    return result


# ============================================================
# SECTION 1: CANDIDATE PERFORMANCE ANALYSIS
# ============================================================


# ------------------------------------------------------------
# TASK 1: ACADEMIC SCORES VS PLACEMENT OUTCOME
# ------------------------------------------------------------

def academic_scores_vs_placement():

    print("\n1. ACADEMIC SCORES VS PLACEMENT OUTCOME")

    academic_columns = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage"
    ]

    available_columns = [
        col for col in academic_columns
        if col in df.columns
    ]

    # Calculate average academic scores by placement status
    result = (
        df.groupby("status")[available_columns]
        .mean()
        .round(2)
    )

    print("\nAverage Academic Scores by Placement Status:")
    print(result)

    result.to_csv(
        OUTPUT_DIR / "academic_scores_by_placement.csv"
    )

    # Convert data into long format for visualization
    plot_data = df[
        available_columns + ["status"]
    ].melt(
        id_vars="status",
        var_name="Academic_Score",
        value_name="Percentage"
    )

    plt.figure(figsize=(11, 6))

    sns.boxplot(
        data=plot_data,
        x="Academic_Score",
        y="Percentage",
        hue="status"
    )

    plt.title(
        "Academic Scores vs Placement Outcome",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Academic Qualification")
    plt.ylabel("Percentage")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "academic_scores_vs_placement.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# TASK 2: SKILLS MATCH VS INTERVIEW PERFORMANCE
# ------------------------------------------------------------

def skills_match_vs_interview():

    print("\n2. SKILLS MATCH VS INTERVIEW PERFORMANCE")

    # Compare skills match levels with interview categories
    result = pd.crosstab(
        df["skills_match_level"],
        df["interview_performance_category"]
    )

    print("\nSkills Match vs Interview Performance:")
    print(result)

    result.to_csv(
        OUTPUT_DIR / "skills_vs_interview_counts.csv"
    )

    # Row-wise percentages
    percentage_result = pd.crosstab(
        df["skills_match_level"],
        df["interview_performance_category"],
        normalize="index"
    ) * 100

    print("\nInterview Performance Distribution (%):")
    print(percentage_result.round(2))

    percentage_result.to_csv(
        OUTPUT_DIR / "skills_vs_interview_percentage.csv"
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        percentage_result,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu"
    )

    plt.title(
        "Skills Match vs Interview Performance (%)",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Interview Performance Category")
    plt.ylabel("Skills Match Level")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "skills_match_vs_interview.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# TASK 3: CERTIFICATION IMPACT ON PLACEMENT
# ------------------------------------------------------------

def certification_impact():

    print("\n3. CERTIFICATION IMPACT ON PLACEMENT")

    result = calculate_placement_rate(
        df,
        "certifications_count"
    )

    result = result.sort_values(
        "certifications_count"
    )

    print("\nPlacement Rate by Certification Count:")
    print(result.round(2))

    result.to_csv(
        OUTPUT_DIR / "certification_placement_analysis.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=result,
        x="certifications_count",
        y="Placement_Rate_Percentage",
        color="teal"
    )

    plt.title(
        "Certification Count vs Placement Rate",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Number of Certifications")
    plt.ylabel("Placement Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "certification_impact.png",
        dpi=300
    )

    plt.close()


# ============================================================
# SECTION 2: PLACEMENT ANALYSIS
# ============================================================


# ------------------------------------------------------------
# TASK 4: PLACEMENT RATE BY COMPANY TIER
# ------------------------------------------------------------

def company_tier_placement():

    print("\n4. PLACEMENT RATE BY COMPANY TIER")

    result = calculate_placement_rate(
        df,
        "company_tier"
    )

    print("\nPlacement Rate by Company Tier:")
    print(result.round(2))

    result.to_csv(
        OUTPUT_DIR / "company_tier_placement.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=result,
        x="company_tier",
        y="Placement_Rate_Percentage",
        color="royalblue"
    )

    plt.title(
        "Placement Rate by Company Tier",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Company Tier")
    plt.ylabel("Placement Rate (%)")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "company_tier_placement.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# TASK 5: EXPERIENCE VS PLACEMENT SUCCESS
# ------------------------------------------------------------

def experience_vs_placement():

    print("\n5. EXPERIENCE VS PLACEMENT SUCCESS")

    result = calculate_placement_rate(
        df,
        "experience_category"
    )

    print("\nPlacement Rate by Experience Category:")
    print(result.round(2))

    result.to_csv(
        OUTPUT_DIR / "experience_placement_analysis.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=result,
        x="experience_category",
        y="Placement_Rate_Percentage",
        color="mediumseagreen"
    )

    plt.title(
        "Experience Category vs Placement Rate",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Experience Category")
    plt.ylabel("Placement Rate (%)")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "experience_vs_placement.png",
        dpi=300
    )

    plt.close()


# ============================================================
# SECTION 3: INTERVIEW & EVALUATION ANALYSIS
# ============================================================


# ------------------------------------------------------------
# TASK 6: INTERVIEW SCORES VS PLACEMENT PROBABILITY SCORE
# ------------------------------------------------------------

def interview_score_vs_probability():

    print("\n6. INTERVIEW SCORES VS PLACEMENT PROBABILITY")

    interview_columns = [
        "technical_score",
        "aptitude_score",
        "communication_score"
    ]

    available_columns = [
        col for col in interview_columns
        if col in df.columns
    ]

    score_column = "placement_probability_score"

    if score_column not in df.columns:
        print("Placement probability score is unavailable.")
        return

    # Descriptive correlation analysis
    result = df[
        available_columns + [score_column]
    ].corr(numeric_only=True)

    print("\nCorrelation Matrix:")
    print(result.round(3))

    result.to_csv(
        OUTPUT_DIR / "interview_probability_correlation.csv"
    )

    # Correlation heatmap
    plt.figure(figsize=(9, 6))

    sns.heatmap(
        result,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Interview Scores vs Placement Probability Score",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "interview_probability_correlation.png",
        dpi=300
    )

    plt.close()

    # Individual scatter plots
    for column in available_columns:

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=df,
            x=column,
            y=score_column,
            hue="status",
            alpha=0.6
        )

        plt.title(
            f"{column} vs Placement Probability Score"
        )

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / f"{column}_vs_placement_probability.png",
            dpi=300
        )

        plt.close()


# ------------------------------------------------------------
# TASK 7: EMPLOYABILITY TEST SCORE ANALYSIS
# ------------------------------------------------------------

def employability_test_analysis():

    print("\n7. EMPLOYABILITY TEST SCORE ANALYSIS")

    employability_columns = [
        "technical_score",
        "aptitude_score",
        "communication_score"
    ]

    available_columns = [
        col for col in employability_columns
        if col in df.columns
    ]

    # Overall descriptive statistics
    result = df[available_columns].describe().round(2)

    print("\nEmployability Score Statistics:")
    print(result)

    result.to_csv(
        OUTPUT_DIR / "employability_score_statistics.csv"
    )

    # Score distributions
    fig, axes = plt.subplots(
        1,
        len(available_columns),
        figsize=(15, 5)
    )

    if len(available_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, available_columns):

        sns.histplot(
            data=df,
            x=column,
            kde=True,
            ax=ax,
            color="teal"
        )

        ax.set_title(
            column.replace("_", " ").title()
        )

        ax.set_xlabel("Score")

    plt.suptitle(
        "Employability Test Score Distributions",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "employability_score_distribution.png",
        dpi=300
    )

    plt.close()

    # Compare employability scores by placement status
    plot_data = df[
        available_columns + ["status"]
    ].melt(
        id_vars="status",
        var_name="Score_Type",
        value_name="Score"
    )

    plt.figure(figsize=(11, 6))

    sns.boxplot(
        data=plot_data,
        x="Score_Type",
        y="Score",
        hue="status"
    )

    plt.title(
        "Employability Scores vs Placement Status",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Employability Score Type")
    plt.ylabel("Score")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "employability_scores_vs_status.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_all_analyst_tasks():

    academic_scores_vs_placement()

    skills_match_vs_interview()

    certification_impact()

    company_tier_placement()

    experience_vs_placement()

    interview_score_vs_probability()

    employability_test_analysis()

    print("\n" + "=" * 70)
    print("ALL 7 ANALYST TASKS COMPLETED")
    print("=" * 70)

    print("\nOutput Folder:")
    print(OUTPUT_DIR)

    print("\nCharts Folder:")
    print(CHART_DIR)


if __name__ == "__main__":
    run_all_analyst_tasks()
