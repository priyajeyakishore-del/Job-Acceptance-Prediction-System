
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# TARGET VARIABLE ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
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
    / "target_analysis"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "graphs"
    / "target_analysis"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 75)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 75)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if "status" not in df.columns:
        raise ValueError(
            "Target column 'status' not found in dataset."
        )

    if df["status"].isnull().any():
        raise ValueError(
            "Target column contains missing values."
        )

    print("\nDataset loaded successfully.")
    print("Dataset Shape:", df.shape)

    return df


# ============================================================
# 3. TARGET CLASS DISTRIBUTION
# ============================================================

def analyze_target_distribution(df):

    print("\n" + "=" * 75)
    print("1. TARGET CLASS DISTRIBUTION")
    print("=" * 75)

    target_counts = df["status"].value_counts()

    print("\nCandidate Count:")
    print(target_counts)

    return target_counts


# ============================================================
# 4. TARGET PERCENTAGE DISTRIBUTION
# ============================================================

def calculate_target_percentage(target_counts, total_rows):

    print("\n" + "=" * 75)
    print("2. TARGET PERCENTAGE DISTRIBUTION")
    print("=" * 75)

    target_percentage = (
        target_counts / total_rows * 100
    ).round(2)

    print("\nCandidate Percentage:")
    print(target_percentage)

    target_summary = pd.DataFrame({
        "status": target_counts.index,
        "candidate_count": target_counts.values,
        "percentage": target_percentage.values
    })

    return target_summary


# ============================================================
# 5. CLASS IMBALANCE ANALYSIS
# ============================================================

def analyze_class_imbalance(target_summary):

    print("\n" + "=" * 75)
    print("3. CLASS IMBALANCE ANALYSIS")
    print("=" * 75)

    if len(target_summary) < 2:
        print("\nOnly one target class is present.")
        print("Class imbalance cannot be calculated.")
        return {
            "majority_class": target_summary.iloc[0]["status"],
            "minority_class": None,
            "imbalance_ratio": None,
            "class_balance": "Single class"
        }

    majority = target_summary.loc[
        target_summary["candidate_count"].idxmax()
    ]

    minority = target_summary.loc[
        target_summary["candidate_count"].idxmin()
    ]

    majority_count = majority["candidate_count"]
    minority_count = minority["candidate_count"]

    imbalance_ratio = round(
        majority_count / minority_count, 2
    ) if minority_count else None

    # Basic 80:20 rule for identifying severe imbalance.
    majority_percentage = majority["percentage"]

    if majority_percentage >= 80:
        class_balance = "Highly Imbalanced"
    elif majority_percentage >= 60:
        class_balance = "Moderately Imbalanced"
    else:
        class_balance = "Relatively Balanced"

    print("\nMajority Class:", majority["status"])
    print("Minority Class:", minority["status"])
    print("Majority Count:", majority_count)
    print("Minority Count:", minority_count)
    print("Imbalance Ratio:", imbalance_ratio)
    print("Class Balance:", class_balance)

    return {
        "majority_class": majority["status"],
        "minority_class": minority["status"],
        "imbalance_ratio": imbalance_ratio,
        "class_balance": class_balance
    }


# ============================================================
# 6. TARGET DISTRIBUTION BAR CHART
# ============================================================

def plot_target_bar_chart(target_summary):

    print("\nGenerating target distribution bar chart...")

    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=target_summary,
        x="status",
        y="candidate_count"
    )

    plt.title(
        "Placement Status Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Placement Status")
    plt.ylabel("Number of Candidates")

    # Display candidate counts on bars.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "target_distribution_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. TARGET DISTRIBUTION PIE CHART
# ============================================================

def plot_target_pie_chart(target_summary):

    print("\nGenerating target distribution pie chart...")

    plt.figure(figsize=(7, 7))

    plt.pie(
        target_summary["candidate_count"],
        labels=target_summary["status"],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Placement Status Percentage Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "target_distribution_pie.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. SAVE TARGET ANALYSIS REPORT
# ============================================================

def save_target_report(
    target_summary,
    imbalance_info
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    target_summary.to_csv(
        REPORT_DIR / "target_distribution.csv",
        index=False
    )

    imbalance_report = pd.DataFrame([
        {
            "metric": metric,
            "value": value
        }
        for metric, value in imbalance_info.items()
    ])

    imbalance_report.to_csv(
        REPORT_DIR / "class_imbalance_report.csv",
        index=False
    )

    print("\nTarget analysis reports saved successfully.")


# ============================================================
# 9. MAIN FUNCTION
# ============================================================

def main():

    # Create output directories
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load dataset
    df = load_dataset()

    # Analyse target counts
    target_counts = analyze_target_distribution(df)

    # Calculate percentages
    target_summary = calculate_target_percentage(
        target_counts,
        len(df)
    )

    # Analyse class imbalance
    imbalance_info = analyze_class_imbalance(
        target_summary
    )

    # Generate charts
    plot_target_bar_chart(target_summary)

    plot_target_pie_chart(target_summary)

    # Save reports
    save_target_report(
        target_summary,
        imbalance_info
    )

    print("\n" + "=" * 75)
    print("TARGET ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("\nGraph Location:", GRAPH_DIR)
    print("Report Location:", REPORT_DIR)


# ============================================================
# 10. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
