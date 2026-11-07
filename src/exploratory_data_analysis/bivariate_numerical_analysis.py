
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# BIVARIATE NUMERICAL ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import ttest_ind


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
    / "bivariate_numerical_analysis"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "graphs"
    / "bivariate_numerical_analysis"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 80)
    print("BIVARIATE NUMERICAL ANALYSIS")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if "status" not in df.columns:
        raise ValueError(
            "Target column 'status' is missing from dataset."
        )

    df["status"] = df["status"].astype(str).str.strip()

    print("\nDataset loaded successfully.")
    print("Dataset Shape:", df.shape)

    print("\nTarget Distribution:")
    print(df["status"].value_counts())

    return df


# ============================================================
# 3. IDENTIFY NUMERICAL FEATURES
# ============================================================

def identify_numerical_features(df):

    numerical_features = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Exclude target column if it is numeric.
    numerical_features = [
        feature
        for feature in numerical_features
        if feature != "status"
    ]

    print("\nTotal Numerical Features:", len(numerical_features))

    print("\nNumerical Features:")

    for feature in numerical_features:
        print("-", feature)

    return numerical_features


# ============================================================
# 4. GROUP-WISE STATISTICAL COMPARISON
# ============================================================

def analyze_group_statistics(df, features):

    print("\n" + "=" * 80)
    print("1. GROUP-WISE STATISTICAL COMPARISON")
    print("=" * 80)

    summary_rows = []

    for feature in features:

        for status, group in df.groupby("status"):

            values = group[feature].dropna()

            summary_rows.append({

                "feature": feature,

                "status": status,

                "count": values.count(),

                "mean": values.mean(),

                "median": values.median(),

                "std": values.std(),

                "minimum": values.min(),

                "maximum": values.max(),

                "missing_values": group[feature].isnull().sum()

            })

    summary = pd.DataFrame(summary_rows)

    print("\nGroup-wise Comparison:")
    print(summary.round(2).to_string(index=False))

    return summary


# ============================================================
# 5. WELCH'S INDEPENDENT T-TEST
# ============================================================

def perform_t_tests(df, features):

    print("\n" + "=" * 80)
    print("2. WELCH'S INDEPENDENT T-TEST")
    print("=" * 80)

    summary_rows = []

    placed_data = df[df["status"] == "Placed"]

    not_placed_data = df[df["status"] == "Not Placed"]

    for feature in features:

        placed_values = (
            placed_data[feature]
            .dropna()
            .to_numpy()
        )

        not_placed_values = (
            not_placed_data[feature]
            .dropna()
            .to_numpy()
        )

        if len(placed_values) < 2 or len(not_placed_values) < 2:

            summary_rows.append({

                "feature": feature,
                "placed_mean": np.mean(placed_values) if len(placed_values) else np.nan,
                "not_placed_mean": np.mean(not_placed_values) if len(not_placed_values) else np.nan,
                "t_statistic": np.nan,
                "p_value": np.nan,
                "cohens_d": np.nan,
                "result": "Insufficient data"

            })

            continue

        # Welch's t-test does not assume equal variances.
        t_statistic, p_value = ttest_ind(
            placed_values,
            not_placed_values,
            equal_var=False,
            nan_policy="omit"
        )

        # Calculate Cohen's d using pooled standard deviation.
        n1 = len(placed_values)
        n2 = len(not_placed_values)

        var1 = np.var(placed_values, ddof=1)
        var2 = np.var(not_placed_values, ddof=1)

        pooled_std = np.sqrt(
            (
                (n1 - 1) * var1
                + (n2 - 1) * var2
            )
            / (n1 + n2 - 2)
        )

        if pooled_std == 0 or not np.isfinite(pooled_std):

            cohens_d = np.nan

        else:

            cohens_d = (
                np.mean(placed_values)
                - np.mean(not_placed_values)
            ) / pooled_std

        if p_value < 0.05:

            result = "Statistically significant"

        else:

            result = "Not statistically significant"

        summary_rows.append({

            "feature": feature,

            "placed_mean": np.mean(placed_values),

            "not_placed_mean": np.mean(not_placed_values),

            "t_statistic": t_statistic,

            "p_value": p_value,

            "cohens_d": cohens_d,

            "result": result

        })

    results = pd.DataFrame(summary_rows)

    print("\nWelch's T-Test Results:")
    print(results.round(4).to_string(index=False))

    return results


# ============================================================
# 6. BOXPLOT ANALYSIS
# ============================================================

def plot_boxplots(df, features):

    print("\nGenerating boxplots...")

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="status",
            y=feature
        )

        plt.title(
            f"{feature} vs Placement Status",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel("Placement Status")
        plt.ylabel(feature)

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{feature}_vs_status_boxplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 7. VIOLIN PLOT ANALYSIS
# ============================================================

def plot_violin_plots(df, features):

    print("\nGenerating violin plots...")

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.violinplot(
            data=df,
            x="status",
            y=feature,
            inner="quartile",
            cut=0
        )

        plt.title(
            f"{feature} vs Placement Status",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel("Placement Status")
        plt.ylabel(feature)

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{feature}_vs_status_violin.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 8. SAVE REPORTS
# ============================================================

def save_reports(group_summary, test_summary):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    group_summary.to_csv(
        REPORT_DIR / "numerical_placement_comparison.csv",
        index=False
    )

    test_summary.to_csv(
        REPORT_DIR / "numerical_statistical_test.csv",
        index=False
    )

    print("\nReports saved successfully.")


# ============================================================
# 9. MAIN FUNCTION
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_dataset()

    numerical_features = identify_numerical_features(df)

    if not numerical_features:
        raise ValueError("No numerical features found.")

    group_summary = analyze_group_statistics(
        df,
        numerical_features
    )

    test_summary = perform_t_tests(
        df,
        numerical_features
    )

    plot_boxplots(
        df,
        numerical_features
    )

    plot_violin_plots(
        df,
        numerical_features
    )

    save_reports(
        group_summary,
        test_summary
    )

    print("\n" + "=" * 80)
    print("BIVARIATE NUMERICAL ANALYSIS COMPLETED")
    print("=" * 80)


# ============================================================
# 10. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
