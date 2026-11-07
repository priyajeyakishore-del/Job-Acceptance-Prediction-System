
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# UNIVARIATE NUMERICAL ANALYSIS
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
    / "univariate_numerical_analysis"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "graphs"
    / "univariate_numerical_analysis"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 75)
    print("UNIVARIATE NUMERICAL ANALYSIS")
    print("=" * 75)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nDataset loaded successfully.")
    print("Dataset Shape:", df.shape)

    return df


# ============================================================
# 3. IDENTIFY NUMERICAL FEATURES
# ============================================================

def identify_numerical_features(df):

    numerical_features = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Exclude the target if it happens to be numeric.
    numerical_features = [
        col for col in numerical_features
        if col != "status"
    ]

    print("\nTotal Numerical Features:", len(numerical_features))

    print("\nNumerical Features:")

    for feature in numerical_features:
        print("-", feature)

    return numerical_features


# ============================================================
# 4. DESCRIPTIVE STATISTICAL ANALYSIS
# ============================================================

def analyze_numerical_statistics(df, features):

    print("\n" + "=" * 75)
    print("1. DESCRIPTIVE STATISTICAL ANALYSIS")
    print("=" * 75)

    summary = df[features].describe().T

    summary["median"] = df[features].median()
    summary["mode"] = df[features].mode().iloc[0]
    summary["variance"] = df[features].var()
    summary["skewness"] = df[features].skew()
    summary["kurtosis"] = df[features].kurtosis()
    summary["missing_values"] = df[features].isnull().sum()

    summary = summary.round(2)

    print("\nNumerical Feature Statistics:")
    print(summary.to_string())

    return summary


# ============================================================
# 5. HISTOGRAM ANALYSIS
# ============================================================

def plot_histograms(df, features):

    print("\nGenerating histograms...")

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True
        )

        plt.title(
            f"Distribution of {feature}",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{feature}_histogram.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 6. BOXPLOT ANALYSIS
# ============================================================

def plot_boxplots(df, features):

    print("\nGenerating boxplots...")

    for feature in features:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            x=df[feature]
        )

        plt.title(
            f"Boxplot of {feature}",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel(feature)

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{feature}_boxplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 7. SAVE REPORT
# ============================================================

def save_report(summary):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = (
        REPORT_DIR / "numerical_univariate_summary.csv"
    )

    summary.to_csv(report_file)

    print("\nNumerical summary saved:")
    print(report_file)


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    numerical_features = identify_numerical_features(df)

    if not numerical_features:
        raise ValueError("No numerical features found.")

    summary = analyze_numerical_statistics(
        df,
        numerical_features
    )

    plot_histograms(df, numerical_features)

    plot_boxplots(df, numerical_features)

    save_report(summary)

    print("\n" + "=" * 75)
    print("NUMERICAL UNIVARIATE ANALYSIS COMPLETED")
    print("=" * 75)


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
