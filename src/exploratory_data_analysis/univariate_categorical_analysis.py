
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# UNIVARIATE CATEGORICAL ANALYSIS
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
    / "univariate_categorical_analysis"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "graphs"
    / "univariate_categorical_analysis"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 75)
    print("UNIVARIATE CATEGORICAL ANALYSIS")
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
# 3. IDENTIFY CATEGORICAL FEATURES
# ============================================================

def identify_categorical_features(df):

    categorical_features = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(
        "\nTotal Categorical Features:",
        len(categorical_features)
    )

    print("\nCategorical Features:")

    for feature in categorical_features:
        print("-", feature)

    return categorical_features


# ============================================================
# 4. CATEGORY FREQUENCY ANALYSIS
# ============================================================

def analyze_category_frequency(df, features):

    print("\n" + "=" * 75)
    print("1. CATEGORY FREQUENCY ANALYSIS")
    print("=" * 75)

    summary_rows = []

    for feature in features:

        counts = df[feature].value_counts(
            dropna=False
        )

        for category, count in counts.items():

            percentage = count / len(df) * 100

            summary_rows.append({
                "feature_name": feature,
                "category": (
                    "Missing" if pd.isna(category)
                    else str(category)
                ),
                "count": count,
                "percentage": round(percentage, 2)
            })

    summary = pd.DataFrame(summary_rows)

    print("\nCategorical Frequency Summary:")
    print(summary.to_string(index=False))

    return summary


# ============================================================
# 5. CATEGORY STATISTICAL SUMMARY
# ============================================================

def analyze_categorical_statistics(df, features):

    print("\n" + "=" * 75)
    print("2. CATEGORICAL STATISTICAL SUMMARY")
    print("=" * 75)

    summary_rows = []

    for feature in features:

        mode_values = df[feature].mode()

        mode_value = (
            mode_values.iloc[0]
            if not mode_values.empty
            else None
        )

        summary_rows.append({
            "feature_name": feature,
            "unique_categories": df[feature].nunique(),
            "mode": mode_value,
            "missing_values": df[feature].isnull().sum()
        })

    summary = pd.DataFrame(summary_rows)

    print("\nCategorical Statistics:")
    print(summary.to_string(index=False))

    return summary


# ============================================================
# 6. CATEGORICAL COUNT PLOTS
# ============================================================

def plot_categorical_features(df, features):

    print("\nGenerating categorical count plots...")

    for feature in features:

        # Display missing values explicitly if present.
        plot_data = df[[feature]].copy()

        plot_data[feature] = (
            plot_data[feature]
            .astype("object")
            .fillna("Missing")
        )

        order = (
            plot_data[feature]
            .value_counts()
            .index
        )

        plt.figure(figsize=(9, 5))

        sns.countplot(
            data=plot_data,
            x=feature,
            order=order
        )

        plt.title(
            f"Distribution of {feature}",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel(feature)
        plt.ylabel("Number of Candidates")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{feature}_countplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 7. SAVE REPORTS
# ============================================================

def save_reports(
    frequency_summary,
    statistics_summary
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    frequency_summary.to_csv(
        REPORT_DIR / "categorical_frequency_summary.csv",
        index=False
    )

    statistics_summary.to_csv(
        REPORT_DIR / "categorical_statistics_summary.csv",
        index=False
    )

    print("\nCategorical reports saved successfully.")


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    categorical_features = identify_categorical_features(df)

    if not categorical_features:
        raise ValueError("No categorical features found.")

    frequency_summary = analyze_category_frequency(
        df,
        categorical_features
    )

    statistics_summary = analyze_categorical_statistics(
        df,
        categorical_features
    )

    plot_categorical_features(
        df,
        categorical_features
    )

    save_reports(
        frequency_summary,
        statistics_summary
    )

    print("\n" + "=" * 75)
    print("CATEGORICAL UNIVARIATE ANALYSIS COMPLETED")
    print("=" * 75)


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
