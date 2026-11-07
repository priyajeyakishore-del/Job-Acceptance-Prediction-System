
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# BIVARIATE CATEGORICAL ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import chi2_contingency


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
    / "bivariate_categorical_analysis"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "graphs"
    / "bivariate_categorical_analysis"
)

TARGET_COLUMN = "status"

# Expected target categories
PLACED_LABEL = "Placed"
NOT_PLACED_LABEL = "Not Placed"


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 80)
    print("BIVARIATE CATEGORICAL ANALYSIS")
    print("=" * 80)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype("string")
        .str.strip()
    )

    print("\nDataset loaded successfully.")

    print("Dataset Shape:", df.shape)

    print("\nTarget Distribution:")

    print(df[TARGET_COLUMN].value_counts(dropna=False))

    return df


# ============================================================
# 3. IDENTIFY CATEGORICAL FEATURES
# ============================================================

def identify_categorical_features(df):

    categorical_features = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Exclude target from independent features.
    categorical_features = [
        feature
        for feature in categorical_features
        if feature != TARGET_COLUMN
    ]

    print(
        "\nTotal Categorical Features:",
        len(categorical_features)
    )

    print("\nCategorical Features:")

    for feature in categorical_features:

        print("-", feature)

    return categorical_features


# ============================================================
# 4. CROSS-TABULATION ANALYSIS
# ============================================================

def analyze_cross_tabulation(df, features):

    print("\n" + "=" * 80)
    print("1. CROSS-TABULATION ANALYSIS")
    print("=" * 80)

    all_results = []

    for feature in features:

        # Exclude rows with missing feature or target values.
        valid_data = df[
            [feature, TARGET_COLUMN]
        ].dropna()

        # Count candidates in each category by placement status.
        cross_tab = pd.crosstab(
            valid_data[feature],
            valid_data[TARGET_COLUMN]
        )

        # Ensure both target columns exist.
        for label in [PLACED_LABEL, NOT_PLACED_LABEL]:

            if label not in cross_tab.columns:

                cross_tab[label] = 0

        cross_tab = cross_tab[
            [PLACED_LABEL, NOT_PLACED_LABEL]
        ]

        print(f"\nFeature: {feature}")

        print(cross_tab)

        # Convert cross-tabulation into report format.
        for category, row in cross_tab.iterrows():

            total_candidates = row.sum()

            placed_count = row[PLACED_LABEL]

            not_placed_count = row[NOT_PLACED_LABEL]

            placement_rate = (
                placed_count / total_candidates * 100
                if total_candidates > 0
                else 0
            )

            all_results.append({

                "feature": feature,

                "category": str(category),

                "total_candidates": total_candidates,

                "placed_count": placed_count,

                "not_placed_count": not_placed_count,

                "placement_rate_percentage": round(
                    placement_rate, 2
                )

            })

    return pd.DataFrame(all_results)


# ============================================================
# 5. CHI-SQUARE TEST AND CRAMER'S V
# ============================================================

def perform_chi_square_tests(df, features):

    print("\n" + "=" * 80)
    print("2. CHI-SQUARE TEST OF INDEPENDENCE")
    print("=" * 80)

    results = []

    for feature in features:

        valid_data = df[
            [feature, TARGET_COLUMN]
        ].dropna()

        contingency_table = pd.crosstab(
            valid_data[feature],
            valid_data[TARGET_COLUMN]
        )

        # Remove categories or target columns with zero totals.
        contingency_table = contingency_table.loc[
            contingency_table.sum(axis=1) > 0,
            contingency_table.sum(axis=0) > 0
        ]

        rows, columns = contingency_table.shape

        # Chi-square requires at least 2 rows and 2 columns.
        if rows < 2 or columns < 2:

            results.append({

                "feature": feature,

                "chi_square_statistic": np.nan,

                "degrees_of_freedom": np.nan,

                "p_value": np.nan,

                "cramers_v": np.nan,

                "minimum_expected_frequency": np.nan,

                "result": "Insufficient categories"

            })

            continue

        chi2, p_value, degrees_of_freedom, expected = (
            chi2_contingency(
                contingency_table,
                correction=False
            )
        )

        # Calculate Cramer's V.
        total = contingency_table.to_numpy().sum()

        denominator = total * min(rows - 1, columns - 1)

        cramers_v = (
            np.sqrt(chi2 / denominator)
            if denominator > 0
            else np.nan
        )

        minimum_expected = expected.min()

        if p_value < 0.05:

            interpretation = "Statistically significant association"

        else:

            interpretation = "No statistically significant association"

        results.append({

            "feature": feature,

            "chi_square_statistic": chi2,

            "degrees_of_freedom": degrees_of_freedom,

            "p_value": p_value,

            "cramers_v": cramers_v,

            "minimum_expected_frequency": minimum_expected,

            "result": interpretation

        })

    summary = pd.DataFrame(results)

    print("\nChi-Square Test Results:")

    print(summary.round(4).to_string(index=False))

    return summary


# ============================================================
# 6. STACKED BAR CHARTS
# ============================================================

def plot_stacked_bar_charts(df, features):

    print("\nGenerating stacked bar charts...")

    for feature in features:

        valid_data = df[
            [feature, TARGET_COLUMN]
        ].dropna()

        cross_tab = pd.crosstab(
            valid_data[feature],
            valid_data[TARGET_COLUMN]
        )

        for label in [PLACED_LABEL, NOT_PLACED_LABEL]:

            if label not in cross_tab.columns:

                cross_tab[label] = 0

        cross_tab = cross_tab[
            [PLACED_LABEL, NOT_PLACED_LABEL]
        ]

        if cross_tab.empty:
            continue

        cross_tab.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 6)
        )

        plt.title(
            f"{feature} vs Placement Status",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel(feature)

        plt.ylabel("Number of Candidates")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.legend(
            title="Placement Status"
        )

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{feature}_vs_status_stacked.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 7. PLACEMENT RATE BAR CHARTS
# ============================================================

def plot_placement_rate(df, features):

    print("\nGenerating placement-rate charts...")

    for feature in features:

        valid_data = df[
            [feature, TARGET_COLUMN]
        ].dropna()

        rate_data = (
            valid_data
            .groupby(feature, observed=True)[TARGET_COLUMN]
            .apply(
                lambda values: (
                    values == PLACED_LABEL
                ).mean() * 100
            )
            .sort_values(ascending=False)
        )

        if rate_data.empty:
            continue

        plt.figure(figsize=(10, 6))

        sns.barplot(
            x=rate_data.index.astype(str),
            y=rate_data.values
        )

        plt.title(
            f"Placement Rate by {feature}",
            fontsize=13,
            fontweight="bold"
        )

        plt.xlabel(feature)

        plt.ylabel("Placement Rate (%)")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.ylim(0, 100)

        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{feature}_placement_rate.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 8. SAVE REPORTS
# ============================================================

def save_reports(
    cross_tabulation_summary,
    chi_square_summary
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cross_tabulation_summary.to_csv(
        REPORT_DIR / "categorical_placement_comparison.csv",
        index=False
    )

    chi_square_summary.to_csv(
        REPORT_DIR / "categorical_chi_square_test.csv",
        index=False
    )

    print("\nCategorical reports saved successfully.")


# ============================================================
# 9. MAIN FUNCTION
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_dataset()

    categorical_features = identify_categorical_features(df)

    if not categorical_features:

        raise ValueError(
            "No categorical features found."
        )

    cross_tabulation_summary = analyze_cross_tabulation(
        df,
        categorical_features
    )

    chi_square_summary = perform_chi_square_tests(
        df,
        categorical_features
    )

    plot_stacked_bar_charts(
        df,
        categorical_features
    )

    plot_placement_rate(
        df,
        categorical_features
    )

    save_reports(
        cross_tabulation_summary,
        chi_square_summary
    )

    print("\n" + "=" * 80)
    print("BIVARIATE CATEGORICAL ANALYSIS COMPLETED")
    print("=" * 80)


# ============================================================
# 10. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
