
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# MULTIVARIATE ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


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
    / "multivariate_analysis"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "graphs"
    / "multivariate_analysis"
)

TARGET_COLUMN = "status"


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 80)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 80)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nDataset loaded successfully.")

    print("Dataset Shape:", df.shape)

    print("\nFirst Five Records:")

    print(df.head())

    return df


# ============================================================
# 3. IDENTIFY NUMERICAL FEATURES
# ============================================================

def identify_numerical_features(df):

    numerical_features = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Exclude target if numeric.
    numerical_features = [
        feature
        for feature in numerical_features
        if feature != TARGET_COLUMN
    ]

    print("\nTotal Numerical Features:", len(numerical_features))

    print("\nNumerical Features:")

    for feature in numerical_features:

        print("-", feature)

    return numerical_features


# ============================================================
# 4. CORRELATION MATRIX
# ============================================================

def calculate_correlation_matrix(df, features):

    print("\n" + "=" * 80)
    print("1. CORRELATION MATRIX")
    print("=" * 80)

    correlation_matrix = df[features].corr(
        method="pearson"
    )

    print("\nCorrelation Matrix:")

    print(correlation_matrix.round(2).to_string())

    return correlation_matrix


# ============================================================
# 5. IDENTIFY HIGHLY CORRELATED FEATURES
# ============================================================

def identify_high_correlations(
    correlation_matrix,
    threshold=0.80
):

    print("\n" + "=" * 80)
    print("2. HIGH CORRELATION ANALYSIS")
    print("=" * 80)

    high_correlations = []

    # Check only the upper triangle to avoid duplicate pairs.
    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    for column in upper_triangle.columns:

        for index in upper_triangle.index:

            correlation = upper_triangle.loc[index, column]

            if pd.notna(correlation) and abs(correlation) >= threshold:

                high_correlations.append({

                    "feature_1": index,

                    "feature_2": column,

                    "correlation": correlation,

                    "correlation_strength": (
                        "Strong positive"
                        if correlation > 0
                        else "Strong negative"
                    )

                })

    results = pd.DataFrame(
        high_correlations,
        columns=[
            "feature_1",
            "feature_2",
            "correlation",
            "correlation_strength"
        ]
    )

    if results.empty:

        print("\nNo feature pairs exceeded the correlation threshold.")

    else:

        print("\nHighly Correlated Feature Pairs:")

        print(results.round(3).to_string(index=False))

    return results


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(correlation_matrix):

    print("\nGenerating correlation heatmap...")

    plt.figure(figsize=(18, 14))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title(
        "Numerical Features Correlation Heatmap",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. ACADEMIC FEATURE CORRELATION
# ============================================================

def plot_academic_correlation(df):

    print("\nGenerating academic correlation heatmap...")

    academic_features = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage"
    ]

    available_features = [
        feature
        for feature in academic_features
        if feature in df.columns
    ]

    if len(available_features) < 2:

        print("Insufficient academic features available.")

        return

    academic_corr = df[available_features].corr()

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        academic_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title(
        "Academic Features Correlation",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "academic_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. IMPORTANT FEATURES PAIR PLOT
# ============================================================

def plot_important_features_pairplot(df):

    print("\nGenerating important features pair plot...")

    important_features = [

        "ssc_percentage",

        "hsc_percentage",

        "degree_percentage",

        "technical_score",

        "aptitude_score",

        "communication_score",

        "skills_match_percentage"

    ]

    available_features = [
        feature
        for feature in important_features
        if feature in df.columns
    ]

    if len(available_features) < 2:

        print("Insufficient features for pair plot.")

        return

    # Sample rows to keep pair plot manageable.
    sample_size = min(1000, len(df))

    sample_df = df[
        available_features + [TARGET_COLUMN]
    ].dropna().sample(
        n=min(
            sample_size,
            len(
                df[
                    available_features + [TARGET_COLUMN]
                ].dropna()
            )
        ),
        random_state=42
    )

    pair_plot = sns.pairplot(
        sample_df,
        vars=available_features,
        hue=TARGET_COLUMN,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.5, "s": 15}
    )

    pair_plot.fig.suptitle(
        "Relationships Among Important Numerical Features",
        y=1.02,
        fontsize=15,
        fontweight="bold"
    )

    pair_plot.savefig(
        GRAPH_DIR / "important_features_pairplot.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(pair_plot.fig)


# ============================================================
# 9. VARIANCE INFLATION FACTOR (VIF)
# ============================================================

def calculate_vif(df, features):

    print("\n" + "=" * 80)
    print("3. MULTICOLLINEARITY ANALYSIS - VIF")
    print("=" * 80)

    # VIF is calculated using predictor variables only.
    # Exclude the manually derived placement score because
    # it is a weighted combination of other predictors.
    vif_features = [
        feature
        for feature in features
        if feature != "placement_probability_score"
    ]

    # Remove constant or all-missing columns.
    numeric_data = df[vif_features].copy()

    numeric_data = numeric_data.dropna()

    variable_features = [
        feature
        for feature in numeric_data.columns
        if numeric_data[feature].nunique() > 1
    ]

    numeric_data = numeric_data[variable_features]

    if numeric_data.shape[1] < 2:

        print("Insufficient varying numerical features for VIF.")

        return pd.DataFrame(
            columns=["feature", "vif"]
        )

    # Standardization improves numerical stability.
    means = numeric_data.mean()

    standard_deviations = numeric_data.std()

    numeric_data = (
        numeric_data - means
    ) / standard_deviations

    numeric_data = add_constant(
        numeric_data,
        has_constant="add"
    )

    vif_results = []

    for index, feature in enumerate(numeric_data.columns):

        if feature == "const":

            continue

        try:

            vif_value = variance_inflation_factor(
                numeric_data.values,
                index
            )

        except Exception:

            vif_value = np.nan

        vif_results.append({

            "feature": feature,

            "vif": vif_value

        })

    vif_df = pd.DataFrame(vif_results)

    vif_df = vif_df.sort_values(
        by="vif",
        ascending=False
    )

    def interpret_vif(value):

        if pd.isna(value):

            return "Unable to calculate"

        elif np.isinf(value):

            return "Perfect multicollinearity"

        elif value >= 10:

            return "High multicollinearity"

        elif value >= 5:

            return "Moderate multicollinearity"

        else:

            return "Low multicollinearity"

    vif_df["interpretation"] = vif_df["vif"].apply(
        interpret_vif
    )

    print("\nVIF Results:")

    print(vif_df.round(3).to_string(index=False))

    return vif_df


# ============================================================
# 10. SAVE REPORTS
# ============================================================

def save_reports(
    correlation_matrix,
    high_correlation_pairs,
    vif_results
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    correlation_matrix.to_csv(
        REPORT_DIR / "correlation_matrix.csv"
    )

    high_correlation_pairs.to_csv(
        REPORT_DIR / "high_correlation_pairs.csv",
        index=False
    )

    vif_results.to_csv(
        REPORT_DIR / "vif_analysis.csv",
        index=False
    )

    print("\nMultivariate reports saved successfully.")


# ============================================================
# 11. MAIN FUNCTION
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

    numerical_features = identify_numerical_features(df)

    if len(numerical_features) < 2:

        raise ValueError(
            "At least two numerical features are required."
        )

    correlation_matrix = calculate_correlation_matrix(
        df,
        numerical_features
    )

    high_correlation_pairs = identify_high_correlations(
        correlation_matrix,
        threshold=0.80
    )

    plot_correlation_heatmap(
        correlation_matrix
    )

    plot_academic_correlation(df)

    plot_important_features_pairplot(df)

    vif_results = calculate_vif(
        df,
        numerical_features
    )

    save_reports(
        correlation_matrix,
        high_correlation_pairs,
        vif_results
    )

    print("\n" + "=" * 80)
    print("MULTIVARIATE ANALYSIS COMPLETED")
    print("=" * 80)


# ============================================================
# 12. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
