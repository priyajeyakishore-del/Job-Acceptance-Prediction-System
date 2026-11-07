
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# MODULE: MISSING VALUE ANALYSIS
# PURPOSE:
# 1. Analyze numerical feature symmetry and outliers.
# 2. Recommend mean or median for numerical features.
# 3. Analyze categorical feature distributions.
# 4. Recommend mode or Unknown for categorical features.
# 5. Save analysis reports and graphs.
#
# IMPORTANT:
# This module only analyzes data.
# It does NOT fill missing values.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step3_missing_indicators.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_cleaning"
    / "missing_value_analysis"
)

GRAPH_DIR = PROJECT_ROOT / "output" / "graphs" / "missing_value_analysis"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("MISSING VALUE ANALYSIS")
print("=" * 75)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Run missing_indicators.py first."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. IDENTIFY TARGET, FLAGS AND FEATURE COLUMNS
# ============================================================

TARGET_COLUMN = "status"

# Missingness indicators must not be analyzed as original features.
flag_columns = [
    column
    for column in df.columns
    if column.endswith("_missing_flag")
]

# Exclude the prediction target from imputation analysis.
feature_columns = [
    column
    for column in df.columns
    if column != TARGET_COLUMN
    and column not in flag_columns
]

# Separate numerical and categorical features.
numeric_columns = df[feature_columns].select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df[feature_columns].select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

print("\nNumerical feature count:", len(numeric_columns))
print("Categorical feature count:", len(categorical_columns))
print("Missingness flag count:", len(flag_columns))


# ============================================================
# 4. NUMERICAL FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("NUMERICAL FEATURE ANALYSIS")
print("=" * 75)

numeric_results = []

for column in numeric_columns:

    series = pd.to_numeric(df[column], errors="coerce")
    valid_values = series.dropna()

    missing_count = int(series.isna().sum())
    missing_percentage = (
        missing_count / len(df) * 100 if len(df) else 0
    )

    if valid_values.empty:
        numeric_results.append({
            "Feature": column,
            "Missing_Count": missing_count,
            "Missing_Percentage": round(missing_percentage, 2),
            "Mean": np.nan,
            "Median": np.nan,
            "Standard_Deviation": np.nan,
            "Skewness": np.nan,
            "Lower_IQR_Bound": np.nan,
            "Upper_IQR_Bound": np.nan,
            "Outlier_Count": 0,
            "Outlier_Percentage": np.nan,
            "Recommended_Method": "Review: no valid values",
            "Reason": "No non-missing numerical values."
        })
        continue

    mean_value = valid_values.mean()
    median_value = valid_values.median()
    std_value = valid_values.std()
    skewness = valid_values.skew()

    # --------------------------------------------------------
    # Detect potential outliers using the IQR method
    # --------------------------------------------------------

    q1 = valid_values.quantile(0.25)
    q3 = valid_values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (valid_values < lower_bound)
        | (valid_values > upper_bound)
    )

    outlier_count = int(outlier_mask.sum())

    outlier_percentage = (
        outlier_count / len(valid_values) * 100
    )

    # --------------------------------------------------------
    # Recommend a baseline imputation method
    #
    # These thresholds are screening rules, not universal laws.
    # --------------------------------------------------------

    if pd.isna(skewness):
        recommended_method = "Review distribution"
        reason = "Insufficient variation to assess skewness."

    elif abs(skewness) >= 1 or outlier_percentage >= 5:
        recommended_method = "Median (candidate)"
        reason = (
            "High skewness or a relatively high IQR-outlier "
            "percentage; median is more robust."
        )

    else:
        recommended_method = "Mean (candidate)"
        reason = (
            "Skewness and IQR-outlier percentage are not high "
            "under the screening rules; verify the histogram."
        )

    numeric_results.append({
        "Feature": column,
        "Missing_Count": missing_count,
        "Missing_Percentage": round(missing_percentage, 2),
        "Mean": round(mean_value, 4),
        "Median": round(median_value, 4),
        "Standard_Deviation": round(std_value, 4)
        if pd.notna(std_value) else np.nan,
        "Skewness": round(skewness, 4)
        if pd.notna(skewness) else np.nan,
        "Lower_IQR_Bound": round(lower_bound, 4),
        "Upper_IQR_Bound": round(upper_bound, 4),
        "Outlier_Count": outlier_count,
        "Outlier_Percentage": round(outlier_percentage, 2),
        "Recommended_Method": recommended_method,
        "Reason": reason
    })

    print(f"\nFeature: {column}")
    print("Missing values:", missing_count)
    print("Mean:", round(mean_value, 3))
    print("Median:", round(median_value, 3))
    print("Skewness:", round(skewness, 3)
          if pd.notna(skewness) else "Not available")
    print("Potential IQR outliers:", outlier_count)
    print("Recommended method:", recommended_method)

numeric_report = pd.DataFrame(numeric_results)

numeric_report.to_csv(
    REPORT_DIR / "numerical_missing_value_analysis.csv",
    index=False
)

print("\nNumerical report saved.")


# ============================================================
# 5. NUMERICAL HISTOGRAMS AND BOXPLOTS
# ============================================================

print("\nGenerating numerical graphs...")

for column in numeric_columns:

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        continue

    # Histogram for distribution shape
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    safe_name = "".join(
        char if char.isalnum() or char in "_-" else "_"
        for char in column
    )

    fig.savefig(
        GRAPH_DIR / f"{safe_name}_histogram.png",
        dpi=150
    )
    plt.close(fig)

    # Boxplot for potential outliers
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot(values, orientation="horizontal")
    ax.set_title(f"Boxplot of {column}")
    ax.set_xlabel(column)
    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / f"{safe_name}_boxplot.png",
        dpi=150
    )
    plt.close(fig)

print("Numerical graphs saved.")


# ============================================================
# 6. CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 75)

categorical_results = []
frequency_results = []

for column in categorical_columns:

    series = df[column]

    missing_count = int(series.isna().sum())
    missing_percentage = (
        missing_count / len(df) * 100 if len(df) else 0
    )

    # Exclude missing values when calculating category frequencies.
    frequencies = series.dropna().value_counts()
    non_missing_count = int(frequencies.sum())

    unique_count = int(series.nunique(dropna=True))

    if non_missing_count > 0:
        top_category = frequencies.index[0]
        top_category_count = int(frequencies.iloc[0])
        top_category_percentage = (
            top_category_count / non_missing_count * 100
        )
    else:
        top_category = None
        top_category_count = 0
        top_category_percentage = 0

    # --------------------------------------------------------
    # Tentative recommendation
    #
    # Missingness percentage and category concentration alone
    # cannot establish why values are missing.
    # --------------------------------------------------------

    if missing_count == 0:
        recommended_method = "No imputation needed"
        reason = "No missing values found."

    elif (
        missing_percentage <= 5
        and top_category_percentage >= 60
    ):
        recommended_method = "Mode (candidate)"
        reason = (
            "Low missingness and a dominant category; "
            "use mode only if missingness is incidental."
        )

    else:
        recommended_method = "Unknown (candidate)"
        reason = (
            "Preserve missingness as a separate category "
            "when the reason for non-response is unclear."
        )

    categorical_results.append({
        "Feature": column,
        "Missing_Count": missing_count,
        "Missing_Percentage": round(missing_percentage, 2),
        "Unique_Category_Count": unique_count,
        "Most_Frequent_Category": top_category,
        "Most_Frequent_Count": top_category_count,
        "Most_Frequent_Percentage": round(
            top_category_percentage, 2
        ),
        "Recommended_Method": recommended_method,
        "Reason": reason
    })

    # Save every category's frequency for detailed inspection.
    for category, count in frequencies.items():
        frequency_results.append({
            "Feature": column,
            "Category": category,
            "Count": int(count),
            "Percentage_of_Non_Missing": round(
                count / non_missing_count * 100, 2
            ) if non_missing_count else 0
        })

    print(f"\nFeature: {column}")
    print("Missing values:", missing_count)
    print("Unique categories:", unique_count)
    print("Most frequent category:", top_category)
    print("Recommended method:", recommended_method)

categorical_report = pd.DataFrame(categorical_results)

categorical_report.to_csv(
    REPORT_DIR / "categorical_missing_value_analysis.csv",
    index=False
)

frequency_report = pd.DataFrame(frequency_results)

frequency_report.to_csv(
    REPORT_DIR / "categorical_value_frequencies.csv",
    index=False
)

print("\nCategorical reports saved.")


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("MISSING VALUE ANALYSIS COMPLETED")
print("=" * 75)

print("\nReports folder:")
print(REPORT_DIR)

print("\nGenerated files:")
print("- numerical_missing_value_analysis.csv")
print("- categorical_missing_value_analysis.csv")
print("- categorical_value_frequencies.csv")
print("- graphs/ (histograms and boxplots)")

print("\nNo missing values were filled by this module.")
print("Review the recommendations before running imputation.")
