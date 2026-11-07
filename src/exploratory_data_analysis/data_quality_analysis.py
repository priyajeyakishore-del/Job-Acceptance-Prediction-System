
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# DATA QUALITY ANALYSIS
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
    / "step7_feature_engineered.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "reports"
    / "data_quality_analysis"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 75)
    print("DATA QUALITY ANALYSIS")
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
# 3. MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(df):

    print("\n" + "=" * 75)
    print("1. MISSING VALUE ANALYSIS")
    print("=" * 75)

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df) * 100
    ).round(2)

    missing_summary = pd.DataFrame({
        "column_name": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_percentage.values
    })

    missing_summary = missing_summary.sort_values(
        by="missing_count",
        ascending=False
    )

    print("\nMissing Value Summary:")
    print(missing_summary.to_string(index=False))

    total_missing = missing_count.sum()

    print("\nTotal Missing Values:", total_missing)

    return missing_summary


# ============================================================
# 4. DUPLICATE RECORD ANALYSIS
# ============================================================

def analyze_duplicates(df):

    print("\n" + "=" * 75)
    print("2. DUPLICATE RECORD ANALYSIS")
    print("=" * 75)

    duplicate_count = df.duplicated().sum()

    duplicate_percentage = (
        duplicate_count / len(df) * 100
    ).round(2)

    print("\nTotal Duplicate Rows:", duplicate_count)

    print(
        "Duplicate Percentage:",
        duplicate_percentage,
        "%"
    )

    duplicate_summary = pd.DataFrame({
        "total_rows": [len(df)],
        "duplicate_rows": [duplicate_count],
        "duplicate_percentage": [duplicate_percentage]
    })

    return duplicate_summary


# ============================================================
# 5. UNIQUE VALUE ANALYSIS
# ============================================================

def analyze_unique_values(df):

    print("\n" + "=" * 75)
    print("3. UNIQUE VALUE ANALYSIS")
    print("=" * 75)

    unique_summary = pd.DataFrame({
        "column_name": df.columns,
        "unique_values": [
            df[column].nunique(dropna=False)
            for column in df.columns
        ]
    })

    print("\nUnique Value Summary:")
    print(unique_summary.to_string(index=False))

    return unique_summary


# ============================================================
# 6. CONSTANT COLUMN ANALYSIS
# ============================================================

def analyze_constant_columns(df):

    print("\n" + "=" * 75)
    print("4. CONSTANT COLUMN ANALYSIS")
    print("=" * 75)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:

        print("\nConstant Columns Found:")

        for column in constant_columns:
            print("-", column)

    else:

        print("\nNo constant columns found.")

    constant_summary = pd.DataFrame({
        "constant_column": constant_columns
    })

    return constant_summary


# ============================================================
# 7. DATA TYPE ANALYSIS
# ============================================================

def analyze_data_types(df):

    print("\n" + "=" * 75)
    print("5. DATA TYPE ANALYSIS")
    print("=" * 75)

    data_type_summary = pd.DataFrame({
        "column_name": df.columns,
        "data_type": df.dtypes.astype(str).values
    })

    data_type_summary["column_category"] = (
        data_type_summary["data_type"].apply(
            lambda dtype:
            "Numerical"
            if dtype in ["int64", "float64"]
            else (
                "Categorical"
                if dtype in ["object", "category", "bool"]
                else "Other"
            )
        )
    )

    print("\nData Type Summary:")
    print(data_type_summary.to_string(index=False))

    print("\nColumn Category Counts:")
    print(
        data_type_summary["column_category"]
        .value_counts()
    )

    return data_type_summary


# ============================================================
# 8. OVERALL DATA QUALITY SUMMARY
# ============================================================

def create_quality_summary(
    df,
    missing_summary,
    duplicate_summary,
    constant_summary
):

    print("\n" + "=" * 75)
    print("6. OVERALL DATA QUALITY SUMMARY")
    print("=" * 75)

    total_missing = missing_summary[
        "missing_count"
    ].sum()

    total_duplicates = duplicate_summary[
        "duplicate_rows"
    ].iloc[0]

    total_constant_columns = len(constant_summary)

    quality_summary = pd.DataFrame({
        "metric": [
            "Total Rows",
            "Total Columns",
            "Total Missing Values",
            "Total Duplicate Rows",
            "Total Constant Columns"
        ],
        "value": [
            len(df),
            len(df.columns),
            total_missing,
            total_duplicates,
            total_constant_columns
        ]
    })

    print("\n", quality_summary.to_string(index=False))

    return quality_summary


# ============================================================
# 9. SAVE REPORTS
# ============================================================

def save_reports(
    missing_summary,
    duplicate_summary,
    unique_summary,
    constant_summary,
    data_type_summary,
    quality_summary
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    missing_summary.to_csv(
        REPORT_DIR / "missing_value_summary.csv",
        index=False
    )

    duplicate_summary.to_csv(
        REPORT_DIR / "duplicate_summary.csv",
        index=False
    )

    unique_summary.to_csv(
        REPORT_DIR / "unique_value_summary.csv",
        index=False
    )

    constant_summary.to_csv(
        REPORT_DIR / "constant_column_summary.csv",
        index=False
    )

    data_type_summary.to_csv(
        REPORT_DIR / "data_type_summary.csv",
        index=False
    )

    quality_summary.to_csv(
        REPORT_DIR / "data_quality_summary.csv",
        index=False
    )

    print("\n" + "=" * 75)
    print("ALL DATA QUALITY REPORTS SAVED")
    print("=" * 75)

    print("\nReport Location:")
    print(REPORT_DIR)


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():

    # Load dataset
    df = load_dataset()

    # Missing value analysis
    missing_summary = analyze_missing_values(df)

    # Duplicate analysis
    duplicate_summary = analyze_duplicates(df)

    # Unique value analysis
    unique_summary = analyze_unique_values(df)

    # Constant column analysis
    constant_summary = analyze_constant_columns(df)

    # Data type analysis
    data_type_summary = analyze_data_types(df)

    # Overall quality summary
    quality_summary = create_quality_summary(
        df,
        missing_summary,
        duplicate_summary,
        constant_summary
    )

    # Save reports
    save_reports(
        missing_summary,
        duplicate_summary,
        unique_summary,
        constant_summary,
        data_type_summary,
        quality_summary
    )

    print("\n" + "=" * 75)
    print("DATA QUALITY ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 75)


# ============================================================
# 11. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
