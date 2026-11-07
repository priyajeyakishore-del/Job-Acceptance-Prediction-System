
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS
# DATASET OVERVIEW
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
    / "dataset_overview"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 75)
    print("DATASET OVERVIEW")
    print("=" * 75)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nDataset loaded successfully.")

    return df


# ============================================================
# 3. DATASET SHAPE
# ============================================================

def analyze_dataset_shape(df):

    print("\n" + "=" * 75)
    print("1. DATASET SHAPE")
    print("=" * 75)

    rows, columns = df.shape

    print(f"\nTotal Rows    : {rows}")
    print(f"Total Columns : {columns}")

    return {
        "total_rows": rows,
        "total_columns": columns
    }


# ============================================================
# 4. SAMPLE RECORDS
# ============================================================

def display_sample_records(df):

    print("\n" + "=" * 75)
    print("2. SAMPLE RECORDS")
    print("=" * 75)

    print("\nFIRST 5 RECORDS")
    print("-" * 75)

    print(df.head())

    print("\nLAST 5 RECORDS")
    print("-" * 75)

    print(df.tail())


# ============================================================
# 5. COLUMN INFORMATION
# ============================================================

def analyze_column_information(df):

    print("\n" + "=" * 75)
    print("3. COLUMN INFORMATION")
    print("=" * 75)

    column_info = pd.DataFrame({
        "column_name": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "non_null_count": df.notnull().sum().values,
        "null_count": df.isnull().sum().values,
        "unique_values": df.nunique().values
    })

    print("\nColumn Information:")
    print(column_info.to_string(index=False))

    return column_info


# ============================================================
# 6. NUMERICAL SUMMARY
# ============================================================

def analyze_numerical_features(df):

    print("\n" + "=" * 75)
    print("4. NUMERICAL FEATURE SUMMARY")
    print("=" * 75)

    numerical_df = df.select_dtypes(
        include=["number"]
    )

    numerical_summary = numerical_df.describe().T

    print("\nNumerical Summary:")
    print(numerical_summary)

    return numerical_summary


# ============================================================
# 7. CATEGORICAL SUMMARY
# ============================================================

def analyze_categorical_features(df):

    print("\n" + "=" * 75)
    print("5. CATEGORICAL FEATURE SUMMARY")
    print("=" * 75)

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    if categorical_df.empty:
        print("\nNo categorical columns found.")
        return pd.DataFrame()

    categorical_summary = pd.DataFrame({
        "unique_values": categorical_df.nunique(),
        "most_frequent_value": categorical_df.mode().iloc[0],
        "most_frequent_count": categorical_df.apply(
            lambda col: col.value_counts().iloc[0]
            if not col.value_counts().empty else 0
        )
    })

    print("\nCategorical Summary:")
    print(categorical_summary)

    return categorical_summary


# ============================================================
# 8. MEMORY USAGE
# ============================================================

def analyze_memory_usage(df):

    print("\n" + "=" * 75)
    print("6. MEMORY USAGE")
    print("=" * 75)

    memory_bytes = df.memory_usage(deep=True).sum()

    memory_mb = memory_bytes / (1024 ** 2)

    print(f"\nMemory Usage: {memory_mb:.2f} MB")

    return memory_mb


# ============================================================
# 9. TARGET COLUMN INFORMATION
# ============================================================

def analyze_target_column(df):

    print("\n" + "=" * 75)
    print("7. TARGET COLUMN INFORMATION")
    print("=" * 75)

    target_column = "status"

    if target_column not in df.columns:
        print("\nTarget column 'status' not found.")
        return pd.DataFrame()

    target_summary = (
        df[target_column]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="count")
    )

    target_summary["percentage"] = (
        target_summary["count"] / len(df) * 100
    ).round(2)

    print("\nTarget Distribution:")
    print(target_summary.to_string(index=False))

    return target_summary


# ============================================================
# 10. SAVE REPORTS
# ============================================================

def save_reports(
    shape_info,
    column_info,
    numerical_summary,
    categorical_summary,
    target_summary,
    memory_mb
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Dataset shape report
    pd.DataFrame([shape_info]).to_csv(
        REPORT_DIR / "dataset_shape.csv",
        index=False
    )

    # Column information report
    column_info.to_csv(
        REPORT_DIR / "column_information.csv",
        index=False
    )

    # Numerical summary report
    numerical_summary.to_csv(
        REPORT_DIR / "numerical_summary.csv"
    )

    # Categorical summary report
    categorical_summary.to_csv(
        REPORT_DIR / "categorical_summary.csv"
    )

    # Target distribution report
    target_summary.to_csv(
        REPORT_DIR / "target_distribution.csv",
        index=False
    )

    # Memory usage report
    pd.DataFrame({
        "memory_usage_mb": [round(memory_mb, 2)]
    }).to_csv(
        REPORT_DIR / "memory_usage.csv",
        index=False
    )

    print("\n" + "=" * 75)
    print("REPORTS SAVED SUCCESSFULLY")
    print("=" * 75)

    print(f"\nReport Location: {REPORT_DIR}")


# ============================================================
# 11. MAIN FUNCTION
# ============================================================

def main():

    # Load dataset
    df = load_dataset()

    # Dataset shape
    shape_info = analyze_dataset_shape(df)

    # Sample records
    display_sample_records(df)

    # Column information
    column_info = analyze_column_information(df)

    # Numerical summary
    numerical_summary = analyze_numerical_features(df)

    # Categorical summary
    categorical_summary = analyze_categorical_features(df)

    # Memory usage
    memory_mb = analyze_memory_usage(df)

    # Target information
    target_summary = analyze_target_column(df)

    # Save all reports
    save_reports(
        shape_info,
        column_info,
        numerical_summary,
        categorical_summary,
        target_summary,
        memory_mb
    )

    print("\n" + "=" * 75)
    print("DATASET OVERVIEW COMPLETED SUCCESSFULLY")
    print("=" * 75)


# ============================================================
# 12. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
