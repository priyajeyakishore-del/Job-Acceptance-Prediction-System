
# ============================================================
# STEP 3: DATA PREPROCESSING
# Project: Job Acceptance Prediction System
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD TRAINING AND TESTING DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: DATA PREPROCESSING")
print("=" * 70)

X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
X_test = pd.read_csv(SPLIT_DIR / "X_test.csv")

y_train = pd.read_csv(SPLIT_DIR / "y_train.csv")
y_test = pd.read_csv(SPLIT_DIR / "y_test.csv")

print("\nDatasets loaded successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ------------------------------------------------------------
# 3. VALIDATE TRAINING AND TESTING FEATURES
# ------------------------------------------------------------

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing datasets have different columns."
    )

if X_train.empty or X_test.empty:
    raise ValueError("Training or testing dataset is empty.")

if y_train.shape[1] != 1 or y_test.shape[1] != 1:
    raise ValueError("Target files must contain exactly one column.")

print("\nFeature validation successful!")


# ------------------------------------------------------------
# 4. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Identify binary numerical columns, such as missing flags.
binary_columns = [
    col for col in numerical_columns
    if set(X_train[col].dropna().unique()).issubset({0, 1})
]

# Remaining numerical columns are continuous numerical features.
continuous_columns = [
    col for col in numerical_columns
    if col not in binary_columns
]

print("\nFEATURE CLASSIFICATION")
print("-" * 50)

print("Continuous numerical features:", len(continuous_columns))
print("Binary numerical features:", len(binary_columns))
print("Categorical features:", len(categorical_columns))

print("\nContinuous columns:")
print(continuous_columns)

print("\nBinary columns:")
print(binary_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ------------------------------------------------------------
# 5. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ------------------------------------------------------------
# 6. BINARY FEATURE PREPROCESSING
# ------------------------------------------------------------

binary_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ]
)


# ------------------------------------------------------------
# 7. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------------------------
# 8. COMBINE PREPROCESSING PIPELINES
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, continuous_columns),
        ("binary", binary_pipeline, binary_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ],
    remainder="drop"
)


# ------------------------------------------------------------
# 9. FIT ON TRAINING DATA AND TRANSFORM TRAINING DATA
# ------------------------------------------------------------

print("\nFitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

print("Training data preprocessing completed!")


# ------------------------------------------------------------
# 10. TRANSFORM TESTING DATA
# ------------------------------------------------------------

print("\nTransforming testing data...")

X_test_processed = preprocessor.transform(X_test)

print("Testing data preprocessing completed!")


# ------------------------------------------------------------
# 11. GET PROCESSED FEATURE NAMES
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# ------------------------------------------------------------
# 12. SAVE PREPROCESSED DATA
# ------------------------------------------------------------

X_train_processed_df.to_csv(
    OUTPUT_DIR / "X_train_processed.csv",
    index=False
)

X_test_processed_df.to_csv(
    OUTPUT_DIR / "X_test_processed.csv",
    index=False
)

# Save target files unchanged.
y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# ------------------------------------------------------------
# 13. SAVE FITTED PREPROCESSING PIPELINE
# ------------------------------------------------------------

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessing_pipeline.pkl"
)

print("\nPreprocessing pipeline saved successfully!")


# ------------------------------------------------------------
# 14. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

print("Original training shape:", X_train.shape)
print("Original testing shape:", X_test.shape)

print("Processed training shape:", X_train_processed_df.shape)
print("Processed testing shape:", X_test_processed_df.shape)

print("\nMissing values in processed training data:",
      X_train_processed_df.isnull().sum().sum())

print("Missing values in processed testing data:",
      X_test_processed_df.isnull().sum().sum())

print("\nOutput directory:", OUTPUT_DIR)
print("Saved preprocessing pipeline:",
      MODEL_DIR / "preprocessing_pipeline.pkl")

print("\nSTEP 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)
