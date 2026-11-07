
# ============================================================
# STEP 2: TRAIN-TEST SPLIT
# Project: Job Acceptance Prediction System
# ============================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 2: TRAIN-TEST SPLIT")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. VALIDATE TARGET COLUMN
# ------------------------------------------------------------

if "status" not in df.columns:
    raise ValueError("Target column 'status' not found.")

if df["status"].isnull().any():
    raise ValueError("Target column contains missing values.")

if not set(df["status"].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

print("\nTarget column validation successful!")

print("\nTarget distribution:")
print(df["status"].value_counts())


# ------------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

y = df["status"]

X = df.drop(
    columns=["status", "placement_probability_score"],
    errors="ignore"
)

print("\nFeatures and target separated successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nExcluded columns:")
print("- status (target variable)")
print("- placement_probability_score (derived score)")


# ------------------------------------------------------------
# 5. SPLIT DATASET
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed!")

print("\nTRAINING DATA")
print("-" * 40)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTESTING DATA")
print("-" * 40)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ------------------------------------------------------------
# 6. CHECK TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))


# ------------------------------------------------------------
# 7. SAVE SPLIT DATASETS
# ------------------------------------------------------------

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_frame().to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_frame().to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("TRAIN-TEST SPLIT SUMMARY")
print("=" * 65)

print("Original dataset:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])

print("\nFiles saved successfully!")
print("Output folder:", OUTPUT_DIR)

print("\nSTEP 2 COMPLETED SUCCESSFULLY!")
print("=" * 65)
