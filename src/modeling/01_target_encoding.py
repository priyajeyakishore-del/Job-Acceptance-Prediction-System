
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step7_feature_engineered.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("TARGET VARIABLE ENCODING")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)

# --------------------------------------------------
# 3. CHECK TARGET VARIABLE
# --------------------------------------------------

if "status" not in df.columns:
    raise ValueError("Target column 'status' not found!")

print("\nOriginal target distribution:")
print(df["status"].value_counts(dropna=False))

# --------------------------------------------------
# 4. ENCODE TARGET VARIABLE
# --------------------------------------------------

target_mapping = {
    "Placed": 1,
    "Not Placed": 0
}

df["status"] = df["status"].map(target_mapping)

# Check for invalid or missing target values
if df["status"].isnull().any():
    raise ValueError(
        "Unexpected or missing target values found. "
        "Please inspect the original status column."
    )

df["status"] = df["status"].astype("int8")

# --------------------------------------------------
# 5. VERIFY ENCODING
# --------------------------------------------------

print("\nEncoded target distribution:")
print(df["status"].value_counts())

print("\nTarget percentages:")
print(
    (df["status"].value_counts(normalize=True) * 100)
    .round(2)
)

# --------------------------------------------------
# 6. SAVE MODELING DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nFinal dataset shape:", df.shape)
print("\nTarget encoding completed successfully!")
print("Saved file:", OUTPUT_PATH)
