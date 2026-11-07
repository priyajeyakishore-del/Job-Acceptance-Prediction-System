
# ============================================================
# STEP 5: FEATURE SELECTION
# Project: Job Acceptance Prediction System
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

PREPROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports" / "machine_learning" / "feature_seloection"
GRAPH_DIR = REPORT_DIR / "graphs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD TRAINING DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5: FEATURE SELECTION")
print("=" * 70)

X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")

y_train = pd.read_csv(SPLIT_DIR / "y_train.csv")["status"]

X_train_processed = pd.read_csv(
    PREPROCESSED_DIR / "X_train_processed.csv"
)

preprocessor = joblib.load(
    MODEL_DIR / "preprocessing_pipeline.pkl"
)

print("\nTraining datasets loaded successfully!")

print("Original X_train shape:", X_train.shape)
print("Processed X_train shape:", X_train_processed.shape)
print("y_train shape:", y_train.shape)


# ------------------------------------------------------------
# 3. VALIDATE DATA
# ------------------------------------------------------------

if len(X_train) != len(y_train):
    raise ValueError("Training features and target row counts differ.")

if len(X_train_processed) != len(y_train):
    raise ValueError("Processed features and target row counts differ.")

if X_train_processed.isnull().any().any():
    raise ValueError("Processed training data contains missing values.")

if X_train_processed.columns.duplicated().any():
    raise ValueError("Processed feature names contain duplicates.")

if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

print("\nData validation successful!")


# ------------------------------------------------------------
# 4. IDENTIFY CONSTANT FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CONSTANT FEATURE ANALYSIS")
print("=" * 70)

unique_counts = X_train.nunique(dropna=False)

constant_features = unique_counts[
    unique_counts <= 1
].index.tolist()

constant_report = pd.DataFrame({
    "Feature": constant_features,
    "Unique_Values": [
        unique_counts[col] for col in constant_features
    ]
})

constant_report.to_csv(
    REPORT_DIR / "constant_features.csv",
    index=False
)

if constant_features:
    print("\nConstant features found:")
    print(constant_report.to_string(index=False))
else:
    print("\nNo constant features found.")

# We report constant features but do not automatically remove them.


# ------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERICAL CORRELATION ANALYSIS")
print("=" * 70)

numerical_data = X_train.select_dtypes(include=["number"])

correlation_matrix = numerical_data.corr()

correlation_threshold = 0.90

correlation_pairs = []

columns = correlation_matrix.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        correlation_value = correlation_matrix.iloc[i, j]

        if abs(correlation_value) >= correlation_threshold:

            correlation_pairs.append({
                "Feature_1": columns[i],
                "Feature_2": columns[j],
                "Correlation": round(correlation_value, 4)
            })

correlation_report = pd.DataFrame(
    correlation_pairs,
    columns=["Feature_1", "Feature_2", "Correlation"]
)

correlation_report.to_csv(
    REPORT_DIR / "high_correlation_pairs.csv",
    index=False
)

if not correlation_report.empty:
    print("\nHighly correlated feature pairs:")
    print(correlation_report.to_string(index=False))
else:
    print("\nNo highly correlated numerical feature pairs found.")

print("\nCorrelation threshold:", correlation_threshold)

# High correlation is a warning for possible redundancy.
# We will not automatically delete either feature.


# ------------------------------------------------------------
# 6. MAP PROCESSED FEATURES TO ORIGINAL FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MAPPING PROCESSED FEATURES")
print("=" * 70)

original_feature_names = []

for transformer_name, transformer, columns_used in (
    preprocessor.transformers_
):

    if transformer_name == "remainder":
        continue

    columns_used = list(columns_used)

    if transformer_name == "categorical":

        encoder = transformer.named_steps["encoder"]

        for column, categories in zip(
            columns_used,
            encoder.categories_
        ):
            original_feature_names.extend(
                [column] * len(categories)
            )

    else:
        original_feature_names.extend(columns_used)


if len(original_feature_names) != X_train_processed.shape[1]:
    raise ValueError(
        "Could not map processed columns to original features. "
        "Check the preprocessing pipeline."
    )

print("Processed features mapped successfully!")


# ------------------------------------------------------------
# 7. RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest for feature importance...")

rf_model.fit(
    X_train_processed,
    y_train
)

print("Random Forest training completed!")


# ------------------------------------------------------------
# 8. GET PROCESSED FEATURE IMPORTANCE
# ------------------------------------------------------------

transformed_importance = pd.DataFrame({
    "Processed_Feature": X_train_processed.columns,
    "Original_Feature": original_feature_names,
    "Importance": rf_model.feature_importances_
})

transformed_importance = transformed_importance.sort_values(
    by="Importance",
    ascending=False
)

transformed_importance.to_csv(
    REPORT_DIR / "transformed_feature_importance.csv",
    index=False
)


# ------------------------------------------------------------
# 9. AGGREGATE IMPORTANCE BY ORIGINAL FEATURE
# ------------------------------------------------------------

feature_importance = (
    transformed_importance
    .groupby("Original_Feature", as_index=False)["Importance"]
    .sum()
    .sort_values(by="Importance", ascending=False)
)

total_importance = feature_importance["Importance"].sum()

if total_importance > 0:
    feature_importance["Importance_Percentage"] = (
        feature_importance["Importance"]
        / total_importance * 100
    )

feature_importance["Rank"] = range(
    1, len(feature_importance) + 1
)

feature_importance = feature_importance[
    [
        "Rank",
        "Original_Feature",
        "Importance",
        "Importance_Percentage"
    ]
]

feature_importance.to_csv(
    REPORT_DIR / "feature_importance_report.csv",
    index=False
)

print("\nTop 15 important original features:")
print(feature_importance.head(15).to_string(index=False))


# ------------------------------------------------------------
# 10. GENERATE FEATURE IMPORTANCE GRAPH
# ------------------------------------------------------------

top_features = feature_importance.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Original_Feature"],
    top_features["Importance"]
)

plt.xlabel("Aggregated Feature Importance")
plt.ylabel("Features")
plt.title("Top 15 Features - Random Forest Importance")

plt.tight_layout()

importance_graph = GRAPH_DIR / "feature_importance.png"

plt.savefig(
    importance_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance graph saved:")
print(importance_graph)


# ------------------------------------------------------------
# 11. SAVE RANKED FEATURE LIST
# ------------------------------------------------------------

ranked_features = feature_importance[
    ["Rank", "Original_Feature", "Importance_Percentage"]
].copy()

ranked_features.to_csv(
    REPORT_DIR / "ranked_features.csv",
    index=False
)

print("\nRanked feature list saved!")


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SELECTION SUMMARY")
print("=" * 70)

print("Original feature count:", X_train.shape[1])
print("Processed feature count:", X_train_processed.shape[1])

print("Constant features:", len(constant_features))
print("Highly correlated pairs:", len(correlation_report))
print("Ranked original features:", len(feature_importance))

print("\nGenerated reports:")
print("- constant_features.csv")
print("- high_correlation_pairs.csv")
print("- transformed_feature_importance.csv")
print("- feature_importance_report.csv")
print("- ranked_features.csv")
print("- graphs/feature_importance.png")

print("\nNOTE:")
print("Feature rankings are recommendations, not automatic deletions.")
print("Final feature selection should be validated using model performance.")

print("\nSTEP 5 COMPLETED SUCCESSFULLY!")
print("=" * 70)
