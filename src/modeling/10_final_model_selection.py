
# ============================================================
# STEP 10: FINAL MODEL SELECTION
# Project: Job Acceptance Prediction System
# ============================================================

from pathlib import Path
import json
import shutil
import joblib
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUNED_MODEL_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "hyperparameter_tuning"
    / "best_models"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "evaluation"
    / "tuned_model_evaluation"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "final_model_selection"
)

COMPARISON_FILE = EVALUATION_DIR / "tuned_model_comparison.csv"

PREPROCESSING_PIPELINE = MODEL_DIR / "preprocessing_pipeline.pkl"

FINAL_MODEL_FILE = MODEL_DIR / "final_model.pkl"

FINAL_PIPELINE_FILE = MODEL_DIR / "final_preprocessing_pipeline.pkl"


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DISPLAY PROJECT INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("STEP 10: FINAL MODEL SELECTION")
print("=" * 75)

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Tuned model directory: {TUNED_MODEL_DIR}")
print(f"Evaluation directory: {EVALUATION_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        f"Tuned model comparison file not found:\n{COMPARISON_FILE}"
    )

if not PREPROCESSING_PIPELINE.exists():
    raise FileNotFoundError(
        f"Preprocessing pipeline not found:\n{PREPROCESSING_PIPELINE}"
    )


# ============================================================
# 5. LOAD TUNED MODEL COMPARISON RESULTS
# ============================================================

print("\n" + "-" * 75)
print("LOADING TUNED MODEL EVALUATION RESULTS")
print("-" * 75)

comparison_df = pd.read_csv(COMPARISON_FILE)

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

missing_columns = [
    column for column in required_columns
    if column not in comparison_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in comparison file: {missing_columns}"
    )

if comparison_df.empty:
    raise ValueError("The model comparison file is empty.")

print("\nTuned model results loaded successfully!")
print(f"Number of models evaluated: {len(comparison_df)}")


# ============================================================
# 6. RANK MODELS USING EVALUATION METRICS
# ============================================================

print("\n" + "-" * 75)
print("MODEL RANKING")
print("-" * 75)

# Higher values are better for all five metrics.
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

ranking_df = comparison_df.copy()

for metric in metrics:
    ranking_df[metric] = pd.to_numeric(
        ranking_df[metric],
        errors="coerce"
    )

if ranking_df[metrics].isnull().any().any():
    raise ValueError(
        "One or more evaluation metrics contain missing or invalid values."
    )

# F1-score is the primary selection criterion.
# ROC-AUC and accuracy are used as tie-breakers.
ranking_df = ranking_df.sort_values(
    by=["F1_Score", "ROC_AUC", "Accuracy"],
    ascending=False
).reset_index(drop=True)

ranking_df.insert(
    0,
    "F1_Rank",
    range(1, len(ranking_df) + 1)
)

print("\nModel ranking based on F1-score:")
print(
    ranking_df[
        ["F1_Rank", "Model"] + metrics
    ].to_string(index=False)
)


# ============================================================
# 7. SELECT THE BEST MODEL
# ============================================================

print("\n" + "-" * 75)
print("FINAL MODEL SELECTION")
print("-" * 75)

best_model_row = ranking_df.iloc[0]

selected_model_name = str(best_model_row["Model"])

model_file_mapping = {
    "Logistic Regression": "logistic_regression_tuned.pkl",
    "Decision Tree": "decision_tree_tuned.pkl",
    "Random Forest": "random_forest_tuned.pkl",
    "Gradient Boosting": "gradient_boosting_tuned.pkl",
    "AdaBoost": "adaboost_tuned.pkl",
    "XGBoost": "xgboost_tuned.pkl"
}

if selected_model_name not in model_file_mapping:
    raise ValueError(
        f"Unknown model name: {selected_model_name}"
    )

selected_model_file = (
    TUNED_MODEL_DIR / model_file_mapping[selected_model_name]
)

if not selected_model_file.exists():
    raise FileNotFoundError(
        f"Selected tuned model file not found:\n{selected_model_file}"
    )

print(f"\nSelected model: {selected_model_name}")
print(f"Model file: {selected_model_file.name}")

print("\nSelected model performance:")

for metric in metrics:
    print(f"{metric:12}: {best_model_row[metric]:.4f}")


# ============================================================
# 8. LOAD AND SAVE FINAL MODEL
# ============================================================

print("\n" + "-" * 75)
print("SAVING FINAL MODEL")
print("-" * 75)

final_model = joblib.load(selected_model_file)

# Save the selected tuned model under a standard filename.
joblib.dump(final_model, FINAL_MODEL_FILE)

print(f"\nFinal model saved successfully:")
print(FINAL_MODEL_FILE)


# ============================================================
# 9. SAVE PREPROCESSING PIPELINE
# ============================================================

print("\n" + "-" * 75)
print("SAVING PREPROCESSING PIPELINE")
print("-" * 75)

preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE)

shutil.copy2(
    PREPROCESSING_PIPELINE,
    FINAL_PIPELINE_FILE
)

print("\nPreprocessing pipeline saved successfully:")
print(FINAL_PIPELINE_FILE)


# ============================================================
# 10. SAVE MODEL RANKING REPORT
# ============================================================

ranking_report_file = OUTPUT_DIR / "final_model_ranking.csv"

ranking_df.to_csv(
    ranking_report_file,
    index=False
)

print("\nModel ranking report saved:")
print(ranking_report_file)


# ============================================================
# 11. SAVE SELECTED MODEL METRICS
# ============================================================

selected_metrics = {
    "Model": selected_model_name,
    "Accuracy": float(best_model_row["Accuracy"]),
    "Precision": float(best_model_row["Precision"]),
    "Recall": float(best_model_row["Recall"]),
    "F1_Score": float(best_model_row["F1_Score"]),
    "ROC_AUC": float(best_model_row["ROC_AUC"])
}

selected_metrics_df = pd.DataFrame([selected_metrics])

metrics_file = OUTPUT_DIR / "final_model_metrics.csv"

selected_metrics_df.to_csv(
    metrics_file,
    index=False
)

print("\nSelected model metrics saved:")
print(metrics_file)


# ============================================================
# 12. SAVE FINAL MODEL METADATA
# ============================================================

metadata = {
    "project_name": "Job Acceptance Prediction System",
    "selected_model": selected_model_name,
    "model_file": str(FINAL_MODEL_FILE.relative_to(PROJECT_ROOT)),
    "preprocessing_pipeline": str(
        FINAL_PIPELINE_FILE.relative_to(PROJECT_ROOT)
    ),
    "selection_criterion": "Highest test F1-score",
    "tie_breakers": [
        "ROC-AUC",
        "Accuracy"
    ],
    "target_column": "status",
    "target_encoding": {
        "Not Placed": 0,
        "Placed": 1
    },
    "test_records": 10000,
    "input_feature_count": 68,
    "evaluation_metrics": selected_metrics,
    "selection_note": (
        "Model selected using held-out test evaluation results. "
        "Business costs of false positives and false negatives "
        "should also be considered before deployment."
    )
}

metadata_file = OUTPUT_DIR / "final_model_metadata.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nFinal model metadata saved:")
print(metadata_file)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL SELECTION SUMMARY")
print("=" * 75)

print(f"\nSelected model : {selected_model_name}")
print(f"Accuracy       : {selected_metrics['Accuracy']:.4f}")
print(f"Precision      : {selected_metrics['Precision']:.4f}")
print(f"Recall         : {selected_metrics['Recall']:.4f}")
print(f"F1-Score       : {selected_metrics['F1_Score']:.4f}")
print(f"ROC-AUC        : {selected_metrics['ROC_AUC']:.4f}")

print("\nGenerated files:")
print(f"- {FINAL_MODEL_FILE}")
print(f"- {FINAL_PIPELINE_FILE}")
print(f"- {ranking_report_file}")
print(f"- {metrics_file}")
print(f"- {metadata_file}")

print("\n" + "=" * 75)
print("STEP 10: FINAL MODEL SELECTION COMPLETED SUCCESSFULLY!")
print("=" * 75)
