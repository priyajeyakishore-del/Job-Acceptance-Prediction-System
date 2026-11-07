
# ============================================================
# STEP 09: TUNED MODEL EVALUATION
# Job Acceptance Prediction System
# ============================================================

from pathlib import Path
import warnings

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

# Step 08 output: use the path confirmed by your console
TUNED_MODEL_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "hyperparameter_tuning"
    / "best_models"
)

# Step 06 original evaluation results
ORIGINAL_RESULTS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "evaluation"
    / "model_comparison.csv"
)

# Step 09 output
REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "evaluation"
    / "tuned_model_evaluation"
)

GRAPH_DIR = REPORT_DIR / "graphs"
PREDICTION_DIR = REPORT_DIR / "predictions"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL FILES
# ============================================================

model_files = {
    "Logistic Regression": "logistic_regression_tuned.pkl",
    "Decision Tree": "decision_tree_tuned.pkl",
    "Random Forest": "random_forest_tuned.pkl",
    "Gradient Boosting": "gradient_boosting_tuned.pkl",
    "AdaBoost": "adaboost_tuned.pkl",
    "XGBoost": "xgboost_tuned.pkl"
}


# ============================================================
# 3. HELPER FUNCTION: LOAD TARGET
# ============================================================

def load_target(file_path):

    target_df = pd.read_csv(file_path)

    target_df = target_df.loc[
        :,
        ~target_df.columns.str.startswith("Unnamed:")
    ]

    if target_df.shape[1] != 1:
        raise ValueError(
            f"Expected one target column in {file_path}"
        )

    target = target_df.iloc[:, 0]

    if target.dtype == "object":
        target = (
            target.astype(str)
            .str.strip()
            .map({
                "Placed": 1,
                "Not Placed": 0
            })
        )

    if target.isna().any():
        raise ValueError(
            "Target contains missing or unrecognized values."
        )

    target = target.astype(int)

    if set(target.unique()) != {0, 1}:
        raise ValueError(
            "Expected binary target values 0 and 1."
        )

    return target


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

print("\n" + "=" * 75)
print("STEP 09: TUNED MODEL EVALUATION")
print("=" * 75)

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Test data directory: {DATA_DIR}")
print(f"Tuned model directory: {TUNED_MODEL_DIR}")
print(f"Output directory: {REPORT_DIR}")

X_TEST_FILE = DATA_DIR / "X_test_processed.csv"
Y_TEST_FILE = DATA_DIR / "y_test.csv"

if not X_TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test features file not found:\n{X_TEST_FILE}"
    )

if not Y_TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test target file not found:\n{Y_TEST_FILE}"
    )

X_test = pd.read_csv(X_TEST_FILE)

X_test = X_test.loc[
    :,
    ~X_test.columns.str.startswith("Unnamed:")
]

y_test = load_target(Y_TEST_FILE)

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test have different row counts."
    )

print("\nTest data loaded successfully!")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 5. EVALUATE TUNED MODELS
# ============================================================

results = []
classification_reports = []
confusion_matrices = {}

for model_name, model_file in model_files.items():

    print("\n" + "-" * 75)
    print(f"Evaluating tuned model: {model_name}")
    print("-" * 75)

    model_path = TUNED_MODEL_DIR / model_file

    if not model_path.exists():
        print(f"WARNING: Model file not found: {model_path}")
        continue

    # --------------------------------------------------------
    # Load tuned model
    # --------------------------------------------------------

    model = joblib.load(model_path)

    print(f"Model file: {model_file}")

    # --------------------------------------------------------
    # Validate feature compatibility
    # --------------------------------------------------------

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != X_test.shape[1]:
            raise ValueError(
                f"{model_name}: Expected "
                f"{model.n_features_in_} features, "
                f"but received {X_test.shape[1]}."
            )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Positive class probability: Placed = 1
    # --------------------------------------------------------

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model_name} does not support predict_proba."
        )

    probabilities = model.predict_proba(X_test)

    if 1 not in model.classes_:
        raise ValueError(
            f"{model_name} does not contain class 1."
        )

    positive_class_index = list(model.classes_).index(1)

    y_probability = probabilities[:, positive_class_index]

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    confusion_matrices[model_name] = cm

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    })

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Placed", "Placed"],
        output_dict=True,
        zero_division=0
    )

    for class_name, values in report.items():

        if isinstance(values, dict):

            classification_reports.append({
                "Model": model_name,
                "Class": class_name,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1_Score": values["f1-score"],
                "Support": values["support"]
            })

    # --------------------------------------------------------
    # Save individual prediction file
    # --------------------------------------------------------

    safe_name = model_name.lower().replace(" ", "_")

    prediction_df = pd.DataFrame({
        "Actual_Status": y_test.to_numpy(),
        "Predicted_Status": y_pred,
        "Placed_Probability": y_probability
    })

    prediction_file = (
        PREDICTION_DIR
        / f"{safe_name}_tuned_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(f"Test records: {len(y_test)}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nEvaluation completed successfully!")


# ============================================================
# 6. SAVE TUNED MODEL EVALUATION RESULTS
# ============================================================

if not results:
    raise RuntimeError(
        "No tuned models were evaluated. "
        "Check the model paths."
    )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
).reset_index(drop=True)

results_file = REPORT_DIR / "tuned_model_comparison.csv"

results_df.to_csv(
    results_file,
    index=False
)

print("\n" + "=" * 75)
print("TUNED MODEL PERFORMANCE COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ============================================================
# 7. SAVE CLASSIFICATION REPORTS
# ============================================================

classification_df = pd.DataFrame(
    classification_reports
)

classification_file = (
    REPORT_DIR / "tuned_classification_reports.csv"
)

classification_df.to_csv(
    classification_file,
    index=False
)

print("\nClassification reports saved successfully!")


# ============================================================
# 8. BEST TUNED MODEL FOR EACH METRIC
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

best_metric_results = []

print("\n" + "=" * 75)
print("BEST TUNED MODEL FOR EACH METRIC")
print("=" * 75)

for metric in metric_columns:

    best_row = results_df.loc[
        results_df[metric].idxmax()
    ]

    print(
        f"{metric}: {best_row['Model']} "
        f"({best_row[metric]:.4f})"
    )

    best_metric_results.append({
        "Metric": metric,
        "Best_Model": best_row["Model"],
        "Best_Score": best_row[metric]
    })

best_metric_df = pd.DataFrame(best_metric_results)

best_metric_file = (
    REPORT_DIR / "best_tuned_model_by_metric.csv"
)

best_metric_df.to_csv(
    best_metric_file,
    index=False
)


# ============================================================
# 9. COMPARE ORIGINAL AND TUNED MODELS
# ============================================================

print("\n" + "=" * 75)
print("ORIGINAL VS TUNED MODEL COMPARISON")
print("=" * 75)

if not ORIGINAL_RESULTS_FILE.exists():

    print(
        "\nWARNING: Original evaluation results not found."
    )

    print(
        "Tuned model evaluation is complete, "
        "but the original-versus-tuned comparison was skipped."
    )

else:

    original_df = pd.read_csv(ORIGINAL_RESULTS_FILE)

    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in original_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Original comparison file is missing columns: "
            f"{missing_columns}"
        )

    # Rename metrics for original models
    original_df = original_df[
        required_columns
    ].rename(columns={
        "Accuracy": "Original_Accuracy",
        "Precision": "Original_Precision",
        "Recall": "Original_Recall",
        "F1_Score": "Original_F1_Score",
        "ROC_AUC": "Original_ROC_AUC"
    })

    # Rename metrics for tuned models
    tuned_comparison_df = results_df.rename(columns={
        "Accuracy": "Tuned_Accuracy",
        "Precision": "Tuned_Precision",
        "Recall": "Tuned_Recall",
        "F1_Score": "Tuned_F1_Score",
        "ROC_AUC": "Tuned_ROC_AUC"
    })

    comparison_df = pd.merge(
        original_df,
        tuned_comparison_df,
        on="Model",
        how="inner"
    )

    # Calculate improvement in percentage points
    for metric in metric_columns:

        comparison_df[f"{metric}_Change"] = (
            comparison_df[f"Tuned_{metric}"]
            - comparison_df[f"Original_{metric}"]
        ) * 100

    comparison_file = (
        REPORT_DIR / "original_vs_tuned_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    print(
        comparison_df[
            [
                "Model",
                "Original_F1_Score",
                "Tuned_F1_Score",
                "F1_Score_Change"
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print(
        "\nPositive F1 change means the tuned model "
        "performed better on the test set."
    )


# ============================================================
# 10. PLOT CONFUSION MATRICES
# ============================================================

print("\nGenerating confusion matrix graphs...")

number_of_models = len(confusion_matrices)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

axes = axes.flatten()

for index, (model_name, cm) in enumerate(
    confusion_matrices.items()
):

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Placed", "Placed"]
    )

    display.plot(
        ax=axes[index],
        cmap="Blues",
        colorbar=False
    )

    axes[index].set_title(model_name)

# Hide unused axes if fewer than six models were evaluated
for index in range(number_of_models, len(axes)):
    axes[index].set_visible(False)

plt.suptitle(
    "Confusion Matrices - Tuned Models",
    fontsize=16
)

plt.tight_layout()

confusion_graph = (
    GRAPH_DIR / "tuned_models_confusion_matrices.png"
)

plt.savefig(
    confusion_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. PLOT TUNED MODEL METRIC COMPARISON
# ============================================================

print("Generating model comparison graphs...")

for metric in metric_columns:

    plt.figure(figsize=(11, 6))

    sorted_df = results_df.sort_values(
        by=metric,
        ascending=False
    )

    plt.bar(
        sorted_df["Model"],
        sorted_df[metric]
    )

    plt.title(
        f"Tuned Models - {metric.replace('_', ' ')} Comparison"
    )

    plt.xlabel("Model")
    plt.ylabel(metric.replace("_", " "))

    plt.ylim(0, 1)

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    graph_file = (
        GRAPH_DIR
        / f"tuned_{metric.lower()}_comparison.png"
    )

    plt.savefig(
        graph_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. ORIGINAL VS TUNED F1 COMPARISON GRAPH
# ============================================================

if ORIGINAL_RESULTS_FILE.exists():

    plt.figure(figsize=(12, 6))

    comparison_plot_df = comparison_df.sort_values(
        by="Tuned_F1_Score",
        ascending=False
    )

    x = np.arange(len(comparison_plot_df))
    width = 0.35

    plt.bar(
        x - width / 2,
        comparison_plot_df["Original_F1_Score"],
        width,
        label="Original Models"
    )

    plt.bar(
        x + width / 2,
        comparison_plot_df["Tuned_F1_Score"],
        width,
        label="Tuned Models"
    )

    plt.title("Original vs Tuned Models - F1-Score")
    plt.xlabel("Model")
    plt.ylabel("F1-Score")

    plt.xticks(
        x,
        comparison_plot_df["Model"],
        rotation=25,
        ha="right"
    )

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    comparison_graph = (
        GRAPH_DIR / "original_vs_tuned_f1_comparison.png"
    )

    plt.savefig(
        comparison_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. FINAL MODEL SELECTION
# ============================================================

best_model_row = results_df.loc[
    results_df["F1_Score"].idxmax()
]

print("\n" + "=" * 75)
print("FINAL TUNED MODEL EVALUATION SUMMARY")
print("=" * 75)

print(f"Total tuned models evaluated: {len(results_df)}")
print(f"Test records per model: {len(y_test)}")

print(f"\nBest tuned model by F1-score: {best_model_row['Model']}")
print(f"Accuracy : {best_model_row['Accuracy']:.4f}")
print(f"Precision: {best_model_row['Precision']:.4f}")
print(f"Recall   : {best_model_row['Recall']:.4f}")
print(f"F1-Score : {best_model_row['F1_Score']:.4f}")
print(f"ROC-AUC  : {best_model_row['ROC_AUC']:.4f}")

print(
    "\nNote: Final model selection should consider "
    "the cost of false positives and false negatives, "
    "not F1-score alone."
)


# ============================================================
# 14. GENERATED FILES
# ============================================================

print("\n" + "=" * 75)
print("GENERATED FILES")
print("=" * 75)

print(f"- {results_file}")
print(f"- {classification_file}")
print(f"- {best_metric_file}")

if ORIGINAL_RESULTS_FILE.exists():
    print(f"- {comparison_file}")

print(f"- {GRAPH_DIR}")

print(f"\nPredictions saved in: {PREDICTION_DIR}")

print("\n" + "=" * 75)
print("STEP 09: TUNED MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 75)
