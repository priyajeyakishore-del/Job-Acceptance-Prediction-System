
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STEP 06: MODEL EVALUATION AND COMPARISON
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input directory: Existing model prediction reports
MODEL_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "models"
)

# Output directory: Step 06 evaluation reports
REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "evaluation"
)

# Output directory: Evaluation graphs
GRAPH_DIR = REPORT_DIR / "graphs"

# Create output directories automatically
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 75)
print("STEP 06: MODEL EVALUATION AND COMPARISON")
print("=" * 75)

print("\nProject root:", PROJECT_ROOT)
print("Input directory:", MODEL_REPORT_DIR)
print("Output directory:", REPORT_DIR)
print("Graph directory:", GRAPH_DIR)


# ============================================================
# 2. MODEL PREDICTION FILES
# ============================================================

model_files = {

    "Logistic Regression":
        Path("logistic_regression") /
        "logistic_regression_predictions.csv",

    "Decision Tree":
        Path("decision_tree") /
        "decision_tree_predictions.csv",

    "Random Forest":
        Path("random_forest") /
        "random_forest_predictions.csv",

    "Gradient Boosting":
        Path("gradient_boost") /
        "gradient_boosting_predictions.csv",

    "AdaBoost":
        Path("ada_boost") /
        "adaboost_predictions.csv",

    "XGBoost":
        Path("xgboost") /
        "xgboost_predictions.csv"
}

# ============================================================
# 3. FUNCTION TO CONVERT TARGET LABELS
# ============================================================

def convert_status_to_binary(series, column_name):

    """
    Convert target labels into binary values.

    Not Placed = 0
    Placed = 1
    """

    numeric_values = pd.to_numeric(series, errors="coerce")

    # Handle numeric labels
    if numeric_values.notna().all():

        if not numeric_values.isin([0, 1]).all():
            raise ValueError(
                f"Unexpected numeric values in {column_name}"
            )

        return numeric_values.astype(int)

    # Handle text labels
    text_values = series.astype(str).str.strip().str.lower()

    mapping = {
        "not placed": 0,
        "not_placed": 0,
        "notplaced": 0,
        "0": 0,
        "placed": 1,
        "1": 1
    }

    converted = text_values.map(mapping)

    if converted.isnull().any():
        raise ValueError(
            f"Unexpected target labels in {column_name}: "
            f"{series.unique()}"
        )

    return converted.astype(int)


# ============================================================
# 4. FUNCTION TO FIND PROBABILITY COLUMN
# ============================================================

def find_probability_column(df):

    """
    Identify the probability column.

    Different model scripts may use different column names.
    """

    possible_columns = [
        "Placed_Probability",
        "Probability_Placed",
        "placed_probability",
        "probability_placed"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Placed probability column not found. "
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# 5. LOAD AND EVALUATE ALL MODELS
# ============================================================

model_results = []

confusion_matrices = {}

reference_actual = None
reference_length = None

classification_reports = []


for model_name, relative_path in model_files.items():

    file_path = MODEL_REPORT_DIR / relative_path

    print("\n" + "-" * 75)
    print("Evaluating:", model_name)
    print("-" * 75)

    # --------------------------------------------------------
    # Check prediction file
    # --------------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n{file_path}"
        )

    # --------------------------------------------------------
    # Load prediction data
    # --------------------------------------------------------

    df = pd.read_csv(file_path)

    print("Prediction file:", file_path.name)

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "Actual_Status",
        "Predicted_Status"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"{column} not found in {file_path.name}. "
                f"Available columns: {df.columns.tolist()}"
            )

    # --------------------------------------------------------
    # Identify probability column
    # --------------------------------------------------------

    probability_column = find_probability_column(df)

    # --------------------------------------------------------
    # Convert target labels into binary values
    # --------------------------------------------------------

    actual = convert_status_to_binary(
        df["Actual_Status"],
        "Actual_Status"
    )

    predicted = convert_status_to_binary(
        df["Predicted_Status"],
        "Predicted_Status"
    )

    # --------------------------------------------------------
    # Convert probability values into numeric format
    # --------------------------------------------------------

    probability = pd.to_numeric(
        df[probability_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Validate prediction data
    # --------------------------------------------------------

    if probability.isnull().any():

        raise ValueError(
            f"Invalid probability values in {file_path.name}"
        )

    if not probability.between(0, 1).all():

        raise ValueError(
            f"Probability values outside 0 to 1 in {file_path.name}"
        )

    if len(actual) != len(predicted) or len(actual) != len(probability):

        raise ValueError(
            f"Row count mismatch in {file_path.name}"
        )

    if actual.nunique() != 2:

        raise ValueError(
            f"Both target classes are required in {file_path.name}"
        )

    # --------------------------------------------------------
    # Ensure all models use the same actual test labels
    # --------------------------------------------------------

    if reference_actual is None:

        reference_actual = actual.reset_index(drop=True)
        reference_length = len(actual)

    else:

        if len(actual) != reference_length:

            raise ValueError(
                f"Test record count mismatch in {file_path.name}"
            )

        if not reference_actual.equals(
            actual.reset_index(drop=True)
        ):

            raise ValueError(
                f"Actual test labels differ in {file_path.name}"
            )

    # ========================================================
    # 6. CALCULATE EVALUATION METRICS
    # ========================================================

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        actual,
        probability
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    model_results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc

    })

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    confusion_matrices[model_name] = cm

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Not Placed", "Placed"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.insert(
        0,
        "Model",
        model_name
    )

    classification_reports.append(report_df)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("Probability column:", probability_column)
    print("Test records:", len(actual))

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nEvaluation completed successfully!")


# ============================================================
# 7. CREATE MODEL COMPARISON REPORT
# ============================================================

comparison_df = pd.DataFrame(model_results)

comparison_df = comparison_df.sort_values(
    by="F1_Score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

comparison_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)


# ============================================================
# 8. BEST MODEL FOR EACH METRIC
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

best_models = []

print("\n" + "=" * 75)
print("BEST MODEL FOR EACH METRIC")
print("=" * 75)

for metric in metrics:

    best_row = comparison_df.loc[
        comparison_df[metric].idxmax()
    ]

    best_models.append({

        "Metric": metric,
        "Best_Model": best_row["Model"],
        "Score": best_row[metric]

    })

    print(
        f"{metric}: {best_row['Model']} "
        f"({best_row[metric]:.4f})"
    )

best_models_df = pd.DataFrame(best_models)

best_models_df.to_csv(
    REPORT_DIR / "best_model_by_metric.csv",
    index=False
)


# ============================================================
# 9. SAVE CLASSIFICATION REPORTS
# ============================================================

all_classification_reports = pd.concat(
    classification_reports,
    ignore_index=False
)

all_classification_reports.to_csv(
    REPORT_DIR / "all_models_classification_reports.csv",
    index=True,
    index_label="Class"
)

print("\nClassification reports saved successfully!")


# ============================================================
# 10. ALL METRICS COMPARISON GRAPH
# ============================================================

comparison_long = comparison_df.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(14, 7))

sns.barplot(
    data=comparison_long,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title(
    "Performance Comparison of All Machine Learning Models"
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Evaluation Score")

plt.ylim(0, 1.05)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.legend(title="Evaluation Metrics")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "all_models_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. F1-SCORE COMPARISON
# ============================================================

f1_sorted = comparison_df.sort_values(
    by="F1_Score",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    f1_sorted["Model"],
    f1_sorted["F1_Score"]
)

plt.title("F1-Score Comparison of All Models")
plt.xlabel("F1-Score")
plt.ylabel("Model")

plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "model_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. ROC-AUC COMPARISON
# ============================================================

roc_sorted = comparison_df.sort_values(
    by="ROC_AUC",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    roc_sorted["Model"],
    roc_sorted["ROC_AUC"]
)

plt.title("ROC-AUC Comparison of All Models")
plt.xlabel("ROC-AUC")
plt.ylabel("Model")

plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "model_roc_auc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. CONFUSION MATRIX COMPARISON
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

axes = axes.flatten()

for index, (model_name, cm) in enumerate(
    confusion_matrices.items()
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Placed", "Placed"],
        yticklabels=["Not Placed", "Placed"],
        ax=axes[index]
    )

    axes[index].set_title(model_name)
    axes[index].set_xlabel("Predicted")
    axes[index].set_ylabel("Actual")

plt.suptitle(
    "Confusion Matrix Comparison - All Models",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "all_models_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. FINAL EVALUATION SUMMARY
# ============================================================

best_accuracy = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

best_precision = comparison_df.loc[
    comparison_df["Precision"].idxmax()
]

best_recall = comparison_df.loc[
    comparison_df["Recall"].idxmax()
]

best_f1 = comparison_df.loc[
    comparison_df["F1_Score"].idxmax()
]

best_auc = comparison_df.loc[
    comparison_df["ROC_AUC"].idxmax()
]

print("\n" + "=" * 75)
print("FINAL MODEL EVALUATION SUMMARY")
print("=" * 75)

print("Total models evaluated:", len(comparison_df))
print("Test records per model:", reference_length)

print(
    f"\nBest Accuracy Model: {best_accuracy['Model']} "
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    f"Best Precision Model: {best_precision['Model']} "
    f"({best_precision['Precision']:.4f})"
)

print(
    f"Best Recall Model: {best_recall['Model']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Best F1-Score Model: {best_f1['Model']} "
    f"({best_f1['F1_Score']:.4f})"
)

print(
    f"Best ROC-AUC Model: {best_auc['Model']} "
    f"({best_auc['ROC_AUC']:.4f})"
)


# ============================================================
# 15. GENERATED FILES
# ============================================================

print("\n" + "=" * 75)
print("GENERATED FILES")
print("=" * 75)

generated_files = [
    REPORT_DIR / "model_comparison.csv",
    REPORT_DIR / "best_model_by_metric.csv",
    REPORT_DIR / "all_models_classification_reports.csv",
    GRAPH_DIR / "all_models_comparison.png",
    GRAPH_DIR / "model_f1_comparison.png",
    GRAPH_DIR / "model_roc_auc_comparison.png",
    GRAPH_DIR / "all_models_confusion_matrices.png"
]

for file in generated_files:

    print("-", file.relative_to(PROJECT_ROOT))


print("\nSTEP 06: MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 75)
