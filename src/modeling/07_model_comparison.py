
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STEP 07: MODEL COMPARISON AND FINAL MODEL SELECTION
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input directory: Step 06 evaluation reports
EVALUATION_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "evaluation"
)

# Output directory: Step 07 model comparison
REPORT_DIR = EVALUATION_DIR / "model_comparison"

# Output directory: Step 07 graphs
GRAPH_DIR = REPORT_DIR / "graphs"

# Step 06 input file
INPUT_FILE = EVALUATION_DIR / "model_comparison.csv"

# Create output directories
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "=" * 75)
print("STEP 07: MODEL COMPARISON AND FINAL MODEL SELECTION")
print("=" * 75)

print("\nProject root:", PROJECT_ROOT)
print("Input file:", INPUT_FILE)
print("Output directory:", REPORT_DIR)


# ============================================================
# 2. LOAD MODEL EVALUATION RESULTS
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Step 06 evaluation file not found:\n{INPUT_FILE}\n"
        "Please run 06_model_evaluation.py first."
    )

comparison_df = pd.read_csv(INPUT_FILE)

print("\nEvaluation results loaded successfully!")

print("\nOriginal evaluation results:")
print(comparison_df.to_string(index=False))


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

missing_columns = [
    column
    for column in required_columns
    if column not in comparison_df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if comparison_df.empty:

    raise ValueError("The evaluation dataset is empty.")

if comparison_df["Model"].duplicated().any():

    raise ValueError("Duplicate model names found.")

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

for column in metric_columns:

    comparison_df[column] = pd.to_numeric(
        comparison_df[column],
        errors="coerce"
    )

    if comparison_df[column].isnull().any():

        raise ValueError(
            f"Invalid values found in {column}"
        )

    if not comparison_df[column].between(0, 1).all():

        raise ValueError(
            f"Values outside 0 to 1 found in {column}"
        )


# ============================================================
# 4. MODEL RANKING
# ============================================================

print("\n" + "=" * 75)
print("MODEL RANKING")
print("=" * 75)

# Rank models separately for each metric.
# Rank 1 indicates the best performance.

ranking_df = comparison_df.copy()

for metric in metric_columns:

    ranking_df[f"{metric}_Rank"] = (
        ranking_df[metric]
        .rank(method="min", ascending=False)
        .astype(int)
    )

# Calculate average rank across all metrics.
rank_columns = [
    f"{metric}_Rank"
    for metric in metric_columns
]

ranking_df["Average_Rank"] = (
    ranking_df[rank_columns].mean(axis=1)
)

# Lower average rank indicates better overall performance.
ranking_df = ranking_df.sort_values(
    by=["Average_Rank", "F1_Score"],
    ascending=[True, False]
).reset_index(drop=True)

ranking_df.insert(
    0,
    "Overall_Rank",
    range(1, len(ranking_df) + 1)
)

print(
    ranking_df[
        ["Overall_Rank", "Model", "Average_Rank"]
        + rank_columns
    ].to_string(index=False)
)

ranking_df.to_csv(
    REPORT_DIR / "model_ranking.csv",
    index=False
)


# ============================================================
# 5. IDENTIFY BEST MODEL FOR EACH METRIC
# ============================================================

print("\n" + "=" * 75)
print("BEST MODEL FOR EACH EVALUATION METRIC")
print("=" * 75)

best_models = []

for metric in metric_columns:

    best_row = comparison_df.loc[
        comparison_df[metric].idxmax()
    ]

    best_models.append({

        "Metric": metric,
        "Best_Model": best_row["Model"],
        "Best_Score": best_row[metric]

    })

    print(
        f"{metric}: {best_row['Model']} "
        f"({best_row[metric]:.4f})"
    )

best_models_df = pd.DataFrame(best_models)

best_models_df.to_csv(
    REPORT_DIR / "best_model_summary.csv",
    index=False
)


# ============================================================
# 6. OVERALL MODEL COMPARISON SUMMARY
# ============================================================

best_overall = ranking_df.iloc[0]

summary_data = {

    "Total_Models_Evaluated": len(comparison_df),

    "Best_Overall_Model": best_overall["Model"],

    "Best_Overall_Average_Rank":
        best_overall["Average_Rank"],

    "Best_Accuracy_Model":
        comparison_df.loc[
            comparison_df["Accuracy"].idxmax(),
            "Model"
        ],

    "Best_Precision_Model":
        comparison_df.loc[
            comparison_df["Precision"].idxmax(),
            "Model"
        ],

    "Best_Recall_Model":
        comparison_df.loc[
            comparison_df["Recall"].idxmax(),
            "Model"
        ],

    "Best_F1_Model":
        comparison_df.loc[
            comparison_df["F1_Score"].idxmax(),
            "Model"
        ],

    "Best_ROC_AUC_Model":
        comparison_df.loc[
            comparison_df["ROC_AUC"].idxmax(),
            "Model"
        ]

}

summary_df = pd.DataFrame(
    [summary_data]
)

summary_df.to_csv(
    REPORT_DIR / "model_comparison_summary.csv",
    index=False
)


# ============================================================
# 7. ALL METRICS COMPARISON GRAPH
# ============================================================

comparison_long = comparison_df.melt(
    id_vars="Model",
    value_vars=metric_columns,
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
    "Overall Performance Comparison of Machine Learning Models"
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
    GRAPH_DIR / "overall_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. OVERALL MODEL RANKING GRAPH
# ============================================================

ranking_plot = ranking_df.sort_values(
    by="Average_Rank",
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=ranking_plot,
    x="Average_Rank",
    y="Model"
)

plt.title("Overall Machine Learning Model Ranking")
plt.xlabel("Average Rank (Lower is Better)")
plt.ylabel("Model")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "overall_model_ranking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. INDIVIDUAL METRIC COMPARISON GRAPHS
# ============================================================

for metric in metric_columns:

    sorted_df = comparison_df.sort_values(
        by=metric,
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=sorted_df,
        x=metric,
        y="Model"
    )

    plt.title(f"{metric} Comparison Across Models")
    plt.xlabel(metric)
    plt.ylabel("Model")

    plt.xlim(0, 1)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{metric.lower()}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. FINAL MODEL SELECTION
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL SELECTION")
print("=" * 75)

print("\nBest Overall Model:", best_overall["Model"])

print(
    "Average Rank:",
    round(best_overall["Average_Rank"], 2)
)

selected_model = comparison_df.loc[
    comparison_df["Model"] == best_overall["Model"]
].iloc[0]

print("\nSelected Model Performance:")

for metric in metric_columns:

    print(
        f"{metric}: {selected_model[metric]:.4f}"
    )

print(
    "\nNote: The overall ranking uses equal weight for all five metrics."
)

print(
    "Final model selection should also consider the business objective "
    "and the cost of false positives and false negatives."
)


# ============================================================
# 11. GENERATED FILES
# ============================================================

print("\n" + "=" * 75)
print("GENERATED FILES")
print("=" * 75)

generated_files = [

    REPORT_DIR / "model_ranking.csv",

    REPORT_DIR / "best_model_summary.csv",

    REPORT_DIR / "model_comparison_summary.csv",

    GRAPH_DIR / "overall_model_comparison.png",

    GRAPH_DIR / "overall_model_ranking.png",

]

for metric in metric_columns:

    generated_files.append(
        GRAPH_DIR / f"{metric.lower()}_comparison.png"
    )

for file in generated_files:

    print("-", file.relative_to(PROJECT_ROOT))


print("\nSTEP 07: MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 75)
