
# ==============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# CLASS IMBALANCE ANALYSIS
# ORIGINAL XGBOOST vs SMOTE + XGBOOST
# ==============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ==============================================================
# 1. PROJECT PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed"
)

SPLIT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "splits"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports" /
    "machine_learning" / "class_imbalance"
)

GRAPH_DIR = REPORT_DIR / "graphs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# 2. CONFIGURATION
# ==============================================================

TARGET_COLUMN = "status"

RANDOM_STATE = 42

CLASS_NAMES = {
    0: "Not Placed",
    1: "Placed"
}

# SMOTE configuration
SMOTE_K_NEIGHBORS = 5

# Use the same hyperparameters as your existing XGBoost
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": -1
}


# ==============================================================
# 3. LOAD TARGET FILES
# ==============================================================

def find_target_file(dataset_type):

    possible_names = [
        f"y_{dataset_type}.csv",
        f"y_{dataset_type}_processed.csv",
        f"{dataset_type}_target.csv",
        f"{dataset_type}_target_processed.csv"
    ]

    search_directories = [
        PREPROCESSED_DIR,
        SPLIT_DIR
    ]

    for directory in search_directories:

        for filename in possible_names:

            file_path = directory / filename

            if file_path.exists():
                return file_path

    raise FileNotFoundError(
        f"Could not find {dataset_type} target file.\n"
        f"Checked folders: {PREPROCESSED_DIR} and {SPLIT_DIR}\n"
        f"Expected one of: {possible_names}"
    )


# ==============================================================
# 4. LOAD PREPROCESSED TRAINING AND TESTING DATA
# ==============================================================

def load_split_data():

    print("\n" + "=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)

    # Load processed features
    X_train_path = (
        PREPROCESSED_DIR / "X_train_processed.csv"
    )

    X_test_path = (
        PREPROCESSED_DIR / "X_test_processed.csv"
    )

    if not X_train_path.exists():
        raise FileNotFoundError(
            f"Training feature file not found: {X_train_path}"
        )

    if not X_test_path.exists():
        raise FileNotFoundError(
            f"Testing feature file not found: {X_test_path}"
        )

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Locate and load corresponding target files
    y_train_path = find_target_file("train")
    y_test_path = find_target_file("test")

    y_train_df = pd.read_csv(y_train_path)
    y_test_df = pd.read_csv(y_test_path)

    print(f"\nTraining target file: {y_train_path.name}")
    print(f"Testing target file: {y_test_path.name}")

    # Extract target column
    if TARGET_COLUMN in y_train_df.columns:
        y_train = y_train_df[TARGET_COLUMN]
    else:
        y_train = y_train_df.iloc[:, 0]

    if TARGET_COLUMN in y_test_df.columns:
        y_test = y_test_df[TARGET_COLUMN]
    else:
        y_test = y_test_df.iloc[:, 0]

    # Reset indexes
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Check row counts
    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train have different row counts."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different row counts."
        )

    # Check feature consistency
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    # Check target labels
    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "Training target must contain numeric labels 0 and 1."
        )

    if set(y_test.unique()) != {0, 1}:
        raise ValueError(
            "Testing target must contain numeric labels 0 and 1."
        )

    # Check categorical columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if categorical_columns:

        print("\nCategorical columns detected:")

        for col in categorical_columns:
            print(
                f"{col}: "
                f"{X_train[col].dropna().unique()[:10]}"
            )

        raise ValueError(
            "Categorical columns still exist. "
            "Complete categorical encoding in your preprocessing "
            "pipeline before applying ordinary SMOTE."
        )

    # Check missing values
    if X_train.isnull().any().any():
        raise ValueError("Missing values found in X_train.")

    if X_test.isnull().any().any():
        raise ValueError("Missing values found in X_test.")

    # Check infinite values
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Infinite values found in X_train.")

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Infinite values found in X_test.")

    print("\nData loaded successfully!")

    print(f"Training records: {X_train.shape[0]}")
    print(f"Testing records: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    print("\nTraining shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==============================================================
# 5. CLASS IMBALANCE ANALYSIS
# ==============================================================

def analyze_class_imbalance(y_train, y_test):

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    majority_count = train_counts.max()
    minority_count = train_counts.min()

    imbalance_ratio = majority_count / minority_count

    print(f"\nTraining records: {len(y_train)}")
    print(f"Testing records: {len(y_test)}")

    print(f"Majority class count: {majority_count}")
    print(f"Minority class count: {minority_count}")

    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    print("Analysis result: Class imbalance detected"
          if imbalance_ratio > 1
          else "Analysis result: Classes are balanced")

    distribution = []

    for dataset_name, target in [
        ("Training", y_train),
        ("Testing", y_test)
    ]:

        counts = target.value_counts().sort_index()

        for class_value, count in counts.items():

            distribution.append({
                "Dataset": dataset_name,
                "Class": CLASS_NAMES[class_value],
                "Class_Value": class_value,
                "Count": count,
                "Percentage": round(
                    count / len(target) * 100, 2
                )
            })

    distribution_df = pd.DataFrame(distribution)

    print("\nCLASS DISTRIBUTION:")
    print(distribution_df.to_string(index=False))

    distribution_df.to_csv(
        REPORT_DIR / "class_distribution.csv",
        index=False
    )

    # Class distribution graph
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=distribution_df,
        x="Dataset",
        y="Count",
        hue="Class",
        hue_order=["Not Placed", "Placed"]
    )

    plt.title(
        "Class Distribution - Training vs Testing",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Dataset")
    plt.ylabel("Number of Candidates")

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return distribution_df


# ==============================================================
# 6. APPLY SMOTE
# ==============================================================

def apply_smote(X_train, y_train):

    print("\n" + "=" * 60)
    print("APPLYING SMOTE")
    print("=" * 60)

    print("\nBefore SMOTE:")
    print(y_train.value_counts().sort_index())

    minority_count = y_train.value_counts().min()

    if minority_count <= SMOTE_K_NEIGHBORS:
        raise ValueError(
            "Insufficient minority-class records for SMOTE."
        )

    smote = SMOTE(
        sampling_strategy="auto",
        random_state=RANDOM_STATE,
        k_neighbors=SMOTE_K_NEIGHBORS
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts().sort_index())

    print("\nBefore SMOTE:", X_train.shape)
    print("After SMOTE:", X_train_smote.shape)

    # Save SMOTE distribution
    smote_distribution = []

    for stage, target in [
        ("Before SMOTE", y_train),
        ("After SMOTE", pd.Series(y_train_smote))
    ]:

        counts = target.value_counts().sort_index()

        for class_value, count in counts.items():

            smote_distribution.append({
                "Stage": stage,
                "Class": CLASS_NAMES[class_value],
                "Class_Value": class_value,
                "Count": count
            })

    smote_df = pd.DataFrame(smote_distribution)

    smote_df.to_csv(
        REPORT_DIR / "smote_class_distribution.csv",
        index=False
    )

    # SMOTE graph
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=smote_df,
        x="Stage",
        y="Count",
        hue="Class",
        hue_order=["Not Placed", "Placed"]
    )

    plt.title(
        "Class Distribution Before and After SMOTE",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Training Data")
    plt.ylabel("Number of Candidates")

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "smote_class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return X_train_smote, y_train_smote


# ==============================================================
# 7. TRAIN ORIGINAL XGBOOST
# ==============================================================

def train_original_xgboost(X_train, y_train):

    print("\n" + "=" * 60)
    print("TRAINING ORIGINAL XGBOOST")
    print("=" * 60)

    model = XGBClassifier(**XGB_PARAMS)

    model.fit(X_train, y_train)

    print("Original XGBoost training completed.")

    return model


# ==============================================================
# 8. TRAIN SMOTE + XGBOOST
# ==============================================================

def train_smote_xgboost(X_train_smote, y_train_smote):

    print("\n" + "=" * 60)
    print("TRAINING SMOTE + XGBOOST")
    print("=" * 60)

    model = XGBClassifier(**XGB_PARAMS)

    model.fit(X_train_smote, y_train_smote)

    print("SMOTE + XGBoost training completed.")

    return model


# ==============================================================
# 9. EVALUATE MODEL
# ==============================================================

def evaluate_model(model, X_test, y_test, model_name):

    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 60)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

    print("\nPerformance Metrics:")

    for metric, value in results.items():

        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Placed", "Placed"],
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_filename = (
        "original_xgboost_report.txt"
        if model_name == "Original XGBoost"
        else "smote_xgboost_report.txt"
    )

    with open(
        REPORT_DIR / report_filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"MODEL: {model_name}\n\n")

        file.write(
            pd.Series(results).to_string()
        )

        file.write("\n\nClassification Report:\n")
        file.write(report)

    return results, y_pred, y_prob


# ==============================================================
# 10. CONFUSION MATRIX COMPARISON
# ==============================================================

def plot_confusion_matrices(
    y_test,
    y_pred_original,
    y_pred_smote
):

    print("\nGenerating confusion matrices...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    predictions = [
        (y_pred_original, "Original XGBoost", "Blues"),
        (y_pred_smote, "SMOTE + XGBoost", "Greens")
    ]

    confusion_data = []

    for ax, (pred, title, cmap) in zip(axes, predictions):

        cm = confusion_matrix(
            y_test,
            pred,
            labels=[0, 1]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Placed", "Placed"]
        )

        display.plot(
            ax=ax,
            cmap=cmap,
            colorbar=False,
            values_format="d"
        )

        ax.set_title(title, fontweight="bold")

        tn, fp, fn, tp = cm.ravel()

        confusion_data.append({
            "Model": title,
            "True_Negative": tn,
            "False_Positive": fp,
            "False_Negative": fn,
            "True_Positive": tp
        })

    plt.suptitle(
        "Confusion Matrix Comparison",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "confusion_matrix_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    confusion_df = pd.DataFrame(confusion_data)

    confusion_df.to_csv(
        REPORT_DIR / "confusion_matrices.csv",
        index=False
    )

    print(confusion_df.to_string(index=False))


# ==============================================================
# 11. MODEL PERFORMANCE COMPARISON
# ==============================================================

def compare_models(original_results, smote_results):

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame([
        original_results,
        smote_results
    ])

    print("\n")
    print(comparison.round(4).to_string(index=False))

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    # Performance comparison graph
    comparison_melted = comparison.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=comparison_melted,
        x="Metric",
        y="Score",
        hue="Model"
    )

    plt.title(
        "Original XGBoost vs SMOTE + XGBoost",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "model_performance_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Metric improvement report
    original = comparison.iloc[0]
    smote = comparison.iloc[1]

    improvement = []

    for metric in metrics:

        difference = smote[metric] - original[metric]

        improvement.append({
            "Metric": metric,
            "Original_XGBoost": original[metric],
            "SMOTE_XGBoost": smote[metric],
            "Difference": difference,
            "Improvement_Percentage_Points": difference * 100
        })

    improvement_df = pd.DataFrame(improvement)

    improvement_df.to_csv(
        REPORT_DIR / "metric_improvement.csv",
        index=False
    )

    print("\nMETRIC IMPROVEMENT:")
    print(improvement_df.round(4).to_string(index=False))

    # Select based on F1-score
    best_index = comparison["F1_Score"].idxmax()

    best_model = comparison.loc[best_index, "Model"]

    print("\nBEST MODEL BASED ON F1-SCORE:")
    print(best_model)

    return comparison, best_model


# ==============================================================
# 12. MAIN FUNCTION
# ==============================================================

def main():

    print("\n")
    print("=" * 60)
    print("JOB ACCEPTANCE PREDICTION SYSTEM")
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)

    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_split_data()

    # Step 2: Class imbalance analysis
    analyze_class_imbalance(y_train, y_test)

    # Step 3: Original XGBoost
    original_model = train_original_xgboost(
        X_train,
        y_train
    )

    original_results, y_pred_original, y_prob_original = (
        evaluate_model(
            original_model,
            X_test,
            y_test,
            "Original XGBoost"
        )
    )

    # Step 4: Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(
        X_train,
        y_train
    )

    # Step 5: SMOTE + XGBoost
    smote_model = train_smote_xgboost(
        X_train_smote,
        y_train_smote
    )

    smote_results, y_pred_smote, y_prob_smote = (
        evaluate_model(
            smote_model,
            X_test,
            y_test,
            "SMOTE + XGBoost"
        )
    )

    # Step 6: Confusion matrix comparison
    plot_confusion_matrices(
        y_test,
        y_pred_original,
        y_pred_smote
    )

    # Step 7: Model comparison
    comparison, best_model = compare_models(
        original_results,
        smote_results
    )

    # Final output
    print("\n" + "=" * 60)
    print("CLASS IMBALANCE ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"\nBest model based on F1-score: {best_model}")

    print(f"\nReports saved at:\n{REPORT_DIR}")

    print(f"\nGraphs saved at:\n{GRAPH_DIR}")


# ==============================================================
# 13. EXECUTION
# ==============================================================

if __name__ == "__main__":
    main()
