
# ============================================================
# LOGISTIC REGRESSION MODEL
# Project: Job Acceptance Prediction System
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREPROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "machine_learning" / "models" / "logistic_regression"
GRAPH_DIR = REPORT_DIR / "graphs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION MODEL")
print("=" * 70)

X_train = pd.read_csv(
    PREPROCESSED_DIR / "X_train_processed.csv"
)

X_test = pd.read_csv(
    PREPROCESSED_DIR / "X_test_processed.csv"
)

y_train = pd.read_csv(
    PREPROCESSED_DIR / "y_train.csv"
)["status"]

y_test = pd.read_csv(
    PREPROCESSED_DIR / "y_test.csv"
)["status"]

print("\nDatasets loaded successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ------------------------------------------------------------
# 3. VALIDATE DATA
# ------------------------------------------------------------

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match."
    )

if len(X_train) != len(y_train):
    raise ValueError("Training features and target sizes differ.")

if len(X_test) != len(y_test):
    raise ValueError("Testing features and target sizes differ.")

if X_train.isnull().any().any():
    raise ValueError("Training data contains missing values.")

if X_test.isnull().any().any():
    raise ValueError("Testing data contains missing values.")

if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Training target must contain only 0 and 1.")

if not set(y_test.unique()).issubset({0, 1}):
    raise ValueError("Testing target must contain only 0 and 1.")

print("\nData validation successful!")


# ------------------------------------------------------------
# 4. CREATE LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
    solver="lbfgs"
)

print("\nLogistic Regression model created!")

print("Class weight: balanced")
print("Maximum iterations: 2000")
print("Solver: lbfgs")


# ------------------------------------------------------------
# 5. TRAIN THE MODEL
# ------------------------------------------------------------

print("\nTraining Logistic Regression model...")

logistic_model.fit(
    X_train,
    y_train
)

print("Model training completed successfully!")


# ------------------------------------------------------------
# 6. MAKE PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = logistic_model.predict(X_test)

# Probability of class 1 (Placed)
y_probability = logistic_model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")


# ------------------------------------------------------------
# 7. EVALUATE MODEL PERFORMANCE
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, y_probability
)

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION EVALUATION")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ------------------------------------------------------------
# 8. CLASSIFICATION REPORT
# ------------------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Not Placed", "Placed"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print("-" * 60)
print(report)


# Save classification report
report_dict = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Not Placed", "Placed"],
    output_dict=True,
    zero_division=0
)

classification_df = pd.DataFrame(report_dict).transpose()

classification_df.to_csv(
    REPORT_DIR / "logistic_regression_classification_report.csv"
)


# ------------------------------------------------------------
# 9. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Placed", "Placed"]
)

disp.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. ROC CURVE
# ------------------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title("Logistic Regression - ROC Curve")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "logistic_regression_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 11. SAVE MODEL EVALUATION METRICS
# ------------------------------------------------------------

metrics = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc]
})

metrics.to_csv(
    REPORT_DIR / "logistic_regression_metrics.csv",
    index=False
)


# ------------------------------------------------------------
# 12. SAVE TEST PREDICTIONS
# ------------------------------------------------------------

predictions_df = pd.DataFrame({
    "Actual_Status": y_test.values,
    "Predicted_Status": y_pred,
    "Placed_Probability": y_probability
})

predictions_df.to_csv(
    REPORT_DIR / "logistic_regression_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 13. SAVE TRAINED MODEL
# ------------------------------------------------------------

joblib.dump(
    logistic_model,
    MODEL_DIR / "logistic_regression_model.pkl"
)

print("\nTrained model saved successfully!")


# ------------------------------------------------------------
# 14. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION MODEL SUMMARY")
print("=" * 70)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])

print("\nModel Performance:")
print(metrics.to_string(index=False))

print("\nGenerated files:")
print("- models/logistic_regression_model.pkl")
print("- reports/machine_learning/logistic_regression_metrics.csv")
print("- reports/machine_learning/logistic_regression_classification_report.csv")
print("- reports/machine_learning/logistic_regression_predictions.csv")
print("- graphs/logistic_regression_confusion_matrix.png")
print("- graphs/logistic_regression_roc_curve.png")

print("\nLOGISTIC REGRESSION TRAINING COMPLETED!")
print("=" * 70)
