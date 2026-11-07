
# ============================================================
# JOB PLACEMENT PREDICTION SYSTEM
# GRADIENT BOOSTING CLASSIFIER
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports" / "machine_learning" / "models" / "gradient_boost"

GRAPH_DIR = REPORT_DIR / "graphs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING CLASSIFIER")
print("=" * 70)

X_train = pd.read_csv(DATA_DIR / "X_train_processed.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_processed.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("\nDatasets loaded successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 3. DATA VALIDATION
# ============================================================

if X_train.shape[0] != len(y_train):
    raise ValueError("Training features and target row counts do not match!")

if X_test.shape[0] != len(y_test):
    raise ValueError("Testing features and target row counts do not match!")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature columns do not match!")

if X_train.isnull().sum().sum() > 0:
    raise ValueError("Missing values found in X_train!")

if X_test.isnull().sum().sum() > 0:
    raise ValueError("Missing values found in X_test!")

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training target must contain both 0 and 1!")

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Testing target must contain both 0 and 1!")

print("\nData validation successful!")


# ============================================================
# 4. HANDLE CLASS IMBALANCE
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("\nClass imbalance handled using sample weights!")

print("Not Placed samples receive higher weight.")
print("Placed samples receive higher weight.")


# ============================================================
# 5. CREATE GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

print("\nGradient Boosting model created!")

print("Number of estimators: 200")
print("Learning rate: 0.05")
print("Maximum depth: 3")
print("Minimum samples split: 20")
print("Minimum samples leaf: 10")
print("Subsample: 0.8")


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining Gradient Boosting model...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print("Model training completed successfully!")


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, y_probability)


print("\n" + "=" * 70)
print("GRADIENT BOOSTING MODEL EVALUATION")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Not Placed", "Placed"],
    output_dict=True,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Placed", "Placed"],
        zero_division=0
    )
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    REPORT_DIR / "gradient_boosting_classification_report.csv"
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nCONFUSION MATRIX")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Placed", "Placed"]
)

fig, ax = plt.subplots(figsize=(7, 5))

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("Gradient Boosting - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "gradient_boosting_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_probability)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Gradient Boosting (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Gradient Boosting - ROC Curve")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "gradient_boosting_roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    REPORT_DIR / "gradient_boosting_feature_importance.csv",
    index=False
)

top_features = feature_importance.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")

plt.title("Gradient Boosting - Top 15 Feature Importances")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "gradient_boosting_feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

with open(
    MODEL_DIR / "gradient_boosting_model.pkl",
    "wb"
) as file:

    pickle.dump(model, file)

print("\nTrained model saved successfully!")


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "Actual_Status": y_test.values,
    "Predicted_Status": y_pred,
    "Placed_Probability": y_probability
})

predictions_df["Actual_Status"] = predictions_df[
    "Actual_Status"
].map({0: "Not Placed", 1: "Placed"})

predictions_df["Predicted_Status"] = predictions_df[
    "Predicted_Status"
].map({0: "Not Placed", 1: "Placed"})

predictions_df.to_csv(
    REPORT_DIR / "gradient_boosting_predictions.csv",
    index=False
)


# ============================================================
# 15. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame([{
    "Model": "Gradient Boosting",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc
}])

metrics_df.to_csv(
    REPORT_DIR / "gradient_boosting_metrics.csv",
    index=False
)


# ============================================================
# 16. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING MODEL SUMMARY")
print("=" * 70)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])

print("\nModel Performance:")

print(metrics_df.to_string(index=False))

print("\nGenerated files:")

print("- models/gradient_boosting_model.pkl")

print("- reports/machine_learning/gradient_boosting_metrics.csv")

print("- reports/machine_learning/gradient_boosting_classification_report.csv")

print("- reports/machine_learning/gradient_boosting_predictions.csv")

print("- reports/machine_learning/gradient_boosting_feature_importance.csv")

print("- reports/machine_learning/graphs/gradient_boosting_confusion_matrix.png")

print("- reports/machine_learning/graphs/gradient_boosting_roc_curve.png")

print("- reports/machine_learning/graphs/gradient_boosting_feature_importance.png")

print("\nGRADIENT BOOSTING TRAINING COMPLETED!")

print("=" * 70)
