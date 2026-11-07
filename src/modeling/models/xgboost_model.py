
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# MODEL: XGBOOST CLASSIFIER
# ============================================================

from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "machine_learning" / "models" / "xgboost"
GRAPH_DIR = REPORT_DIR / "graphs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST CLASSIFIER")
print("=" * 70)

X_train = pd.read_csv(DATA_DIR / "X_train_processed.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_processed.csv")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze("columns")

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze("columns")

print("\nDatasets loaded successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 3. DATA VALIDATION
# ============================================================

assert len(X_train) == len(y_train), \
    "Training row mismatch!"

assert len(X_test) == len(y_test), \
    "Testing row mismatch!"

assert list(X_train.columns) == list(X_test.columns), \
    "Training and testing feature columns do not match!"

assert not X_train.isnull().values.any(), \
    "Missing values found in X_train!"

assert not X_test.isnull().values.any(), \
    "Missing values found in X_test!"

assert set(y_train.unique()).issubset({0, 1}), \
    "Unexpected target values in y_train!"

assert set(y_test.unique()).issubset({0, 1}), \
    "Unexpected target values in y_test!"

print("\nData validation successful!")


# ============================================================
# 4. HANDLE CLASS IMBALANCE
# ============================================================

not_placed_count = (y_train == 0).sum()
placed_count = (y_train == 1).sum()

scale_pos_weight = not_placed_count / placed_count

print("\nClass imbalance analysis:")
print("Not Placed records:", not_placed_count)
print("Placed records:", placed_count)
print(f"Scale positive weight: {scale_pos_weight:.4f}")

print("\nMinority class (Placed) receives higher weight.")


# ============================================================
# 5. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\nXGBoost model created!")

print("Number of estimators: 300")
print("Learning rate: 0.05")
print("Maximum depth: 4")
print("Minimum child weight: 5")
print("Subsample: 0.8")
print("Column sample by tree: 0.8")
print(f"Scale positive weight: {scale_pos_weight:.4f}")


# ============================================================
# 6. TRAIN THE MODEL
# ============================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train,
    verbose=False
)

print("Model training completed successfully!")


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

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

roc_auc = roc_auc_score(y_test, y_prob)


print("\n" + "=" * 70)
print("XGBOOST MODEL EVALUATION")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Placed", "Placed"],
        zero_division=0
    )
)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Not Placed", "Placed"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "xgboost_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("XGBoost - ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "xgboost_roc_curve.png",
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
    REPORT_DIR / "xgboost_feature_importance.csv",
    index=False
)

top_features = (
    feature_importance.head(15)
    .sort_values(by="Importance", ascending=True)
)

plt.figure(figsize=(9, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("XGBoost - Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "xgboost_feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model_path = MODEL_DIR / "xgboost_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nTrained model saved successfully!")


# ============================================================
# 14. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{
    "Model": "XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc
}])

metrics_df.to_csv(
    REPORT_DIR / "xgboost_metrics.csv",
    index=False
)


# ============================================================
# 15. SAVE CLASSIFICATION REPORT
# ============================================================

report_df.to_csv(
    REPORT_DIR / "xgboost_classification_report.csv"
)


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "Actual_Status": y_test.values,
    "Predicted_Status": y_pred,
    "Probability_Placed": y_prob
})

predictions_df["Actual_Status"] = (
    predictions_df["Actual_Status"]
    .map({0: "Not Placed", 1: "Placed"})
)

predictions_df["Predicted_Status"] = (
    predictions_df["Predicted_Status"]
    .map({0: "Not Placed", 1: "Placed"})
)

predictions_df.to_csv(
    REPORT_DIR / "xgboost_predictions.csv",
    index=False
)


# ============================================================
# 17. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST MODEL SUMMARY")
print("=" * 70)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])

print("\nModel Performance:")
print(metrics_df.to_string(index=False))

print("\nGenerated files:")
print("- models/xgboost_model.pkl")
print("- reports/machine_learning/xgboost_metrics.csv")
print("- reports/machine_learning/xgboost_classification_report.csv")
print("- reports/machine_learning/xgboost_predictions.csv")
print("- reports/machine_learning/xgboost_feature_importance.csv")
print("- reports/machine_learning/graphs/xgboost_confusion_matrix.png")
print("- reports/machine_learning/graphs/xgboost_roc_curve.png")
print("- reports/machine_learning/graphs/xgboost_feature_importance.png")

print("\nXGBOOST TRAINING COMPLETED!")
print("=" * 70)
