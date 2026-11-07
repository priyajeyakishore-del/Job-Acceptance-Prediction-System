
# ============================================================
# STEP 08: HYPERPARAMETER TUNING
# Job Acceptance Prediction System
# ============================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"

X_TRAIN_FILE = DATA_DIR / "X_train_processed.csv"
Y_TRAIN_FILE = DATA_DIR / "y_train.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "machine_learning"
    / "hyperparameter_tuning"
)

MODEL_DIR = OUTPUT_DIR / "best_models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 75)
print("STEP 08: HYPERPARAMETER TUNING")
print("=" * 75)

print("\nLoading training data...")

X_train = pd.read_csv(X_TRAIN_FILE)
y_train_df = pd.read_csv(Y_TRAIN_FILE)

# Remove accidental CSV index columns
X_train = X_train.loc[
    :, ~X_train.columns.str.startswith("Unnamed:")
]

y_train_df = y_train_df.loc[
    :, ~y_train_df.columns.str.startswith("Unnamed:")
]

# Target must contain one column
if y_train_df.shape[1] != 1:
    raise ValueError(
        "y_train.csv must contain exactly one target column."
    )

y_train = y_train_df.iloc[:, 0]

# Convert text labels if required
if y_train.dtype == "object":
    y_train = y_train.astype(str).str.strip().map({
        "Placed": 1,
        "Not Placed": 0
    })

if y_train.isna().any():
    raise ValueError(
        "Target contains missing or unrecognized values."
    )

y_train = y_train.astype(int)

if set(y_train.unique()) != {0, 1}:
    raise ValueError(
        "Expected binary target values: 0 and 1."
    )

if X_train.isnull().any().any():
    raise ValueError(
        "X_train contains missing values. "
        "Check the preprocessing output."
    )

if len(X_train) != len(y_train):
    raise ValueError(
        "X_train and y_train have different row counts."
    )

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print("\nTarget distribution:")
print(y_train.value_counts().sort_index())


# ============================================================
# 3. CROSS-VALIDATION CONFIGURATION
# ============================================================

cv_strategy = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Class weights are calculated from training data only.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Positive class: Placed = 1
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 4. DEFINE MODELS AND PARAMETER SEARCH SPACES
# ============================================================

model_configs = {

    "Logistic Regression": {
        "model": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ),
        "params": {
            "C": [0.01, 0.1, 1, 5, 10, 20],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"]
        },
        "fit_params": {}
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "params": {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [4, 6, 8, 10, 12, 15, None],
            "min_samples_split": [10, 20, 30, 50],
            "min_samples_leaf": [5, 10, 15, 20],
            "max_features": [None, "sqrt", "log2"]
        },
        "fit_params": {}
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ),
        "params": {
            "n_estimators": [100, 200, 300, 400],
            "criterion": ["gini", "entropy"],
            "max_depth": [8, 10, 12, 15, 20, None],
            "min_samples_split": [10, 20, 30],
            "min_samples_leaf": [5, 10, 15],
            "max_features": ["sqrt", "log2"]
        },
        "fit_params": {}
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(
            random_state=42
        ),
        "params": {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_samples_split": [10, 20, 30],
            "min_samples_leaf": [5, 10, 15],
            "subsample": [0.7, 0.8, 0.9, 1.0]
        },
        "fit_params": {
            "sample_weight": sample_weights
        }
    },

    "AdaBoost": {
        "model": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=1,
                random_state=42
            ),
            random_state=42
        ),
        "params": {
            "n_estimators": [50, 100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
            "estimator__max_depth": [1, 2]
        },
        "fit_params": {
            "sample_weight": sample_weights
        }
    },

    "XGBoost": {
        "model": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1
        ),
        "params": {
            "n_estimators": [100, 200, 300, 400],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "min_child_weight": [1, 3, 5, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "scale_pos_weight": [1.0, 2.0, 2.3, 3.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0]
        },
        "fit_params": {}
    }
}


# ============================================================
# 5. PERFORM RANDOMIZED SEARCH
# ============================================================

results = []
best_models = {}
best_parameters = {}

for model_name, config in model_configs.items():

    print("\n" + "=" * 75)
    print(f"TUNING: {model_name.upper()}")
    print("=" * 75)

    search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=10,
        scoring=SCORING,
        refit="f1",
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise"
    )

    try:
        search.fit(
            X_train,
            y_train,
            **config["fit_params"]
        )

        best_model = search.best_estimator_

        best_models[model_name] = best_model
        best_parameters[model_name] = search.best_params_

        best_index = search.best_index_

        cv_accuracy = search.cv_results_[
            "mean_test_accuracy"
        ][best_index]

        cv_precision = search.cv_results_[
            "mean_test_precision"
        ][best_index]

        cv_recall = search.cv_results_[
            "mean_test_recall"
        ][best_index]

        cv_f1 = search.cv_results_[
            "mean_test_f1"
        ][best_index]

        cv_roc_auc = search.cv_results_[
            "mean_test_roc_auc"
        ][best_index]

        results.append({
            "Model": model_name,
            "Best_CV_Accuracy": cv_accuracy,
            "Best_CV_Precision": cv_precision,
            "Best_CV_Recall": cv_recall,
            "Best_CV_F1": cv_f1,
            "Best_CV_ROC_AUC": cv_roc_auc,
            "Best_Parameters": json.dumps(
                search.best_params_,
                default=str
            )
        })

        # Save the best estimator fitted on all training data
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )

        model_path = MODEL_DIR / f"{safe_name}_tuned.pkl"

        joblib.dump(best_model, model_path)

        print("\nBest parameters:")
        print(search.best_params_)

        print(f"\nBest CV F1-score: {cv_f1:.4f}")
        print(f"Saved model: {model_path}")

    except Exception as error:
        print(f"\nError while tuning {model_name}: {error}")


# ============================================================
# 6. SAVE TUNING RESULTS
# ============================================================

if not results:
    raise RuntimeError(
        "No models were successfully tuned. "
        "Check the errors above."
    )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Best_CV_F1",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "Rank",
    range(1, len(results_df) + 1)
)

results_file = OUTPUT_DIR / "hyperparameter_tuning_results.csv"

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# 7. SAVE BEST PARAMETERS
# ============================================================

parameters_file = OUTPUT_DIR / "best_hyperparameters.json"

with open(parameters_file, "w", encoding="utf-8") as file:
    json.dump(
        best_parameters,
        file,
        indent=4,
        default=str
    )


# ============================================================
# 8. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 75)

display_columns = [
    "Rank",
    "Model",
    "Best_CV_Accuracy",
    "Best_CV_Precision",
    "Best_CV_Recall",
    "Best_CV_F1",
    "Best_CV_ROC_AUC"
]

print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

best_model_row = results_df.iloc[0]

print("\n" + "=" * 75)
print("BEST MODEL AFTER HYPERPARAMETER TUNING")
print("=" * 75)

print(f"Model: {best_model_row['Model']}")
print(f"Cross-validation F1: {best_model_row['Best_CV_F1']:.4f}")
print(f"Cross-validation ROC-AUC: {best_model_row['Best_CV_ROC_AUC']:.4f}")

print("\nBest parameters:")
print(best_model_row["Best_Parameters"])


# ============================================================
# 9. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("STEP 08 COMPLETED SUCCESSFULLY")
print("=" * 75)

print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"Best models saved to: {MODEL_DIR}")
print(f"Results CSV: {results_file}")
print(f"Parameters JSON: {parameters_file}")
