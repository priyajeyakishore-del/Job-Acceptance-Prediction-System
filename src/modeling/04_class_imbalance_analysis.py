
# ============================================================
# STEP 4: CLASS IMBALANCE ANALYSIS
# Project: Job Acceptance Prediction System
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

REPORT_DIR = PROJECT_ROOT / "reports" / "machine_learning" / "class_imbalance"
GRAPH_DIR = REPORT_DIR / "graphs" 

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD TARGET DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4: CLASS IMBALANCE ANALYSIS")
print("=" * 70)

y_train = pd.read_csv(SPLIT_DIR / "y_train.csv")
y_test = pd.read_csv(SPLIT_DIR / "y_test.csv")

if "status" not in y_train.columns or "status" not in y_test.columns:
    raise ValueError("Expected target column 'status' in both files.")

y_train = y_train["status"]
y_test = y_test["status"]

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError("Target data contains missing values.")

if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Training target must contain only 0 and 1.")

if not set(y_test.unique()).issubset({0, 1}):
    raise ValueError("Testing target must contain only 0 and 1.")

print("\nTarget datasets loaded successfully!")
print("Training records:", len(y_train))
print("Testing records:", len(y_test))


# ------------------------------------------------------------
# 3. DEFINE TARGET LABELS
# ------------------------------------------------------------

target_labels = {
    1: "Placed",
    0: "Not Placed"
}


# ------------------------------------------------------------
# 4. CREATE CLASS DISTRIBUTION SUMMARY
# ------------------------------------------------------------

def calculate_class_distribution(target, dataset_name):

    counts = target.value_counts().reindex([0, 1], fill_value=0)

    percentages = (
        target.value_counts(normalize=True)
        .reindex([0, 1], fill_value=0) * 100
    )

    summary = pd.DataFrame({
        "Dataset": dataset_name,
        "Class": [target_labels[0], target_labels[1]],
        "Class_Value": [0, 1],
        "Count": [counts[0], counts[1]],
        "Percentage": [
            round(percentages[0], 2),
            round(percentages[1], 2)
        ]
    })

    return summary


train_summary = calculate_class_distribution(
    y_train, "Training"
)

test_summary = calculate_class_distribution(
    y_test, "Testing"
)

distribution_summary = pd.concat(
    [train_summary, test_summary],
    ignore_index=True
)


# ------------------------------------------------------------
# 5. DISPLAY CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(distribution_summary.to_string(index=False))


# ------------------------------------------------------------
# 6. CALCULATE IMBALANCE RATIO
# ------------------------------------------------------------

train_counts = y_train.value_counts()

majority_count = train_counts.max()
minority_count = train_counts.min()

if minority_count == 0:
    imbalance_ratio = float("inf")
else:
    imbalance_ratio = majority_count / minority_count

print("\n" + "=" * 70)
print("IMBALANCE RATIO")
print("=" * 70)

print("Majority class count:", majority_count)
print("Minority class count:", minority_count)
print("Imbalance ratio:", round(imbalance_ratio, 2))

if imbalance_ratio < 1.5:
    imbalance_status = "Approximately balanced"
elif imbalance_ratio < 2:
    imbalance_status = "Mild imbalance"
else:
    imbalance_status = "Class imbalance detected"

print("Analysis result:", imbalance_status)


# ------------------------------------------------------------
# 7. COMPARE TRAINING AND TESTING DISTRIBUTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING VS TESTING DISTRIBUTION")
print("=" * 70)

comparison = pd.DataFrame({
    "Training_Percentage": (
        y_train.value_counts(normalize=True)
        .reindex([0, 1], fill_value=0) * 100
    ),
    "Testing_Percentage": (
        y_test.value_counts(normalize=True)
        .reindex([0, 1], fill_value=0) * 100
    )
})

comparison.index = ["Not Placed", "Placed"]

print(comparison.round(2))


# ------------------------------------------------------------
# 8. VISUALIZE CLASS DISTRIBUTION
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

datasets = [
    (y_train, "Training Data"),
    (y_test, "Testing Data")
]

for ax, (target, title) in zip(axes, datasets):

    counts = (
        target.value_counts()
        .reindex([0, 1], fill_value=0)
    )

    labels = ["Not Placed", "Placed"]

    bars = ax.bar(labels, counts.values)

    ax.set_title(title)
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=10)

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom"
        )

plt.suptitle("Class Distribution Analysis")
plt.tight_layout()

chart_path = GRAPH_DIR / "class_distribution.png"

plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nClass distribution chart saved:")
print(chart_path)


# ------------------------------------------------------------
# 9. SAVE CLASS DISTRIBUTION REPORT
# ------------------------------------------------------------

report_path = REPORT_DIR / "class_distribution_report.csv"

distribution_summary.to_csv(
    report_path,
    index=False
)

comparison_path = REPORT_DIR / "train_test_class_comparison.csv"

comparison.to_csv(comparison_path)


# ------------------------------------------------------------
# 10. SAVE ANALYSIS SUMMARY
# ------------------------------------------------------------

summary_path = REPORT_DIR / "class_imbalance_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("CLASS IMBALANCE ANALYSIS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Training records: {len(y_train)}\n")
    file.write(f"Testing records: {len(y_test)}\n")

    file.write(
        f"Majority class count: {majority_count}\n"
    )

    file.write(
        f"Minority class count: {minority_count}\n"
    )

    file.write(
        f"Imbalance ratio: {imbalance_ratio:.2f}\n"
    )

    file.write(
        f"Analysis result: {imbalance_status}\n\n"
    )

    file.write("CLASS DISTRIBUTION:\n")
    file.write(distribution_summary.to_string(index=False))


# ------------------------------------------------------------
# 11. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS IMBALANCE ANALYSIS SUMMARY")
print("=" * 70)

print("Imbalance ratio:", round(imbalance_ratio, 2))
print("Result:", imbalance_status)

print("\nGenerated files:")
print("1.", report_path)
print("2.", comparison_path)
print("3.", summary_path)
print("4.", chart_path)

print("\nSTEP 4 COMPLETED SUCCESSFULLY!")
print("=" * 70)
