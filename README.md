# Job Acceptance Prediction System

### Machine Learning-Based Job Placement Prediction

A data science and machine learning project that analyzes candidate academic performance, skills, experience, and interview-related attributes to predict job placement outcomes.

---

## 1. Project Overview

The **Job Acceptance Prediction System** is a machine learning project developed to identify the key factors influencing candidate placement outcomes.

The project follows a structured end-to-end data science workflow, starting from raw dataset exploration and data cleaning to feature engineering, model development, evaluation, and deployment through an interactive Streamlit application.

The system aims to predict whether a candidate is likely to be placed or not placed based on available candidate attributes.

## 2. Project Objectives

- Analyze candidate academic, professional, and interview-related information.
- Identify patterns and factors influencing job placement.
- Perform comprehensive data cleaning and validation.
- Conduct Exploratory Data Analysis (EDA).
- Engineer meaningful features to improve predictive performance.
- Apply statistical and machine learning techniques for feature selection.
- Train and compare multiple classification algorithms.
- Evaluate models using appropriate classification metrics.
- Develop an interactive application for placement prediction.

## 3. Dataset Information

| Description | Details |
|---|---|
| Dataset | HR Job Placement Dataset |
| Initial Records | 51,500 |
| Initial Features | 26 |
| Exact Duplicate Records | 1,376 |
| Records After Duplicate Removal | 50,124 |
| Final Records | 50,000 |
| Target Variable | `status` |
| Problem Type | Binary Classification |

**Target variable:**

- `Placed` – Candidate successfully placed.
- `Not Placed` – Candidate not placed.

*Note: Dataset records and feature counts should be updated if the final preprocessing pipeline changes.*

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Statistical Analysis | SciPy |
| Application Development | Streamlit |
| Development Environment | VS Code, PyCharm |
| Version Control | Git, GitHub |

## 5. Project Workflow

The project follows an end-to-end machine learning pipeline.

1. Data Collection
2. Dataset Overview
3. Data Quality Assessment
4. Exploratory Data Analysis
5. Data Cleaning and Preprocessing
6. Missing Value Analysis and Treatment
7. Duplicate Removal
8. Outlier Detection and Analysis
9. Data Validation
10. Feature Engineering
11. Feature Selection
12. Model Development
13. Model Evaluation and Comparison
14. Model Selection
15. Streamlit Application Development

## 6. Data Cleaning and Preprocessing

A modular data cleaning pipeline was developed to improve data quality and maintainability.

Key operations include:

- Duplicate record identification and removal.
- Missing value analysis.
- Missing value treatment using suitable statistical methods.
- Missing value indicator creation.
- Data type validation.
- Outlier detection using statistical techniques.
- Business rule validation.
- Final dataset quality checks.

The cleaning process is divided into individual Python modules instead of a single monolithic script.

## 7. Exploratory Data Analysis

EDA was performed to understand the dataset and identify meaningful relationships.

### Analysis performed

- Dataset overview and statistical summary.
- Data quality analysis.
- Target variable distribution.
- Numerical feature analysis.
- Categorical feature analysis.
- Numerical feature relationships.
- Categorical feature relationships.
- Multivariate analysis.
- Correlation analysis.

The analysis focuses on identifying factors associated with successful job placement.

## 8. Feature Engineering

Meaningful features were developed to improve model learning and interpretability.

| Engineered Feature | Description |
|---|---|
| Experience Category | Groups candidates based on experience |
| Academic Performance Band | Categorizes academic performance |
| Skills Match Level | Represents the candidate's skill alignment |
| Interview Performance Category | Categorizes interview performance |
| Placement Probability Score | Represents estimated placement likelihood |

## 9. Machine Learning Models

Multiple classification algorithms are evaluated and compared to identify the best-performing model.

Models considered include:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Other suitable classification algorithms

### Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

## 10. Model Performance

The following XGBoost results were obtained in the reported model evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 91.79% |
| Precision | 86.93% |
| Recall | 85.76% |
| F1-Score | 86.34% |
| ROC-AUC | 97.62% |

XGBoost demonstrated strong classification performance in the reported evaluation.

*These metrics should be confirmed against the final saved model comparison report before publication.*

## 11. Project Structure

The project follows a modular directory structure to separate source code, datasets, reports, machine learning models, and application components.

```text
Job Acceptance Prediction System/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_cleaning_pipeline/
│   ├── exploratory_data_analysis/
│   ├── feature_engineering/
│   ├── feature_selection/
│   └── machine_learning/
│
├── models/
│
├── reports/
│   ├── data_cleaning/
│   ├── exploratory_data_analysis/
│   ├── feature_engineering/
│   ├── feature_selection/
│   ├── machine_learning/
│   └── project_report/
│
├── app/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above is a high-level overview. Refer to the actual repository for the complete directory structure.*

## 12. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/priyajeyakishore-del/Job-Acceptance-Prediction-System.git
```

### Step 2: Navigate to the project directory

```bash
cd Job-Acceptance-Prediction-System
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run <your_streamlit_entry_file.py>
```

Replace the placeholder with the actual Streamlit application filename.

## 13. Key Learnings

- Building modular data preprocessing pipelines.
- Handling missing values and duplicate records.
- Performing comprehensive exploratory data analysis.
- Applying statistical techniques for outlier detection.
- Developing meaningful features.
- Performing feature selection.
- Training and comparing classification models.
- Evaluating classification performance.
- Developing interactive machine learning applications.
- Managing projects using Git and GitHub.

## 14. Future Enhancements

- Improve model generalization through hyperparameter optimization.
- Add explainable AI techniques such as SHAP.
- Enhance prediction transparency.
- Implement additional candidate-level insights.
- Deploy the application to a cloud platform.
- Monitor model performance and prediction quality.

## 15. Author

**Priya Jeyakishore**

Data Science & Machine Learning Enthusiast

GitHub: [priyajeyakishore-del](https://github.com/priyajeyakishore-del)

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.
